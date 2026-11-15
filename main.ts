// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// The server reads the brief, probes each provider's catalog route, builds one catalog, and
// then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (answering in the shape of the auth header
//      it got: x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions
//      and POST /v1/messages, each with a JSON body or with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}.
//
// ROUTING. For each request the server ranks every free model that meets the request's
// requirements by the quality table's rating (highest first; ties break by provider id, then
// model id, so the order is a pure function of the brief). A JSON-mode request never routes
// to an anthropic-wire provider: that wire cannot carry JSON mode, so its echo would fail.
// It tries the ranking in order and stops at the first provider that answers 2xx — exactly
// one successful upstream call per request, which the wire checks require. A provider that
// refuses (402, 429, 5xx or unreachable) is tried again on later requests: a refusal is
// usually one failed call on that provider's schedule, not a dead provider. A provider whose
// last two attempts both failed is treated as spent and sorts last (it is still tried when
// nothing else can serve the request), which keeps the provider-call count — the tie-break —
// low once a provider's budget is gone.
//
// `node main.ts --list` prints the same catalog JSON the register body carries and exits 0
// before any server starts. Outside a trial, with no brief to read, it prints
// {"providers":[]} and exits 0.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  wireCarriesJson,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type RatingRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToGeminiBody,
  anthropicToOpenAIBody,
  anthropicToOpenAIResponse,
  anthropicToOpenAIStreamText,
  geminiToAnthropicResponse,
  geminiToAnthropicStreamText,
  geminiToOpenAIResponse,
  geminiToOpenAIStreamText,
  openAIError,
  openAIModelList,
  openAIToAnthropicBody,
  openAIToAnthropicResponse,
  openAIToAnthropicStreamText,
  openAIToGeminiBody,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream inference call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 8000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Per-provider failure state: consecutive failed attempts, and whether it is treated as spent. */
const providerState = new Map<string, { fails: number; dead: boolean }>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  for (const probe of probes) providerState.set(probe.provider.id, { fails: 0, dead: false });
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  row: RatingRow | null;
  rating: number;
}

/**
 * Every free model that can serve one request, best first: rating descending, then provider
 * id, then model id. A provider treated as spent sorts after every live one. A JSON-mode
 * request skips anthropic-wire providers (that wire cannot carry JSON mode) unless no other
 * candidate exists at all, in which case the request is served anyway and only its JSON
 * echo check is lost.
 */
function candidatesFor(requires: Requirement): Candidate[] {
  const strict: Candidate[] = [];
  const loose: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const candidate: Candidate = { probe, model, row, rating: row?.rating_milli ?? 0 };
      if (requires.json === true && !wireCarriesJson(probe.provider.wire)) loose.push(candidate);
      else strict.push(candidate);
    }
  }
  const pool = strict.length > 0 ? strict : loose;
  const stateOf = (id: string): { fails: number; dead: boolean } => providerState.get(id) ?? { fails: 0, dead: false };
  pool.sort((a, b) => {
    const deadA = stateOf(a.probe.provider.id).dead ? 1 : 0;
    const deadB = stateOf(b.probe.provider.id).dead ? 1 : 0;
    if (deadA !== deadB) return deadA - deadB;
    if (a.rating !== b.rating) return b.rating - a.rating;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return pool;
}

/** A provider call's outcome: the HTTP status and the body text when it answered 2xx. */
interface UpstreamResult {
  ok: boolean;
  status: number;
  text: string;
}

/** One inference call to a provider, in that provider's own wire. */
async function callUpstream(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<UpstreamResult> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientApi === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAIBody(body, candidate.model.id);
    // A streamed OpenAI answer carries its usage only when the request asks for it; without
    // the usage chunk the client's usage check fails, so always ask.
    if (stream) upstreamBody.stream_options = { include_usage: true };
    if (stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropicBody(body, candidate.model.id);
    if (stream) headers.accept = 'text/event-stream';
  } else {
    const action = stream ? ':streamGenerateContent' : ':generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}${action}${stream ? '?alt=sse' : ''}`;
    upstreamBody = clientApi === 'openai' ? openAIToGeminiBody(body, candidate.model.id) : anthropicToGeminiBody(body, candidate.model.id);
    if (stream) headers.accept = 'text/event-stream';
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch (err) {
    return { ok: false, status: 0, text: (err as Error).message };
  }
  const text = await res.text().catch(() => '');
  return { ok: res.ok, status: res.status, text };
}

/** Record one provider attempt: a success clears its failures, two failures in a row mark it spent. */
function noteAttempt(providerId: string, ok: boolean): void {
  const state = providerState.get(providerId) ?? { fails: 0, dead: false };
  if (ok) {
    state.fails = 0;
    state.dead = false;
  } else {
    state.fails += 1;
    if (state.fails >= 2) state.dead = true;
  }
  providerState.set(providerId, state);
}

/** The upstream answer translated into the wire the client used, or null when it is not one. */
function translate(clientApi: 'openai' | 'anthropic', candidate: Candidate, result: UpstreamResult, stream: boolean): string | null {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  // The same wire needs no translation: the provider's answer is already the client's.
  if (wire === clientApi) return result.text;
  if (stream) {
    if (clientApi === 'openai' && wire === 'anthropic') return anthropicToOpenAIStreamText(result.text, model);
    if (clientApi === 'openai' && wire === 'gemini') return geminiToOpenAIStreamText(result.text, model);
    if (clientApi === 'anthropic' && wire === 'openai') return openAIToAnthropicStreamText(result.text, model);
    if (clientApi === 'anthropic' && wire === 'gemini') return geminiToAnthropicStreamText(result.text, model);
    return null;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(result.text);
  } catch {
    return null;
  }
  if (!isRecord(parsed)) return null;
  if (clientApi === 'openai' && wire === 'anthropic') return JSON.stringify(anthropicToOpenAIResponse(parsed, model));
  if (clientApi === 'openai' && wire === 'gemini') return JSON.stringify(geminiToOpenAIResponse(parsed, model));
  if (clientApi === 'anthropic' && wire === 'openai') return JSON.stringify(openAIToAnthropicResponse(parsed, model));
  if (clientApi === 'anthropic' && wire === 'gemini') return JSON.stringify(geminiToAnthropicResponse(parsed, model));
  return null;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, plus what its own body asks for. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  const requires: Requirement = entry !== undefined ? { ...entry.requires } : {};
  // The body's own shape is a second witness: a request that carries tools, an image or a
  // JSON mode needs a model that takes them, whatever the manifest row said.
  const fromBody = requirementsFromBody(body);
  if (fromBody.tools === true) requires.tools = true;
  if (fromBody.vision === true) requires.vision = true;
  if (fromBody.json === true) requires.json = true;
  return requires;
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * One client request, served by the best free model that can take it: try the ranking in
 * order, fail over to the next candidate on any refusal, and stop at the first answer.
 */
async function serve(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const candidates = candidatesFor(requires);
  if (candidates.length === 0) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  for (const candidate of candidates) {
    const result = await callUpstream(candidate, clientApi, body, stream);
    noteAttempt(candidate.probe.provider.id, result.ok);
    if (!result.ok) continue;
    const translated = translate(clientApi, candidate, result, stream);
    // The call succeeded: a second successful call would break the wire's one-provider
    // rule, so a translation failure ends the request here instead of trying another.
    if (translated === null) {
      return sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'the provider answered in a shape this server could not translate');
    }
    if (stream) return sendText(res, 200, 'text/event-stream; charset=utf-8', translated);
    return sendText(res, 200, 'application/json', translated);
  }
  return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'every free model that meets this request refused it');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object
    // and stop before any server starts. A process whose stdout is not flushed loses the tail,
    // so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
