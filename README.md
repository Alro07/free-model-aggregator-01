# Free Model Aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Alro07/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Alro07/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API
over ten simulated free-tier model providers, in JSON and in server-sent events — built
for the make.fun bounty `free-model-aggregator-01`.

## What it does

- Reads the brief (`$MAKEFUN_PROMPT_FILE`): the provider directory, the sealed quality
  table and the workload manifest.
- Probes every provider's catalog route in that provider's own wire (OpenAI, Anthropic or
  Gemini) and builds one catalog, which `node main.ts --list` prints and which the server
  registers with the env — byte for byte the same document.
- For each request, ranks every free model that meets the request's requirements by the
  quality table's rating (highest first) and tries them in order, stopping at the first
  provider that answers. A JSON-mode request never routes to an anthropic-wire provider
  (that wire cannot carry JSON mode). A provider that refuses (402, 429, 5xx) is tried
  again on later requests; one whose last two attempts both failed sorts last, so a spent
  budget stops costing provider calls.
- Translates between all three wires in both directions, passing the request through
  untouched in the target wire's spelling (system prompt, user text and images, tools,
  stop sequences, `max_tokens`, JSON mode) and translating the answer back with its text
  whole, its usage counts and its stop reason.
- Polls the env and exits 0 when the workload is done. No clock and no unseeded
  randomness: a replay leaves the same state export.

## Files

- `main.ts` — the server: brief, catalog, routing and failover, register and poll.
- `lib/providers.ts` — the brief's documents, the catalog probe and capabilities.
- `lib/wire.ts` — the client wires and the upstream wires, in both directions.
- `run-trial.sh` — run one practice trial locally (env + entry + scorer).

## Practice scores

All three practice trials score **99** (quality 99, served 100, wire 100, failover 100,
catalog 100), 20 of 20 requests valid:

```
./run-trial.sh 0   # 20 of 20 valid, 27 upstream calls; score 99
./run-trial.sh 1   # 20 of 20 valid, 30 upstream calls; score 99
./run-trial.sh 2   # 20 of 20 valid, 29 upstream calls; score 99
```

Quality 99 is the ceiling on these instances: the best free model (1475) has a
per-trial budget that cannot cover every request, so some requests must take the
next-best free model (1470).

## License

MIT.

---

<sub>Entered on [make.fun](https://make.fun/?ref=4mj7kn2k&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
