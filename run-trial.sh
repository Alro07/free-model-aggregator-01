#!/bin/bash
# Run one practice trial locally, the way the grader does (process driver):
#   ./run-trial.sh 0   — trial-0
# The env serves the providers and drives the 20-request client; the entry (this repo's
# main.ts) registers, serves, and exits 0 when done; the env exports state.json on SIGTERM;
# the scorer scores it against the trial's reference.
set -u
TRIAL="${1:-0}"
KIT="$HOME/workspace/makefun/free-model-aggregator-01-kit"
REPO="$HOME/workspace/makefun/free-model-aggregator-01"
TDIR="$KIT/practice/trial-$TRIAL"
STATE="/tmp/fma-trial-$TRIAL"
SCORES="/tmp/fma-scores-$TRIAL"
rm -rf "$STATE" "$SCORES"
mkdir -p "$STATE" "$SCORES"

# The env on 127.0.0.1:8080 (process driver: MAKEFUN_STATE_DIR set).
MAKEFUN_ENV_DATA_DIR="$TDIR/input/env" MAKEFUN_STATE_DIR="$STATE" MAKEFUN_ENV_PORT=8080 \
  node "$KIT/program/env/server.mjs" > "$STATE/env.log" 2>&1 &
ENV_PID=$!
# Wait for the env's ready file.
for i in $(seq 1 50); do [ -f "$STATE/ready" ] && break; sleep 0.1; done
if [ ! -f "$STATE/ready" ]; then echo "env did not start:"; cat "$STATE/env.log"; kill $ENV_PID 2>/dev/null; exit 1; fi

# The entry.
cd "$REPO"
MAKEFUN_ENV_URL="http://127.0.0.1:8080" MAKEFUN_PROMPT_FILE="$TDIR/input/prompt.txt" \
  node main.ts > "$STATE/entry.log" 2>&1
ENTRY_RC=$?
echo "entry exited: $ENTRY_RC"
sleep 0.5
kill -TERM $ENV_PID 2>/dev/null
wait $ENV_PID 2>/dev/null
echo "--- entry log ---"
cat "$STATE/entry.log"
echo "--- env log ---"
cat "$STATE/env.log"
if [ ! -f "$STATE/export/state.json" ]; then echo "NO state.json exported"; exit 1; fi
echo "--- score ---"
MAKEFUN_OUTPUT_DIR="$STATE/export" MAKEFUN_REFERENCE_DIR="$TDIR/reference" MAKEFUN_SCORES_DIR="$SCORES" \
  node "$KIT/program/scorer/score.mjs"
echo "--- scores.json ---"
cat "$SCORES/scores.json"
echo "--- per-request ---"
python3 - "$STATE/export/state.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
for r in s['requests']:
    bad = [c['name'] for c in r['checks'] if not c['pass']]
    print(r['id'], 'valid' if r['valid'] else 'INVALID', r['provider'], r['model'], 'rating', r['rating_milli'], 'calls', r['upstream_calls'], 'failures', len(r['failures']), ('FAILED: ' + ','.join(bad)) if bad else '')
    if not r['valid']:
        for c in r['checks']:
            if not c['pass']: print('   ', c['name'], '-', c['detail'])
print('catalog_route:', s['catalog_route'])
print('totals:', s['totals'])
EOF
