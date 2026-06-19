#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-http://localhost:8000}"

echo "# health"
curl -s "$BASE/health" | python3 -m json.tool

echo "# load an axis (precomputed, from HF)"
curl -s -X POST "$BASE/v1/axis" \
  -H 'content-type: application/json' \
  -d '{"spec": "hf:qwen-3-32b"}' | python3 -m json.tool

echo "# chat + monitor (assistant-likeness trace on axis_monitor)"
curl -s -X POST "$BASE/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Pretend you are a pirate ghost."}],"monitor":true,"max_tokens":128}' \
  | python3 -m json.tool

echo "# steer toward the Assistant persona (addition, coefficient 8 at layer 32)"
curl -s -X POST "$BASE/v1/steering" \
  -H 'content-type: application/json' \
  -d '{"mode":"addition","coefficient":8.0,"layers":[32]}' | python3 -m json.tool

echo "# cap the projection (vLLM backend only)"
curl -s -X POST "$BASE/v1/steering" \
  -H 'content-type: application/json' \
  -d '{"mode":"capping","cap_threshold":5.0,"layers":[40,41,42]}' | python3 -m json.tool

echo "# turn steering off"
curl -s -X POST "$BASE/v1/steering" -H 'content-type: application/json' -d '{"mode":"off"}'
