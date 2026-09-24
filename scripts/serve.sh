#!/usr/bin/env bash
# bash scripts/serve.sh <adapter_dir with plan/ action/ reflect/> [prelrshared|rebaseshared]  (omitted: NonShared)
set -euo pipefail
ADAPTERS=$(realpath "${1:?adapter_dir}"); METHOD=${2:-}
LOG=$(realpath -m "${LOG_DIR:-logs}"); mkdir -p "$LOG"
export LOGDIR=$LOG    # FastChat writes its own logs here
CP=${CONTROLLER_PORT:-21001}; AP=${API_PORT:-8888}; WP=${WORKER_PORT:-31022}

python -m fastchat.serve.controller --host localhost --port $CP > "$LOG/controller.log" 2>&1 &
sleep 5
python -m fastchat.serve.openai_api_server --host localhost --port $AP \
    --controller-address http://localhost:$CP > "$LOG/api.log" 2>&1 &
# run from the adapter folder: FastChat names each LoRA after its path, and names may not contain dots
cd "$ADAPTERS"
PEFT_SHARE_BASE_WEIGHTS=true PRECACHE_METHOD=$METHOD \
python -m fastchat.serve.multi_model_worker --host localhost --port $WP \
    --worker-address http://localhost:$WP --controller-address http://localhost:$CP \
    --dtype float16 --limit-worker-concurrency 1 \
    --model-path plan --model-names plan \
    --model-path action --model-names action \
    --model-path reflect --model-names reflect > "$LOG/worker.log" 2>&1 &
WORKER=$!
echo "waiting for worker..."
until curl -sf -X POST http://localhost:$WP/worker_get_status > /dev/null; do
    kill -0 $WORKER 2>/dev/null || { echo "worker failed, see $LOG/worker.log"; kill 0; }
    sleep 5
done
echo "ready: ${METHOD:-nonshared} on $ADAPTERS"
wait
