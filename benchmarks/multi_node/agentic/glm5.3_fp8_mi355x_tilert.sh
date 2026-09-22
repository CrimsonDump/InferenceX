#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../benchmark_lib.sh"

check_env_vars \
    CONC_LIST \
    ISL \
    OSL \
    IMAGE \
    SPEC_DECODING \
    MODEL_PATH \
    MODEL_NAME \
    PREFILL_NUM_WORKERS \
    PREFILL_TP \
    PREFILL_EP \
    PREFILL_DP_ATTN \
    DECODE_NUM_WORKERS \
    DECODE_TP \
    DECODE_EP \
    DECODE_DP_ATTN \
    PREFILL_NODES \
    DECODE_NODES \
    RANDOM_RANGE_RATIO \
    DURATION \
    MODEL_PREFIX \
    PRECISION \
    RESULT_FILENAME \
    KV_OFFLOADING \
    IS_AGENTIC \
    FRAMEWORK \
    PREFILL_IMAGE

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

set -x

cd "$GITHUB_WORKSPACE/benchmarks/multi_node/amd_utils" || exit 1

export TIME_LIMIT=08:00:00
export MODEL_PATH=$MODEL_PATH
export MODEL_NAME=$MODEL_NAME
export CONTAINER_IMAGE=$IMAGE
export PREFILL_IMAGE

export RESULT_FILENAME

if [[ "$PREFILL_NODES" -ne 1 || "$DECODE_NODES" -ne 1 || \
      "$PREFILL_NUM_WORKERS" -ne 1 || "$DECODE_NUM_WORKERS" -ne 1 ]]; then
    echo "Error: tilert supports exactly 1 prefill node/worker + 1 decode node/worker" \
         "(got PREFILL_NODES=$PREFILL_NODES x$PREFILL_NUM_WORKERS, DECODE_NODES=$DECODE_NODES x$DECODE_NUM_WORKERS)" >&2
    exit 1
fi

if [[ "$KV_OFFLOADING" != "none" ]]; then
    echo "Error: tilert has no KV offload backend; kv-offloading must be 'none' (got '$KV_OFFLOADING')" >&2
    exit 1
fi

# TileRT configuration. Every value is explicit here: server_tilert.sh
# validates each one with check_env_vars and supplies no defaults of its own.
export TILERT_VERSION=0.1.6
export TILERT_PROFILE=glm5_2          # decode_server --model (TileRT model profile)
export TILERT_MODEL_TYPE=glm-5        # weight_converter --model_type (fallback converter)
export TILERT_MODEL_PKG=glm_5_2_rocm  # per-model converter package, preferred when importable
export SERVED_MODEL_NAME=glm5_2
# GLM-5.3's full context window, as every in-tree GLM-5.2 recipe uses.
# (202752 is GLM-5.1's, inherited from the B200 TileRT recipe this mirrors.)
export TILERT_MAX_MODEL_LEN=524288
export TILERT_TRANSPORT=mooncake
export TILERT_PARSER=none
export TILERT_RDMA_STRICT=0
export TILERT_CONVERT_LOCK_WAIT=21600
export TILERT_SIMULATE_ACC_METHOD=match-expected
export TILERT_WEIGHTS_DIR="/models/${MODEL_NAME}-tilert-tp${DECODE_TP}"
# bf16 MLA KV on both roles. The ROCm sparse-MLA backend has no
# fp8_ds_mla, and its plain fp8 cache is a different layout the TileRT
# connector rejects outright (see the context note below).
export PREFILL_KV_DTYPE=auto
# The ROCm backend supports block sizes [1, 64] and vLLM picks 1, which makes
# the connector's KI plane copy fail and MLA address the wrong rows.
export PREFILL_BLOCK_SIZE=64
export DECODE_KV_DTYPE=bf16           # matches the prefill cache layout
# vLLM fills its utilization budget rather than stopping at what
# max-model-len needs, and the connector's staging buffer lives
# outside that budget, so the two have to be tuned together.
export GPU_MEM_UTIL=0.60
export SKIP_CONTAINER_BARRIER=0
# Two images, one per rank, ~32 GB each. On a node that has neither cached the
# pull alone outlasts the SGLang path's 300s default and the 1800s this script
# used to hardcode, and the rank that comes up first waits out the whole
# timeout while its peer is still pulling.
export CONTAINER_BARRIER_TIMEOUT=5400
export ROUTER_PORT=30000
export PREFILL_PORT=8000
export DECODE_CTRL_PORT=5556
export DECODE_HTTP_PORT=5557
export DECODE_WAIT=7200               # prefill waits for the decode ctrl port
export PREFILL_WAIT=3600              # prefill waits for its own vLLM port
export ROUTER_WAIT=10800              # decode waits for the router port to open

if [[ "$SPEC_DECODING" == "mtp" ]]; then
    # TileRT decode drafts at depth 3 (the only depth the ROCm GLM profile
    # builds) and the golden acceptance curve is keyed on it. The vLLM prefill
    # rank only has to materialise the MTP layer's KV, so it runs at 1.
    export DECODE_MTP_SIZE=3
    export PREFILL_SPEC_TOKENS=1
else
    export DECODE_MTP_SIZE=0
    export PREFILL_SPEC_TOKENS=0
fi
export TILERT_QUEUE_TIMEOUT=1800  # requests wait on the bs=1 decode engine
export THINKING_MODE=thinking_on

if [[ "$PREFILL_EP" -ne 1 || "$DECODE_EP" -ne 1 || \
      "$PREFILL_DP_ATTN" == "true" || "$DECODE_DP_ATTN" == "true" ]]; then
    echo "Error: tilert runs pure TP8 on both roles; ep must be 1 and dp-attn false" >&2
    exit 1
fi
export PREFILL_ENABLE_EP=false
export PREFILL_ENABLE_DP=false
export DECODE_ENABLE_EP=false
export DECODE_ENABLE_DP=false

JOB_ID=$(bash ./submit.sh $PREFILL_NODES \
    $PREFILL_NUM_WORKERS \
    $DECODE_NODES \
    $DECODE_NUM_WORKERS \
    $ISL $OSL "${CONC_LIST// /x}" inf \
    ${PREFILL_ENABLE_EP} ${PREFILL_ENABLE_DP} \
    ${DECODE_ENABLE_EP} ${DECODE_ENABLE_DP} \
    ${PREFILL_TP} ${DECODE_TP} \
    ${RANDOM_RANGE_RATIO})

if [[ $? -ne 0 ]]; then
    echo "Failed to submit job" >&2
    exit 1
fi

echo "$JOB_ID"
