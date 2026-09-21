#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    CONC_LIST \
    ISL \
    OSL \
    IMAGE \
    SPEC_DECODING \
    MODEL_PATH \
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
    FRAMEWORK \
    PREFILL_IMAGE

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

set -x

cd "$GITHUB_WORKSPACE/benchmarks/multi_node/amd_utils" || exit 1

export TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
export MODEL_PATH=$MODEL_PATH
export MODEL_NAME=$MODEL_NAME
export CONTAINER_IMAGE=$IMAGE
export PREFILL_IMAGE

if [[ "$PREFILL_NODES" -ne 1 || "$DECODE_NODES" -ne 1 || \
      "$PREFILL_NUM_WORKERS" -ne 1 || "$DECODE_NUM_WORKERS" -ne 1 ]]; then
    echo "Error: tilert supports exactly 1 prefill node/worker + 1 decode node/worker" \
         "(got PREFILL_NODES=$PREFILL_NODES x$PREFILL_NUM_WORKERS, DECODE_NODES=$DECODE_NODES x$DECODE_NUM_WORKERS)" >&2
    exit 1
fi

export TILERT_VERSION="${TILERT_VERSION:-0.1.6}"
export TILERT_PROFILE="${TILERT_PROFILE:-glm5_2}"
export TILERT_MODEL_TYPE="${TILERT_MODEL_TYPE:-glm-5}"
export TILERT_MAX_MODEL_LEN="${TILERT_MAX_MODEL_LEN:-${MAX_MODEL_LEN:-202752}}"
export DECODE_MTP_SIZE="${DECODE_MTP_SIZE:-$([[ "$SPEC_DECODING" == "mtp" ]] && echo 1 || echo 0)}"
export TILERT_TRANSPORT="${TILERT_TRANSPORT:-mooncake}"
export PREFILL_KV_DTYPE="${PREFILL_KV_DTYPE:-auto}"
export DECODE_KV_DTYPE="${DECODE_KV_DTYPE:-bf16}"
export TILERT_PARSER="${TILERT_PARSER:-none}"
export ROUTER_PORT="${ROUTER_PORT:-30000}"
export TILERT_QUEUE_TIMEOUT="${TILERT_QUEUE_TIMEOUT:-0}"

if [[ "${PREFILL_EP:-1}" -ne 1 || "${DECODE_EP:-1}" -ne 1 || \
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
