#!/usr/bin/env bash
# ============================================================================
# benchmarks/multi_node/glm5.1_fp8_b200_tilert-disagg.sh
# tileRT 正式榜路线（方案 A，多节点 PD 分离）的**薄入口**，对标 ATOM 的 *_atom-disagg.sh：
#   校验 env → 交给 tilert_utils/submit.sh 做 slurm 编排（salloc/enroot/srun）。
#   容器内每节点的角色分派与三进程（decode / prefill+router+bench）在 tilert_utils/run_node.sh。
#
# tileRT PD：vLLM prefill(挂 TileRTConnector) + TileRT decode(独占 8×B200,TP8) + pd_router(OpenAI :23333)。
# 编排是裸 slurm（非 Dynamo，用不了 srtctl），与 dgxc 单节点路径同 house style；命令 flag 对齐 TileRT README。
#
# 契约：master config 的 disagg 块经 benchmark-multinode-tmpl.yml 注入 env
#   （IMAGE/MODEL_PATH/ISL/OSL/CONC_LIST/*_TP/*_NUM_WORKERS/RANDOM_RANGE_RATIO/RESULT_FILENAME/KV_P2P_TRANSFER/…）。
# ============================================================================

source "$(dirname "$0")/../benchmark_lib.sh"

# 注：PREFILL_NODES/DECODE_NODES 不由模板注入（只给 *_NUM_WORKERS/*_TP），由 launcher 的 tilert 分支导出；
# 此处不硬性要求，缺省按 num-worker（每 worker=TP8=1 整节点）推。
check_env_vars \
    CONC_LIST ISL OSL IMAGE MODEL_PATH FRAMEWORK RANDOM_RANGE_RATIO RESULT_FILENAME \
    PREFILL_NUM_WORKERS PREFILL_TP DECODE_NUM_WORKERS DECODE_TP
export PREFILL_NODES=${PREFILL_NODES:-${PREFILL_NUM_WORKERS:-1}}
export DECODE_NODES=${DECODE_NODES:-${DECODE_NUM_WORKERS:-1}}

# 交给 slurm 提交器（宿主机侧 salloc/enroot/srun）。
exec bash "$(dirname "$0")/tilert_utils/submit.sh"
