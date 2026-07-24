#!/usr/bin/env bash
#
# 必需环境变量: MODEL TP CONC ISL OSL RANDOM_RANGE_RATIO RESULT_FILENAME
# 可选: DP(默认=TP) EP(默认1=关; >1走deepep) SPEC_NUM_STEPS(=MTP-N,默认5)
#       MTP(默认1) MTP_ACC(=accept_len,默认6) REPS MAX_RETRY

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ "$TP" -ne 8 ]]; then
    echo "提示: GLM-5.2 MTP 在本机主要验证过 TP=8; TP=$TP 为实验配置, 若不收敛请回退 TP=8" >&2
fi

nvidia-smi || true

# ---- MTP / EAGLE 配置 ----
# EAGLE (topk=1, chain): draft 步数 = MTP-N; 每步 1 个 draft token,
# 验证后接受 token 数 <= N+1 (含保底 1). N 与 accept_len 的固定见下方.
MTP="${MTP:-1}"
SPEC_NUM_STEPS="${SPEC_NUM_STEPS:-5}"          # = MTP-N (-n), 默认 5
SPEC_EAGLE_TOPK="${SPEC_EAGLE_TOPK:-1}"
# accept_len (-a): 每个 decode step 强制恰好接受的 token 数 (含保底), clamp 到 [1, N+1].
# 用 sglang 内置的 "模拟接受长度" 特性 (SGLANG_SIMULATE_ACC_*), 而非改源码:
#   match-expected + 整数 -> 逐步恒定 (spec_utils.generate_simulated_accept_index).
# 这样 N (=--speculative-num-steps) 和 accept_len 都被硬固定且可复现.
MTP_ACC="${MTP_ACC:-6}"

# 并行 (TP/DP/EP 由 orchestrator 的 --parallel 预设翻译好; sglang 语义: 总卡数=tp, dp/ep 叠加).
#   DP>1 -> attention-DP(attn_tp=tp/dp); EP>1 -> MoE deepep 专家并行, 否则 MoE 张量并行.
DP="${DP:-$TP}"
EP="${EP:-1}"


# 本次扫描的最大 batch(用于 max-running-requests 与 CUDA graph 捕获上限, 不写死).
MAX_C=$(echo "$CONC" | tr ',' '\n' | tr -d ' ' | grep -E '^[0-9]+$' | sort -n | tail -1)
[[ -z "$MAX_C" || "$MAX_C" -lt 1 ]] && MAX_C=256
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-$MAX_C}"

# CUDA graph 捕获上限. 两个约束:
#  - 关 dp-attn(DP=1): 每卡存全量 KV, 按过大 bs 捕获会 OOM -> 用 MAX_C 自适应, 不写死.
#  - 开 dp-attn(DP>1): sglang 按 global_bs/DP 算每卡 capture batch, MAX_C<DP 会得到
#    capture_bs=[0] 直接崩. 故 dp-attn 下把上限抬到至少 DP -> 单跑 -c 1(dp=8) 也安全.
CUDA_GRAPH_MAX_BS="$MAX_C"
[[ "$DP" -gt 1 && "$CUDA_GRAPH_MAX_BS" -lt "$DP" ]] && CUDA_GRAPH_MAX_BS="$DP"

# 静态显存占比(权重 + KV 池). 是 "KV池 ↔ 运行时余量" 的旋钮(权重固定 ~106G/卡):
#   高(0.85) -> KV 池大但运行时紧: EP 时 batch256 的 prefill MHA 激活(~3G)OOM;
#   低(0.75) -> 运行时松但 KV 池小: batch256 decode 触发 KV retract -> STPS 崩.
# 实测 dep(ep8,ISL8192): 0.80 两头都够(256 peak KV usage 0.80, retract=0 oom=0, 干净跑通),
#   KV 阈值 ~314 并发. 故 EP 默认 0.80; 非 EP(dpa-tp 无 deepep 通信缓冲, 运行时更宽)默认 0.85.
#   显式 MEM_FRAC env 优先. (1024 仍装不下 KV, 需 offer+排队, 见 cookbook high-throughput.)
if [[ -n "${MEM_FRAC:-}" ]]; then :; elif [[ "$EP" -gt 1 ]]; then MEM_FRAC=0.80; else MEM_FRAC=0.85; fi

# 关闭 client 侧 warmup(benchmark_lib.sh 默认 --num-warmups=2*concurrency):
# 本项目靠稳态窗口(max-TTFT ~ min-finish)刨 prefill, 不靠 warmup 请求; JIT/cudagraph
# 预热由下方 SERVER_WARMUP 块负责. 关掉省一波冗余负载(默认 0, 可用 NUM_WARMUPS 覆盖).
export NUM_WARMUPS="${NUM_WARMUPS:-0}"

# 结果/日志直接写 outdir(bench.sh 以绝对路径传入, 在 /tilert 挂载下容器可写),
# 不再落在 /workspace(=仓库 InferenceX 目录). 否则被中断的 run 会把 json/.cmd/.serverlog
# 留在仓库里, 和历史结果混淆. 未传 RESULT_DIR 时回退 /workspace(兼容单独调试).
RESULT_DIR="${RESULT_DIR:-/workspace}"
SERVER_LOG="$RESULT_DIR/server.log"

echo "CONC=$CONC ISL=$ISL OSL=$OSL RANGE=$RANDOM_RANGE_RATIO MTP=$MTP N=$SPEC_NUM_STEPS ACC=$MTP_ACC TP=$TP DP=$DP EP=$EP RESULT_DIR=$RESULT_DIR"

SERVER_ARGS=(
    --model-path="$MODEL"
    --trust-remote-code
    --host=0.0.0.0 --port="$PORT"
    --enable-metrics
    --tensor-parallel-size="$TP"
    --mem-fraction-static="$MEM_FRAC"
    --max-running-requests="$MAX_RUNNING_REQUESTS"
    --chunked-prefill-size 32768
    # 下面是本地 checkpoint / 本 sglang build 必需 (cookbook 的 zai-org 模型/新版默认不需要):
    --page-size=64            # NSA KV pool 硬断言 page_size==64, 不设会 AssertionError 崩
    --context-length=16384    # 不设用模型 max_position=1048576(1M!)→ KV池/并发估算按1M算, 异常
    # 注意: 不要设 --kv-cache-dtype=fp8. 本 build 的 NSA prefill 走 fp8 KV 时会进
    #   _get_mla_kv_buffer_from_fp8_for_nsa, 引用不存在的 page_table_1_flattened -> 大 prefill
    #   batch(如 c=64, 64x8192 打进一个 chunk)时 scheduler 崩. 走默认 bf16 KV 即可绕开;
    #   MLA 是压缩 latent(kv_lora_rank=512), bf16 KV 也很小, 无显存压力.
    --cuda-graph-max-bs="$CUDA_GRAPH_MAX_BS"  # 不设 sglang 默认~160, 高并发退 eager 变慢. (dp-attn 的 >=DP 保护见上)
)

# attention-DP (DP>1)
if [[ "$DP" -gt 1 ]]; then
    SERVER_ARGS+=( --data-parallel-size="$DP" --enable-dp-attention )
fi
# MoE 并行: EP>1 走专家并行(deepep + deep_gemm); 否则 triton runner (已验证配置)
if [[ "$EP" -gt 1 ]]; then
    # deepep 额外 env. 单节点 unified 无 RDMA GIN, deepep-v2 会报 "NCCL GIN unavailable";
    # 故默认走 v1 buffer (仅 NVLink). 多机带 IB 时可 SGLANG_DEEPEP_USE_V2=1 覆盖.
    export SGLANG_DEEPEP_USE_V2="${SGLANG_DEEPEP_USE_V2:-0}"
    # deepep low-latency 每-rank dispatch 缓冲上限, 必须 >= 单 rank 一次 forward 派发的最大 token 数:
    #   decode+MTP 下 = ceil(cuda_graph_max_bs / DP) * num_draft_tokens(=N+1).
    # 写死 256 时, 大 batch(如 sweep 到 1024, dp8 -> 每 rank 128*6=768)会在 CUDA graph 捕获阶段
    # 触发 deepep buffer.hpp 断言 (x.size(0) <= num_max_dispatch_tokens_per_rank) -> 整个 scheduler 崩.
    # 故按本次 cuda-graph-max-bs 自适应, 向上取 128 的倍数留余量.
    _ndt=1; [[ "$MTP" == "1" ]] && _ndt=$(( SPEC_NUM_STEPS + 1 ))
    _need=$(( ( ( ( (CUDA_GRAPH_MAX_BS + DP - 1) / DP ) * _ndt ) + 127 ) / 128 * 128 ))
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-$_need}"
    # nvshmem QP 深度必须 >= (num_max_dispatch+1)*2 (deep_ep/buffer.py 断言), 默认 1024 只够 dispatch<=511.
    # 抬 dispatch 后必须同步抬 QP 深度, 否则换一个断言(nvshmem_qp_depth)继续崩. 向上取 2 的幂.
    _qpneed=$(( (SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK + 1) * 2 ))
    _qp=1024; while (( _qp < _qpneed )); do _qp=$(( _qp * 2 )); done
    export NVSHMEM_QP_DEPTH="${NVSHMEM_QP_DEPTH:-$_qp}"
    echo "deepep: EP=$EP num_max_dispatch=$SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK NVSHMEM_QP_DEPTH=$NVSHMEM_QP_DEPTH (cuda_graph_max_bs=$CUDA_GRAPH_MAX_BS DP=$DP ndt=$_ndt)"
    SERVER_ARGS+=( --ep-size="$EP" --moe-a2a-backend=deepep )
fi

if [[ "$MTP" == "1" ]]; then
    # 本地 checkpoint 的 MTP 层单独在 mtp/ 目录, 必须显式指定 draft 路径
    # (cookbook 的 zai-org/GLM-5.2-FP8 把 MTP 烘焙进主权重, 故其命令无此 flag).
    MTP_DRAFT_PATH="${MTP_DRAFT_PATH:-${MODEL%/model}/mtp}"
    SERVER_ARGS+=( --speculative-algorithm=EAGLE
                   --speculative-num-steps="$SPEC_NUM_STEPS"
                   --speculative-eagle-topk="$SPEC_EAGLE_TOPK"
                   --speculative-draft-model-path="$MTP_DRAFT_PATH"
                   )
fi

# 固定 accept_len: 必须在启动 sglang 进程前设好 (spec_utils 在 import 时读取该 env).
if [[ "$MTP" == "1" && "$MTP_ACC" -gt 0 ]]; then
    export SGLANG_SIMULATE_ACC_LEN="$MTP_ACC"
    export SGLANG_SIMULATE_ACC_METHOD="match-expected"
    echo "固定 MTP: N(num-steps)=$SPEC_NUM_STEPS, accept_len(SIMULATE_ACC_LEN)=$MTP_ACC (match-expected)"
fi

# watchdog: 偶发 stall 会触发 sglang scheduler watchdog. 降到 120s 让崩溃更快暴露、
# 便于重启恢复 (OSL 下正常单次 forward 远 <120s, 不误伤).
SERVER_ARGS+=( --watchdog-timeout "${WATCHDOG_TIMEOUT:-120}" )

# .cmd 文件 = launch_server 的复现记录: 先列该 server 进程实际读取、但不在命令行里的环境变量
# (每行 export VAR=值 # 说明), 再列启服务命令. 注: client/编排参数(ISL/OSL/CONC/NUM_WARMUPS/
# REPS/MAX_RETRY 等)不影响 server 进程, 不在此列(它们记在 run_config.json). 供报告"测试配置"展示.
{
  echo "# ===== 环境变量 (launch_server 进程读取, 但不在命令行里, 需 export 才能复现) ====="
  echo "# --- 容器/运行时 (bench.sh 注入) ---"
  emit_env PYTHONUNBUFFERED     "python 输出不缓冲, 日志实时"
  emit_env PYTHONNOUSERSITE     "忽略用户 site-packages, 用容器内干净环境"
  emit_env TORCH_CUDA_ARCH_LIST "目标 SM 架构 = B200 sm_100, 不编多余 arch"
  emit_env CUDA_DEVICE_ORDER    "按 PCI 总线枚举 GPU, 卡号稳定"
  emit_env NCCL_SHM_DISABLE     "关 NCCL 共享内存传输 (容器 /dev/shm 受限, 否则易挂/降速)"
  emit_env GLM_XGRAMMAR_BACKEND_CACHE_MAX_MB  "GLM xgrammar 语法后端缓存上限(MB)"
  emit_env GLM_GRAMMAR_OBJECT_CACHE_MAX_COUNT "GLM 语法对象缓存条数上限"
  echo "# --- MTP accept_len hack (sglang 进程 import 时读取) ---"
  emit_env SGLANG_SIMULATE_ACC_LEN    "固定 MTP accept_len 的 hack: 每 decode step 恒接受该值"
  emit_env SGLANG_SIMULATE_ACC_METHOD "配合上者逐步恒定 (match-expected)"
  echo "# --- EP(deepep) 专用, 仅 --parallel dep/tep 出现 ---"
  emit_env SGLANG_DEEPEP_USE_V2 "deepep v1 buffer; 单机无 RDMA GIN, v2 报 NCCL GIN unavailable"
  emit_env SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK "每 rank dispatch token 缓冲上限 = ceil(cuda_graph_max_bs/DP)*(N+1)"
  emit_env NVSHMEM_QP_DEPTH     "nvshmem QP 深度, 必须 >= (dispatch+1)*2"
  echo ""
  echo "# ===== sglang 启服务命令 (所有 server 参数都在这) ====="
  echo "python3 -m sglang.launch_server ${SERVER_ARGS[*]}"
} > "$RESULT_DIR/${RESULT_FILENAME}.cmd" 2>/dev/null || true

start_server() {
    stop_server
    set -x
    PYTHONNOUSERSITE=1 python3 -m sglang.launch_server "${SERVER_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    set +x
    wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"
}
stop_server() {
    [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true
    # 等 GPU 显存释放再重启, 避免 OOM
    [[ -n "${SERVER_PID:-}" ]] && { for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done; }
    SERVER_PID=""
}
# 判定一次测量是否成功: 结果 JSON 存在且 completed>0 (watchdog 崩溃时 completed=0)
run_succeeded() {
    local jf="$RESULT_DIR/$1.json"
    python3 -c "import json,sys; d=json.load(open('$jf')); sys.exit(0 if d.get('completed',0)>0 else 1)" 2>/dev/null
}

start_gpu_monitor
start_server
# datasets/pandas: 本 client 跑 random 数据集其实用不到(全脚本 0 处 import), 仅为兼容其他数据集
# 做 best-effort 安装. 必须加 timeout: 无网环境(容器连不上 PyPI)不限时会无限重试卡死整个 run.
# sglang 镜像已预装, 这里通常是秒级 no-op; 装不上就跳过, random 照跑.
timeout 60 pip install -q datasets pandas >/dev/null 2>&1 || echo "(跳过 datasets/pandas 安装: 无网/已装; random 数据集不需要)"

# server 侧预热: sweep 前跑一遍最大 batch(丢弃), 触发 JIT/autotune/cudagraph replay,
# 避免头几个测点冷启动污染 TTFT. 输出到 /tmp 不被采集. (与 vllm 侧对齐)
if [[ "${SERVER_WARMUP:-1}" == "1" ]]; then
    echo "===== server warmup: conc=$MAX_C osl=64 (丢弃) ====="
    run_benchmark_serving \
        --model "$MODEL" --port "$PORT" --backend vllm \
        --input-len "$ISL" --output-len 64 \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts "$MAX_C" --max-concurrency "$MAX_C" \
        --result-filename "_warmup" --result-dir /tmp/ \
        --trust-remote-code --server-pid "$SERVER_PID" || true
    kill -0 "$SERVER_PID" 2>/dev/null || start_server
fi

# CONC 支持逗号列表: server 起一次, 每个 batch 循环压测, 描出整条曲线.
# REPS: 每点重复次数 (默认3), 全部原始 JSON 保留: ${prefix}_c${c}_r${rep}.json
# 健壮性: 若某次 watchdog 崩溃 (server 死 / completed=0), 重启 server 并重试该点, 最多 MAX_RETRY 次.
REPS="${REPS:-3}"
MAX_RETRY="${MAX_RETRY:-2}"
BENCH_RC=0
IFS=',' read -ra CONC_ARR <<< "$CONC"
for c in "${CONC_ARR[@]}"; do
    c="${c// /}"
    [[ -z "$c" ]] && continue
    for rep in $(seq 1 "$REPS"); do
        rf="${RESULT_FILENAME}_c${c}_r${rep}"
        ok=0
        for attempt in $(seq 0 "$MAX_RETRY"); do
            # server 若已死 (上一次崩溃), 先重启
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                echo "!! server 不在, 重启 (c=$c rep=$rep attempt=$attempt)"; start_server
            fi
            echo "===== sweep batch CONC=$c rep=$rep/$REPS attempt=$attempt ISL=$ISL OSL=$OSL ====="
            run_benchmark_serving \
                --model "$MODEL" --port "$PORT" --backend vllm \
                --input-len "$ISL" --output-len "$OSL" \
                --random-range-ratio "$RANDOM_RANGE_RATIO" \
                --num-prompts "$c" --max-concurrency "$c" \
                --result-filename "$rf" --result-dir "$RESULT_DIR/" \
                --trust-remote-code --server-pid "$SERVER_PID" || true
            if kill -0 "$SERVER_PID" 2>/dev/null && run_succeeded "$rf"; then
                ok=1; break
            fi
            echo "!! c=$c rep=$rep attempt=$attempt 失败 (崩溃/completed=0), 重启重试"
            # 保留失败时的 server.log (start_server 会覆盖它), 便于事后定位崩因.
            cp -f "$SERVER_LOG" "$RESULT_DIR/${rf}.serverlog.a${attempt}" 2>/dev/null || true
            start_server
        done
        [[ "$ok" == "1" ]] || { echo "!! c=$c rep=$rep 重试 $MAX_RETRY 次仍失败, 放弃该点"; BENCH_RC=1; }
    done
done

stop_gpu_monitor
stop_server
exit $BENCH_RC
