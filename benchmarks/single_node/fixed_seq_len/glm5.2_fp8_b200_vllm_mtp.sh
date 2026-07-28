#!/usr/bin/env bash
# GLM-5.2 FP8 on B200, vLLM, MTP, 固定 seqlen (UTPS-STPS 曲线用).
#
# 在容器内执行 (由 bench.sh 起容器并调用). 与 sglang 侧同口径的 vLLM 对照:
# 单节点合一 (unified, 非 PD 分离): 一个 vllm engine 同时 prefill+decode,
# benchmark client 也在同容器内跑. 口径 (稳态窗口/固定 MTP) 与 sglang 完全一致.
#
# GLM-5.2 = deepseek_v32 (DSA) 架构; vLLM 的 glm52 镜像原生支持 (registry 里
# DeepseekV32ForCausalLM / GlmMoeDsaForCausalLM 均已注册), 无需 --hf-overrides.
# 并行由 orchestrator(bench.sh) 的 --parallel 预设翻译成 vllm 语义传入 (TP/DP/EP):
#   - vllm world = TP×DP; DP>1 = attention 数据并行; EP>=1 = MoE 专家并行(否则 MoE 张量并行).
#   - DSA sparse attention / KV dtype / chunked-prefill / MoE kernel 等【全用 vLLM 默认】,
#     不强行对齐 sglang —— 让 vLLM 走自己的最优路径取分(曾强按 sglang 的 chunk-size 反而拖累, 已移除).
#   - **必须** VLLM_ALLREDUCE_USE_SYMM_MEM=0(由 bench.sh 注入) + --compilation-config
#     fuse_allreduce_rms=false(下方): 否则 torch 对称内存 rendezvous 在本机 8×B200 死锁.
#
# ---- MTP (与 sglang 完全对齐的固定口径) ----
# 本地 checkpoint 把 MTP 层单独拆到 mtp/ 目录 (主模型 model/ 无 nextn 权重, 实测
# weight_map 里 0 个 nextn key; mtp/ 里是 layers.78.eh_proj/enorm/hnorm 的 MTP 层),
# 故 speculative-config 要显式指 "model"=<draft路径> (等价 sglang 的 --speculative-draft-model-path).
#   - N (草稿步数) = num_speculative_tokens (脚本 -n, 默认 5).
#   - accept_len 固定: 本 glm52 镜像的 vllm/config/speculative.py 内置
#     rejection_sample_method="synthetic" + synthetic_acceptance_length (∈[1,N+1]),
#     即 sglang SGLANG_SIMULATE_ACC_LEN 的 vLLM 等价物 (无需 hack 源码).
#     synthetic_acceptance_length=6 => 每 decode step 恒接受 6 token (5 draft + 1),
#     与 sglang accept_len=6 逐步恒定完全一致. MTP_ACC<=0 则不设 -> 走自然接受(对照用).
#
# 必需环境变量: MODEL TP CONC ISL OSL RANDOM_RANGE_RATIO RESULT_FILENAME
# 可选: DP(=TP, 仅 EP>1 时用) EP(默认1=关) SPEC_NUM_STEPS(=MTP-N,默认5)
#       MTP(默认1) MTP_ACC(=accept_len,默认6) MTP_DRAFT_PATH REPS MAX_RETRY

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

nvidia-smi || true

# ---- MTP / 并行配置 ----
MTP="${MTP:-1}"
SPEC_NUM_STEPS="${SPEC_NUM_STEPS:-5}"          # = MTP-N (-n), 默认 5
MTP_DRAFT_PATH="${MTP_DRAFT_PATH:-${MODEL%/model}/mtp}"
MTP_ACC="${MTP_ACC:-6}"                          # accept_len (-a), synthetic, [1,N+1]
# accept_len 可为小数(cookbook low-latency 用 3.5) -> 不能用 bash 整数比较
# ([[ 3.5 -gt 0 ]] 是算术语法错, 会静默不设 synthetic = 没固定住). 统一走 awk 数值比较.
acc_gt0() { awk -v v="${1:-0}" 'BEGIN{exit !(v + 0 > 0)}'; }
# vllm 原生语义(由 bench.sh 的 --parallel 预设翻译好): DP=attention 数据并行, EP=0/1 专家并行开关
DP="${DP:-1}"
EP="${EP:-0}"

# 客户端 warmup 请求数 (由 bench.sh 注入, 默认 0). 主要靠稳态窗口刨 prefill/ramp.
export NUM_WARMUPS="${NUM_WARMUPS:-0}"

CONTEXT_LEN="${CONTEXT_LEN:-16384}"            # 远大于 ISL+OSL 即可
MEM_FRAC="${MEM_FRAC:-0.9}"                     # vllm --gpu-memory-utilization (recipe 默认 0.9)
# max-num-seqs 按本次扫描的最大 batch 自适应(cuda-graph 捕获等其余走 vLLM 默认)
MAX_C=$(echo "$CONC" | tr ',' '\n' | tr -d ' ' | grep -E '^[0-9]+$' | sort -n | tail -1)
[[ -z "$MAX_C" || "$MAX_C" -lt 1 ]] && MAX_C=256
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$MAX_C}"

# DeepGEMM 预热跳过, 加快 server 启动 (GLM-5.2 FP8 需 DeepGEMM)
export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"

# 结果/日志直接写 outdir(bench.sh 以绝对路径传入, /tilert 挂载下可写), 不落在 /workspace
# (=仓库目录), 否则中断的 run 会把 json/.cmd 留在仓库和历史结果混淆. 未传时回退 /workspace.
RESULT_DIR="${RESULT_DIR:-/workspace}"
SERVER_LOG="$RESULT_DIR/server.log"

echo "CONC=$CONC ISL=$ISL OSL=$OSL RANGE=$RANDOM_RANGE_RATIO MTP=$MTP N=$SPEC_NUM_STEPS ACC=$MTP_ACC TP=$TP DP=$DP EP=$EP"

# vllm serve 的 model 是位置参数; 不设 --served-model-name, 使 model id = $MODEL 路径,
# 与 client 的 --model "$MODEL" 对齐 (否则 404 model not found).
SERVER_ARGS=(
    "$MODEL"
    --port "$PORT"
    --trust-remote-code
    --tensor-parallel-size "$TP"
    --gpu-memory-utilization "$MEM_FRAC"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-model-len "$CONTEXT_LEN"
    --no-enable-prefix-caching
    --async-scheduling
    --distributed-executor-backend mp
)

# 并行 (vllm 原生, world=tp*dp): DP>1 -> attention 数据并行; MoE 按 tp*dp 分片,
#   EP>=1 -> MoE 走专家并行(--enable-expert-parallel); 否则 MoE 走张量并行(纯 TP).
if [[ "$DP" -gt 1 ]]; then
    SERVER_ARGS+=( --data-parallel-size "$DP" )
fi
if [[ "${EP:-0}" -ge 1 ]]; then
    SERVER_ARGS+=( --enable-expert-parallel )
fi

# MTP: 拆分的 draft 路径 + 固定 accept_len (synthetic).
if [[ "$MTP" == "1" ]]; then
    SPEC="{\"method\":\"mtp\",\"model\":\"$MTP_DRAFT_PATH\",\"num_speculative_tokens\":$SPEC_NUM_STEPS"
    if acc_gt0 "$MTP_ACC"; then
        SPEC="$SPEC,\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":$MTP_ACC"
        echo "固定 MTP: N(num_speculative_tokens)=$SPEC_NUM_STEPS, accept_len(synthetic_acceptance_length)=$MTP_ACC"
    else
        echo "MTP=1, accept_len 自然接受 (synthetic 未设, MTP_ACC=$MTP_ACC)"
    fi
    SPEC="$SPEC}"
    SERVER_ARGS+=( --speculative-config "$SPEC" )
fi

# 关掉 allreduce+rms 融合 pass: 该 pass 在 torch.compile 编译期创建 flashinfer all-reduce
# workspace, 走 torch 对称内存(_symmetric_memory) rendezvous, 在本机 8×B200 上死锁——
# 与 VLLM_ALLREDUCE_USE_SYMM_MEM 是同一个 torch symm_mem bug 的另一入口 (compile post-pass).
# 关掉即可正常编译 (仅损失 allreduce+rmsnorm 融合这一个优化, 对吞吐影响很小).
SERVER_ARGS+=( --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}' )

# .cmd 文件 = vllm serve 的复现记录: 先列该 server 进程实际读取、但不在命令行里的环境变量
# (每行 export VAR=值 # 说明), 再列启服务命令. 注: client/编排参数(ISL/OSL/CONC/NUM_WARMUPS/
# REPS/MAX_RETRY 等)不影响 server 进程, 不在此列(它们记在 run_config.json). 供报告"测试配置"展示.
{
  echo "# ===== 环境变量 (vllm serve 进程读取, 但不在命令行里, 需 export 才能复现) ====="
  echo "# --- 容器/运行时 (bench.sh 注入) ---"
  emit_env PYTHONUNBUFFERED     "python 输出不缓冲, 日志实时"
  emit_env PYTHONNOUSERSITE     "忽略用户 site-packages, 用容器内干净环境"
  emit_env TORCH_CUDA_ARCH_LIST "目标 SM 架构 = B200 sm_100, 不编多余 arch"
  emit_env CUDA_DEVICE_ORDER    "按 PCI 总线枚举 GPU, 卡号稳定"
  emit_env PYTHONFAULTHANDLER   "崩溃时打印 python 栈 (SIGABRT/段错误便于定位)"
  echo "# --- vllm 关键/hack env (本机 8xB200 必需) ---"
  emit_env VLLM_ALLREDUCE_USE_SYMM_MEM     "关对称内存: 否则 all-reduce rendezvous 死锁"
  emit_env VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE "关 inductor max-autotune: 否则首次编译卡死"
  emit_env VLLM_DEEP_GEMM_WARMUP           "跳过 DeepGEMM 预热, 加快 server 启动"
  echo "# 注: 固定 accept_len 的 hack 在 --speculative-config 的 synthetic_acceptance_length (见启服务命令)"
  echo ""
  echo "# ===== vllm 启服务命令 (所有 server 参数都在这) ====="
  echo "vllm serve ${SERVER_ARGS[*]}"
} > "$RESULT_DIR/${RESULT_FILENAME}.cmd" 2>/dev/null || true

start_server() {
    stop_server
    set -x
    vllm serve "${SERVER_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    set +x
    wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"
}
stop_server() {
    [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true
    # vllm 用 mp executor 会起子进程, 一并清掉; 等显存释放再重启避免 OOM
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    [[ -n "${SERVER_PID:-}" ]] && { for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done; }
    SERVER_PID=""
}
# 判定一次测量是否成功: 结果 JSON 存在且 completed>0
run_succeeded() {
    local jf="$RESULT_DIR/$1.json"
    python3 -c "import json,sys; d=json.load(open('$jf')); sys.exit(0 if d.get('completed',0)>0 else 1)" 2>/dev/null
}

# GPU 指标也写 outdir(不传 --output 会落在 /workspace, 即仓库根, 污染仓库)
start_gpu_monitor --output "$RESULT_DIR/${RESULT_FILENAME}.gpu_metrics.csv"
start_server
# datasets/pandas: 本 client 跑 random 数据集其实用不到(全脚本 0 处 import), 仅为兼容其他数据集
# 做 best-effort 安装. 必须加 timeout: vllm 镜像未预装 datasets 且容器常连不上 PyPI, 不限时会
# 无限重试卡死整个 run(server 已就绪却永远不发请求). 装不上就跳过, random 照跑.
timeout 60 pip install -q datasets pandas >/dev/null 2>&1 || echo "(跳过 datasets/pandas 安装: 无网/已装; random 数据集不需要)"

# server 侧预热: sweep 前跑一遍最大 batch(丢弃), 触发 JIT/autotune/torch.compile 特化/
# cudagraph replay, 避免头几个测点的冷启动污染 TTFT. 输出到 /tmp 不被采集.
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
# REPS: 每点重复次数; watchdog/崩溃 (server 死 / completed=0) 重启重试, 最多 MAX_RETRY 次.
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
            start_server
        done
        [[ "$ok" == "1" ]] || { echo "!! c=$c rep=$rep 重试 $MAX_RETRY 次仍失败, 放弃该点"; BENCH_RC=1; }
    done
done

stop_gpu_monitor
stop_server
exit $BENCH_RC
