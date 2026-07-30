#!/usr/bin/env bash
# GLM-5.2 FP8 on B200, vLLM, MTP, 固定 seqlen (UTPS-STPS 曲线用).
#
# 在容器内执行 (由 bench.sh 起容器并调用). 与 sglang 侧同口径的 vLLM 对照:
# 单节点合一 (unified, 非 PD 分离): 一个 vllm engine 同时 prefill+decode,
# benchmark client 也在同容器内跑. 口径 (稳态窗口/固定 MTP) 与 sglang 完全一致.
#
# GLM-5.2 = deepseek_v32 (DSA) 架构; registry 里 DeepseekV32ForCausalLM /
# GlmMoeDsaForCausalLM 均已注册, 无需 --hf-overrides. 注: 这【不是 glm52 镜像特有的】——
# 上游官方 release 也早就有(实测 0.19.0~0.26.0 的 sdist 里两个名字都在 registry.py),
# 官方 docs 的 supported models 也列了 GlmMoeDsaForCausalLM -> GLM-5/5.1/5.2.
# ★而且在 vllm 里 architectures 名对本模型【毫无影响】★: registry 把两个名字都指向
# deepseek_v2.py, 且 `class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM): pass` 与
# `DeepseekV3ForCausalLM` 逐字相同(空子类). vllm 里唯一的 GLM 专属分支是
# _get_moe_router_dtype() 按 `config.model_type == "glm_moe_dsa"` 强制 fp32 路由 ——
# 我们这份权重 model_type=deepseek_v32, 所以【走不到】那条, 按 DeepSeek-V3.2 处理.
# (sglang 那边相反: DSA 白名单是按 architectures 判的, 所以才需要 arch 旁路改名.)
# 并行由 orchestrator(bench.sh) 的 --parallel 预设翻译成 vllm 语义传入 (TP/DP/EP):
#   - vllm world = TP×DP; DP>1 = attention 数据并行; EP>=1 = MoE 专家并行(否则 MoE 张量并行).
#   - chunked-prefill / cudagraph / MoE kernel 等仍【用 vLLM 默认】, 不强行对齐 sglang ——
#     让 vLLM 走自己的最优路径取分(曾强按 sglang 的 chunk-size 反而拖累, 已移除).
#   - ★但 KV dtype 与 MoE all-to-all 后端【不能靠默认】★(2026-07-29 定位, 见下方两节):
#     vLLM 的默认值(auto=bf16 KV / allgather_reducescatter)与 sglang 侧实际在跑的
#     (fp8_e4m3 / deepep)根本不是同一配置, 却都能跑成功 -> 三 backend 对比图直接失真.
#   - 曾经无条件下发的 VLLM_ALLREDUCE_USE_SYMM_MEM=0 + fuse_allreduce_rms=false 是为
#     node071 的多播死锁打的规避, 该死锁已于 2026-07-27 修好(GPU6 单卡 reset), 故两者都已
#     改成【默认不下发】; 复发时 AR_FUSION=0 + curve env 里 VLLM_ALLREDUCE_USE_SYMM_MEM=0.
#
# ---- MTP (与 sglang 完全对齐的固定口径) ----
# 本地 checkpoint 把 MTP 层单独拆到 mtp/ 目录 (主模型 model/ 无 nextn 权重, 实测
# weight_map 里 0 个 nextn key; mtp/ 里是 layers.78.eh_proj/enorm/hnorm 的 MTP 层),
# 故 speculative-config 要显式指 "model"=<draft路径> (等价 sglang 的 --speculative-draft-model-path).
#   - N (草稿步数) = num_speculative_tokens (脚本 -n, 默认 5).
#   - accept_len 固定: vllm/config/speculative.py 的 rejection_sample_method="synthetic"
#     + synthetic_acceptance_length (∈[1,N+1]), 即 sglang SGLANG_SIMULATE_ACC_LEN 的
#     vLLM 等价物 (无需 hack 源码). ★这是【上游功能】不是本镜像特有★ —— 实测官方
#     0.26.0 sdist 里 vllm/config/speculative.py + tests/v1/e2e/spec_decode 都有它.
#     synthetic_acceptance_length=6 => 每 decode step 恒接受 6 token (5 draft + 1),
#     与 sglang accept_len=6 逐步恒定完全一致. MTP_ACC<=0 则不设 -> 走自然接受(对照用).
#
# 必需环境变量: MODEL TP CONC ISL OSL RANDOM_RANGE_RATIO RESULT_FILENAME
# 可选: DP(=TP, 仅 EP>1 时用) EP(默认1=关) SPEC_NUM_STEPS(=MTP-N,默认5)
#       MTP(默认1) MTP_ACC(=accept_len,默认6) MTP_DRAFT_PATH REPS MAX_RETRY
#       KV_CACHE_DTYPE(=curve 的 kv_cache_dtype, 默认 fp8_e4m3) ALL2ALL_BACKEND(=curve 的
#       all2all_backend, 不给则 EP 曲线用 deepep_low_latency) AR_FUSION(默认1=不下发规避)

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

# ★把"loader 注入了但本 backend 不实现"的旋钮显式喊出来★(与 fp8 那份保持一致)
# config.json 的一等字段是【全 backend 共用的】, loader 对任何曲线都会注入; 而下面这几个是
# sglang 语义、vllm 没有对应物(或本脚本刻意不下发). 不喊就是【配了没生效还不报错】——
# 与 loader 对未知字段直接报错的设计初衷相悖, 也最难事后发现.
for _k in CHUNKED_PREFILL_SIZE MAX_RUNNING_REQUESTS CUDA_GRAPH_MAX_BS NUM_PROMPTS_LIST; do
    if [[ -n "${!_k:-}" ]]; then
        case "$_k" in
          CHUNKED_PREFILL_SIZE) _why="vllm 用自己的 chunked prefill 默认(曾强按 sglang 的 chunk-size 反而拖累, 已移除)";;
          MAX_RUNNING_REQUESTS) _why="vllm 侧等价物是 --max-num-seqs, 本脚本按本曲线最大并发自动取";;
          CUDA_GRAPH_MAX_BS)    _why="vllm 自己按 max-num-seqs 推 cudagraph 捕获尺寸, 无此 flag";;
          NUM_PROMPTS_LIST)     _why="逐点 num_prompts 目前只有 sglang 脚本实现, vllm 的 sweep 循环恒用 num_prompts==并发";;
        esac
        echo "⚠️  忽略 $_k=${!_k}: $_why" >&2
    fi
done

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

# 客户端 warmup 请求数 (由 bench.sh 注入, 默认 0; defaults.num_warmups 覆盖, cookbook 用 64).
export NUM_WARMUPS="${NUM_WARMUPS:-0}"

# context length -> --max-model-len. ★名字对齐★(2026-07-29 修): loader 按 curve 的
# context_length 注入的是 CONTEXT_LENGTH(与 sglang 脚本同名), 这里早前只读 CONTEXT_LEN, 差一截
# -> curve 里配的 context_length 在 vllm 曲线上【静默无效】, 恒是 16384。★这意味着此前所有
# 写了 context_length:"off" 的 vllm 曲线其实都跑的 16384★, 修好后行为会变(不下发 -> 按权重
# max_position=1M 算 KV 池, 日志里的 "Maximum concurrency" 会大不相同), 对比历史数据时注意。
# 保留 CONTEXT_LEN 仅为脱离 bench.sh 单独调试时的兜底。
CONTEXT_LEN="${CONTEXT_LENGTH:-${CONTEXT_LEN:-16384}}"
CONTEXT_LEN_ARGS=()
case "$CONTEXT_LEN" in
    off|OFF|default|0|"") echo "CONTEXT_LENGTH=off: 不下发 --max-model-len (按权重 max_position=1M)";;
    *) CONTEXT_LEN_ARGS=( --max-model-len "$CONTEXT_LEN" );;
esac
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
# ★逐曲线一份日志★(与 sglang 那份一致): 共用 server.log 的话, 同一次 run 里的多条 vllm 曲线
# 会互相覆盖, 事后只剩最后一条的日志 —— 而排查(崩在哪/有没有 preemption/KV 不够)全靠它。
SERVER_LOG="$RESULT_DIR/${RESULT_FILENAME:-server}.serverlog"

echo "CONC=$CONC ISL=$ISL OSL=$OSL RANGE=$RANDOM_RANGE_RATIO MTP=$MTP N=$SPEC_NUM_STEPS ACC=$MTP_ACC TP=$TP DP=$DP EP=$EP"

# 量化方式(= defaults.quantization -> QUANTIZATION). 不给则让 vllm 按权重目录的
# quantization_config 自动识别(fp8 权重就这样, 行为与以前完全一致); modelopt 导出的 NVFP4
# 权重(nvidia/GLM-5.2-NVFP4)必须显式给 modelopt_fp4 才走 NVFP4 kernel —— 0.26.0 的
# QUANTIZATION_METHODS 里确认有 modelopt_fp4.
QUANT_ARGS=()
if [[ -n "${QUANTIZATION:-}" ]]; then
    QUANT_ARGS=( --quantization "$QUANTIZATION" )
    echo "显式 --quantization $QUANTIZATION"
fi

# vllm serve 的 model 是位置参数; 不设 --served-model-name, 使 model id = $MODEL 路径,
# 与 client 的 --model "$MODEL" 对齐 (否则 404 model not found).
SERVER_ARGS=(
    "$MODEL"
    --port "$PORT"
    --trust-remote-code
    ${QUANT_ARGS[@]+"${QUANT_ARGS[@]}"}
    --tensor-parallel-size "$TP"
    --gpu-memory-utilization "$MEM_FRAC"
    --max-num-seqs "$MAX_NUM_SEQS"
    ${CONTEXT_LEN_ARGS[@]+"${CONTEXT_LEN_ARGS[@]}"}
    --no-enable-prefix-caching
    --distributed-executor-backend mp
)

# ---- KV cache dtype (★曾经漏给, 是三 backend 对比失真的最大单一原因★) ----
# 不给 --kv-cache-dtype 时 vllm 用 auto = bf16 KV, 而 sglang 走 DSA 自动配置选 fp8_e4m3
# (见 CLAUDE.md「DSA 路径」), 于是同一张图上两个 backend 的 KV 根本不是一种精度:
#   - decode 是 memory-bound 的, KV 字节直接翻倍;
#   - 更致命的是【KV 池减半】: 实测 tp 下 `GPU KV cache size: 742,528 tokens`, 而
#     ISL8192+OSL1024 = 9216 tok/req -> 只装得下 ~80 条; c=256 那格日志逐行是
#     `Running: 81~86, Waiting: 1~4, GPU KV cache usage: 99.4%` —— offered 256 但在飞只有
#     ~85, 其余在【排队】. 注意 `grep -ic preempt` = 0: vllm 这里是排队不是抢占, 所以
#     "没有 preempt 所以不是 KV 问题"是错的判据(memory vllm-glm52-highbatch-cliff 的根因).
#   - dep 没有这个悬崖只是因为 attn DP 把 KV 摊到 8 rank(每 rank 367,872 tok)装得下.
# vLLM 官方 recipe(recipes.vllm.ai/zai-org/GLM-5.2)的 B200 命令里本来就有这个 flag.
# "off" = 不下发 -> 回到 auto(bf16), 仅用于复现旧数/做 A/B.
#
# ★默认值必须跟着 attention backend 走★(踩过, 一次白等 5 分钟的启动失败): 两个稀疏 MLA 后端
# 支持的 KV dtype 【没有交集】——
#   FLASHINFER_MLA_SPARSE(自动选中的): auto/float16/bfloat16/fp8/fp8_e4m3
#   FLASHMLA_SPARSE:                   auto/bfloat16/fp8_ds_mla/fp8(= fp8_ds_mla 的别名)
# 拿 fp8_e4m3 去配 FLASHMLA_SPARSE 会在 worker 里直接抛
#   "Selected backend AttentionBackendEnum.FLASHMLA_SPARSE is not valid ... ['kv_cache_dtype not supported']".
# ⚠️ 还有个坑: 字符串 "fp8" 在两个后端里【含义不同】(FlashInfer 下 = e4m3, FlashMLA 下 = ds_mla
# 打包格式), 所以别用 "fp8" 图省事, 显式写全名.
case "${ATTENTION_BACKEND:-}" in
    FLASHMLA_SPARSE|flashmla_sparse) _kv_default=fp8_ds_mla ;;
    *)                               _kv_default=fp8_e4m3 ;;
esac
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-$_kv_default}"
case "$KV_CACHE_DTYPE" in
    off|OFF|default) echo "KV_CACHE_DTYPE=off: 不下发 --kv-cache-dtype (回到 vllm auto=bf16)";;
    *) SERVER_ARGS+=( --kv-cache-dtype "$KV_CACHE_DTYPE" );;
esac

# ---- MoE all-to-all 后端 ----
# vllm 默认 all2all_backend="allgather_reducescatter"(config/parallel.py), 即 naive a2a ——
# 日志 `Using AgRsAll2AllManager all2all manager` + `MoEPrepareAndFinalizeNaiveDPEPMonolithic`.
# 但 parallel 预设 dep/tep 的【定义】就是 "MoE EP(deepep)"(sglang 侧下发 --moe-a2a-backend=deepep),
# 所以不给这个 flag 时同一个 "dep" 在两个 backend 上跑的不是一回事. 官方 0.26.0 镜像里
# deep_ep 已装(实测 importlib 找得到; pplx_kernels 没有).
# 选 low_latency 而非 high_throughput: 这是 decode benchmark —— 同样的教训在 sglang 侧付过
# 学费(deepep-mode=normal 会强制关 cudagraph, 见 memory vllm-glm52-highbatch-cliff 里那条更正).
if [[ -n "${ALL2ALL_BACKEND:-}" ]]; then
    case "$ALL2ALL_BACKEND" in
        off|OFF|default) echo "ALL2ALL_BACKEND=off: 不下发 --all2all-backend (回到 vllm 默认 allgather_reducescatter)";;
        *) SERVER_ARGS+=( --all2all-backend "$ALL2ALL_BACKEND" );;
    esac
elif [[ "${EP:-0}" -ge 1 ]]; then
    # 与 fp8 那份保持一致(别让两个 quant 分支漂移). ★依据是 fp8 上的实测★
    # (dep c=64, ISL8192/OSL1024: deepep_low_latency 21.60ms TPOT / 10606ms TTFT / STPS 1999
    #  vs flashinfer_nvlink_two_sided 21.59ms / 8016ms / STPS 2435, 即 +21.8%):
    # 两者 TPOT 逐位相同, 差别 100% 在 prefill —— DeepEP 的 low_latency 是给【不做 prefill 的
    # PD-decode 节点】用的, 我们单机合一要 prefill, 照抄就把 TTFT 拖垮.
    # ⚠️ fp4 上【尚未复验】: 机理与量化格式无关(是 a2a 模式之分, 不是权重之分), 故沿用同一默认,
    #    但真要下结论请在 nvfp4 上重跑一次 A/B.
    SERVER_ARGS+=( --all2all-backend flashinfer_nvlink_two_sided )
    echo "EP 曲线未指定 all2all_backend -> 默认 flashinfer_nvlink_two_sided (fp8 实测优于 deepep_low_latency; fp4 未复验)"
fi

# ---- 注意力后端 ----
# 不给则用 vllm 自动选择(博客明说 Blackwell 上自动选 FlashInfer 系, 无需手动 flag), 本模型上
# 自动选中 FLASHINFER_MLA_SPARSE + "标准 fp8" KV 布局.
# ★为什么需要这个旋钮★: FLASHMLA_SPARSE 【不在自动候选里】, 必须显式点名, 而它换的是 KV 布局
# (DeepSeek 的 fp8_ds_mla 打包格式)与 decode kernel —— vllm 自己在日志里提示:
#   "Using standard fp8 KV cache format. To use DeepSeek's fp8_ds_mla KV cache format,
#    please set `--attention-backend FLASHMLA_SPARSE`"
# 0.26.0 已【没有】VLLM_ATTENTION_BACKEND 环境变量, 只能走 CLI, 所以必须做成脚本旋钮.
if [[ -n "${ATTENTION_BACKEND:-}" ]]; then
    case "$ATTENTION_BACKEND" in
        off|OFF|default|auto) echo "ATTENTION_BACKEND=$ATTENTION_BACKEND: 不下发 --attention-backend (走 vllm 自动选择)";;
        *) SERVER_ARGS+=( --attention-backend "$ATTENTION_BACKEND" );;
    esac
fi

# ---- 每步 prefill token 预算 ----
# 不给则走 vllm 默认. 官方 GLM-5.2-NVFP4 博客的 decode 节点给的是 1024 —— 但它是 PD 分离,
# decode 引擎几乎不跑 prefill; 我们是单节点 unified, 给小值会把 ISL8192 切成更多 chunk:
#   -> TTFT 变差, 且 whole-run 的 mean_TPOT(含被 prefill 抢走的 step)结构也变,
#      故【设了它的点与没设的点不可逐格比】, 报告里必须标注.
# 用途: 规避 attn-DP + spec 下 shm_broadcast 的死锁(vllm #41530 / vllm-ascend #9405, 上游未修;
#   已知调大 max_chunks 无效, 关 graph capture 也无效 —— 降低 prefill 压力是仅剩的结构性差异).
if [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]]; then
    case "$MAX_NUM_BATCHED_TOKENS" in
        off|OFF|default) echo "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS: 不下发(走 vllm 默认)";;
        *) SERVER_ARGS+=( --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" )
           echo "--max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS";;
    esac
fi

# ---- NvFp4 MoE kernel ----
# vllm 的 auto 选择顺序是 FLASHINFER_TRTLLM > CUTEDSL > CUTEDSL_BATCHED > CUTLASS > ...(取第一个
# 支持当前配置的), 本模型上自动选中 FLASHINFER_TRTLLM(日志 "Using 'FLASHINFER_TRTLLM' NvFp4 MoE
# backend out of potential backends: [...]"). 小 batch 下 trtllm 的 tile 可能浪费, 故留旋钮.
# 注: FLASHINFER_B12X 被上游从自动候选里排除(CUTLASS SM121 guard), 与 B200(SM100) 无关.
if [[ -n "${MOE_BACKEND:-}" ]]; then
    case "$MOE_BACKEND" in
        off|OFF|default|auto) echo "MOE_BACKEND=$MOE_BACKEND: 不指定 moe_backend (走 vllm 自动选择)";;
        *) SERVER_ARGS+=( --kernel-config "{\"moe_backend\":\"$MOE_BACKEND\"}" )
           echo "--kernel-config {\"moe_backend\":\"$MOE_BACKEND\"}";;
    esac
fi

# 异步调度: 默认开(吞吐更好)。ASYNC_SCHEDULING=0 可关 —— 排查用: dp-attention + spec 组合下
# 实测 sample_tokens 的 collective RPC 会超过 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS(默认 300s)
# 而把 EngineCore 打死(EngineDeadError), 怀疑与异步调度 + spec 的交互有关。
if [[ "${ASYNC_SCHEDULING:-1}" == "1" ]]; then
    SERVER_ARGS+=( --async-scheduling )
else
    echo "ASYNC_SCHEDULING=0: 不下发 --async-scheduling (排查 DP+spec 的 RPC 超时)"
fi

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
    # 逃生阀: 往 --speculative-config 里追加实验性键值(逗号分隔的 JSON 片段, 不含外层花括号).
    # 例: SPEC_EXTRA='"use_local_argmax_reduction":true'
    #   —— 把 draft token 的 logits 通信从 O(vocab) 降到 O(2*tp)(只对 greedy + 非树形 spec 生效,
    #      我们两条都满足); N=5 时每个 decode step 有 5 次全 vocab all-gather, 故对 tp 曲线有意义.
    # 之所以用 env 而不是一等字段: 这些是【一次性 A/B 的实验量】, 验证有效才该升字段.
    if [[ -n "${SPEC_EXTRA:-}" ]]; then
        SPEC="$SPEC,$SPEC_EXTRA"
        echo "speculative-config 追加: $SPEC_EXTRA"
    fi
    SPEC="$SPEC}"
    SERVER_ARGS+=( --speculative-config "$SPEC" )
fi

# allreduce+rms 融合 pass: 该 pass 在 torch.compile 编译期创建 flashinfer all-reduce
# workspace, 走 torch 对称内存(_symmetric_memory) rendezvous —— 在【多播坏掉的节点上】会死锁,
# 与 VLLM_ALLREDUCE_USE_SYMM_MEM 是同一个 symm_mem 入口的两处(一处编译期一处运行期).
# ★默认不再关它★: 根因(node071 GPU6 掉 NVLink -> FM 编不出 8 卡多播路由)已于 2026-07-27 修好,
# sglang 侧同类规避也全撤了; 继续无条件关等于白丢 allreduce+rmsnorm 融合, 而 tp 曲线每层一次
# allreduce × 78 层, 低并发时是纯延迟主导(实测 tp c=1 step 15.54ms vs sglang 11.63ms).
# 复发时 AR_FUSION=0 关回去(还要同时在 curve env 里给 VLLM_ALLREDUCE_USE_SYMM_MEM=0,
# 两个入口得一起堵). 判据见 CLAUDE.md「多播死锁的根因与修法」: 卡在 torch.compile 且无报错.
#
# ★--compilation-config 只能给一次★, 所以下面把两处来源(cudagraph_mode 与 AR_FUSION 的
# pass_config)合成【一个】JSON —— 之前它们各自 SERVER_ARGS+= 会互相覆盖(后者胜), 静默丢配置.
#
# CUDAGRAPH_MODE: vllm 默认会同时捕 "mixed prefill-decode(PIECEWISE)" 与 "decode(FULL)" 两套图;
#   官方 GLM-5.2-NVFP4 的 decode 节点用 FULL_DECODE_ONLY(见 vllm.ai/blog/2026-07-23-glm-5.2-nvfp4-b300-pd),
#   纯 decode benchmark 下更贴近它。不设则走 vllm 默认。
_cc_parts=()
if [[ -n "${CUDAGRAPH_MODE:-}" ]]; then
    case "$CUDAGRAPH_MODE" in
        off|OFF|default) echo "CUDAGRAPH_MODE=$CUDAGRAPH_MODE: 不指定 cudagraph_mode (走 vllm 默认)";;
        *) _cc_parts+=("\"cudagraph_mode\":\"$CUDAGRAPH_MODE\""); echo "cudagraph_mode=$CUDAGRAPH_MODE";;
    esac
fi
# pass_config 有两个来源(AR_FUSION 与 PASS_CONFIG_EXTRA), ★必须并进同一个对象★ ——
# JSON 里同名键出现两次时后者覆盖前者, 各自 append 会静默丢掉一边.
_pc_parts=()
if [[ "${AR_FUSION:-1}" == "0" ]]; then
    _pc_parts+=('"fuse_allreduce_rms":false')
    echo "按 AR_FUSION=0 关闭 allreduce+rms 融合 pass (坏 fabric 节点用; 记得同时设 VLLM_ALLREDUCE_USE_SYMM_MEM=0)"
fi
# 逃生阀: 往 pass_config 里追加实验性融合开关(逗号分隔的 JSON 片段, 不含外层花括号).
# 例: PASS_CONFIG_EXTRA='"fuse_rope_kvcache_cat_mla":true,"enable_qk_norm_rope_fusion":true'
# ⚠️ 注意哪些是【ROCm 专用】、开了也会被平台判定关掉(实测源码 config/compilation.py):
#   fuse_act_padding / fuse_mla_dual_rms_norm / fuse_rope_kvcache / fuse_qk_norm_rope_kvcache
# 而 enable_sp / fuse_gemm_comms 在本模型上【必然被禁】: Blackwell 要求 hidden_size>=8192,
#   GLM-5.2 只有 6144 -> get_sequence_parallelism_threshold 返回 None -> 两者一起 False.
if [[ -n "${PASS_CONFIG_EXTRA:-}" ]]; then
    _pc_parts+=("$PASS_CONFIG_EXTRA")
    echo "pass_config 追加: $PASS_CONFIG_EXTRA"
fi
if ((${#_pc_parts[@]})); then
    _cc_parts+=("\"pass_config\":{$(IFS=,; echo "${_pc_parts[*]}")}")
fi
if ((${#_cc_parts[@]})); then
    _cc="{$(IFS=,; echo "${_cc_parts[*]}")}"
    SERVER_ARGS+=( --compilation-config "$_cc" )
    echo "--compilation-config $_cc"
fi

# .cmd 文件 = vllm serve 的复现记录: 先列该 server 进程实际读取、但不在命令行里的环境变量
# (每行 export VAR=值 # 说明), 再列启服务命令. 注: client/编排参数(ISL/OSL/CONC/NUM_WARMUPS/
# REPS/MAX_RETRY 等)不影响 server 进程, 不在此列(它们记在 run_config.json). 供报告"测试配置"展示.
{
  echo "# ===== 环境变量 ====="
  echo "# --- 容器/运行时 (bench.sh 注入) ---"
  emit_env PYTHONUNBUFFERED     "python 输出不缓冲, 日志实时"
  emit_env PYTHONNOUSERSITE     "忽略用户 site-packages, 用容器内干净环境"
  emit_env TORCH_CUDA_ARCH_LIST "目标 SM 架构 = B200 sm_100, 不编多余 arch"
  emit_env CUDA_DEVICE_ORDER    "按 PCI 总线枚举 GPU, 卡号稳定"
  emit_env PYTHONFAULTHANDLER   "崩溃时打印 python 栈 (SIGABRT/段错误便于定位)"
  echo "# --- vllm 关键/hack env ---"
  # 下面两条只在【坏 fabric 的逃生阀被打开时】才会出现(默认不设 -> emit_env 自动不打印):
  # 多播修好后两者都不该在, 若在 .cmd 里看到它们, 说明这次 run 是退化配置, 数不能与常态比.
  emit_env VLLM_ALLREDUCE_USE_SYMM_MEM     "★逃生阀★关对称内存(坏 fabric 才需要): 会退回 CUSTOM all-reduce, tp 低并发变慢"
  emit_env AR_FUSION                       "★逃生阀★=0 时关 allreduce+rms 融合 pass (与上一条配对使用)"
  emit_env VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE "关 inductor max-autotune: 否则首次编译卡死"
  emit_env VLLM_DEEP_GEMM_WARMUP           "跳过 DeepGEMM 预热, 加快 server 启动"
  # config 的 env 块里本脚本没显式列到的项也要进 .cmd (见 benchmark_lib.sh::emit_cfg_env)
  emit_cfg_env
  echo "# 注: 固定 accept_len 的 hack 在 --speculative-config 的 synthetic_acceptance_length (见启服务命令)"
  echo ""
  echo "# ===== vllm 启服务命令 ====="
  echo "vllm serve ${SERVER_ARGS[*]}"
} > "$RESULT_DIR/${RESULT_FILENAME}.cmd" 2>/dev/null || true

start_server() {
    stop_server
    set -x
    # ★追加而非截断★: 崩溃重试会再次调用 start_server, 用 > 会把上一次的崩溃现场覆盖掉。
    echo "===== start_server @ $(date '+%F %T') =====" >> "$SERVER_LOG"
    vllm serve "${SERVER_ARGS[@]}" >> "$SERVER_LOG" 2>&1 &
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
