#!/usr/bin/env bash
#
# 容器内脚本: 起一次 sglang server, 在 CONC 列表上循环压测. 所有配置由编排脚本
# bench.sh 按 runners/config.json 注入为环境变量 (本脚本不读 config.json, 也不解析
# 命令行 —— 配置的唯一入口是 config.json, 见 bench.sh 文件头).
#
# 必需: MODEL TP CONC ISL OSL RANDOM_RANGE_RATIO RESULT_FILENAME
# 可选(由 config.json 的 curve/defaults 落下来):
#   DP(默认=TP) EP(默认1=关; >1走deepep) MTP(默认1) SPEC_NUM_STEPS(=MTP的N) MTP_ACC(=accept_len)
#   MTP_DRAFT_PATH(默认 <MODEL 同级>/mtp) REPS MAX_RETRY RESULT_DIR
# 由 bench.sh 直接给(不在 config.json 里): BENCH_SGL_SRC / BENCH_SGL_COMMIT (build from source 的源码树
#   与 revision; 为空则用镜像自带引擎)
# 可选(config.json 的 `env` 字段, 调试/逃生阀): ARCH_RENAME(默认1) ARCH_BASE
#   KV_CACHE_DTYPE MEM_FRAC MAX_RUNNING_REQUESTS NUM_WARMUPS SERVER_WARMUP FLUSH_CACHE
#   WATCHDOG_TIMEOUT SGLANG_DISABLE_RAGGED FLASHINFER_PREWARM AR_FUSION

source "$(dirname "$0")/../../benchmark_lib.sh"
source "$(dirname "$0")/glm52_arch_bypass.sh"

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
SPEC_NUM_STEPS="${SPEC_NUM_STEPS:-5}"          # = curve.mtp_n; 默认 5 仅单独调试时用
SPEC_EAGLE_TOPK="${SPEC_EAGLE_TOPK:-1}"
# accept_len (= curve.mtp_acc): 每个 decode step 强制接受的 token 数 (含保底), clamp 到 [1, N+1].
# 用 sglang 内置的 "模拟接受长度" 特性 (SGLANG_SIMULATE_ACC_*), 而非改源码:
#   match-expected + 整数   -> 逐步恒定;
#   match-expected + 小数(如 cookbook low-latency 的 3.5) -> 按 lower/upper 权重伯努利抽样,
#     期望恰为该值 (spec_utils.generate_simulated_accept_index; env 是 EnvFloat, 接受小数).
# 这样 N (=--speculative-num-steps) 和 accept_len 都被硬固定且可复现.
# 默认值只在"脱离 bench.sh 单独调试"时生效, 取 cookbook low-latency 那格(= tp 曲线);
# 正常跑由 config.json 的 curve.mtp_n / curve.mtp_acc 逐曲线给定.
MTP_ACC="${MTP_ACC:-3.5}"
# MTP_ACC 可能是小数 -> 不能用 bash 整数比较 ([[ 3.5 -gt 0 ]] 是算术语法错, 会静默走 else
# 分支导致"以为固定了其实没固定"). 统一用 awk 做数值比较.
acc_gt0() { awk -v v="${1:-0}" 'BEGIN{exit !(v + 0 > 0)}'; }

# 并行 (TP/DP/EP 由 bench.sh 从 curve.parallel 翻译好; sglang 语义: 总卡数=tp, dp/ep 叠加).
#   DP>1 -> attention-DP(attn_tp=tp/dp); EP>1 -> MoE deepep 专家并行, 否则 MoE 张量并行.
DP="${DP:-$TP}"
EP="${EP:-1}"


# 本次扫描的最大 batch(用于 max-running-requests 与 CUDA graph 捕获上限, 不写死).
MAX_C=$(echo "$CONC" | tr ',' '\n' | tr -d ' ' | grep -E '^[0-9]+$' | sort -n | tail -1)
[[ -z "$MAX_C" || "$MAX_C" -lt 1 ]] && MAX_C=256
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-$MAX_C}"

# ★低并发 + EP 的两个下限(都实测崩过, 2026-07-28 的 c=1 扫描)★
# sglang 在 EP/dp-attn 下会把这两个值再除/过滤一遍, 低并发时会算出 0 或空集:
#  ① dep(DP>1): max_running_requests 按 attn_dp_size 整除(pool_configurator.py::num_reqs),
#     1//8=0 -> 8 个 scheduler 全抛 `AssertionError: max_running_request is zero`,
#     然后 `Rank 0 scheduler died during initialization`.
#  ② tep(EP>1 且 attn 走 TP): EP 后端要 gathered buffer -> get_batch_sizes_to_capture 里
#     mul_base = attn_tp_size = 8, 过滤 `bs*req_width % 8 == 0`; cuda-graph-max-bs=1 且
#     spec 的 req_width=6 -> 6%8!=0 -> `AssertionError: capture_bs=[]`.
# 下限取 DP(dp-attn) 或 TP(attn 走 TP): bs=TP 时 TP*任何 req_width 必被 TP 整除, 恒合法。
# 【不影响"max-running == 并发、不许排队"的口径】: 只有 1 个请求在飞时不可能排队, 这里抬的是
# 引擎的内部下限, 不是人为放大 offered 并发。
_ep_floor=1
if [[ "$EP" -gt 1 ]]; then
    if [[ "$DP" -gt 1 ]]; then _ep_floor="$DP"; else _ep_floor="$TP"; fi
fi
[[ "$MAX_RUNNING_REQUESTS" -lt "$_ep_floor" ]] && MAX_RUNNING_REQUESTS="$_ep_floor"

# CUDA graph 捕获上限. 约束:
#  - 关 dp-attn(DP=1): 每卡存全量 KV, 按过大 bs 捕获会 OOM -> 用 MAX_C 自适应, 不写死.
#  - offer 的并发可以超过 max-running-requests(cookbook high-throughput: offer 1024/跑 256),
#    此时按 MAX_C 捕图纯浪费 -> 先夹到 MAX_RUNNING_REQUESTS.
#  - 再抬到上面的 _ep_floor(见 ①②), 否则 capture_bs 会被算成 [0] 或空集.
CUDA_GRAPH_MAX_BS="$MAX_C"
[[ "$CUDA_GRAPH_MAX_BS" -gt "$MAX_RUNNING_REQUESTS" ]] && CUDA_GRAPH_MAX_BS="$MAX_RUNNING_REQUESTS"
[[ "$CUDA_GRAPH_MAX_BS" -lt "$_ep_floor" ]] && CUDA_GRAPH_MAX_BS="$_ep_floor"
[[ "$DP" -gt 1 && "$CUDA_GRAPH_MAX_BS" -lt "$DP" ]] && CUDA_GRAPH_MAX_BS="$DP"

# 静态显存占比(权重 + KV 池). 是 "KV池 ↔ 运行时余量" 的旋钮(权重固定 ~106G/卡):
#   高(0.85) -> KV 池大但运行时紧: EP 时 batch256 的 prefill MHA 激活(~3G)OOM;
#   低(0.75) -> 运行时松但 KV 池小: batch256 decode 触发 KV retract -> STPS 崩.
# 实测 dep(ep8,ISL8192): 0.80 两头都够(256 peak KV usage 0.80, retract=0 oom=0, 干净跑通),
#   KV 阈值 ~314 并发. 故 EP 默认 0.80; 非 EP(dpa-tp 无 deepep 通信缓冲, 运行时更宽)默认 0.85.
#   config.json 的 env.MEM_FRAC 优先. (1024 仍装不下 KV, 需 offer+排队, 见 cookbook high-throughput.)
if [[ -n "${MEM_FRAC:-}" ]]; then :; elif [[ "$EP" -gt 1 ]]; then MEM_FRAC=0.80; else MEM_FRAC=0.85; fi

# chunked prefill: 默认 32768; config.json 的 env.CHUNKED_PREFILL_SIZE=off/0 则【不下发该 flag】
# (用 sglang 自己的默认, B200 上 16384) —— cookbook low-latency 那格就是这样.
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-32768}"
CHUNKED_PREFILL_ARGS=()
case "$CHUNKED_PREFILL_SIZE" in
    off|OFF|default|0|"") ;;
    *) CHUNKED_PREFILL_ARGS=(--chunked-prefill-size "$CHUNKED_PREFILL_SIZE") ;;
esac

# 关闭 client 侧 warmup(benchmark_lib.sh 默认 --num-warmups=2*concurrency):
# 本项目靠稳态窗口(max-TTFT ~ min-finish)刨 prefill, 不靠 warmup 请求; JIT/cudagraph
# 预热由下方 SERVER_WARMUP 块负责. 关掉省一波冗余负载(默认 0, 可用 NUM_WARMUPS 覆盖).
export NUM_WARMUPS="${NUM_WARMUPS:-0}"

# 结果/日志直接写 outdir(bench.sh 以绝对路径传入, 在 /tilert 挂载下容器可写),
# 不再落在 /workspace(=仓库 InferenceX 目录). 否则被中断的 run 会把 json/.cmd/.serverlog
# 留在仓库里, 和历史结果混淆. 未传 RESULT_DIR 时回退 /workspace(兼容单独调试).
RESULT_DIR="${RESULT_DIR:-/workspace}"
SERVER_LOG="$RESULT_DIR/${RESULT_FILENAME:-server}.serverlog"

echo "CONC=$CONC ISL=$ISL OSL=$OSL RANGE=$RANDOM_RANGE_RATIO MTP=$MTP N=$SPEC_NUM_STEPS ACC=$MTP_ACC TP=$TP DP=$DP EP=$EP RESULT_DIR=$RESULT_DIR"

# ====== architectures 旁路: 走通 sglang 的 DSA 自动配置 (对齐 cookbook) ======
# 本地 checkpoint 标 architectures=DeepseekV32ForCausalLM, 不在 sglang
# _handle_model_specific_adjustments 的白名单里 -> 整个 DSA 块被跳过(KV 退 bf16、
# attention 退 flashinfer、page_size 留 1), 跟 cookbook 不是一条路径. 改名成
# GlmMoeDsaForCausalLM(同一个实现类)后自动配置生效: nsa 后端 + trtllm NSA
# prefill/decode + page_size=64 + kv_cache_dtype=fp8_e4m3. 详见 glm52_arch_bypass.sh.
# 主模型与 draft(mtp/) 都要改: 上游 _config_draft_model 的 NextN 改写白名单也不含 V32,
# 不改 draft 会按整模型(78 层)加载 -> 崩/OOM.
# ARCH_RENAME=0 可退回旧行为(错配路径, 仅供 A/B 对照).
# 本地 checkpoint 的 MTP 层单独在 mtp/ 目录, 必须显式指定 draft 路径
# (cookbook 的 zai-org/GLM-5.2-FP8 把 MTP 烘焙进主权重, 故其命令无此 flag).
MTP_DRAFT_PATH="${MTP_DRAFT_PATH:-${MODEL%/model}/mtp}"
ARCH_RENAME="${ARCH_RENAME:-1}"
ARCH_RENAMED=0
if [[ "$ARCH_RENAME" == "1" ]]; then
    _m="$(apply_arch_bypass "$MODEL")" || { echo "!! 主模型 arch 旁路失败, 中止"; exit 1; }
    MODEL="$_m"; ARCH_RENAMED=1
    if [[ "$MTP" == "1" ]]; then
        _d="$(apply_arch_bypass "$MTP_DRAFT_PATH")" || { echo "!! draft arch 旁路失败, 中止"; exit 1; }
        MTP_DRAFT_PATH="$_d"
    fi
else
    echo "ARCH_RENAME=0: 用原样 architectures (DSA 自动配置不生效, 见 glm52_arch_bypass.sh)"
fi

# ====== build from source: 用源码那份 sglang 覆盖镜像自带版本 ======
# 镜像(lmsysorg/sglang:latest)只当运行时基座(CUDA/torch/flashinfer/deepep/nvshmem 等依赖);
# 引擎本体用 config.json 里钉死的 commit 从源码装成 editable, 好跟 cookbook 数字可比且版本可复现.
# 源码树由 bench.sh 在宿主侧准备好并挂进来(BENCH_SGL_SRC), 这里只做 editable 安装:
#   --no-deps        依赖全部来自基座镜像, 不动它 (也避免容器无外网时卡死)
#   --no-build-isolation  不新建构建环境(否则要联网装 setuptools 等)
# 装完必须校验【实际生效的】是源码那份: import sglang 的路径要落在 BENCH_SGL_SRC 下,
# 否则说明 editable 没盖住镜像自带的 site-packages 版本 -> 直接中止, 不能拿错版本的数字.
# BENCH_SGL_SRC 为空 = 不从源码构建, 用镜像自带版本 (config.json 里 commit=null 的情形).
if [[ -n "${BENCH_SGL_SRC:-}" ]]; then
    [[ -d "$BENCH_SGL_SRC/python" ]] || { echo "!! BENCH_SGL_SRC=$BENCH_SGL_SRC 下没有 python/ 子目录, 不像 sglang 源码树"; exit 1; }
    # ---- 先对齐 sgl-kernel 版本(否则装上了也跑不起来) ----
    # 源码树的 pyproject 钉死了它要的 kernel 版本(`sglang-kernel==X.Y.Z`), 而基座镜像带的是
    # 【镜像自己那个 sglang 版本】要的 kernel. 两者不一致时不会有友好报错:
    #   - 版本太低 -> engine.py::_set_envs_and_config 的最低版本断言直接抛异常(还算明显);
    #   - 版本太高 -> 最低版本断言【能过】, 但符号已改名/删除, import 才炸. 实测 latest 基座
    #     带 0.4.5, 而 09ca4fc 要 sgl_kernel.fp8_blockwise_scaled_mm, 0.4.5 里已没有.
    # 所以按源码的 pin 校准(只在不一致时装, 装完复查). pip 走 PIP_PROXY(node071 出网受限);
    # 【只给 pip 加 --proxy, 不设容器级 HTTP_PROXY】—— 后者会把 client 连 localhost:PORT 的
    # 压测请求也代理走, 直接把 run 弄坏.
    _kpin="$(grep -oE 'sglang-kernel==[0-9][0-9a-zA-Z.]*' "$BENCH_SGL_SRC/python/pyproject.toml" | head -1 | cut -d= -f3)"
    _kcur="$(python3 -c 'import importlib.metadata as m;print(m.version("sglang-kernel"))' 2>/dev/null)"
    echo "sgl-kernel: 源码要求=${_kpin:-未标注}  基座现装=${_kcur:-无}"
    if [[ -n "$_kpin" && "$_kpin" != "$_kcur" ]]; then
        echo "===== 对齐 sgl-kernel -> $_kpin (基座是 $_kcur) ====="
        pip install -q ${PIP_PROXY:+--proxy "$PIP_PROXY"} "sglang-kernel==$_kpin" 2>&1 | grep -vE "^WARNING: Running pip|dependency resolver|sglang 0" || true
        _kcur="$(python3 -c 'import importlib.metadata as m;print(m.version("sglang-kernel"))' 2>/dev/null)"
        [[ "$_kcur" == "$_kpin" ]] || { echo "!! sgl-kernel 对齐失败(现为 ${_kcur:-无}, 需 $_kpin), 中止"; exit 1; }
    fi
    echo "===== build from source: pip install -e $BENCH_SGL_SRC/python (commit=${BENCH_SGL_COMMIT:-?}) ====="
    pip install -e "$BENCH_SGL_SRC/python" --no-deps --no-build-isolation -q \
        || { echo "!! sglang editable 安装失败"; exit 1; }
    _sgl_file="$(python3 -c 'import sglang,sys;sys.stdout.write(sglang.__file__)' 2>/dev/null)"
    _sgl_ver="$(python3 -c 'import sglang,sys;sys.stdout.write(str(sglang.__version__))' 2>/dev/null)"
    echo "sglang.__file__=$_sgl_file  __version__=$_sgl_ver  sgl-kernel=$_kcur"
    case "$_sgl_file" in
        "$BENCH_SGL_SRC"/*) echo "校验通过: 生效的是源码那份 ($BENCH_SGL_COMMIT)";;
        *) echo "!! 校验失败: import sglang 落在 $_sgl_file, 不在 $BENCH_SGL_SRC 下 -> 仍是镜像自带版本, 中止"; exit 1;;
    esac
fi

# ====== flashinfer cubin symlink 预热 (消 8-rank 竞态) ======
# DSA 自动配置生效后 MoE 走 trtllm-gen fp8 kernel, 首次用会在
# FLASHINFER_CUBIN_DIR/flashinfer/trtllm/batched_gemm/ 下建 symlink 指向 cubin 目录.
# flashinfer 的 jit/cubin_loader.py::ensure_symlink 是 TOCTOU 的("不存在才建"),
# 8 个 TP rank 在 kernel_warmup->_flashinfer_autotune 里同时首次触发 -> 抢建同一 symlink,
# 输的那些 rank 抛 FileExistsError, scheduler 全崩 (实测 tp 曲线 100% 复现).
# 修法: 起 server 前用【单进程】先触发一次模块生成把 symlink 建好; ensure_symlink 见到
# "已存在且指向正确"会直接 return, 之后 8 个 rank 都走这条早退路径, 竞态消失.
# 非致命: 预热失败(如换了 flashinfer 版本没这个函数)只告警, 让 server 自己去撞, 便于暴露.
if [[ "${FLASHINFER_PREWARM:-1}" == "1" ]]; then
    echo "===== flashinfer cubin symlink 预热 (单进程, 消 8-rank TOCTOU 竞态) ====="
    python3 -c '
from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module as g
g()
print("prewarm ok: trtllm_gen_fused_moe_sm100 cubin symlink 就位")
' || echo "(预热跳过: flashinfer 侧无此路径或取 cubin 失败, 见上方报错)"
fi

SERVER_ARGS=(
    --model-path="$MODEL"
    --trust-remote-code
    --host=0.0.0.0 --port="$PORT"
    --enable-metrics
    --tensor-parallel-size="$TP"
    --mem-fraction-static="$MEM_FRAC"
    --max-running-requests="$MAX_RUNNING_REQUESTS"
    ${CHUNKED_PREFILL_ARGS[@]+"${CHUNKED_PREFILL_ARGS[@]}"}
    # 本地 checkpoint 必需 (cookbook 的 zai-org 模型不需要): 不设则用模型
    # max_position=1048576(1M!) → KV池/并发估算按 1M 算, 异常.
    --context-length=16384
    --cuda-graph-max-bs="$CUDA_GRAPH_MAX_BS"  # 不设 sglang 默认~160, 高并发退 eager 变慢. (dp-attn 的 >=DP 保护见上)
)

# page_size / kv_cache_dtype / attention_backend 全部【交给 sglang 的 DSA 自动配置】——
# 前提是上面的 architectures 改名生效(ARCH_RENAMED=1), 此时 _handle_model_specific_adjustments
# 会自动设 page_size=64 + attention_backend=nsa + NSA prefill/decode=trtllm +
# kv_cache_dtype=fp8_e4m3(Blackwell), 与 cookbook 同路径, 故这里【不再手动指定】.
# ARCH_RENAME=0 的对照跑走的是错配路径, 必须补回手动 page-size(NSA KV pool 硬断言
# page_size==64, 不设 AssertionError 崩) 且【不能】设 fp8 KV(flashinfer MLA + DSA pool
# 错配下 NSA prefill 会进 _get_mla_kv_buffer_from_fp8_for_nsa 引用不存在的
# page_table_1_flattened -> 大 prefill batch 时 scheduler 崩).
if [[ "$ARCH_RENAMED" != "1" ]]; then
    SERVER_ARGS+=( --page-size=64 )
fi
# 逃生阀: 自动选的 fp8 KV 若在某个 build 上翻车, 用 KV_CACHE_DTYPE=bfloat16 显式压回.
if [[ -n "${KV_CACHE_DTYPE:-}" ]]; then
    SERVER_ARGS+=( --kv-cache-dtype="$KV_CACHE_DTYPE" )
    echo "显式 --kv-cache-dtype=$KV_CACHE_DTYPE (覆盖 DSA 自动选择)"
fi

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
    # MTP_DRAFT_PATH 已在上面的 arch 旁路段解析好(旁路生效时指向改名后的 draft 目录)
    SERVER_ARGS+=( --speculative-algorithm=EAGLE
                   --speculative-num-steps="$SPEC_NUM_STEPS"
                   --speculative-eagle-topk="$SPEC_EAGLE_TOPK"
                   --speculative-draft-model-path="$MTP_DRAFT_PATH"
                   )
fi

# 固定 accept_len: 必须在启动 sglang 进程前设好 (spec_utils 在 import 时读取该 env).
if [[ "$MTP" == "1" ]] && acc_gt0 "$MTP_ACC"; then
    export SGLANG_SIMULATE_ACC_LEN="$MTP_ACC"
    export SGLANG_SIMULATE_ACC_METHOD="match-expected"
    echo "固定 MTP: N(num-steps)=$SPEC_NUM_STEPS, accept_len(SIMULATE_ACC_LEN)=$MTP_ACC (match-expected)"
fi

# watchdog: 偶发 stall 会触发 sglang scheduler watchdog. 降到 120s 让崩溃更快暴露、
# 便于重启恢复 (OSL 下正常单次 forward 远 <120s, 不误伤).
SERVER_ARGS+=( --watchdog-timeout "${WATCHDOG_TIMEOUT:-120}" )

# ---- NVLink 多播 / 对称内存: 默认【用上游默认行为】, 只在坏 fabric 的节点上显式关 ----
# 新版 sglang 有三条路会用 NVLink 多播 / torch 对称内存, 在 node071 上 8 卡 rendezvous 死锁
# (node074 正常, 见 CLAUDE.md「node071 上要关掉两处多播对称内存路径」):
#   1) flashinfer AllReduce Fusion —— SM90/SM100 + 本模型 arch + tp>1 + 非 dp-attn + MoE 非 a2a
#      时【自动打开】(日志 "Auto-enabling FlashInfer AllReduce Fusion"), Blackwell 上 auto→mnnvl.
#      只有 tp 曲线满足这些条件. 关法: config env 里 AR_FUSION=0.
#   2) custom all-reduce v2 —— CUDA 上默认走 v2, 其 _init_workspace 会 _allocate_symmetric_memory
#      (torch _SymmetricMemory.empty_strided_p2p + rendezvous). 关法: config env 里
#      SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0 (退回 v1 的 cudaIpc, 仍保留 custom AR 加速).
#   3) logits 的 multimem all-gather (logits_processor -> triton_symm_mem_ag) —— **没有开关**,
#      所以坏 fabric 的节点上这个版本的 tp 曲线跑不了, 只能换节点.
# 这三项都是性能特性, 关掉会偏慢也偏离 cookbook 环境 -> 默认不动, 谁需要谁在 config 里关.
if [[ "${AR_FUSION:-1}" == "0" ]] \
   && python3 -m sglang.launch_server --help 2>/dev/null | grep -q -- "--enforce-disable-flashinfer-allreduce-fusion"; then
    SERVER_ARGS+=( --enforce-disable-flashinfer-allreduce-fusion )
    echo "按 AR_FUSION=0 显式关闭 flashinfer AllReduce Fusion (坏 fabric 节点用)"
fi
if [[ -n "${SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2:-}" ]]; then
    export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2
    echo "custom all-reduce: SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=$SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2 (0=退回 v1 的 cudaIpc)"
fi

# tep(attn-TP8 + MoE-EP) + MTP 下, sglang 的 NSA/MLA ragged prefill 会崩:
#   "q.shape[0](8) does not match qo_indptr[-1](6)" (spec verify 的 ragged 形状不符).
# 关掉 ragged MLA prefill 走 padded 路径绕开 (只影响 prefill, decode 稳态口径无损).
# 这是 flashinfer MLA 后端 + DSA KV pool 错配态下的产物: arch 改名 / 新版 sglang 下
# attention_backend 自动走 dsa|nsa(trtllm prefill), 该 flag 不再适用 -> 默认【不加】.
# 但保留逃生阀: config 的 env 里显式 SGLANG_DISABLE_RAGGED=1 就强制加回(tep 若仍崩用它),
# 反之 =0 可在未改名时也不加. 仅 tep(EP>1 且 DP<=1, 即 attn 非 DP) 有意义; dep/tp 不动.
_dr_default=1; [[ "$ARCH_RENAMED" == "1" ]] && _dr_default=0
if [[ "${SGLANG_DISABLE_RAGGED:-$_dr_default}" == "1" && "${EP:-1}" -gt 1 && "${DP:-$TP}" -le 1 ]]; then
    SERVER_ARGS+=( --flashinfer-mla-disable-ragged )
    echo "tep+MTP: 加 --flashinfer-mla-disable-ragged 绕开 NSA ragged prefill 形状崩"
fi

# .cmd 文件 = launch_server 的复现记录: 先列该 server 进程实际读取、但不在命令行里的环境变量
# (每行 export VAR=值 # 说明), 再列启服务命令. 注: client/编排参数(ISL/OSL/CONC/NUM_WARMUPS/
# REPS/MAX_RETRY 等)不影响 server 进程, 不在此列(它们记在 run_config.json). 供报告"测试配置"展示.
{
  echo "# ===== 引擎版本 ====="
  if [[ -n "${BENCH_SGL_SRC:-}" ]]; then
    echo "# build from source: commit=${BENCH_SGL_COMMIT:-?}  源码树=$BENCH_SGL_SRC"
    echo "# 实际生效: $(python3 -c 'import sglang,sys;sys.stdout.write(sglang.__file__+" v"+str(sglang.__version__))' 2>/dev/null)"
  else
    echo "# 用镜像自带 sglang (config.json 里该 backend/curve 的 commit=null)"
    echo "# 实际生效: $(python3 -c 'import sglang,sys;sys.stdout.write(sglang.__file__+" v"+str(sglang.__version__))' 2>/dev/null)"
  fi
  echo "# architectures 旁路: ARCH_RENAMED=$ARCH_RENAMED (1=已改名成 GlmMoeDsaForCausalLM, DSA 自动配置生效)"
  echo ""
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
  emit_env SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2 "custom all-reduce 退回 v1(cudaIpc); v2 的对称内存 rendezvous 在 node071 8卡死锁"
  emit_env SGLANG_DEEPEP_USE_V2 "deepep v1 buffer; 单机无 RDMA GIN, v2 报 NCCL GIN unavailable (注: ≥0.5.16 源码已无此 env, 留给老 build)"
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

# GPU 指标也写 outdir(不传 --output 会落在 /workspace, 即仓库根, 污染仓库)
start_gpu_monitor --output "$RESULT_DIR/${RESULT_FILENAME}.gpu_metrics.csv"
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
