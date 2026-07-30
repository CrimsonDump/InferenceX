#!/usr/bin/env bash
# GLM-5.2 FP8 on B200, TokenSpeed (lightseek), MTP, 固定 seqlen (UTPS-STPS 曲线用).
#
# 在容器内执行 (镜像由 config.json 的 backends.tokenspeed.image 决定, 当前 :glm-radix ——
# GLM-5.2 的 DSA 需要 radix 调度器; :tml 那个是 Inkling flat-KV 构建, 不行). 与 sglang/vllm
# 同口径的第三条对照曲线: 单节点合一 (unified, 非 PD 分离), 一个 ts engine 同时
# prefill+decode, benchmark client 也在同容器内跑. 发压方式与固定 MTP 的做法完全一致.
#
# TokenSpeed CLI = `ts serve` / `tokenspeed serve` (entry point tokenspeed.cli:main),
# 参数命名贴近 vLLM. GLM-5.2 原生支持 (models/glm5.py; recipe: docs/recipes/models.md
# 的 "GLM5 / GLM5.2" 节). OpenAI 兼容 /v1/completions + /health (control_server.py).
#
# ---- MTP 固定口径 (与 sglang/vllm 对齐) ----
# N (草稿步数) = --speculative-num-steps (脚本 -n, 默认 5); EAGLE topk=1 chain,
#   num_draft_tokens = N+1 (--speculative-num-draft-tokens 显式钉死, 不靠默认推导).
# accept_len 固定: TokenSpeed **无** sglang(SGLANG_SIMULATE_ACC_LEN) / vllm
#   (synthetic_acceptance_length) 那种内置模拟接受长度的开关. 故用一处**源码补丁**
#   (patch_ts_accept, 幂等) 在 model_executor.ModelExecutor._run_sampling 的 verify
#   返回处强制把 accept_lengths 填成常数 (= min(TS_SIMULATE_ACC_LEN, spec_num_tokens)),
#   等价 sglang 的 match-expected 逐步恒定. 补丁只在 TS_SIMULATE_ACC_LEN>0 时生效;
#   MTP_ACC<=0 则不打补丁 -> 走自然接受(对照用). 见文件末尾补丁块注释.
# 本地 checkpoint 把 MTP/NextN 层单独拆到 mtp/ 目录(主 model/ 无 nextn 权重),
#   故显式 --speculative-draft-model-path=<mtp路径>(等价 sglang/vllm 的 draft 路径).
#
# 必需环境变量: MODEL TP CONC ISL OSL RANDOM_RANGE_RATIO RESULT_FILENAME
# 可选: DP(=1; >1 走 attention-DP) EP(默认0=关; >=1 走 MoE 专家并行)
#       SPEC_NUM_STEPS(=MTP-N,默认5) MTP(默认1) MTP_ACC(=accept_len,默认6)
#       MTP_DRAFT_PATH REPS MAX_RETRY MEM_FRAC CONTEXT_LEN

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

nvidia-smi || true

# ---- DeepGEMM JIT (DSA indexer kernel, CUDA graph 捕获期用 nvcc 编) 的 CUDA 环境 ----
# :tml/glm-radix 镜像里 CUDA_HOME 空、nvcc 不在 PATH: deep_gemm 回退 cuda_home=/usr/local/cuda
# 且其 JIT 编 paged_mqa_logits 时报 `cuda/std/cstdint: No such file` —— CUDA 13 的 CCCL 头
# 在 targets/<arch>/include/cccl/ 下, 未进 nvcc include. 修: 指定 CUDA_HOME + PATH, 并用
# NVCC_APPEND_FLAGS 把 cccl include 强行追加到每次 nvcc 调用 (兜住 deep_gemm 自造的编译命令).
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$PATH"
_CCCL_INC="$CUDA_HOME/targets/x86_64-linux/include/cccl"
[[ -d "$_CCCL_INC" ]] && export NVCC_APPEND_FLAGS="-I$_CCCL_INC ${NVCC_APPEND_FLAGS:-}"
echo "CUDA_HOME=$CUDA_HOME nvcc=$(command -v nvcc || echo MISSING) NVCC_APPEND_FLAGS=${NVCC_APPEND_FLAGS:-}"

# ---- torch 对称内存后端 (仅在需要时覆盖) ----
# TokenSpeed 的 RSAG (MoE EP dispatch) 走 torch._symmetric_memory + multimem 多播 PTX, 需要
# 8-way NVLink 多播 rendezvous 可用. node074 默认(CUDA)后端 8 卡已验证 OK -> 默认不覆盖.
# 坏 fabric 的节点上默认后端会 8 卡多播死锁(当年 node071 就是, 根因 GPU6 掉链路, 已修好但
# tokenspeed 未在那台复验): 可试 TS_SYMMMEM_BACKEND=NVSHMEM(4 卡验证过, 但当年即使换后端整
# 链路仍卡, 见 memory tokenspeed-glm52-bringup). 用 TS_SYMMMEM_BACKEND 显式指定才覆盖.
if [[ -n "${TS_SYMMMEM_BACKEND:-}" ]]; then
    export TORCH_SYMMMEM="$TS_SYMMMEM_BACKEND"
    echo "TORCH_SYMMMEM=$TORCH_SYMMMEM (显式覆盖)"
else
    echo "TORCH_SYMMMEM 未覆盖 (用 torch 默认后端; node074 8卡已验证可用)"
fi

# ---- harmony/tiktoken 离线 vocab (smg router 启动必需) ----
# smg (rust router) 启动时加载 Harmony 编码, 默认从 openaipublic.blob.core.windows.net 下
# o200k_base.tiktoken; 无外网会超时崩(rc=1, 引擎其实已就绪). 指 TIKTOKEN_ENCODINGS_BASE 到本地
# 目录(含 o200k_base.tiktoken, sha256 446a9538...)即离线加载. 见 rust .so 的 TIKTOKEN_ENCODINGS_BASE.
_TKB="${TIKTOKEN_ENCODINGS_BASE:-/tilert/xbj/tiktoken_enc}"
if [[ -f "$_TKB/o200k_base.tiktoken" ]]; then
    export TIKTOKEN_ENCODINGS_BASE="$_TKB"
    echo "TIKTOKEN_ENCODINGS_BASE=$_TKB (离线 harmony vocab)"
else
    echo "!! 警告: $_TKB/o200k_base.tiktoken 不存在, smg 可能因下载 harmony vocab 失败而崩" >&2
fi

# ---- MTP / 并行配置 ----
MTP="${MTP:-1}"
SPEC_NUM_STEPS="${SPEC_NUM_STEPS:-5}"          # = MTP-N (-n), 默认 5
MTP_DRAFT_PATH="${MTP_DRAFT_PATH:-${MODEL%/model}/mtp}"
MTP_ACC="${MTP_ACC:-6}"                          # accept_len (-a), 常数, [1,N+1]
NUM_DRAFT_TOKENS=$(( SPEC_NUM_STEPS + 1 ))       # EAGLE topk=1 chain
# 并行(vllm 语义, 由 bench.sh 的 --parallel 预设翻译好): DP=attn 数据并行, EP=0/1 专家并行开关.
#   TokenSpeed 语义: DP>1 -> --data-parallel-size DP (attn DP, attn_tp=world/DP);
#                    DP==1 -> --tensor-parallel-size TP (attn TP);
#                    EP>=1 -> --enable-expert-parallel (MoE 专家并行) + flashinfer_trtllm.
DP="${DP:-1}"
EP="${EP:-0}"

# 客户端 warmup 请求数 (默认 0; defaults.num_warmups 覆盖, 复现 cookbook 时用 64).
export NUM_WARMUPS="${NUM_WARMUPS:-0}"

CONTEXT_LEN="${CONTEXT_LEN:-16384}"            # 远大于 ISL+OSL; 小 context 便于高并发 KV 估算
MEM_FRAC="${MEM_FRAC:-0.9}"                     # --gpu-memory-utilization (recipe 默认 0.9)
CHUNKED_PREFILL="${CHUNKED_PREFILL:-8192}"     # recipe 默认
# max-num-seqs 按本次扫描的最大 batch 自适应
MAX_C=$(echo "$CONC" | tr ',' '\n' | tr -d ' ' | grep -E '^[0-9]+$' | sort -n | tail -1)
[[ -z "$MAX_C" || "$MAX_C" -lt 1 ]] && MAX_C=256
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$MAX_C}"
# attention-DP(dep) 要求 max_num_seqs >= attn_dp_size(=DP), 否则 ts 启动即
#   "ValueError: max_num_seqs must be >= attn_dp_size". 单跑低并发(如 -c 1)时兜底抬到 DP.
[[ "$DP" -gt 1 && "$MAX_NUM_SEQS" -lt "$DP" ]] && MAX_NUM_SEQS="$DP"

# 结果/日志直接写 outdir(bench.sh 以绝对路径传入, /tilert 挂载下可写), 不落在 /workspace.
RESULT_DIR="${RESULT_DIR:-/workspace}"
# ★逐曲线一份日志★(与 sglang 那份一致): 共用 server.log 的话, 同一次 run 里的多条 tokenspeed
# 曲线会互相覆盖, 事后只剩最后一条的日志 —— 而排查全靠它。
SERVER_LOG="$RESULT_DIR/${RESULT_FILENAME:-server}.serverlog"

echo "CONC=$CONC ISL=$ISL OSL=$OSL RANGE=$RANDOM_RANGE_RATIO MTP=$MTP N=$SPEC_NUM_STEPS ACC=$MTP_ACC TP=$TP DP=$DP EP=$EP"

# ====== MTP 固定 accept_len 的源码补丁 (幂等) ======
# TokenSpeed 无内置模拟接受长度开关; 在 _run_sampling 的 verify 返回处强制常数 accept.
# 只在 MTP=1 且 MTP_ACC>0 时打; 运行期由 TS_SIMULATE_ACC_LEN 控制生效值.
patch_ts_accept() {
python3 - <<'PYEOF'
import importlib.util, sys
spec = importlib.util.find_spec("tokenspeed.runtime.execution.model_executor")
if spec is None or not spec.origin:
    print("!! 找不到 model_executor, 跳过 accept 补丁", file=sys.stderr); sys.exit(0)
path = spec.origin
src = open(path).read()
if "_apply_simulate_acc_len" in src:
    print(f"accept 补丁已存在: {path}"); sys.exit(0)

lines = src.splitlines(keepends=True)
out = []
inserted_calls = 0
for ln in lines:
    stripped = ln.strip()
    indent = ln[:len(ln) - len(ln.lstrip())]
    # 在每处 `... = self._apply_force_single_token_verify(` 之前插入 sim 调用,
    # 对同名变量 (accept_lengths / decode_accept) 生效.
    if stripped.startswith("accept_lengths = self._apply_force_single_token_verify("):
        out.append(f"{indent}accept_lengths = self._apply_simulate_acc_len(accept_lengths)\n")
        inserted_calls += 1
    elif stripped.startswith("decode_accept = self._apply_force_single_token_verify("):
        out.append(f"{indent}decode_accept = self._apply_simulate_acc_len(decode_accept)\n")
        inserted_calls += 1
    # 在 _apply_force_single_token_verify 方法定义前插入 _apply_simulate_acc_len 方法定义.
    if stripped.startswith("def _apply_force_single_token_verify("):
        method = (
            f"{indent}def _apply_simulate_acc_len(self, accept_lengths):\n"
            f"{indent}    # 固定 MTP accept_len 的 hack (等价 sglang SGLANG_SIMULATE_ACC_LEN /\n"
            f"{indent}    # vllm synthetic_acceptance_length): 每 decode step 强制恰好接受 K 个\n"
            f"{indent}    # (K = min(TS_SIMULATE_ACC_LEN, spec_num_tokens), clamp 到候选宽度内).\n"
            f"{indent}    # in-place: drafter 读同一 buffer 定下一 block 大小 (见 _cap_accept 注释).\n"
            f"{indent}    #\n"
            f"{indent}    # ★★ 本函数是在 cuda graph 【捕获期】执行的 (torch.cuda.graph(...) 内),\n"
            f"{indent}    # replay 时 python 完全不跑 —— 所以只能用 device 侧算子, 且严禁任何\n"
            f"{indent}    # .item()/同步 ★★  (fp8 那边踩过, 见 pitfalls/tokenspeed.md 第 2 条):\n"
            f"{indent}    #   1) 用 python 的 random 抽 k 再 fill_(k): 被当成【常量 fill】捕获进图,\n"
            f"{indent}    #      每个 bs 桶的 k 就此永久冻结 -> 非整数 acc 永远打不到目标值,\n"
            f"{indent}    #      UTPS 系统性偏低 ~15% 且【无任何报错】. 整数 acc(如 5) 恰好无害,\n"
            f"{indent}    #      所以这个 bug 在 acc=5 的 fp4 曲线上藏了很久。\n"
            f"{indent}    #   2) 调 .item() 看中间值: 捕获期同步 -> 进程当场 abort\n"
            f"{indent}    #      (captures_underway.empty() INTERNAL ASSERT FAILED)。\n"
            f"{indent}    # torch.rand 落在捕获的图里是安全的: torch.cuda.graph 会注册 generator\n"
            f"{indent}    # 状态, 每次 replay 推进 philox offset -> 逐 step 真正重新抽。\n"
            f"{indent}    import math as _math, os as _os\n"
            f"{indent}    import torch as _torch\n"
            f"{indent}    try:\n"
            f"{indent}        _a = float(_os.environ.get('TS_SIMULATE_ACC_LEN', '0') or '0')\n"
            f"{indent}    except ValueError:\n"
            f"{indent}        _a = 0.0\n"
            f"{indent}    if _a <= 0 or not accept_lengths.numel():\n"
            f"{indent}        return accept_lengths\n"
            f"{indent}    _w = int(getattr(self.config, 'spec_num_tokens', 0) or 0)\n"
            f"{indent}    if _w > 0:\n"
            f"{indent}        _a = min(_a, float(_w))\n"
            f"{indent}    # 小数(cookbook low-latency 的 3.5)按 lower/upper 伯努利抽, 期望恰为该值\n"
            f"{indent}    # —— 与 sglang match-expected / vllm 的 synthetic 接受率同分布.\n"
            f"{indent}    _lo = int(_math.floor(_a)); _hi = _lo + 1 if _lo < _a else _lo\n"
            f"{indent}    _p = _a - _lo\n"
            f"{indent}    if _os.environ.get('TS_ACC_DEBUG'):\n"
            f"{indent}        # 无同步的自检行: capturing=True 就说明本函数确实跑在图捕获期.\n"
            f"{indent}        print('[TS_ACC_DEBUG] a=%s lo=%d hi=%d p=%.3f rows=%d capturing=%s' % (_a, _lo, _hi, _p, accept_lengths.numel(), _torch.cuda.is_current_stream_capturing()), flush=True)\n"
            f"{indent}    if _hi == _lo:\n"
            f"{indent}        accept_lengths.fill_(max(1, _lo))\n"
            f"{indent}        return accept_lengths\n"
            f"{indent}    # 逐行伯努利: P(_hi)=_p, P(_lo)=1-_p. 全 device 侧, 可被图捕获.\n"
            f"{indent}    _r = _torch.rand(accept_lengths.shape, device=accept_lengths.device)\n"
            f"{indent}    _sel = (_r < _p).to(accept_lengths.dtype)\n"
            f"{indent}    accept_lengths.copy_((_sel * (_hi - _lo) + _lo).clamp_(min=1))\n"
            f"{indent}    return accept_lengths\n\n"
        )
        out.append(method)
    out.append(ln)

new_src = "".join(out)
if "_apply_simulate_acc_len(self" not in new_src or inserted_calls == 0:
    print(f"!! accept 补丁注入失败 (method_def={'_apply_simulate_acc_len(self' in new_src} calls={inserted_calls}); "
          f"TokenSpeed 版本可能变动, 请核对 model_executor._run_sampling", file=sys.stderr)
    sys.exit(1)
open(path, "w").write(new_src)
print(f"accept 补丁已注入: {path} (插入 {inserted_calls} 处调用)")
PYEOF
}

# ====== device_id 补丁 (修 B200 对称内存 rendezvous 死锁, 幂等) ======
# distributed_initializer.py 在 NVIDIA 上把 init_process_group 的 device_id 写死 None
# ("Device-scoped NCCL init is only required and tested on AMD"). 但 torch 会因此
# "Guessing device ID ... can cause a hang", 且 TritonRSAG/TRT-LLM 的对称内存 collective
# 在多进程/CUDA graph 捕获期 rendezvous 死锁 (tep/dep/eager 均复现, 卡在 RSAG symm buffer).
# 显式传 device_id=cuda:gpu_id 即修 (与 AMD 分支同). 修好后无需 --enforce-eager, 图可捕获.
patch_ts_device_id() {
python3 - <<'PYEOF'
import importlib.util, re, sys
spec = importlib.util.find_spec("tokenspeed.runtime.execution.distributed_initializer")
if spec is None or not spec.origin:
    print("!! 找不到 distributed_initializer, 跳过 device_id 补丁", file=sys.stderr); sys.exit(0)
path = spec.origin
src = open(path).read()
if "PATCHED_DEVICE_ID_B200" in src:
    print(f"device_id 补丁已存在: {path}"); sys.exit(0)
pat = re.compile(
    r"device_id\s*=\s*\(\s*torch\.device\(config\.device,\s*config\.gpu_id\)\s*"
    r"if current_platform\(\)\.is_amd\s*else None\s*\)")
new = ("device_id = torch.device(config.device, config.gpu_id)  "
       "# PATCHED_DEVICE_ID_B200: 对称内存 rendezvous 在多进程/图捕获需显式 device_id, 否则死锁")
src2, n = pat.subn(new, src)
if n != 1:
    print(f"!! device_id 补丁未命中 (n={n}); TokenSpeed 版本可能变动, 请核对 distributed_initializer", file=sys.stderr)
    sys.exit(1)
open(path, "w").write(src2)
print(f"device_id 补丁已注入: {path}")
PYEOF
}
patch_ts_device_id || { echo "!! device_id 补丁失败, 中止"; exit 1; }

# ====== 关 allreduce fusion 补丁 (幂等) ======
# resolve_communication 在 attn-TP 单节点上自动开 allreduce fusion, 其 TRT-LLM Lamport
# 对称内存 collective 在本机 8×B200 死锁 (同 vllm 需关对称内存). 本机永不想开它, 直接 neuter
# 自动开启. (纯 TP 关 fusion 后 allreduce 走 custom_ar 的 CUDA IPC 路径, 不碰对称内存.)
patch_ts_no_fusion() {
python3 - <<'PYEOF'
import importlib.util, sys
spec = importlib.util.find_spec("tokenspeed.runtime.utils.server_args")
if spec is None or not spec.origin:
    print("!! 找不到 server_args, 跳过 no-fusion 补丁", file=sys.stderr); sys.exit(0)
path = spec.origin
src = open(path).read()
if "PATCHED_NO_FUSION_B200" in src:
    print(f"no-fusion 补丁已存在: {path}"); sys.exit(0)
needle = '            self.enable_allreduce_fusion = True\n            logger.info("Auto-enabled allreduce fusion")'
repl = ('            self.enable_allreduce_fusion = False  # PATCHED_NO_FUSION_B200\n'
        '            logger.info("Auto-enable allreduce fusion SKIPPED (PATCHED_NO_FUSION_B200: TRT-LLM Lamport 对称内存在本机死锁)")')
if needle not in src:
    print("!! no-fusion 补丁未命中 anchor; TokenSpeed 版本可能变动", file=sys.stderr); sys.exit(1)
open(path, "w").write(src.replace(needle, repl, 1))
print(f"no-fusion 补丁已注入: {path}")
PYEOF
}
patch_ts_no_fusion || { echo "!! no-fusion 补丁失败, 中止"; exit 1; }

# MTP_ACC 可为小数 -> awk 数值比较 ([[ 3.5 -gt 0 ]] 是 bash 算术语法错, 会静默走 else).
# 注意补丁按 "_apply_simulate_acc_len 是否已存在" 判幂等: 若镜像里已 commit 过旧版补丁,
# 需先在容器内删掉旧方法(或重建镜像)才能吃到这里的小数支持.
acc_gt0() { awk -v v="${1:-0}" 'BEGIN{exit !(v + 0 > 0)}'; }
if [[ "$MTP" == "1" ]] && acc_gt0 "$MTP_ACC"; then
    patch_ts_accept || { echo "!! accept 补丁失败, 中止"; exit 1; }
    export TS_SIMULATE_ACC_LEN="$MTP_ACC"
    echo "固定 MTP: N(num-steps)=$SPEC_NUM_STEPS, num_draft_tokens=$NUM_DRAFT_TOKENS, accept_len(TS_SIMULATE_ACC_LEN)=$MTP_ACC"
else
    echo "MTP=$MTP MTP_ACC=$MTP_ACC: 不打 accept 补丁 (走自然接受, 对照用)"
fi

# ====== 架构名重映射 (DeepseekV32ForCausalLM -> GlmMoeDsaForCausalLM) ======
# 本地 checkpoint 的 config.json 把 GLM-5.2 标成 architectures=["DeepseekV32ForCausalLM"]
# (deepseek_v32), 但 TokenSpeed 只按 **GlmMoeDsaForCausalLM** 注册 GLM-5.2 (model registry +
# DSA attention family + NextN 检测 + get_config 的 _restore_raw_glm_dsa_fields 全按此名路由;
# 官方 recipe 用的 zai-org/GLM-5.2-FP8 即此名). vllm 的 glm52 镜像两名都注册故无此问题.
# 修法(隔离, 不动 /mnt/ramweights 共享 config, 免破坏 sglang/vllm 跑): 造一个只改 config.json
# (architectures->GlmMoeDsaForCausalLM)、其余文件全 symlink 的旁路目录, --model 指向它.
#   - draft(mtp/) 同样只标 GlmMoeDsaForCausalLM: get_config(is_draft_worker) 会自动 +NextN
#     -> GlmMoeDsaForCausalLMNextN (已注册), 同时 raw==GlmMoeDsaForCausalLM 让 DSA 字段 restore 生效.
# 实现在 glm52_arch_bypass.sh (与 sglang 那份共用同一套旁路目录; sglang 侧改名是为了走通
# DSA 自动配置, 原因不同但做法与产物完全一样, 故一份实现两处用). 落点默认 ${src}_glmarch,
# 源在只读区时用 ARCH_BASE(兼容旧名 TS_ARCH_BASE) 指到可写区.
MODEL="$(apply_arch_bypass "$MODEL")" || { echo "!! 主模型 arch 旁路目录失败, 中止"; exit 1; }
if [[ "$MTP" == "1" ]]; then
    MTP_DRAFT_PATH="$(apply_arch_bypass "$MTP_DRAFT_PATH")" \
        || { echo "!! draft arch 旁路目录失败, 中止"; exit 1; }
    echo "draft 的 +NextN 由 get_config(is_draft_worker) 自动补"
fi

# 量化方式(= defaults.quantization -> QUANTIZATION). tokenspeed 的 --quantization 枚举里有
# nvfp4; 不给则让它按权重目录的 quantization_config 自动识别(fp8 权重就这样).
# 注: tokenspeed 把 draft 的量化【单独】管 —— --speculative-draft-model-quantization 默认
# "unquant"(转成 None), 正好对上 nvidia/GLM-5.2-NVFP4 的 layer 78 是 BF16 这一事实
# (vllm 那边就是因为把全局 moe_backend 套到未量化的 draft 上, flashinfer_cutedsl 直接起不来).
QUANT_ARGS=()
if [[ -n "${QUANTIZATION:-}" ]]; then
    QUANT_ARGS=( --quantization "$QUANTIZATION" )
    echo "显式 --quantization $QUANTIZATION"
fi

# ====== ts serve 参数 ======
# model id = $MODEL 路径 (served_model_name 默认=model), 与 client --model "$MODEL" 对齐.
SERVER_ARGS=(
    serve
    --model "$MODEL"
    --trust-remote-code
    ${QUANT_ARGS[@]+"${QUANT_ARGS[@]}"}
    --host 0.0.0.0 --port "$PORT"
    --max-model-len "$CONTEXT_LEN"
    --chunked-prefill-size "$CHUNKED_PREFILL"
    --max-num-seqs "$MAX_NUM_SEQS"
    --gpu-memory-utilization "$MEM_FRAC"
    --enable-metrics               # 读 accept rate / Decoded Tok/Iter
    # ★钉死 smg 网关的 Prometheus 端口★(踩过, 2026-07-30, tokenspeed 0.1.0):
    # 上游 serve_smg._gateway_args_with_default_prometheus_port 会用 get_free_port() 现挑一个
    # 空闲端口再交给 smg 去 bind —— 典型 TOCTOU: 8 个 DP engine 自己也在抢端口(gRPC/权重传输/
    # dist), 中间被抢掉就 panic `metrics server bind failed: ... Address already in use` ->
    # 网关启动失败 -> engine 明明 SERVING 也被拖死("Server died before becoming healthy").
    # 注意与 --enable-metrics 无关: 网关无条件起这个 server, 关 metrics 也躲不开。
    # 本 bench 同一时刻只有一个 server, 且没人 scrape 这个端口(不会产生 TIME_WAIT), 故固定安全。
    --prometheus-port "${PROM_PORT:-$((PORT + 2000))}"
    # ★钉死 KVStore 的 host 池大小★(踩过, 2026-07-30, fp4 + 0.1.0): **`--disable-kvstore` 挡不住
    # host 池的分配** —— 日志里 `enable_kvstore=False`/`disable_kvstore=True` 照样走
    # MemoryExecutor.__init__ 建 host_pool. 而 `kvstore_size=0`(默认) 时按 `kvstore_ratio=2.0`
    # 自动定尺寸: DSA 每 token 55,224 B × (设备池 1,882,112 tok × 2.0) => **138 GB/rank**.
    # 自动定尺寸带 cgroup/可用内存的封顶(_auto_capped_host_size_tokens), 但 **per_rank_budget =
    # 可用内存 / nprocs_per_node 是各 rank 各自采样的** —— 8 个 rank 同时按"当时"的可用内存算,
    # 谁也不知道另外 7 个刚要吃掉多少, 于是 8×138 GB 把 host RAM 榨干, 紧随其后的 draft L2 池
    # (只要 1.77 GB)必崩: `ValueError: Not enough host memory available. Requesting 1.77 GB but
    # only have 0.62 GB free`. node076 上尤其容易撞: tmpfs(/mnt/ramweights 1.5T + /dev/shm 388G)
    # 已占掉 1.8T/3T RAM, 只剩 ~1.1T, 正好卡在 8×138 GB 的边上(日志里 8 条 Capping 也没救回来).
    # ★但不能砍到"够小"★(踩过, 同日): 这个池不只是 L2 前缀缓存, 还是**超容时的卸载目标** ——
    # tokenspeed 处理"装不下"的方式是把被抢占请求的 KV 卸到 host 池(不是 sglang 的重算式 retract、
    # 也不是 vllm 的排队). 池太小 -> `[Scheduler] Retract failed for request …: host capacity
    # exhausted, aborting request` -> 请求被 abort、流断(Broken pipe) -> **我们的 client 收不到
    # 终止事件, 永久挂住**(实测 tp c=256: 进度条停在 221/256 不动 29 分钟, 服务端 GPU 0%).
    # 定尺寸的依据 = 超容缺口: c 并发 × (ISL+OSL) - 设备池 tokens, 再 × 每 token 字节.
    #   实测 tp/ISL8192/OSL1024: c=256 需 2,359,296 tok/rank, 设备池 1,882,112 -> 缺口 26.4 GB/rank.
    # 48 GB/rank 给了 ~1.8× 余量, 8×48=384 GB, host RAM(~1 TB 可用)放得下, 也仍远小于自动的 138 GB。
    # `--kvstore-size` 是 GB, 显式给了就覆盖 kvstore_ratio, 绕开上面那套逐 rank 自动定尺寸。
    # ⚠ 口径提醒: 到了要卸载的并发档(tp c=256), 该点的 TPOT 里就含 host<->device 拷贝, 与 sglang
    #   (重算)/vllm(排队)的超容行为不同 —— 三家在那一格比的是"超容策略", 不是纯 decode 速度。
    --kvstore-size "${KVSTORE_SIZE_GB:-48}"
    # ★权重加载: 必开 prefetch, 否则在共享盘上慢 10 倍★(实测 2026-07-30, node076 + CephFS):
    # tokenspeed 用 mmap 读 safetensors(8 个 rank 的 rchar 各只 1 GiB, 字节全靠 page fault 进来),
    # 于是 8 个 rank 各自去 fault 全部 47 个 shard 里属于自己的【跨步切片】 -> 共享盘上退化成大量
    # 小的随机网络读, 同一段字节还被多个 rank 反复读. 实测同一份 433 GB nvfp4 权重:
    #   vllm(显式顺序读整文件再切片) 134 s (~3.2 GB/s)  vs  tokenspeed 默认 **24 分 20 秒**.
    # 这个 flag 让 8 个本地 rank 把 shard 列表分掉(sorted_files[rank::world_size], 各约 6 个),
    # 每 rank 起 N 个线程【顺序整文件】读进 OS page cache, 之后 mmap fault 全部命中内存.
    # 默认 False, 所以不给就一直付那 24 分钟(一次 run 两条曲线 = 白等近 1 小时).
    --weight-loader-prefetch-checkpoints
    --weight-loader-prefetch-num-threads "${WEIGHT_PREFETCH_THREADS:-8}"
    --disable-kvstore              # 关 host-offload L2 KV cache: bench 不需要; 省 host 内存池(tep KVStore 分配曾崩); recipe(V4/Inkling)亦用
    --disable-prefill-graph        # 关 prefill CUDA graph: tep(attn-TP) 的 GLM DSA prefill graph 有 "token count mismatch" bug; 走 eager prefill 绕开. 对本 bench 无损(稳态窗口本就刨 prefill; decode 图仍用). recipe(MiniMax)亦用.
)

# KV dtype (= curve 的一等字段 kv_cache_dtype). 默认 fp8 = GLM-5.2 recipe 的值, 也与 sglang
# 侧 DSA 自动选的 fp8_e4m3 同精度. ★取值用 backend 原生的名字★: TokenSpeed 认 "fp8",
# 不认 vllm 那个 "fp8_e4m3" —— 所以跨 backend 同图时这个字段通常【不要写在 defaults 级别】,
# 要写就逐曲线写. "off" = 不下发, 走 TokenSpeed 自己的默认.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
case "$KV_CACHE_DTYPE" in
    off|OFF|default) echo "KV_CACHE_DTYPE=off: 不下发 --kv-cache-dtype";;
    *) SERVER_ARGS+=( --kv-cache-dtype "$KV_CACHE_DTYPE" );;
esac

# CUDA graph 捕获在本机 8×B200 会卡死: 捕获期 TritonRSAG 的 torch 对称内存 collective
# (+ NCCL "Guessing device ID ... can cause a hang" 警告) rendezvous 死锁, tep/dep 均复现
# (与 vllm 的对称内存死锁同源). ENFORCE_EAGER=1 走 eager(不捕获图)绕开——代价是 decode
# 吞吐偏低(无图融合/CPU 开销大), 非代表性最优, 但可端到端跑通/出曲线. 图捕获修复待查.
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
    SERVER_ARGS+=( --enforce-eager )
    echo "ENFORCE_EAGER=1: 走 eager (跳过 CUDA graph 捕获, 绕开对称内存捕获死锁)"
fi

# 并行: attn TP vs DP
if [[ "$DP" -gt 1 ]]; then
    # dp_size>1 时 ts 要求显式 --dist-init-addr (单节点用 loopback 任一空闲端口).
    SERVER_ARGS+=( --data-parallel-size "$DP" --dist-init-addr "127.0.0.1:${DIST_INIT_PORT:-40001}" )
else
    SERVER_ARGS+=( --tensor-parallel-size "$TP" )
fi
# MoE 并行 + backend
if [[ "${EP:-0}" -ge 1 ]]; then
    # 专家并行: 每卡持完整专家, flashinfer_trtllm (recipe 默认).
    SERVER_ARGS+=( --enable-expert-parallel --moe-backend "${MOE_BACKEND:-flashinfer_trtllm}" )
    # MoE all-to-all: 默认 none(用 TritonRSAG 对称内存, 本机死锁); 可设 deepep 走 NVSHMEM
    #   (本机 NVLink 已由 sglang 验证可用), 绕开对称内存 RSAG dispatch.
    # ★env 名必须是 ALL2ALL_BACKEND★: 它是 config 里 curve 的 all2all_backend 字段, bench.sh
    #   对三个 backend 统一注入这个名字. 早前这里写的是 A2A_BACKEND -> 【config 配了也静默无效】
    #   (见 pitfalls/tokenspeed.md 第 3 条, fp8 脚本已改, 这里同步). "off" = 不下发.
    if [[ -n "${ALL2ALL_BACKEND:-}" && "${ALL2ALL_BACKEND}" != "off" ]]; then
        SERVER_ARGS+=( --all2all-backend "$ALL2ALL_BACKEND" )
        [[ -n "${DEEPEP_MODE:-}" ]] && SERVER_ARGS+=( --deepep-mode "$DEEPEP_MODE" )
    fi
else
    # MoE 张量并行(纯 tp): intermediate 被切分, flashinfer_trtllm 的 fp8 block-scale kernel
    #   断言 gemm1_weights_scale 形状 (期望完整专家) 会崩; 用 triton MoE (支持 TP 切分).
    SERVER_ARGS+=( --moe-backend "${MOE_BACKEND:-triton}" )
fi

# MTP: 拆分 draft 路径 + 固定 N (num-steps / num-draft-tokens) + eagle topk=1.
if [[ "$MTP" == "1" ]]; then
    SERVER_ARGS+=( --speculative-algorithm MTP
                   --speculative-num-steps "$SPEC_NUM_STEPS"
                   --speculative-eagle-topk 1
                   --speculative-num-draft-tokens "$NUM_DRAFT_TOKENS"
                   --speculative-draft-model-path "$MTP_DRAFT_PATH" )
fi

# ====== 只给 server 进程的出网代理 (flashinfer trtllm-gen cubin 运行期下载) ======
# ★踩过两次 (2026-07-30)★: tokenspeed 0.1.0 的 flashinfer-python 会把 trtllm-gen 的 cubin
# 【运行期按需从 edge.urm.nvidia.com 下载】(旧 :glm-radix 用自带的 tokenspeed-trtllm-kernel,
# 完全离线, 故没这问题). 无外网时 fp4 这边两条曲线各崩在一颗不同的 cubin 上:
#   parallel=tp    : DSA decode 的 fmha —— flashinfer.jit: Downloading fmhaSm100fKernel_QkvE4m3...
#                    -> RuntimeError: Failed to load cubin (崩在 cudagraph capture, server 起不来)
#   parallel=dpa-tp: MoE 的 batched GEMM —— trtllm_batched_gemm_runner.cu:305
#                    Error occurred when running GEMM! -> engine worker 挂 -> 网关 gRPC 断
# 该 host 经节点代理可达. ★绝不设成容器级 HTTP_PROXY★: 那会把 client 打向 0.0.0.0:$PORT 的
# 压测请求也代理走(NO_PROXY 写 127.0.0.1/localhost 盖不住 0.0.0.0), 整个 run 直接废掉。
# 所以只在 setsid 启 server 时用 env 注入。不同并发会选到不同 tile 的 kernel(= 不同 cubin),
# 故整条 sweep 都要留着这条出网, 不能只在第一个点预热一次。
# 注: bench.sh 把 /mnt/ramweights/jitp/cache 挂成容器的 /root/.cache, 所以下过的 cubin 会
#     持久留在宿主 cache 里, 后续 run 即便无代理也能命中。
SERVER_PROXY_ENV=()
if [[ -n "${SERVER_PROXY:-}" ]]; then
    SERVER_PROXY_ENV=( "HTTPS_PROXY=$SERVER_PROXY" "HTTP_PROXY=$SERVER_PROXY"
                       "https_proxy=$SERVER_PROXY" "http_proxy=$SERVER_PROXY"
                       "NO_PROXY=127.0.0.1,localhost,0.0.0.0,10.0.0.0/8"
                       "no_proxy=127.0.0.1,localhost,0.0.0.0,10.0.0.0/8" )
    echo "SERVER_PROXY=$SERVER_PROXY (只注入 server 进程, 供 flashinfer 下 cubin; client 不受影响)"
else
    echo "SERVER_PROXY 未设: 若 flashinfer 需要的 trtllm-gen cubin 不在 /root/.cache 里会下载失败 -> 起不来/GEMM 崩"
fi

# .cmd 文件 = 复现记录: 先列 server 进程读取但不在命令行的 env, 再列启服务命令.
{
  echo "# ===== 环境变量 ====="
  echo "# --- 容器/运行时 (bench.sh 注入) ---"
  emit_env PYTHONUNBUFFERED     "python 输出不缓冲, 日志实时"
  emit_env PYTHONNOUSERSITE     "忽略用户 site-packages, 用容器内干净环境"
  emit_env TORCH_CUDA_ARCH_LIST "目标 SM 架构 = B200 sm_100"
  emit_env CUDA_DEVICE_ORDER    "按 PCI 总线枚举 GPU, 卡号稳定"
  echo "# --- MTP 固定 accept_len hack (源码补丁 model_executor._apply_simulate_acc_len 运行期读取) ---"
  emit_env TS_SIMULATE_ACC_LEN  "固定 MTP accept_len: 每 decode step 强制恰好接受该值 (= min(值, num_draft_tokens))"
  # config 的 env 块里本脚本没显式列到的项也要进 .cmd (见 benchmark_lib.sh::emit_cfg_env)
  emit_cfg_env
  echo ""
  echo "# ===== TokenSpeed 启服务命令 ====="
  echo "# 注: 固定 accept_len 靠容器内源码补丁 (model_executor._apply_simulate_acc_len), 见脚本 patch_ts_accept"
  if [[ ${#SERVER_PROXY_ENV[@]} -gt 0 ]]; then
    echo "# 注: 下面这些 *_PROXY 只注入 server 进程(flashinfer 下 trtllm-gen cubin 用),"
    echo "#     ★不能设成容器级★, 否则 client 连 0.0.0.0:PORT 的压测请求也会被代理走。"
    echo "env ${SERVER_PROXY_ENV[*]} \\"
  fi
  echo "ts ${SERVER_ARGS[*]}"
} > "$RESULT_DIR/${RESULT_FILENAME}.cmd" 2>/dev/null || true

start_server() {
    stop_server
    set -x
    # setsid: ts 主进程成为**新进程组组长** (pgid==pid), per-GPU worker 子进程继承该组,
    # 故 stop_server 用 `kill -- -PID` 一次干掉整组, 不必按名 pkill (脚本路径含 "tokenspeed"
    # 会被 `pkill -f tokenspeed` 误杀, 之前踩过).
    # ★追加而非截断★: 崩溃重试会再次调用 start_server, 用 > 的话新 server 的日志会把上一次的
    # 崩溃现场覆盖掉 —— 而那正是最需要的证据(踩过: c=64 崩了, 重启后日志变 0 字节)。
    echo "===== start_server @ $(date '+%F %T') =====" >> "$SERVER_LOG"
    setsid env "${SERVER_PROXY_ENV[@]}" ts "${SERVER_ARGS[@]}" >> "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    set +x
    wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"
    # smg HTTP 网关(/v1)可能比 /health 晚就绪 ~百秒: /health 通了但 /v1/completions 仍
    # 秒回 503(网关未接上 engine). 直接开压会全 503 -> 判失败 -> 重启循环. 故再探真实
    # /v1/completions(1 token) 直到 200, 最多 ~180s, 之后才认为可压测.
    # 探真实 /v1/completions: TokenSpeed 只收**字符串** prompt(不收 token-id 数组), 且客户端
    # 还带 ignore_eos. 用字符串 prompt + ignore_eos 探到 200 才算网关真就绪(顺带验证 ignore_eos 被接受).
    echo "探测 smg 网关 /v1 就绪(字符串 prompt + ignore_eos)..."
    local _ok=0 _code _body
    local _probe="/tmp/_ts_probe_body.$$"
    for _ in $(seq 1 90); do
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "!! server 进程已退, 停止探测"; break; }
        _code=$(curl -s -o "$_probe" -w '%{http_code}' -m 10 \
            -H 'Content-Type: application/json' \
            --data "{\"model\":\"$MODEL\",\"prompt\":\"Hello, world. This is a warmup probe.\",\"max_tokens\":4,\"ignore_eos\":true,\"temperature\":0,\"stream\":false}" \
            "http://0.0.0.0:$PORT/v1/completions" 2>/dev/null)
        if [[ "$_code" == "200" ]]; then _ok=1; echo "网关 /v1 就绪 (200)"; break; fi
        sleep 2
    done
    if [[ "$_ok" != "1" ]]; then
        echo "!! 网关 /v1 探测未见 200 (最后 code=$_code) body: $(head -c 400 "$_probe" 2>/dev/null)"
    fi
    rm -f "$_probe" 2>/dev/null
}
stop_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
        kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    fi
    # 精准兜底 (只匹配 ts 启动器, 不匹配 bench 脚本路径), 等显存释放再重启避免 OOM.
    pkill -9 -f "/usr/local/bin/ts" 2>/dev/null || true
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
# 调试用: SERVE_ONLY=1 只起服务并保活(供手动 curl 探 API), 不跑 sweep.
if [[ "${SERVE_ONLY:-0}" == "1" ]]; then
    echo "SERVE_ONLY=1: server 已就绪于 :$PORT (model=$MODEL), 保活中. Ctrl-C / docker stop 退出."
    while kill -0 "$SERVER_PID" 2>/dev/null; do sleep 10; done
    exit 0
fi
# datasets/pandas: client 跑 random 用不到, best-effort 安装, 带 timeout 兜底(无网不卡死).
timeout 60 pip install -q datasets pandas >/dev/null 2>&1 || echo "(跳过 datasets/pandas 安装: 无网/已装; random 数据集不需要)"

# server 侧预热: sweep 前跑一遍最大 batch(丢弃), 触发 JIT/autotune/cudagraph, 避免冷启动污染 TTFT.
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

# CONC 逗号列表: server 起一次, 每个 batch 循环压测. 崩溃(completed=0)重启重试, 最多 MAX_RETRY 次.
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
            cp -f "$SERVER_LOG" "$RESULT_DIR/${rf}.serverlog.a${attempt}" 2>/dev/null || true
            start_server
        done
        [[ "$ok" == "1" ]] || { echo "!! c=$c rep=$rep 重试 $MAX_RETRY 次仍失败, 放弃该点"; BENCH_RC=1; }
    done
done

stop_gpu_monitor
stop_server
exit $BENCH_RC
