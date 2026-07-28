#!/usr/bin/env bash
# GLM-5.2 FP8 on B200, TokenSpeed (lightseek), MTP, 固定 seqlen (UTPS-STPS 曲线用).
#
# 在容器内执行 (由 bench.sh 起 lightseekorg/tokenspeed:tml 容器并调用). 与 sglang/vllm
# 同口径的第三条对照曲线: 单节点合一 (unified, 非 PD 分离), 一个 ts engine 同时
# prefill+decode, benchmark client 也在同容器内跑. 口径(稳态窗口/固定 MTP)完全一致.
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
# node071 默认后端 8-way 多播死锁(节点级 fabric 问题); 那台可试 TS_SYMMMEM_BACKEND=NVSHMEM
# (4卡验证过; 但 node071 即使换后端整链路仍卡, 见 memory). 用 TS_SYMMMEM_BACKEND 显式指定才覆盖.
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

# 客户端 warmup 请求数 (默认 0, 靠稳态窗口刨 prefill/ramp).
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
SERVER_LOG="$RESULT_DIR/server.log"

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
            f"{indent}    import math as _math, os as _os, random as _random\n"
            f"{indent}    try:\n"
            f"{indent}        _a = float(_os.environ.get('TS_SIMULATE_ACC_LEN', '0') or '0')\n"
            f"{indent}    except ValueError:\n"
            f"{indent}        _a = 0.0\n"
            f"{indent}    if _a > 0 and accept_lengths.numel():\n"
            f"{indent}        _w = int(getattr(self.config, 'spec_num_tokens', 0) or 0)\n"
            f"{indent}        if _w > 0:\n"
            f"{indent}            _a = min(_a, float(_w))\n"
            f"{indent}        # accept_len 可为小数(cookbook low-latency 的 3.5): 按 lower/upper 权重\n"
            f"{indent}        # 伯努利抽, 期望恰为该值 —— 对齐 sglang match-expected 的做法.\n"
            f"{indent}        _lo = int(_math.floor(_a)); _hi = _lo + 1 if _lo < _a else _lo\n"
            f"{indent}        _k = _lo if (_hi == _lo or _random.random() >= _a - _lo) else _hi\n"
            f"{indent}        accept_lengths.fill_(max(1, _k))\n"
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

# ====== ts serve 参数 ======
# model id = $MODEL 路径 (served_model_name 默认=model), 与 client --model "$MODEL" 对齐.
SERVER_ARGS=(
    serve
    --model "$MODEL"
    --trust-remote-code
    --host 0.0.0.0 --port "$PORT"
    --max-model-len "$CONTEXT_LEN"
    --chunked-prefill-size "$CHUNKED_PREFILL"
    --max-num-seqs "$MAX_NUM_SEQS"
    --gpu-memory-utilization "$MEM_FRAC"
    --kv-cache-dtype fp8            # GLM-5.2 recipe 默认 (让 TokenSpeed 走自身最优路径, 同 vllm 不强对齐 sglang)
    --enable-metrics               # 读 accept rate / Decoded Tok/Iter
    --disable-kvstore              # 关 host-offload L2 KV cache: bench 不需要; 省 host 内存池(tep KVStore 分配曾崩); recipe(V4/Inkling)亦用
    --disable-prefill-graph        # 关 prefill CUDA graph: tep(attn-TP) 的 GLM DSA prefill graph 有 "token count mismatch" bug; 走 eager prefill 绕开. 对本 bench 无损(稳态窗口本就刨 prefill; decode 图仍用). recipe(MiniMax)亦用.
)

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
    # MoE all-to-all: 默认 none(用 TritonRSAG 对称内存, 本机死锁); 可设 A2A_BACKEND=deepep
    #   走 NVSHMEM(本机 NVLink 已由 sglang 验证可用), 绕开对称内存 RSAG dispatch.
    if [[ -n "${A2A_BACKEND:-}" ]]; then
        SERVER_ARGS+=( --all2all-backend "$A2A_BACKEND" )
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

# .cmd 文件 = 复现记录: 先列 server 进程读取但不在命令行的 env, 再列启服务命令.
{
  echo "# ===== 环境变量 (ts serve 进程读取, 但不在命令行里, 需 export 才能复现) ====="
  echo "# --- 容器/运行时 (bench.sh 注入) ---"
  emit_env PYTHONUNBUFFERED     "python 输出不缓冲, 日志实时"
  emit_env PYTHONNOUSERSITE     "忽略用户 site-packages, 用容器内干净环境"
  emit_env TORCH_CUDA_ARCH_LIST "目标 SM 架构 = B200 sm_100"
  emit_env CUDA_DEVICE_ORDER    "按 PCI 总线枚举 GPU, 卡号稳定"
  echo "# --- MTP 固定 accept_len hack (源码补丁 model_executor._apply_simulate_acc_len 运行期读取) ---"
  emit_env TS_SIMULATE_ACC_LEN  "固定 MTP accept_len: 每 decode step 强制恰好接受该值 (= min(值, num_draft_tokens))"
  echo ""
  echo "# ===== TokenSpeed 启服务命令 (所有 server 参数都在这) ====="
  echo "# 注: 固定 accept_len 靠容器内源码补丁 (model_executor._apply_simulate_acc_len), 见脚本 patch_ts_accept"
  echo "ts ${SERVER_ARGS[*]}"
} > "$RESULT_DIR/${RESULT_FILENAME}.cmd" 2>/dev/null || true

start_server() {
    stop_server
    set -x
    # setsid: ts 主进程成为**新进程组组长** (pgid==pid), per-GPU worker 子进程继承该组,
    # 故 stop_server 用 `kill -- -PID` 一次干掉整组, 不必按名 pkill (脚本路径含 "tokenspeed"
    # 会被 `pkill -f tokenspeed` 误杀, 之前踩过).
    setsid ts "${SERVER_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
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
