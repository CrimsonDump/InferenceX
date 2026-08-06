#!/usr/bin/env bash
# ============================================================================
# tilert_utils/run_node.sh —— 容器内、每节点执行的角色运行器（对标 amd_utils/job.slurm + server.sh）。
#
# 由 tilert_utils/submit.sh 通过 `srun --ntasks-per-node=1 --container-image=…` 在每个节点的容器里拉起，
# 按 $SLURM_PROCID 分角色（0=decode 节点, 1=prefill 节点+router+bench），照抄 ATOM 的 NODE_RANK 范式砍成 1+1。
#
# 命令 flag 已按 TileRT README（Topology A）对齐：--parser glm47 / --transport nixl / fp8_ds_mla↔fp8 /
#   MTP speculative-config / max_seq_len 202752 均为 README 权威值。
#
# 手动 / 无 slurm（你的两台裸机阶段2）：直接在每台跑本脚本、用 TILERT_ROLE 指定角色即可，无需 submit.sh：
#   box A:  TILERT_ROLE=decode  DECODE_HOST=<A> PREFILL_HOST=<B> LOGDIR=<共享盘或本地> bash run_node.sh
#   box B:  TILERT_ROLE=prefill DECODE_HOST=<A> PREFILL_HOST=<B> CONC_LIST=1 ISL=1024 OSL=1024 \
#           RANDOM_RANGE_RATIO=1.0 RESULT_FILENAME=smoke MODEL_PATH=<HF> bash run_node.sh
#   （两台无共享盘时 sentinel 传不过去，decode 侧手动 Ctrl-C；或 LOGDIR 指到 NFS。）
# ============================================================================

source "$(dirname "$0")/../../benchmark_lib.sh"

CONC="${CONC_LIST%% *}"                        # bs=1：conc-list: [1] → CONC=1
DECODE_CTRL_PORT=${DECODE_CTRL_PORT:-5556}
DECODE_HTTP_PORT=${DECODE_HTTP_PORT:-5557}
PREFILL_PORT=${PREFILL_PORT:-8000}
ROUTER_PORT=${PORT:-23333}
TRANSPORT=${TILERT_TRANSPORT:-${KV_P2P_TRANSFER:-nixl}}   # 两端须一致；InferenceX 侧 kv-p2p-transfer: nixl（README 示例也用 nixl）
TILERT_WEIGHTS_DIR=${TILERT_WEIGHTS_DIR:-/workspace/GLM-5-FP8-TileRT}   # weight_converter 产出的 8-shard（decode 用）
TILERT_MODEL_TYPE=${TILERT_MODEL_TYPE:-glm-5}            # weight_converter --model_type（GLM-5 → glm-5）
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-glm5}             # decode --model / prefill --served-model-name / bench --model 三处须一致
MAX_SEQ_LEN=${MAX_SEQ_LEN:-202752}                       # GLM-5.1 上下文；decode --max-seq-len 与 connector tilert_max_seq_len 须一致
# ⛔ router 的输出解析器：打榜必须用 **none**，不能用 README 示例里的 glm47。
# 依据一（平台惯例）：InferenceX 给推理模型跑 bench 一律**不挂 reasoning parser** ——
#   benchmarks/single_node/speedbench/dsr1_fp4_b300_vllm.sh:15-17 明写
#   「NO --reasoning-parser …（serve command is bare）; reasoning-parser is irrelevant here anyway」。
# 依据二（实测）：用 glm47 时 pd_router 走 _event_delta()，把模型输出分流到 **reasoning_content**
#   （pd_router.py:271-272）；而 InferenceX 的 bench 客户端只认 delta.content
#   （utils/bench_serving/backend_request_func.py:400）→ 永远认不出首 token。
#   实测后果：10 个请求全部 success，但 **generated_tokens=0、TTFT=128.7s（≈E2EL）、TPOT=0**，
#   而 ITL=8.49ms 正常 —— 典型的「token 在流、统计口径错」。
# 依据三（源码）：parser=none 时 sess 为 None，走 pd_router.py:336-338
#   `yield _chunk({"content": text})`，全部进 content，bench 与 lm-eval 都能正常识别。
#   evals 同理需要 content：gsm8k 的 filter 正则要在 content 里抓 "#### <number>"。
# glm47 的价值在 tool-call / reasoning 分离，打榜场景用不到（我们的 config 无 agentic 档）。
TILERT_PARSER=${TILERT_PARSER:-none}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.75}                       # README prefill 示例值
# ⛔ 结果 JSON 必须落 **$GITHUB_WORKSPACE 的容器挂载点（/workspace 根）**，不能跟着 LOGDIR 走：
#   CI 在 workspace 根下 glob `${RESULT_FILENAME}_*.json` 判定成败
#   （.github/workflows/benchmark-multinode-tmpl.yml:341 "Run failed: No benchmark result files found"），
#   随后 :356 再遍历同一 glob 调 utils/process_result.py。落到子目录 = job 直接判红。
# 注：dgxc launcher 里那段把结果从 /run_logs 拷回 workspace 并改名的逻辑
#   （runners/launch_b200-dgxc.sh:373-390）在 dynamo 分支，tilert 早返回**不经过它** → 得自己落对地方。
RESULT_DIR=${RESULT_DIR:-/workspace}
LOGDIR=${LOGDIR:-/workspace}                             # 三进程日志落这；wait_for_server_ready 要读 router 日志
DECODE_WAIT=${DECODE_WAIT:-3600}                         # prefill 等 decode ctrl 端口就绪的最长秒数（含首跑权重转换/加载）

# ⛔ prefill 的 --speculative-config 是**无条件必填**，不是"开 MTP 才加"。
# 依据一：README Topology A 的 prefill 命令里它是固定项，注释写明「The MTP speculative config is
#   **required**: the prefill populates the draft-layer KV that decode-side speculation resumes from.」
# 依据二：tilert 的 glm5 profile 把层数写成常量 —— profiles/glm5.py: `NUM_LAYERS = 79 # 78 main + 1 MTP draft`，
#   而 profiles/mla_nsa.py:252 的 classify_layers 会硬校验 vLLM 交出的 KV 层数必须等于 79。
#   不传 speculative-config 时 vLLM 只建 78 层 → 实测直接失败：
#     RuntimeError: glm5 classify: 78 MLA + 78 KI layers (expected 79 each); check --speculative-config
#   也就是说 MTP draft 层是这套 PD **wire format 的固有组成**，不是可选优化。
# 权重侧对得上：GLM-5.1 的 config.json 是 num_hidden_layers=78 + num_nextn_predict_layers=1 = 79，
#   且 HF 目录里 layers 0..78 齐全（layer 78 就是 MTP draft），vLLM 能直接加载。
PREFILL_SPEC=(--speculative-config '{"method":"mtp","num_speculative_tokens":1}')

# ⛔ decode 的 --with-mtp 同样是**无条件必填** —— 它不是"要不要投机"的可选项，而是 PD 能否工作的前提。
# 实测链路：prefill 按 wire format 发 79 层 KV → decode 侧 generator.py:475 `_dst = caches[layer_id*3 + off]`。
#   `caches` 由 end2end.py 组装，而 MTP 那一层的 cache 只在 `if self.with_mtp:` 分支里
#   `caches.extend(mtp.get_cache_vars())` 才追加。不带 --with-mtp 时 caches 只有 78*3=234 项（0..233），
#   注入第 79 层要取 caches[234..236] → 实测 `IndexError: list index out of range`（KV 已成功传到、
#   `all ranks done in 75.7 ms`，就死在注入这一步）。
# README 的 decode 命令同样把 --with-mtp 写成固定项。
# 权重侧无需额外处理：`--model_type glm-5` 的 base 转换已包含 layer 78（MTP 层，244 个 key
#   vs 普通层 204 个），**不需要再跑 --append_mtp**。
#
# ⚠️ 由此推论：tilert 的 PD 形态**永远在跑 MTP**。所以 InferenceX 的 config 应当声明
#    `spec-decoding: mtp`，否则 dashboard 会把它标成 none —— 与实际不符。
DECODE_MTP=(--with-mtp)

# ---- 角色发现 ----
DECODE_HOST=${DECODE_HOST:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | sed -n '1p')}
PREFILL_HOST=${PREFILL_HOST:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | sed -n '2p')}
ROLE="${TILERT_ROLE:-}"
if [[ -z "$ROLE" ]]; then
    if [[ "${SLURM_NTASKS:-1}" -ge 2 ]]; then
        [[ "${SLURM_PROCID:-0}" == "0" ]] && ROLE=decode || ROLE=prefill
    else
        ROLE=all
    fi
fi
[[ "$ROLE" == "all" ]] && { DECODE_HOST=${DECODE_HOST:-127.0.0.1}; PREFILL_HOST=${PREFILL_HOST:-127.0.0.1}; }

# 三进程的日志/结果/sentinel 都落 $LOGDIR，且全靠 `>` 重定向写入 —— 目录不存在则重定向直接失败、
# 进程起不来还没有日志可查。CI 下 LOGDIR 默认 /workspace（已存在）不会踩，但 LOGDIR 指到子目录时必踩。
mkdir -p "$LOGDIR"

# 跨节点完成信号：$GITHUB_WORKSPACE(→/workspace) 在两节点 bind-mount 共享；bench 完 touch，decode 侧据此收尾。
DONE_SENTINEL="$LOGDIR/.tilert_done.${SLURM_JOB_ID:-local}"
echo "[tilert-run_node] ROLE=$ROLE PROCID=${SLURM_PROCID:-NA} host=$(hostname) DECODE_HOST=$DECODE_HOST PREFILL_HOST=$PREFILL_HOST"

# ============================ 工具 ============================
# 起服务前把**完整展开后**的命令打出来，同时写两个地方：
#   1) 本脚本 stdout —— CI job log / `docker logs` 里直接可见
#   2) 该服务自己的日志文件开头 —— 排查时打开 tilert_decode.log 就知道它是怎么被起的，
#      不用再去翻另一个流。日志因此改用 `>>` 追加（header 已先写进去）。
# 用 printf %q 逐参数转义 → 打出来的是**可直接复制粘贴重跑**的命令（--kv-transfer-config
# 那个 JSON 里的引号也会被正确转义，这是靠 `set -x` 拿不到的）。
log_and_run_bg() {    # $1=标签 $2=日志文件 其余=要跑的命令；把 pid 放进 $LAST_BG_PID
    local label="$1" logfile="$2"; shift 2
    # 本函数自己必须不被 xtrace 追踪：否则 `+ printf ' %q' …` 会和 printf 的真实输出
    # 交错到同一行（`[cmd]+ printf …`），打出来的命令就没法复制粘贴了。用完恢复。
    local _xtrace=0; [[ $- == *x* ]] && _xtrace=1
    { set +x; } 2>/dev/null
    { printf '===== [%s] %s =====\n' "$label" "$(date '+%F %T')"
      printf '[cmd]'; printf ' %q' "$@"; printf '\n'
      printf '[cwd] %s\n[host] %s\n\n' "$PWD" "$(hostname)"
    } | tee -a "$logfile"
    "$@" >>"$logfile" 2>&1 &
    LAST_BG_PID=$!
    echo "[$label] pid=$LAST_BG_PID log=$logfile"
    (( _xtrace )) && set -x
    return 0          # 别让 (( _xtrace )) 的假值成为函数返回码
}

# 结果文件名 stem（不带 .json —— benchmark_lib 自己补，见 benchmark_lib.sh:341,535）。
# 必须是 `${RESULT_FILENAME}_…` 前缀 + 带 `_gpus_/_ctx_/_gen_` 字段，两条都是硬要求：
#   前缀 —— CI 的成败 glob 是 `${RESULT_FILENAME}_*.json`（tmpl:341）。直接用 $RESULT_FILENAME 会
#           产出 `${RESULT_FILENAME}.json`，**匹配不上**（glob 要求前缀后至少一个字符）→ 判红。
#   字段 —— tmpl:360-362 用 sed 从**文件名**里抽 gpus/ctx/gen，`if [ -n "$gpus" ]` 不成立就
#           **静默跳过** process_result.py → 跑绿了却没有可 ingest 的产物。
# 命名照抄同为 disagg 的 llm-d（benchmarks/multi_node/llm-d/server.sh:599）：
#   ${RESULT_FILENAME}_c<N>_gpus_<总>_ctx_<prefill>_gen_<decode>
# （用 `_c` 而非 `_conc`：workflow 生成的 $RESULT_FILENAME 本身已含 `_conc<N>_<runner>`，llm-d 同样用 `_c` 避免重复。）
# GPU 数由 workflow 的 tp/worker env 算出（submit.sh 用 --export=ALL 透传）；本地手跑时默认 8+8。
bench_result_stem() {
    local pg=$(( ${PREFILL_TP:-8} * ${PREFILL_NUM_WORKERS:-1} ))
    local dg=$(( ${DECODE_TP:-8} * ${DECODE_NUM_WORKERS:-1} ))
    printf '%s_c%s_gpus_%s_ctx_%s_gen_%s' \
        "${RESULT_FILENAME}" "$CONC" "$(( pg + dg ))" "$pg" "$dg"
}

# ⛔ RDMA / UCX 前置检查 —— **两端都要跑**（NIXL 的传输后端就是 UCX，P 和 D 都要注册 CUDA 显存）。
#
# 为什么要专门做这一步：缺 RDMA 时的失败**又晚又难认**。KV 传输发生在服务全部起好、
# 第一个请求进来之后，报错落在 NIXL/UCX 内部（`UCX  ERROR ... ibv_*` 或干脆握手超时），
# 那时已经烧掉 ~25 分钟的权重加载 + warmup。所以这里在起服务前就断言，报错要能直接照着修。
#
# 三项依赖，缺一不可：
#   1. /dev/infiniband/uverbs* —— 用户态 verbs 设备节点。docker 侧靠 `--device /dev/infiniband`；
#      **pyxis/enroot 下没有等价 flag**，依赖集群把它配成默认透传（多数 NVIDIA 集群的 enroot
#      装了 mellanox hook 会自动带上）。带不上就只能照 ATOM 的做法在 slurm 分配里改用 docker run
#      （amd_utils/job.slurm:673 显式列 uverbs*）——这条已作为 C5 向 maintainer 确认。
#   2. memlock 无上限 —— RDMA 要 pin 住内存；docker 侧靠 `--cap-add CAP_IPC_LOCK`。
#      受限时表现为传大 KV 到一半失败，比完全没有设备节点更难查。
#   3. UCX_NET_DEVICES 已 pin —— 多 NIC 机器上不 pin 会挑到管理网口，慢几个数量级。
#      由 runner launcher 注入（runners/launch_b200-dgxc.sh 的 tilert 分支）。
#
# 默认只**告警不致命**（TILERT_RDMA_STRICT=1 可改成硬失败）：`--transport` 理论上还能退到
# 非 RDMA 后端，而且我们无法确定 dgxc 上 uverbs 的确切暴露方式，不该在没验证过的机器上先自杀。
rdma_preflight() {
    local role="$1" warn=0
    echo "[rdma] role=$role UCX_NET_DEVICES=${UCX_NET_DEVICES:-<未设>} UCX_MEMTYPE_CACHE=${UCX_MEMTYPE_CACHE:-<未设>} UCX_MEMTYPE_REG_WHOLE=${UCX_MEMTYPE_REG_WHOLE:-<未设>}"

    local uverbs=(/dev/infiniband/uverbs*)
    if [[ -e "${uverbs[0]}" ]]; then
        echo "[rdma] verbs 设备: ${uverbs[*]}"
    else
        echo "[rdma] ⚠️ /dev/infiniband/uverbs* 不存在 —— 容器没拿到 RDMA 设备节点。" >&2
        echo "[rdma]    docker: 加 --device /dev/infiniband；pyxis: 需集群侧默认透传（见 C5）" >&2
        warn=1
    fi

    local ml; ml="$(ulimit -l 2>/dev/null)"
    if [[ "$ml" == "unlimited" ]]; then
        echo "[rdma] memlock: unlimited"
    else
        echo "[rdma] ⚠️ memlock=$ml（非 unlimited）—— RDMA pin 内存可能失败。" >&2
        echo "[rdma]    docker: 加 --cap-add CAP_IPC_LOCK（或 --ulimit memlock=-1）" >&2
        warn=1
    fi

    if command -v ibv_devices >/dev/null 2>&1; then
        echo "[rdma] ibv_devices:"; ibv_devices 2>&1 | sed 's/^/[rdma]   /'
    fi

    if (( warn )) && [[ "${TILERT_RDMA_STRICT:-0}" == "1" ]]; then
        echo "[rdma] TILERT_RDMA_STRICT=1 且检查未全通过 → 直接退出" >&2
        return 1
    fi
    return 0
}

# ============================ 角色函数 ============================
stage_tokenizer_files() {
    # decode_server 除了权重，还从 **$TILERT_WEIGHTS_DIR** 加载 tokenizer 和 chat 模板：
    #   tilert/models/glm_5/generator.py:48  AutoTokenizer.from_pretrained(model_weights_dir, trust_remote_code=True)
    #   tilert/models/glm_5/generator.py:56  open(model_weights_dir + "/chat_template.jinja")   ← 硬读，缺了直接抛异常
    # 而 weight_converter **只写 safetensors + index.json，不拷任何辅助文件**（源码里零 copy/shutil）。
    # 不补的话 decode_server 起不来：OSError: Can't load tokenizer for '<weights_dir>'（node071 实测）。
    # 所以从 HF 目录把非权重文件补过去。幂等：已存在则不覆盖（maintainer 预置的产物可能已自带）。
    #
    # ⚠️ 必须排除 HF 侧的 model.safetensors.index.json —— 它描述的是 HF 分片，
    #    拷过去会覆盖掉转换产物的 index，等于把权重索引毁掉。
    local staged=0 f b
    for f in "$MODEL_PATH"/*; do
        [[ -f "$f" ]] || continue
        b="$(basename "$f")"
        [[ "$b" == *.safetensors ]] && continue
        [[ "$b" == "model.safetensors.index.json" ]] && continue
        [[ -e "$TILERT_WEIGHTS_DIR/$b" ]] && continue
        cp -p "$f" "$TILERT_WEIGHTS_DIR/$b" && staged=$((staged+1))
    done
    echo "[stage_tokenizer] 从 $MODEL_PATH 补入 $staged 个辅助文件"
    # 硬校验这两个必需项，否则报可读的错，别留给 decode_server 抛 OSError/FileNotFoundError
    local missing=()
    [[ -f "$TILERT_WEIGHTS_DIR/chat_template.jinja" ]] || missing+=(chat_template.jinja)
    [[ -f "$TILERT_WEIGHTS_DIR/tokenizer_config.json" || -f "$TILERT_WEIGHTS_DIR/tokenizer.json" ]] \
        || missing+=("tokenizer.json/tokenizer_config.json")
    if (( ${#missing[@]} )); then
        echo "[stage_tokenizer] ERROR: $TILERT_WEIGHTS_DIR 缺少 ${missing[*]}"
        echo "[stage_tokenizer]        decode_server 会从该目录加载 tokenizer/chat 模板（generator.py:48,56）。"
        echo "[stage_tokenizer]        请确认 MODEL_PATH=$MODEL_PATH 是包含 tokenizer 的 HF 目录。"
        return 1
    fi
    return 0
}

convert_weights() {   # HF 权重 → TileRT 8-shard（仅 decode 需要；跨 run 复用缓存，幂等）
    # 转换代价很大：读 ~705 GiB + 写 ~700 GiB、纯 CPU、需 ~700 GB 常驻内存，实测数小时。且它跑在
    # 独占的 GPU 分配里 —— 重复转一次等于让 16 张卡空转数小时。所以产物必须缓存复用，
    # 且 $TILERT_WEIGHTS_DIR 必须在 $GITHUB_WORKSPACE 之外（checkout 用 clean:true，
    # git clean -ffdx 会删掉 workspace 里的未跟踪产物 → 放里面等于每个 job 重转）。
    #
    # 缓存命中判据 = 输出目录里有 model.safetensors.index.json。weight_converter 把 index 写在
    # 所有 shard 落盘并改名之后的最后一步，所以它存在 ⇔ 转换完整跑完；反之只看「目录非空」会把
    # 中途崩掉的残缺 shard 当成已转好、静默加载坏权重。maintainer 预置的 8-shard 同样自带 index，
    # 因此预置场景无需改代码即命中缓存、完全不触发转换。
    local index_json="$TILERT_WEIGHTS_DIR/model.safetensors.index.json"
    if [[ -f "$index_json" ]]; then
        echo "[weight_converter] 缓存命中（已有 index.json），跳过转换：$TILERT_WEIGHTS_DIR"
        return 0
    fi

    mkdir -p "$TILERT_WEIGHTS_DIR"
    # 一次 sweep 会展开多个 job（1k1k / 8k1k），可能并发走到这里、同时往同一目录写而互相踩踏。
    # 用 flock 串行化（与 submit.sh 里 enroot import 同一手法）；锁文件放在输出目录内，跟缓存一起
    # 躺在共享盘上，跨 job、跨节点都有效。转换耗时数小时，等锁超时要给足。
    exec 9>"$TILERT_WEIGHTS_DIR/.convert.lock"
    flock -w "${TILERT_CONVERT_LOCK_WAIT:-21600}" 9 || {
        echo "[weight_converter] 等转换锁超时（另一个 job 仍在转？）"; return 1; }
    if [[ -f "$index_json" ]]; then          # 等锁期间已被并发 job 转好
        echo "[weight_converter] 缓存已由并发 job 生成，跳过转换"; exec 9>&-; return 0
    fi
    # 有产物但没 index → 上一次转换半途死了，残缺 shard 必须清掉再重转（保留锁文件本身）
    if [[ -n "$(ls -A "$TILERT_WEIGHTS_DIR" 2>/dev/null | grep -v '^\.convert\.lock$')" ]]; then
        echo "[weight_converter] 发现无 index.json 的残留产物（上次转换未完成），清理后重转"
        find "$TILERT_WEIGHTS_DIR" -mindepth 1 ! -name '.convert.lock' -delete
    fi

    echo "[weight_converter] $MODEL_PATH → $TILERT_WEIGHTS_DIR (model_type=$TILERT_MODEL_TYPE)"
    "${PY:-python}" -m tilert.models.preprocess.weight_converter \
        --model_type "$TILERT_MODEL_TYPE" --model_dir "$MODEL_PATH" --save_dir "$TILERT_WEIGHTS_DIR"
    local rc=$?
    exec 9>&-                                # 释放锁
    # 显式校验：转换失败就地报错退出，比让 decode_server 抱着空目录起来再神秘死掉好排查。
    if [[ $rc -ne 0 || ! -f "$index_json" ]]; then
        echo "[weight_converter] 转换失败（rc=$rc，index.json 未生成）：$TILERT_WEIGHTS_DIR"
        return 1
    fi
    echo "[weight_converter] 转换完成并已缓存：$TILERT_WEIGHTS_DIR"
}

start_decode() {      # TileRT decode（独占 8×B200，TP8）
    local cmd=("${PY:-python}" -m tilert.pd_vllm.decode_server
        --engine tilert --model "$SERVED_MODEL_NAME"
        --model-weights-dir "$TILERT_WEIGHTS_DIR"
        --max-seq-len "$MAX_SEQ_LEN"
        --kv-cache-dtype fp8 --transport "$TRANSPORT"
        --ctrl-port "$DECODE_CTRL_PORT" --http-port "$DECODE_HTTP_PORT"
        "${DECODE_MTP[@]}")
    log_and_run_bg decode "$LOGDIR/tilert_decode.log" "${cmd[@]}"
    DECODE_PID=$LAST_BG_PID
}

start_prefill() {     # stock vLLM prefill（挂 TileRTConnector 把 KV 传给 decode）
    # --enforce-eager 为 README 参考值（禁 CUDA graph，稳但影响 prefill 计算；bs=1 榜单看 decode TPOT，先随 README）。
    local cmd=(vllm serve "$MODEL_PATH"
        --served-model-name "$SERVED_MODEL_NAME" --port "$PREFILL_PORT"
        --tensor-parallel-size "$PREFILL_TP"
        --enforce-eager --trust-remote-code --return-tokens-as-token-ids
        --gpu-memory-utilization "$GPU_MEM_UTIL" --kv-cache-dtype fp8_ds_mla
        "${PREFILL_SPEC[@]}"
        --kv-transfer-config "{\"kv_connector\":\"TileRTConnector\",\"kv_connector_module_path\":\"tilert.pd_vllm.prefill_connector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"tilert_host\":\"$DECODE_HOST\",\"tilert_ctrl_port\":$DECODE_CTRL_PORT,\"tilert_model\":\"$SERVED_MODEL_NAME\",\"tilert_max_seq_len\":$MAX_SEQ_LEN,\"tilert_transport\":\"$TRANSPORT\"}}")
    log_and_run_bg prefill "$LOGDIR/tilert_prefill.log" "${cmd[@]}"
    PREFILL_PID=$LAST_BG_PID
}

start_router() {      # OpenAI 兼容端点（CPU-only，与 prefill 同节点）
    # CUDA_VISIBLE_DEVICES 置空是故意的：router 纯 CPU，不该占卡。用 env 传而不是前缀赋值，
    # 这样它能进 cmd 数组、被完整打印出来（前缀赋值不会出现在 %q 的输出里）。
    local cmd=(env CUDA_VISIBLE_DEVICES= "${PY:-python}" -m tilert.pd_vllm.pd_router
        --vllm-url "http://$PREFILL_HOST:$PREFILL_PORT"
        --decode "$DECODE_HOST:$DECODE_CTRL_PORT:$DECODE_HTTP_PORT"
        --port "$ROUTER_PORT" --model-path "$MODEL_PATH" --parser "$TILERT_PARSER")
    log_and_run_bg router "$LOGDIR/tilert_router.log" "${cmd[@]}"
    ROUTER_PID=$LAST_BG_PID
}

wait_for_tcp() {      # 等 host:port 可连（无 nc 依赖，用 bash /dev/tcp）；$3=超时秒
    local host="$1" port="$2" deadline=$(( SECONDS + ${3:-600} ))
    # 轮询循环会把 xtrace 刷爆，所以临时关掉——但**必须恢复**：以前这里关了不开，
    # 导致 prefill 分支调用本函数之后 start_prefill / start_router 全部丢失 set -x 追踪。
    local _xtrace=0; [[ $- == *x* ]] && _xtrace=1
    { set +x; } 2>/dev/null
    local rc=0
    until (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null; do
        if [[ $SECONDS -ge $deadline ]]; then
            echo "[wait_for_tcp] timeout $host:$port after ${3:-600}s"; rc=1; break
        fi
        sleep 5
    done
    [[ $rc -eq 0 ]] && { exec 3>&- 2>/dev/null || true; echo "[wait_for_tcp] $host:$port ready"; }
    (( _xtrace )) && set -x
    return $rc
}

run_bench_and_eval() {   # 打流量 + eval（在 prefill/all 节点跑，router 在本地 0.0.0.0）
    # router 是 OpenAI 兼容端点 → backend=openai-chat + /v1/chat/completions；--use-chat-template 为 AGENTS.md 硬性要求。
    # --tokenizer 必须显式给 HF 权重目录：bench 的 tokenizer 默认回落到 --model（这里是 served-model-name
    # "glm5"，不是可加载路径）。GLM-5.1 的 tokenizer 走 auto_map(tokenization_glm4moe.ChatGLM4TokenizerFast)
    # 自定义类，故还需 --trust-remote-code。与 ATOM 的 amd_utils/bench.sh 同一写法。
    # wait_for_server_ready / run_benchmark_serving 内部写死 http://0.0.0.0:$port，无 --host 参数。
    wait_for_server_ready --port "$ROUTER_PORT" \
        --server-log "$LOGDIR/tilert_router.log" --server-pid "$ROUTER_PID"
    run_benchmark_serving \
        --model "$SERVED_MODEL_NAME" --port "$ROUTER_PORT" \
        --backend openai-chat --endpoint /v1/chat/completions \
        --input-len "$ISL" --output-len "$OSL" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts "$((CONC * 10))" --max-concurrency "$CONC" \
        --use-chat-template --server-pid "$ROUTER_PID" \
        --tokenizer "$MODEL_PATH" --trust-remote-code \
        --result-filename "$(bench_result_stem)" --result-dir "$RESULT_DIR"
    local rc=$?
    if [ "${RUN_EVAL}" = "true" ]; then
        run_eval --framework lm-eval --port "$ROUTER_PORT"   # run_lm_eval 只认 --port（不认 --host）
        append_lm_eval_summary
    fi
    return $rc
}

# 容器启动时装依赖（官方镜像只带 pinned 构建环境，不含 tilert/vLLM —— 详见 setup_deps.sh 文件头）。
# 用 source 而非执行：conda 激活等环境变更要保留到后面起服务。放在角色分派前、$ROLE 已确定之后。
# shellcheck source=./setup_deps.sh
source "$(dirname "$0")/setup_deps.sh"

# ============================ 角色分派 ============================
set -x
case "$ROLE" in
  decode)
    # node0：转权重 + 起 decode，前台常驻；轮询 DONE_SENTINEL，bench 完则收尾。
    rdma_preflight decode || exit 1   # 起服务前就断言 RDMA/UCX，别等第一个请求才在 NIXL 里炸
    convert_weights || exit 1        # 权重没准备好就别起 decode，省得抱着空目录神秘死掉
    stage_tokenizer_files || exit 1  # tokenizer/chat 模板也在 weights 目录里找（见函数头）
    start_decode
    { set +x; } 2>/dev/null
    while kill -0 "$DECODE_PID" 2>/dev/null; do
        [[ -f "$DONE_SENTINEL" ]] && break
        sleep 5
    done
    if [[ -f "$DONE_SENTINEL" ]]; then
        echo "[decode] 收到完成信号，收尾"; kill "$DECODE_PID" 2>/dev/null || true; exit 0
    fi
    echo "[decode] decode_server 提前退出（见 $LOGDIR/tilert_decode.log）"; exit 1
    ;;
  prefill)
    # node1：等 decode ctrl 端口就绪 → 起 prefill → 起 router → 打流量 → 通知 decode 收尾。
    rdma_preflight prefill || exit 1
    rm -f "$DONE_SENTINEL"
    wait_for_tcp "$DECODE_HOST" "$DECODE_CTRL_PORT" "$DECODE_WAIT" \
        || echo "[prefill] 警告：等 decode ctrl 端口($DECODE_HOST:$DECODE_CTRL_PORT)超时，仍尝试启动"
    start_prefill
    # 等 vLLM 自己的端口起来再起 router / 打流量。原因：run_bench_and_eval 里的
    # wait_for_server_ready 等的是 **router** 的 /health，而 router 是 CPU-only、秒起，
    # prefill 要加载数百 GB 权重要几十分钟 —— 只等 router 会让 bench 在 prefill 就绪前就开打。
    # vLLM 的 HTTP 端口在模型加载完成后才 listen，所以探这个端口是"已加载"的可靠代理。
    wait_for_tcp "$PREFILL_HOST" "$PREFILL_PORT" "${PREFILL_WAIT:-3600}" \
        || echo "[prefill] 警告：等 vLLM 端口($PREFILL_HOST:$PREFILL_PORT)超时，仍继续（见 $LOGDIR/tilert_prefill.log）"
    start_router
    run_bench_and_eval; BENCH_RC=$?
    touch "$DONE_SENTINEL"        # 通知 node0 decode 收尾
    kill "$ROUTER_PID" "$PREFILL_PID" 2>/dev/null || true
    exit $BENCH_RC
    ;;
  all)
    # 单节点 all-in-one（本地冒烟，非榜单）：decode+prefill+router 同机（各需 8 卡，单机只够 PoC/stub）。
    rdma_preflight all || exit 1
    convert_weights || exit 1
    stage_tokenizer_files || exit 1
    start_decode
    wait_for_tcp 127.0.0.1 "$DECODE_CTRL_PORT" "$DECODE_WAIT" || true
    start_prefill
    start_router
    run_bench_and_eval; BENCH_RC=$?
    kill "$ROUTER_PID" "$PREFILL_PID" "$DECODE_PID" 2>/dev/null || true
    exit $BENCH_RC
    ;;
  *)
    echo "未知 ROLE=$ROLE"; exit 2 ;;
esac
set +x
