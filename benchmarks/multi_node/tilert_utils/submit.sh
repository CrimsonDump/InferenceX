#!/usr/bin/env bash
# ============================================================================
# tilert_utils/submit.sh —— tileRT 多节点 slurm 提交器（对标 amd_utils/submit.sh），在宿主机侧执行。
#
# 非 Dynamo 自定义 runtime → 不走 srtctl，直接用裸 slurm 原语（与 dgxc 单节点路径同一 house style）：
#   salloc 2 节点(各 8 卡独占) → 每节点 enroot import 镜像 → srun 每节点 1 task 进容器跑 run_node.sh。
# run_node.sh 内按 $SLURM_PROCID 分角色(decode / prefill+router+bench)。
#
# 入参经 env 传入（launcher 已导出）：IMAGE / MODEL_PATH / GITHUB_WORKSPACE / SLURM_PARTITION /
#   SLURM_ACCOUNT / RUNNER_NAME，以及打榜 env（ISL/OSL/CONC_LIST/… 经 --export=ALL 透传进容器）。
# ============================================================================
set -x
HERE="$(cd "$(dirname "$0")" && pwd)"
NODES="${TILERT_NODES:-2}"                      # prefill 1 节点 + decode 1 节点（各 8 卡）

# 镜像 squash 缓存目录（与 dgxc dynamo/单节点路径同款解析：优先共享盘，不可写退到 workspace 本地）
SQUASH_DIR="${B200_SQUASH_DIR:-/home/sa-shared/containers}"
{ mkdir -p "$SQUASH_DIR" 2>/dev/null && [[ -w "$SQUASH_DIR" ]]; } || SQUASH_DIR="$GITHUB_WORKSPACE/.container-squash"
mkdir -p "$SQUASH_DIR"

# ⛔ P 和 D **必须是两个不同镜像**，不是可选优化：
#   tilert pin transformers==4.46.3，而 vLLM 要 >=5.5.3 —— 装不进同一个 Python 环境。
#   恰好与 PD 物理拓扑吻合：decode 节点用官方 tilert 镜像，prefill 节点用 stock vLLM 镜像。
# master config 的 schema 只有**一个** `image` 顶层字段（configs/CONFIGS.md:81），所以：
#   decode 镜像 = 那个 `image`；prefill 镜像 = 走 `prefill.additional-settings` 直通的
#   `PREFILL_IMAGE=...`（workflow 在跑 launcher 前 `export` 它，见
#   .github/workflows/benchmark-multinode-tmpl.yml:309）。不改 schema，maintainer 也看得见。
# 第二镜像在树里有先例：ATOM 的 CLIENT_IMAGE（amd_utils/job.slurm:533）也是同节点起 sibling 容器。
DECODE_IMAGE="${DECODE_IMAGE:-$IMAGE}"
: "${PREFILL_IMAGE:?PREFILL_IMAGE 未设 —— 应由 master config 的 prefill.additional-settings 提供，如 PREFILL_IMAGE=vllm/vllm-openai:v0.26.0}"

squash_path() { echo "$SQUASH_DIR/$(echo "$1" | sed 's/[\/:@#]/_/g').sqsh"; }
DECODE_SQUASH="$(squash_path "$DECODE_IMAGE")"
PREFILL_SQUASH="$(squash_path "$PREFILL_IMAGE")"

# 1) 申请 2 节点，各独占 8 卡
salloc --partition="$SLURM_PARTITION" --account="$SLURM_ACCOUNT" \
    --nodes="$NODES" --gres=gpu:8 --exclusive --mem=0 \
    --time="${SALLOC_TIME_LIMIT:-480}" --no-shell --job-name="$RUNNER_NAME"
JOB_ID=$(squeue --name="$RUNNER_NAME" -u "$USER" -h -o %A | head -n1)

# 2) host 发现要放在 import 之前 —— 每台只 import 它自己那个角色的镜像（省一半时间和磁盘）。
mapfile -t HOSTS < <(scontrol show hostnames "$(squeue -j "$JOB_ID" -h -o %N)")
[[ "${#HOSTS[@]}" -ge 2 ]] || { echo "expected >=2 nodes, got: ${HOSTS[*]}"; exit 1; }
export DECODE_HOST="${HOSTS[0]}" PREFILL_HOST="${HOSTS[1]}"

# 3) enroot import（public image → local squash）。flock 串行化防并发重复 import；
#    squash 目录可能是共享盘，别的并发 job 也可能在导同一个镜像。
import_image() {   # $1=镜像 $2=squash 路径 $3=目标节点
    srun --jobid="$JOB_ID" --nodelist="$3" --ntasks=1 bash -c "
        export ENROOT_CACHE_PATH=\$HOME/.cache/enroot; mkdir -p \$ENROOT_CACHE_PATH
        exec 9>\"$2.lock\"; flock -w 600 9 || exit 1
        unsquashfs -l \"$2\" >/dev/null 2>&1 || enroot import -o \"$2\" docker://$1
    "
}
import_image "$DECODE_IMAGE"  "$DECODE_SQUASH"  "$DECODE_HOST"  || exit 1
import_image "$PREFILL_IMAGE" "$PREFILL_SQUASH" "$PREFILL_HOST" || exit 1

# 4) tileRT 8-shard 权重缓存（decode 用 weight_converter 的产物，不是 HF 原始权重）。
# 必须满足两条，否则等于每个 job 重转 ~700 GiB / 数小时（详见 run_node.sh 的 convert_weights）：
#   a) 目录在 $GITHUB_WORKSPACE 之外 —— checkout 用 clean:true(git clean -ffdx) 会删掉 workspace 里的未跟踪产物；
#   b) 目录挂进容器 —— 否则写的是容器 overlay，容器一销毁就没了。
# 路径优先由 launcher 按机器给出（dgxc 用 lustre 可写树）；这里的兜底与镜像 squash 缓存同款思路：
# 落到 $HOME/.cache（enroot 缓存也在那儿，是 runner 上持久且可写的位置）。
export TILERT_WEIGHTS_DIR="${TILERT_WEIGHTS_DIR:-$HOME/.cache/tilert/${MODEL_PREFIX:-model}-${PRECISION:-fp8}-tilert-8shard}"
mkdir -p "$TILERT_WEIGHTS_DIR"

# 5) 两个 srun（各自 --nodelist + --container-image）进容器跑角色运行器。
#
# ⚠️ 为什么不能像以前那样用一个 `srun --nodes=2`：那样只能给**一个** --container-image。
#    拆成两个 srun 的代价是 **$SLURM_PROCID 在各自的 srun 里都是 0** ——
#    所以角色**必须显式用 $TILERT_ROLE 指定**，不能再靠 PROCID 自动分派。
#    run_node.sh 的角色解析本来就是 TILERT_ROLE 优先（`ROLE="${TILERT_ROLE:-}"`），无需改动；
#    drafts/mock_test.sh 的 MODE=role 覆盖的就是这条现在成为 CI 主路径的分支。
#
# $GITHUB_WORKSPACE → /workspace bind-mount，两节点共享，供跨节点 sentinel 收尾握手。
run_role() {   # $1=角色 $2=节点 $3=squash
    srun --jobid="$JOB_ID" --nodelist="$2" --ntasks=1 \
        --container-image="$3" \
        --container-mounts="$GITHUB_WORKSPACE:/workspace,$MODEL_PATH:$MODEL_PATH,$TILERT_WEIGHTS_DIR:$TILERT_WEIGHTS_DIR" \
        --container-workdir=/workspace --no-container-entrypoint \
        --export=ALL,TILERT_ROLE="$1",DECODE_HOST="$DECODE_HOST",PREFILL_HOST="$PREFILL_HOST",PORT="${PORT:-23333}" \
        bash "/workspace/benchmarks/multi_node/tilert_utils/run_node.sh"
}

# decode 后台常驻（它自己轮询 DONE_SENTINEL 收尾）；prefill 前台，跑完 bench 就 touch sentinel。
run_role decode "$DECODE_HOST" "$DECODE_SQUASH" &
DECODE_SRUN_PID=$!

run_role prefill "$PREFILL_HOST" "$PREFILL_SQUASH"
PREFILL_RC=$?

# prefill 结束（不论成败）后给 decode 一段时间自己收尾；超时就直接砍掉那个 srun，
# 否则 salloc 会一直挂到 --time 上限，白占 16 张卡。
for _ in $(seq 1 "${TILERT_DECODE_DRAIN:-60}"); do
    kill -0 "$DECODE_SRUN_PID" 2>/dev/null || break
    sleep 1
done
kill -0 "$DECODE_SRUN_PID" 2>/dev/null && { echo "[submit] decode srun 未自行退出，kill"; kill "$DECODE_SRUN_PID" 2>/dev/null; }
wait "$DECODE_SRUN_PID" 2>/dev/null || true

# 成败以 prefill 为准：bench 和结果 JSON 都在它那一侧。
exit "$PREFILL_RC"
