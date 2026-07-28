#!/usr/bin/env bash
# ============================================================================
# bench.sh — GLM-5.2 多 backend (sglang / vllm / tokenspeed) 的
#            每GPU吞吐(STPS) vs 交互速度(UTPS) 对比曲线.
# ============================================================================
# 在装了 8×B200 + docker 的节点上运行(当前 sglang 用 node074: node071 的多播/对称内存
# rendezvous 在 8 卡下死锁, 见 CLAUDE.md). 一次 run 把 config.json 里所有 enabled 的曲线
# 逐条跑完(每条曲线起一次容器), 结果全写同一 outdir, 最后聚合成【一个含所有曲线 +
# 全部原始数据】的自包含 HTML 报告. 同一套口径: max-TTFT 稳态窗口 + 固定 MTP N/accept_len.
#
# ---- 用法: 配置的唯一入口是 config.json, 本脚本【没有配置类命令行开关】 ----
#   bash bench.sh                      # 用 runners/config.json
#   bash bench.sh --config my.json     # 换一份配置(如只跑 smoke 的那份)
#   bash bench.sh --dry-run            # 只打印解析出的计划与将执行的命令, 不跑
#
#   要改模型 / 并行 / MTP / commit / 镜像 / 并发 / ISL / OSL / 重复次数 / 输出目录 —— 全
#   在 config.json 里改. 【刻意不提供命令行覆盖】: 同一个配置项两处可写(命令行 + 配置
#   文件)会让"这次到底跑的是什么"无从判断, 也让 run_config.json 记录失真. 要临时试跑就
#   复制一份 config.json 改完用 --config 指过去(配置文件本身就是配置的单位).
#
# ---- config.json 三层结构 (字段说明见 config.json 内的 _readme / _*_fields) ----
#   defaults = 整次 run 共享的坐标轴与环境 (model / seqlens / batches / osl / reps /
#              gpu_count / outroot|outdir / mtp_draft_path / src_cache / fetch_proxy / env)
#   backends = 每个 backend 的构建来源默认值 (image / repo / commit)
#              commit=null -> 不从源码构建, 用镜像自带引擎(commit 位由镜像探测填)
#   curves   = 图上每条线: backend / parallel / mtp_n / mtp_acc (+ 可选 commit / batches /
#              env / enabled / note). curve 内同名字段覆盖 defaults(仅 batches / env).
#
# ---- 产物命名: 按"图 / 曲线 / 点"三层, 各层只放本层统一的字段 ----
#   图 (= 一份报告/一个目录, 全图统一): <model>_<quant>_i<ISL>o<OSL>_c<最小>-<最大>.html
#         例: glm-5p2_fp8_i8192o1024_c1-256.html
#         model/quant = 从 defaults.model 的权重目录名拆 (…/glm-5p2-fp8/model ->
#                       glm-5p2 + fp8); c 段 = 本次全部曲线并发取值的范围
#         注意: backend / commit / parallel / MTP 【不进图名】—— 同一张图里它们逐曲线不同.
#               故同 model/quant/ISL/OSL/并发范围 的两次 run 会落进同一目录(旧结果进
#               _prev_<时间戳>/); 要并存就在 config 里给 defaults.outdir.
#   曲线 (series, 逐曲线不同): <backend>-<commit>-<parallel>-mtpN<N>A<acc>
#         例: sglang-09ca4fc-tp-mtpN5A3.5 —— 进结果文件名与图例
#         commit = curve.commit > backends.<bk>.commit; 为 null 时从镜像探测:
#                  安装目录 git hash > 镜像 tag 尾部 hash > 模块版本号 > tag 原样
#   点 (逐点不同): concurrency(batch) —— 曲线的横轴, 只进结果文件名的 _c<b>_r<rep> 段
#   目录: defaults.outdir 优先, 否则 <defaults.outroot>/<图名去掉 .html>
#
# 产物(在 outdir): <图名>.html / tps_curve.csv / tps_raw.csv / tps_curve_s<ISL>.svg /
#   run_config.json(含 config.json 全文) / glm52_<series>_s<ISL>_c<b>_r<rep>.json /
#   每曲线一个 glm52_<series>_s<ISL>.cmd (实际启服务命令 + 需 export 的环境变量)
#
# ---- build from source (config 里 commit 非 null 时) ----
#   源码树缓存在 <src_cache>/<backend>-<commit>/, 缺了就按 backends.<bk>.repo 从
#   codeload 下 tarball 解开(经 defaults.fetch_proxy; 见 CLAUDE.md 出网约束), 校验 tar
#   完整性并写下完整 sha. 容器内再 editable 安装并校验"实际生效的是源码那份".
#   按 (backend, commit) 去重: 同一次 run 里相同组合只准备一次, 用到它的曲线共用.
# ============================================================================
set -uo pipefail

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$SELF_DIR/config.json"
DRY_RUN=0
REPO="$(cd "$SELF_DIR/.." && pwd)"        # InferenceX 仓库根(挂进容器当 /workspace)
AGG="$SELF_DIR/aggregate_and_plot.py"

usage() { sed -n '2,56p' "$0"; exit "${1:-0}"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage 0;;
    *) echo "未知参数: $1  (配置全在 config.json 里, 本脚本只有 --config/--dry-run)" >&2; usage 1;;
  esac
done

[[ -f "$CONFIG" ]] || { echo "错误: 找不到配置文件 $CONFIG" >&2; exit 1; }
[[ -f "$AGG" ]] || { echo "错误: 找不到 $AGG" >&2; exit 1; }
command -v docker >/dev/null || { echo "错误: 需要 docker (请在有 GPU 的节点上运行, 如 node074)" >&2; exit 1; }

# ============================ 读 config.json ============================
# python 侧做全部解析与校验(未知字段直接报错, 免得配了个错别字被静默忽略),
# 输出每行一条记录给 bash 消费, 字段用 \x1f(US) 分隔:
#   D <key> <value>                       defaults 标量(列表用逗号连)
#   B <backend> <image> <repo> <commit>   用到的 backend (commit 为空 = 用镜像自带)
#   C <idx> <backend> <parallel> <mtp_n> <mtp_acc> <batches> <commit> <note>
#   E <idx|D> <KEY> <VALUE>               容器额外 env (idx=D 表示 defaults.env)
# 分隔符【不能用 tab】: tab 属于 IFS 空白字符, bash 的 read 会把连续 tab 折叠成一个分隔符,
# 于是 commit 为空(用镜像自带引擎)时该字段被吞掉、后面的字段整体左移. \x1f 非空白, 空字段得以保留.
CFG_TSV="$(python3 - "$CONFIG" <<'PY'
import json, sys

path = sys.argv[1]
with open(path) as f:
    cfg = json.load(f)

def die(msg):
    sys.stderr.write(f"配置错误 ({path}): {msg}\n")
    sys.exit(1)

DEFAULT_KEYS = {"model", "mtp_draft_path", "seqlens", "batches", "osl", "reps",
                "gpu_count", "outroot", "outdir", "src_cache", "fetch_proxy", "env"}
BACKEND_KEYS = {"image", "repo", "commit"}
CURVE_KEYS = {"backend", "commit", "parallel", "mtp_n", "mtp_acc", "enabled",
              "batches", "env", "note", "cookbook_ref"}
PARALLELS = {"tp", "dep", "tep", "dpa-tp"}

# 允许 _ 开头的键作注释/文档(config.json 里的 _readme / _*_fields)
top = {k: v for k, v in cfg.items() if not k.startswith("_")}
unknown = set(top) - {"defaults", "backends", "curves"}
if unknown:
    die(f"顶层未知字段 {sorted(unknown)} (只认 defaults/backends/curves, _ 开头的是注释)")

d = {k: v for k, v in top.get("defaults", {}).items() if not k.startswith("_")}
unknown = set(d) - DEFAULT_KEYS
if unknown:
    die(f"defaults 未知字段 {sorted(unknown)}")
for req in ("model", "seqlens", "batches", "osl", "reps", "gpu_count"):
    if req not in d:
        die(f"defaults 缺字段 {req}")
if "outroot" not in d and "outdir" not in d:
    die("defaults 需要 outroot 或 outdir 之一")

backends = {k: v for k, v in top.get("backends", {}).items() if not k.startswith("_")}
for bk, bv in backends.items():
    unknown = set(k for k in bv if not k.startswith("_")) - BACKEND_KEYS
    if unknown:
        die(f"backends.{bk} 未知字段 {sorted(unknown)}")
    if "image" not in bv:
        die(f"backends.{bk} 缺 image")

curves = [c for c in top.get("curves", []) if c.get("enabled", True)]
if not curves:
    die("curves 里没有 enabled 的曲线")

out = []


def emit(*fields):
    out.append("\x1f".join(str(x) for x in fields))


def join(v):
    return ",".join(str(x) for x in v) if isinstance(v, list) else str(v)


for k in sorted(DEFAULT_KEYS - {"env"}):
    if k in d:
        emit("D", k, join(d[k]))

for k, v in (d.get("env") or {}).items():
    emit("E", "D", k, v)

used = []
for i, c in enumerate(curves):
    unknown = set(k for k in c if not k.startswith("_")) - CURVE_KEYS
    if unknown:
        die(f"curves[{i}] 未知字段 {sorted(unknown)}")
    for req in ("backend", "parallel", "mtp_n", "mtp_acc"):
        if req not in c:
            die(f"curves[{i}] 缺字段 {req}")
    bk = c["backend"]
    if bk not in backends:
        die(f"curves[{i}].backend={bk} 不在 backends 里")
    if c["parallel"] not in PARALLELS:
        die(f"curves[{i}].parallel={c['parallel']} 非法 (应为 {sorted(PARALLELS)})")
    commit = c.get("commit", backends[bk].get("commit"))
    if commit is None:
        commit = ""
    batches = join(c.get("batches", d["batches"]))
    # mtp_n 落到 bash 里要做整数比较(`-gt 0`), 写成 5.0 会让 bash 报语法错 -> 这里就收敛成 int;
    # mtp_acc 保持原样(小数是合法的, 如 cookbook 的 3.5), 只校验它是数
    try:
        mtp_n = int(c["mtp_n"])
    except (TypeError, ValueError):
        die(f"curves[{i}].mtp_n={c['mtp_n']!r} 不是整数")
    if not isinstance(c["mtp_acc"], (int, float)) or isinstance(c["mtp_acc"], bool):
        die(f"curves[{i}].mtp_acc={c['mtp_acc']!r} 不是数")
    emit("C", i, bk, c["parallel"], mtp_n, c["mtp_acc"], batches,
         commit, c.get("note", ""))
    for k, v in (c.get("env") or {}).items():
        emit("E", i, k, v)
    key = (bk, commit)
    if key not in used:
        used.append(key)

for bk, commit in used:
    b = backends[bk]
    emit("B", bk, b["image"], b.get("repo") or "", commit)

print("\n".join(out))
PY
)" || exit 1

# ---- TSV -> bash 变量/数组 ----
declare -A DEF=() BK_IMAGE=() BK_REPO=() BK_CFG_COMMIT=() CURVE_ENV=()
CURVE_LINES=(); BK_LIST=()
while IFS=$'\x1f' read -r kind f2 f3 f4 f5 f6 f7 f8 f9; do
  case "$kind" in
    D) DEF["$f2"]="$f3";;
    E) CURVE_ENV["$f2|$f3"]="$f4";;
    C) CURVE_LINES+=("$f2"$'\x1f'"$f3"$'\x1f'"$f4"$'\x1f'"$f5"$'\x1f'"$f6"$'\x1f'"$f7"$'\x1f'"$f8"$'\x1f'"$f9");;
    B) BK_LIST+=("$f2"$'\x1f'"$f5"); BK_IMAGE["$f2|$f5"]="$f3"; BK_REPO["$f2"]="$f4";;
  esac
done <<< "$CFG_TSV"

MODEL="${DEF[model]}"
MTP_DRAFT_PATH="${DEF[mtp_draft_path]:-}"
SEQLENS="${DEF[seqlens]}"
BATCHES_DEF="${DEF[batches]}"
OSL="${DEF[osl]}"
REPS="${DEF[reps]}"
GPU_COUNT="${DEF[gpu_count]}"
OUTROOT="${DEF[outroot]:-}"
OUTDIR="${DEF[outdir]:-}"
SRC_CACHE="${DEF[src_cache]:-/tilert/xbj/src_cache}"
FETCH_PROXY="${DEF[fetch_proxy]:-}"
GPU_DEV="$(seq -s, 0 $((GPU_COUNT - 1)))"

# GLM-5.2 硬约束预检: FP8 704G 需 ≥8 卡 (无论怎么切, 总卡数不足会 OOM)
if [[ "$GPU_COUNT" -lt 8 ]]; then
  echo "⚠️  GPU 数=$GPU_COUNT < 8, GLM-5.2 FP8(704G) 每卡权重≈$((704 / GPU_COUNT))G > B200 可用(~178G) -> 会 OOM." >&2
fi

# ---- 并行预设: config 的 parallel 名 -> canonical (vllm 语义 world=tp*dp) ----
# PTP=切头, PDP=attn 数据并行, PEP=0/1 专家并行开关. 逐 backend 再翻译成各自 flag.
# MTP 的 (N, acc) 【不在这里】—— 它是逐曲线的 config 字段(curve.mtp_n/mtp_acc), 因为每条
# 曲线跑在它自己的 cookbook 操作点上(tp≈low-latency, dep≈balanced), 不是控 MTP 单变量比并行.
parallel_canonical() {   # $1=parallel 名 -> 设 P_TP/P_DP/P_EP
  case "$1" in
    tp)      P_TP=$GPU_COUNT; P_DP=1;          P_EP=0;;   # 纯TP:    attn TP + MoE TP
    dpa-tp)  P_TP=1;          P_DP=$GPU_COUNT; P_EP=0;;   # attn DP + MoE TP (不入最终图)
    dep)     P_TP=1;          P_DP=$GPU_COUNT; P_EP=1;;   # attn DP + MoE EP(deepep)
    tep)     P_TP=$GPU_COUNT; P_DP=1;          P_EP=1;;   # attn TP + MoE EP
    *) return 1;;
  esac
}

sanitize() { printf '%s' "$1" | tr -c 'A-Za-z0-9.+' '-'; }

# ============================ commit / 镜像 / 源码树 ============================
# commit 进 series 名(属曲线, 不属图 —— 同一张图允许不同 backend/commit 的曲线).
# config 里给了就用它(同时意味着 build from source); 为 null 则从镜像探测.
# 探测顺序: 容器内该模块所在目录的 git short hash (源码 editable 安装才有)
#        -> 镜像 tag 尾部的 commit hash (私有 build 惯例, 如 ...-prerelease-9844dd05)
#        -> 模块 __version__ (前缀 v)  ->  镜像 tag 原样.
detect_commit() {   # $1=backend $2=image
  local bk="$1" img="$2" tag="${2##*:}" c=""
  c="$(timeout 240 docker run --rm --entrypoint bash "$img" -c '
        d=$(python3 -c "import '"$bk"' as M,os;print(os.path.dirname(M.__file__))" 2>/dev/null) || exit 0
        git -C "$d" rev-parse --short=8 HEAD 2>/dev/null || true
      ' 2>/dev/null | tr -dc 'a-f0-9')"
  [[ -z "$c" ]] && c="$(printf '%s' "$tag" | grep -oE '[0-9a-f]{7,12}$' | head -1)"
  if [[ -z "$c" ]]; then
    c="$(timeout 240 docker run --rm --entrypoint python3 "$img" \
           -c "import $bk as M;print('v'+M.__version__)" 2>/dev/null | tr -d '[:space:]')"
  fi
  [[ -z "$c" ]] && c="$tag"
  sanitize "$c"
}

# 准备源码树: <src_cache>/<backend>-<commit>/ (幂等; 已存在就直接用).
# node071 出网受限(见 CLAUDE.md): github.com 直连/代理都不通, 但 codeload.github.com
# 经代理可达, 故用 tarball 而非 git clone. 经代理下载会偶发【截断】(拿到 200 但字节不全),
# 所以每次都用 tar 校验完整性, 不完整就重下.
ensure_src() {   # $1=backend $2=commit -> stdout 源码树路径
  local bk="$1" commit="$2" repo="${BK_REPO[$1]:-}" dir="$SRC_CACHE/$1-$2" url owner_name tgz i
  if [[ -f "$dir/.commit" ]]; then printf '%s\n' "$dir"; return 0; fi
  [[ -n "$repo" ]] || { echo "!! backends.$bk 要从源码构建(commit=$commit)但没配 repo" >&2; return 1; }
  owner_name="${repo#*github.com/}"; owner_name="${owner_name%.git}"; owner_name="${owner_name%/}"
  url="https://codeload.github.com/$owner_name/tar.gz/$commit"
  mkdir -p "$SRC_CACHE" || return 1
  tgz="$SRC_CACHE/.$bk-$commit.tgz"
  echo ">>> 准备源码树 $dir  (repo=$repo commit=$commit)" >&2
  for i in 1 2 3 4 5; do
    curl -sL --max-time 900 ${FETCH_PROXY:+-x "$FETCH_PROXY"} -o "$tgz" "$url" || true
    tar tzf "$tgz" >/dev/null 2>&1 && break
    echo "   下载不完整(第 $i 次), 重试" >&2
    [[ "$i" == "5" ]] && { echo "!! 源码 tarball 下载/校验失败: $url" >&2; return 1; }
  done
  rm -rf "$dir"; mkdir -p "$dir" || return 1
  tar xzf "$tgz" -C "$dir" --strip-components=1 || return 1
  rm -f "$tgz"
  # 记下完整 sha(配置里通常是 short hash), 也当"这棵树准备好了"的标记
  curl -s --max-time 60 ${FETCH_PROXY:+-x "$FETCH_PROXY"} \
    "https://api.github.com/repos/$owner_name/commits/$commit" 2>/dev/null \
    | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["sha"])
except Exception: print("")' > "$dir/.commit" 2>/dev/null || true
  [[ -s "$dir/.commit" ]] || printf '%s\n' "$commit" > "$dir/.commit"
  printf '%s\n' "$dir"
}

# 逐 (backend, commit) 组合解析一次: 镜像 / series 用的 commit / 源码树(如需)
declare -A BK_SERIES_COMMIT=() BK_SRC=()
for bkline in "${BK_LIST[@]}"; do
  IFS=$'\x1f' read -r _bk _commit <<< "$bkline"
  _img="${BK_IMAGE[$_bk|$_commit]}"
  if [[ -n "$_commit" ]]; then
    BK_SERIES_COMMIT["$_bk|$_commit"]="$(sanitize "$_commit")"
    if [[ "$DRY_RUN" == "1" ]]; then
      BK_SRC["$_bk|$_commit"]="$SRC_CACHE/$_bk-$_commit"
    else
      BK_SRC["$_bk|$_commit"]="$(ensure_src "$_bk" "$_commit")" \
        || { echo "错误: 准备 $_bk@$_commit 源码树失败" >&2; exit 1; }
    fi
  else
    BK_SERIES_COMMIT["$_bk|$_commit"]="$(detect_commit "$_bk" "$_img")"
    BK_SRC["$_bk|$_commit"]=""
  fi
done

# ============================ 图名 / 目录 ============================
# 图名只放全图统一的东西 —— model / quant / io seqlen / 并发范围.
# model / quant: 从 defaults.model 的权重目录名拆 (…/glm-5p2-fp8/model -> glm-5p2 + fp8)
_wdir="$(basename "${MODEL%/}")"
[[ "$_wdir" == "model" || "$_wdir" == "weights" ]] && _wdir="$(basename "$(dirname "${MODEL%/}")")"
case "$_wdir" in
  *-nvfp4) QUANT=nvfp4;; *-mxfp4) QUANT=mxfp4;; *-fp4) QUANT=fp4;;
  *-fp8)   QUANT=fp8;;   *-bf16)  QUANT=bf16;;  *-fp16) QUANT=fp16;;
  *-int8|*-w8a8|*-awq|*-gptq) QUANT="${_wdir##*-}";;
  *) QUANT=unk;;
esac
MODEL_NAME="$_wdir"; [[ "$QUANT" != "unk" ]] && MODEL_NAME="${_wdir%-$QUANT}"

# 并发范围取【全部曲线】并发取值的并集 min-max: 逐曲线可有自己的 batches(如 tep 有硬上限),
# 但图名是全图统一的, 只记这张图扫到的范围, 便于区分"扫到 256"与"只扫到 64"两张图.
_all_c=""
for cl in "${CURVE_LINES[@]}"; do
  IFS=$'\x1f' read -r _i _bk _par _n _acc _batches _commit _note <<< "$cl"
  _all_c+="${_all_c:+,}$_batches"
done
_cmin="$(printf '%s' "$_all_c" | tr ',' '\n' | tr -d ' ' | grep -E '^[0-9]+$' | sort -n | head -1)"
_cmax="$(printf '%s' "$_all_c" | tr ',' '\n' | tr -d ' ' | grep -E '^[0-9]+$' | sort -n | tail -1)"
CFIELD="c${_cmin:-0}"; [[ -n "$_cmax" && "$_cmax" != "$_cmin" ]] && CFIELD="c${_cmin}-${_cmax}"
STEM="$(sanitize "$MODEL_NAME")_$(sanitize "$QUANT")"
STEM="${STEM}_i$(printf '%s' "$SEQLENS" | tr -d ' ' | tr ',' '+')o${OSL}_${CFIELD}"
REPORT="${STEM}.html"

[[ -z "$OUTDIR" ]] && OUTDIR="$OUTROOT/$STEM"
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$OUTDIR"
  # 解析成绝对路径: 容器内脚本用 RESULT_DIR 直接把结果写到这里(在 /tilert 挂载下可写),
  # 否则相对路径在容器内会指向 /workspace(仓库目录)造成混淆.
  OUTDIR="$(readlink -f "$OUTDIR")"
  # 隔离本次运行: 旧结果移入 _prev_<时间戳>/ (一次, 对全部曲线)
  if ls "$OUTDIR"/glm52_*_c*.json >/dev/null 2>&1 || ls "$OUTDIR"/tps_curve.csv >/dev/null 2>&1; then
    PREV="$OUTDIR/_prev_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$PREV"
    mv -f "$OUTDIR"/glm52_*_c*.json "$OUTDIR"/glm52_*.cmd "$OUTDIR"/tps_curve.csv "$OUTDIR"/tps_raw.csv \
          "$OUTDIR"/tps_curve_s*.svg "$OUTDIR"/*.html "$PREV/" 2>/dev/null || true
    echo "注意: OUTDIR 已有旧结果, 已移入 $PREV/ 隔离本次运行(原始数据保留)"
  fi
fi

echo "=========================================================="
echo " GLM-5.2 benchmark  (config: $CONFIG)"
echo " model=$MODEL_NAME quant=$QUANT   MODEL=$MODEL"
[[ -n "$MTP_DRAFT_PATH" ]] && echo " MTP draft=$MTP_DRAFT_PATH"
echo " SEQLENS(ISL)=$SEQLENS  OSL=$OSL  REPS=$REPS  GPU=$GPU_COUNT  (默认 batches=$BATCHES_DEF)"
echo " 曲线 (逐条 = 图上一条线):"
for cl in "${CURVE_LINES[@]}"; do
  IFS=$'\x1f' read -r _i _bk _par _n _acc _batches _commit _note <<< "$cl"
  parallel_canonical "$_par"
  echo "   - $_bk@${BK_SERIES_COMMIT[$_bk|$_commit]} $_par (vllm 语义 tp=$P_TP dp=$P_DP ep=$P_EP, world=$((P_TP*P_DP)))  MTP N=$_n acc=$_acc  c=$_batches"
  [[ -n "$_note" ]] && echo "       note: $_note"
  [[ -n "${BK_SRC[$_bk|$_commit]}" ]] && echo "       源码: ${BK_SRC[$_bk|$_commit]} (editable 覆盖镜像自带)"
  [[ $((P_TP * P_DP)) -ne "$GPU_COUNT" ]] && echo "       ⚠️  tp*dp=$((P_TP*P_DP)) != GPU 数($GPU_COUNT)"
done
echo " 输出目录: $OUTDIR"
echo " 报告: $OUTDIR/$REPORT"
[[ "$DRY_RUN" == "1" ]] && echo " (--dry-run: 只打印计划, 不起容器)"
echo "=========================================================="

# run_config.json: 图级配置 + config.json 全文(曲线/backend/commit 全在里面, 便于事后复现)
if [[ "$DRY_RUN" != "1" ]]; then
  _series_json=""
  for cl in "${CURVE_LINES[@]}"; do
    IFS=$'\x1f' read -r _i _bk _par _n _acc _batches _commit _note <<< "$cl"
    _mtptag="mtpN${_n}A${_acc}"; [[ "$_n" -gt 0 ]] || _mtptag="mtpoff"
    _series_json+="${_series_json:+, }\"${_bk}-${BK_SERIES_COMMIT[$_bk|$_commit]}-${_par}-${_mtptag}\""
  done
  python3 - "$CONFIG" "$OUTDIR/run_config.json" <<PY
import json, sys, datetime
cfg = json.load(open(sys.argv[1]))
json.dump({
    "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "report": "$REPORT",
    "config_path": sys.argv[1],
    "model_name": "$MODEL_NAME", "quant": "$QUANT",
    "series": [$_series_json],
    "metric": "steady-state max-TTFT window (decode-only, 刨 prefill/ramp); "
              "UTPS=各请求窗内速率中位数, STPS/gpu=窗内总token/窗长/gpu",
    "note": "各曲线的 MTP (N/accept_len) 按其 cookbook 操作点逐曲线取值, 不是控 MTP 单变量比并行; "
            "启服务的完整命令(含 hack env)见各曲线的 .cmd",
    "config": cfg,
}, open(sys.argv[2], "w"), indent=2, ensure_ascii=False)
PY
fi

# ============================ 逐曲线 × 逐 seqlen 起容器 ============================
IFS=',' read -ra SL_ARR <<< "$SEQLENS"
RUN_RC=0
for cl in "${CURVE_LINES[@]}"; do
  IFS=$'\x1f' read -r idx backend par CN CACC CBATCH commit note <<< "$cl"
  IMAGE="${BK_IMAGE[$backend|$commit]}"
  SRC="${BK_SRC[$backend|$commit]}"
  SERIES_COMMIT="${BK_SERIES_COMMIT[$backend|$commit]}"

  case "$backend" in
    sglang)
      BENCH_REL="benchmarks/single_node/fixed_seq_len/glm5.2_fp8_b200_sglang_mtp.sh"
      # sglang 专属 docker 参数
      BK_DEVICE=( --device /dev/infiniband )
      BK_ENV=( -e NCCL_SHM_DISABLE=1 -e GLM_XGRAMMAR_BACKEND_CACHE_MAX_MB=76800
               -e GLM_GRAMMAR_OBJECT_CACHE_MAX_COUNT=1920 )
      BK_MOUNT=( -v /mnt/ramweights/jitp/cache:/root/.cache
                 -v /mnt/ramweights/jitp/triton:/root/.triton
                 -v /mnt/ramweights/jitp/tilelang:/root/.tilelang
                 -v /mnt/ramweights/jitp/deep_ep:/root/.deep_ep )
      # 源码树(build from source)与其 commit: 容器内 editable 安装并校验实际生效那份.
      # PIP_PROXY 供容器内按源码 pyproject 的 pin 装 sgl-kernel 用(node071 出网受限);
      # 只传给 pip(--proxy), 不设容器级 HTTP_PROXY —— 否则 client 连 localhost 的压测请求
      # 也会被代理走.
      [[ -n "$SRC" ]] && BK_ENV+=( -e BENCH_SGL_SRC="$SRC" -e BENCH_SGL_COMMIT="$commit"
                                   -e PIP_PROXY="$FETCH_PROXY" )
      ;;
    vllm)
      BENCH_REL="benchmarks/single_node/fixed_seq_len/glm5.2_fp8_b200_vllm_mtp.sh"
      # vllm 专属: 关对称内存死锁两处 + 关 inductor max-autotune + 跳过 deepgemm warmup
      BK_DEVICE=()
      BK_ENV=( -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 -e VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=0
               -e VLLM_DEEP_GEMM_WARMUP=skip -e PYTHONFAULTHANDLER=1 )
      BK_MOUNT=( -v /mnt/ramweights/jitp/cache:/root/.cache
                 -v /mnt/ramweights/jitp/triton:/root/.triton )
      ;;
    tokenspeed)
      BENCH_REL="benchmarks/single_node/fixed_seq_len/glm5.2_fp8_b200_tokenspeed_mtp.sh"
      BK_DEVICE=()
      # getting-started 建议 --ipc host (已在通用 docker run 里); 无 vllm/sglang 那种 hack env.
      BK_ENV=( -e PYTHONFAULTHANDLER=1 )
      BK_MOUNT=( -v /mnt/ramweights/jitp/cache:/root/.cache
                 -v /mnt/ramweights/jitp/triton:/root/.triton )
      ;;
    *) echo "⚠️  未知 backend: $backend, 跳过" >&2; continue;;
  esac
  [[ -f "$REPO/$BENCH_REL" ]] || { echo "错误: 找不到 $REPO/$BENCH_REL" >&2; exit 1; }

  # 翻译 canonical(vllm 语义) -> 各 backend 的 flag
  parallel_canonical "$par"
  case "$backend" in
    sglang)
      # sglang: 卡数=tp, dp/ep 叠加. tp=ptp*pdp, dp=pdp(>1 开 dp-attn), ep=全卡 if EP else 1
      BK_TP=$((P_TP * P_DP)); BK_DP=$P_DP
      if [[ "$P_EP" == "1" ]]; then BK_EP=$((P_TP * P_DP)); else BK_EP=1; fi
      ;;
    *)
      # vllm / tokenspeed 直接用 canonical: tp=切头, dp=attn 数据并行, ep=0/1 专家并行开关
      BK_TP=$P_TP; BK_DP=$P_DP; BK_EP=$P_EP
      ;;
  esac

  # MTP: mtp_n<=0 = 不开 spec (cookbook high-throughput 那格无 spec)
  if [[ "$CN" -gt 0 ]]; then MTP_ON=1; mtptag="mtpN${CN}A${CACC}"; else MTP_ON=0; mtptag="mtpoff"; fi

  # config 的 env: defaults.env 打底, curve.env 覆盖同名键 (作用域不同, 不是命令行那种双写)
  CFG_ENV=()
  declare -A _envmap=()
  for k in "${!CURVE_ENV[@]}"; do
    [[ "${k%%|*}" == "D" ]] && _envmap["${k#*|}"]="${CURVE_ENV[$k]}"
  done
  for k in "${!CURVE_ENV[@]}"; do
    [[ "${k%%|*}" == "$idx" ]] && _envmap["${k#*|}"]="${CURVE_ENV[$k]}"
  done
  for k in "${!_envmap[@]}"; do CFG_ENV+=( -e "$k=${_envmap[$k]}" ); done
  unset _envmap

  # series = 图上曲线名 = 结果文件名里的那一段: 逐曲线变化的全部字段
  # (backend / commit / parallel / MTP). 只含 [A-Za-z0-9.+-](accept_len 可能是小数,
  # 如 mtpN5A3.5), 不含下划线 —— 聚合脚本按 _s<ISL>_c<batch> 切分文件名.
  series="${backend}-${SERIES_COMMIT}-${par}-${mtptag}"

  for seqlen in "${SL_ARR[@]}"; do
    seqlen="${seqlen// /}"; [[ -z "$seqlen" ]] && continue
    prefix="glm52_${series}_s${seqlen}"
    CNAME="ix-glm52-${series}-s${seqlen}"
    echo ">>> [$series] ISL=$seqlen  tp=$BK_TP dp=$BK_DP ep=$BK_EP  MTP N=$CN acc=$CACC  c=$CBATCH  (容器 $CNAME)"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "    (dry-run) 镜像=$IMAGE  脚本=$BENCH_REL  额外env=${CFG_ENV[*]:-无}"
      continue
    fi
    docker rm -f "$CNAME" >/dev/null 2>&1 || true
    # ★--ulimit nofile★(踩过): 镜像里 soft 限只有 1024(hard 524288)。client 每个在飞请求一条
    # HTTP 连接, 所以 c=1024 时只开得出 ~1018 条 socket, 余下 6 个请求【静默失败】——
    # 实测 completed=1018/1024(warmup 轮与正式轮都恰好 1018, 确定性), 表现为"少发了请求"
    # 而不是任何报错, 极难发现。c>=1024 的点必须抬这个限。
    docker run --rm --name "$CNAME" \
      --gpus "\"device=$GPU_DEV\"" --network host --ipc host --shm-size 64g \
      --ulimit nofile=524288:524288 \
      --cap-add CAP_IPC_LOCK "${BK_DEVICE[@]}" \
      -e PYTHONUNBUFFERED=1 -e PYTHONNOUSERSITE=1 \
      -e TORCH_CUDA_ARCH_LIST=10.0 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e PORT=30000 \
      "${BK_ENV[@]}" \
      -e MODEL="$MODEL" -e MTP_DRAFT_PATH="$MTP_DRAFT_PATH" \
      -e TP="$BK_TP" -e DP="$BK_DP" -e EP="$BK_EP" \
      -e CONC="$CBATCH" -e ISL="$seqlen" -e OSL="$OSL" \
      -e RANDOM_RANGE_RATIO=1.0 \
      -e REPS="$REPS" -e RESULT_FILENAME="$prefix" -e RESULT_DIR="$OUTDIR" \
      -e MTP="$MTP_ON" -e SPEC_NUM_STEPS="$CN" -e MTP_ACC="$CACC" \
      "${CFG_ENV[@]}" \
      -v "$REPO":/workspace -v /tilert:/tilert -v /mnt/ramweights:/mnt/ramweights \
      "${BK_MOUNT[@]}" \
      -w /workspace --entrypoint /bin/bash \
      "$IMAGE" "$BENCH_REL" \
      || { echo "⚠️  [$series] ISL=$seqlen 容器异常退出, 已产出的原始数据仍会被收集"; RUN_RC=1; }
    # 结果/.cmd/.serverlog 由容器内脚本经 RESULT_DIR 直接写到 $OUTDIR, 无需再 mv.
  done
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo ">>> dry-run 结束 (未起容器, 未写产物)"
  exit 0
fi

echo ">>> 汇总 CSV + 画图 + HTML 报告 (所有曲线合并)"
python3 "$AGG" --results-dir "$OUTDIR" --gpu-count "$GPU_COUNT" --report-name "$REPORT"
echo "=========================================================="
echo " 完成. 产物在 $OUTDIR :"
echo "   - $REPORT   (全部曲线 + 全部原始数据)"
echo "   - tps_curve.csv / tps_raw.csv / tps_curve_s*.svg / run_config.json"
echo "=========================================================="
exit $RUN_RC

# ============================================================================
# USAGE (在 node071 上):
#   RUN=/tilert/xbj/tps_bench/InferenceX/runners/bench.sh
#   bash $RUN                          # 跑 runners/config.json 里所有 enabled 的曲线
#   bash $RUN --dry-run                # 先看解析出的曲线/命名/源码树对不对
#   cp runners/config.json /tilert/xbj/smoke.json   # 试跑: 复制一份改 batches/reps/outdir
#   bash $RUN --config /tilert/xbj/smoke.json
# 换 commit 做 A/B: 在 config.json 的 curves 里加一条同 parallel、commit 不同的曲线
# (两条线进同一张图, 因为 commit 属曲线不属图).
# ============================================================================
