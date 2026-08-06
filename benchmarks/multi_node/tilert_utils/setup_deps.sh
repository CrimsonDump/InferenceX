#!/bin/bash
# =============================================================================
# tilert_utils/setup_deps.sh —— 容器启动时装齐依赖（对标 amd_utils/setup_deps.sh）。
#
# 为什么需要这一步：TileRT 官方镜像（ghcr.io/tile-ai/tilert:cu132-latest）**只带 pinned
# 构建环境**（Python 3.12 / torch 2.11.0+cu130 / transformers 4.46.3 / CUDA 13.2），
# **既不含 `tilert` 也不含 `vllm`**。upstream README 明确把「进容器 pip install
# tilert==0.1.5.post1」列为运行 v0.1.5 的标准步骤 —— 所以这不是给镜像打补丁，而是照官方
# 文档把官方**正式发布的** wheel 装进它自己的基础环境。
#
# 与 docs/PR_REVIEW_CHECKLIST.md 的关系：该清单禁止 patch 引擎/serving stack（.patch、
# 改 site-packages、monkey-patch、覆盖容器文件、装 forked/rebuilt 的引擎 wheel）。这里装的
# 全是 PyPI 上未修改的官方 release wheel、且不改动镜像内任何既有文件，因此不属于被禁范围。
# 在树先例：amd_utils/setup_deps.sh 在打榜路径上装 amd-quark，并为 GLM-5 支持从 git commit
# 装 transformers（"the mori images do not ship it"）—— 本文件严格更保守。
#
# 按角色分装（$ROLE 由 run_node.sh 定）：
#   decode  -> tilert==$TILERT_VERSION 连自身依赖一起装（transformers 会被 pin 到 4.46.3）
#   prefill -> 只装 tilert --no-deps。**必须 --no-deps**：tilert pin `transformers==4.46.3`，
#              而 vLLM 要 `transformers>=5.5.3`，带依赖装会把 vLLM 的 transformers 降级搞崩。
#              prefill 侧只需要 `tilert.pd_vllm.prefill_connector` 这个纯 Python connector 可导入。
#
# 被 run_node.sh **source**（不是执行），以便 conda 激活等环境变更能保留。每步幂等。
# =============================================================================

# ⛔ 必须 >= 0.1.5.post2。post1 有四个会直接打穿 InferenceX bench 的上游缺陷，均已在
# post2 修复（PyPI 已发布）：pd_router 转发 prefill 请求时没剥掉 `stream_options`
# （vLLM 按 OpenAI 规范回 400，bench 全部请求失败）；没清 `max_completion_tokens`
# （该新字段在 vLLM 里优先于 max_tokens=1，prefill 节点会把整段答案重新生成一遍，
# TTFT 慢两个数量级、PD 分离白做）；`ignore_eos` 未端到端透传（输出被 EOS 截短，
# 对不上固定 OSL 的操作点）；收尾 usage chunk 缺 `choices: []`（bench 客户端是
# if/elif，统计恒为 `Total generated tokens: 0`）。
TILERT_VERSION="${TILERT_VERSION:-0.1.5.post2}"
# 可选：指向国内 PyPI 镜像（本地验证用；CI 上留空走默认源即可）。
TILERT_PIP_INDEX_URL="${TILERT_PIP_INDEX_URL:-}"

# tilert wheel 只声明了 einops/numpy/scipy/torch/transformers，但 `tilert.pd_vllm` 实际还需要
# 一套 HTTP 服务栈才能 import —— 官方镜像里没有，不补则 `decode_server`/`pd_router` 直接
# ModuleNotFoundError: No module named 'uvicorn'（实测 node071 官方镜像）。
# fastapi+uvicorn+httpx 会顺带带入 anyio / starlette / pydantic，无需单列。
TILERT_HTTP_DEPS="${TILERT_HTTP_DEPS:-fastapi uvicorn httpx}"
# KV 传输后端：transport.py 里是懒加载（`from nixl._api import ...` / `from mooncake.engine import ...`），
# 所以 import 模块本身不需要它们，但真跑 --transport 时必须有。官方 tilert 镜像不含。
# 注意 `nixl-cu13` 只带 CUDA 原生库、**不提供 `nixl` Python 模块**，API 包 `nixl` 必须单独装。
#
# ⚠️ 版本必须 pin，且要跟 prefill 侧镜像自带的那个对齐：NIXL 是 P/D 两端握手的传输层，
# 两边版本漂移有兼容风险。实测 vllm/vllm-openai:v0.26.0 自带 nixl 1.3.1，而不 pin 时
# decode 侧会装到当时的最新版（1.3.2）→ 造成 1.3.1↔1.3.2 的跨节点错配。这里 pin 成
# prefill 镜像自带的版本；换 prefill 镜像时同步改这个默认值（或用 env 覆盖）。
TILERT_NIXL_VERSION="${TILERT_NIXL_VERSION:-1.3.1}"
TILERT_TRANSPORT_DEPS="${TILERT_TRANSPORT_DEPS:-nixl==$TILERT_NIXL_VERSION}"

_SETUP_INSTALLED=()

# ---------------------------------------------------------------------------
# 0. 激活官方镜像里的 conda 环境。
#    镜像用 ENTRYPOINT=/usr/local/bin/entrypoint.sh 做 `conda activate tilert`，但
#    enroot/pyxis 起容器时通常**不执行镜像 ENTRYPOINT**（直接跑 srun 给的命令），
#    那样 `python` 会是 base env（没有 torch）。这里显式补上，幂等。
# ---------------------------------------------------------------------------
activate_tilert_env() {
    local env_dir=/opt/conda/envs/tilert
    [[ -d "$env_dir" ]] || return 0                     # 非官方镜像（如自建 vLLM 镜像）跳过
    if [[ "$(command -v python)" == "$env_dir/bin/python" ]]; then
        echo "[SETUP] conda env 'tilert' 已激活"
        return 0
    fi
    echo "[SETUP] 激活 conda env 'tilert'（enroot 不跑镜像 ENTRYPOINT，需手动激活）"
    # shellcheck disable=SC1091
    if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
        . /opt/conda/etc/profile.d/conda.sh && conda activate tilert
    fi
    # conda.sh 不可用时退化为直接改 PATH，效果等价
    [[ "$(command -v python)" == "$env_dir/bin/python" ]] || export PATH="$env_dir/bin:$PATH"
    echo "[SETUP] python -> $(command -v python)"
}

# 查已装版本（用 importlib.metadata 而非 import：import tilert 需要 libcuda，
# 在无 GPU 的上下文里会假报“没装”）。
_installed_version() {
    "$PY" - "$1" <<'PY' 2>/dev/null
import sys
from importlib.metadata import version, PackageNotFoundError
try:
    print(version(sys.argv[1]))
except PackageNotFoundError:
    pass
PY
}

_pip_args() {
    local a=(--quiet --no-cache-dir)
    [[ -n "$TILERT_PIP_INDEX_URL" ]] && a+=(--index-url "$TILERT_PIP_INDEX_URL")
    printf '%s\n' "${a[@]}"
}

# ---------------------------------------------------------------------------
# 1. decode 侧：tilert + 自身依赖。decode_server 是纯 tilert，不需要 vLLM。
# ---------------------------------------------------------------------------
install_tilert_decode() {
    mapfile -t _pa < <(_pip_args)
    local have; have="$(_installed_version tilert)"
    if [[ "$have" == "$TILERT_VERSION" ]]; then
        echo "[SETUP] tilert $have 已存在，跳过"
    else
        [[ -n "$have" ]] && echo "[SETUP] 已装 tilert $have，切到 pin 的 $TILERT_VERSION"
        echo "[SETUP] 装 tilert==$TILERT_VERSION（官方 PyPI release wheel）"
        "$PY" -m pip install "${_pa[@]}" "tilert==$TILERT_VERSION" || {
            echo "[SETUP] ERROR: tilert==$TILERT_VERSION 安装失败"; exit 1; }
        have="$(_installed_version tilert)"
        [[ "$have" == "$TILERT_VERSION" ]] || {
            echo "[SETUP] ERROR: 装完仍不是 $TILERT_VERSION（实际: ${have:-未安装}）"; exit 1; }
        _SETUP_INSTALLED+=("tilert==$TILERT_VERSION")
    fi
    # pd_vllm 的 HTTP 栈（wheel 未声明）+ KV 传输后端。两者官方镜像都不带。
    _install_missing "uvicorn" "$TILERT_HTTP_DEPS"
    _install_missing "nixl"    "$TILERT_TRANSPORT_DEPS"
    # ⛔ transformers 必须 >= 5.4：GLM-5.1 官方 checkpoint 的 tokenizer_config.json 用的是
    #    transformers 5.x 才有的 `TokenizersBackend`；镜像自带 4.46.3 会直接报
    #      ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported.
    #    而 decode_server 会从 $TILERT_WEIGHTS_DIR 加载 tokenizer（glm_5/generator.py:48）。
    #    tilert wheel 声明的是 `transformers>=4.46.3`（不是 ==），所以升级合规；4.46.3 只是
    #    镜像自带版本，pip 因为 ">=" 已满足而不会主动升。实测 5.14.1 + torch 2.11.0+cu130 正常。
    local tv; tv="$(_installed_version transformers)"
    if [[ -n "$tv" && "${tv%%.*}" -lt 5 ]]; then
        echo "[SETUP] transformers $tv < 5 —— 官方 checkpoint 的 TokenizersBackend 读不了，升级"
        "$PY" -m pip install "${_pa[@]}" -U "transformers>=5.4.0" || {
            echo "[SETUP] ERROR: transformers 升级失败"; exit 1; }
        _SETUP_INSTALLED+=("transformers>=5.4.0(upgrade from $tv)")
    fi
    # 到这一步 decode_server 必须能导入，否则后面起服务才炸、且看不到根因。
    "$PY" -c "import tilert.pd_vllm.decode_server" 2>/dev/null || {
        echo "[SETUP] ERROR: import tilert.pd_vllm.decode_server 失败，实际报错："
        "$PY" -c "import tilert.pd_vllm.decode_server" 2>&1 | tail -3
        exit 1; }
    echo "[SETUP] tilert.pd_vllm.decode_server 可导入 ✓"
}

# _install_missing <探针模块> <要装的包列表>：探针模块已可导入则跳过，否则装。
_install_missing() {
    local probe="$1" pkgs="$2"
    [[ -n "$pkgs" ]] || return 0
    if "$PY" -c "import $probe" 2>/dev/null; then
        echo "[SETUP] $probe 已存在，跳过（$pkgs）"
        return 0
    fi
    echo "[SETUP] 装 $pkgs（探针 $probe 缺失）"
    mapfile -t _pa < <(_pip_args)
    # shellcheck disable=SC2086
    "$PY" -m pip install "${_pa[@]}" $pkgs || {
        echo "[SETUP] ERROR: 安装失败: $pkgs"; exit 1; }
    _SETUP_INSTALLED+=("$pkgs")
}

# ---------------------------------------------------------------------------
# 2. prefill 侧：vLLM 由镜像自带（本脚本不装，见文件头说明），只补 connector 插件。
# ---------------------------------------------------------------------------
install_tilert_prefill() {
    local vllm_v; vllm_v="$(_installed_version vllm)"
    if [[ -z "$vllm_v" ]]; then
        echo "[SETUP] ERROR: prefill 侧镜像里没有 vLLM。"
        echo "[SETUP]        prefill 需要一个支持 V1 disaggregation + GLM-5/5.1(DSA) +"
        echo "[SETUP]        --kv-cache-dtype fp8_ds_mla 的 vLLM 镜像；官方 tilert 镜像不含 vLLM，"
        echo "[SETUP]        且其 transformers==4.46.3 与 vLLM 的 >=5.5.3 冲突，两侧必须用不同镜像。"
        exit 1
    fi
    echo "[SETUP] prefill 侧 vLLM $vllm_v"
    local have; have="$(_installed_version tilert)"
    if [[ "$have" == "$TILERT_VERSION" ]]; then
        echo "[SETUP] tilert $have 已存在，跳过"
    else
        # --no-deps 是硬要求：否则 tilert 的 transformers==4.46.3 会把 vLLM 需要的 5.x 降级。
        echo "[SETUP] 装 tilert==$TILERT_VERSION --no-deps（只为 connector 插件可导入；不动 transformers）"
        mapfile -t _pa < <(_pip_args)
        "$PY" -m pip install "${_pa[@]}" --no-deps "tilert==$TILERT_VERSION" || {
            echo "[SETUP] ERROR: tilert==$TILERT_VERSION (--no-deps) 安装失败"; exit 1; }
        have="$(_installed_version tilert)"
        [[ "$have" == "$TILERT_VERSION" ]] || {
            echo "[SETUP] ERROR: 装完仍不是 $TILERT_VERSION（实际: ${have:-未安装}）"; exit 1; }
        _SETUP_INSTALLED+=("tilert==$TILERT_VERSION(--no-deps)")
    fi
    # vLLM 镜像自带 fastapi/uvicorn/pydantic（其 OpenAI server 就用这套），故不补 HTTP 栈；
    # 但 KV 传输后端要与 decode 侧对齐，缺则补。
    _install_missing "nixl" "$TILERT_TRANSPORT_DEPS"
    "$PY" -c "import tilert.pd_vllm.prefill_connector" 2>/dev/null || {
        echo "[SETUP] WARN: import tilert.pd_vllm.prefill_connector 失败（vLLM 会在加载 connector 插件时再报）："
        "$PY" -c "import tilert.pd_vllm.prefill_connector" 2>&1 | tail -3; }
}

# ---------------------------------------------------------------------------
# 0b. 统一解释器名。两侧镜像不一样：官方 tilert 镜像的 conda env 提供 `python`，
#     而 vllm/vllm-openai:v0.26.0 **只有 `python3`、没有 `python`**（实测 node076）。
#     写死 `python` 会在 prefill 侧直接 command not found。解析成 $PY 并导出给 run_node.sh 用。
#     必须在 activate_tilert_env 之后解析 —— 激活会改 PATH。
# ---------------------------------------------------------------------------
resolve_python() {
    if [[ -n "${PY:-}" ]] && command -v "$PY" >/dev/null 2>&1; then :
    else
        PY=""
        for c in python python3; do command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }; done
    fi
    [[ -n "$PY" ]] || { echo "[SETUP] ERROR: 找不到 python / python3"; exit 1; }
    export PY
    echo "[SETUP] 解释器 PY=$PY ($(command -v "$PY"))"
}

# ============================ 入口 ============================
activate_tilert_env
resolve_python
case "${ROLE:-}" in
    decode)        install_tilert_decode ;;
    prefill)       install_tilert_prefill ;;
    all)           install_tilert_decode ;;   # 单机 all-in-one 冒烟：与 decode 同环境
    *)             echo "[SETUP] 未知 ROLE='${ROLE:-}'，跳过依赖安装" ;;
esac
if (( ${#_SETUP_INSTALLED[@]} )); then
    echo "[SETUP] 本次安装: ${_SETUP_INSTALLED[*]}"
else
    echo "[SETUP] 无需安装（依赖已就绪）"
fi
