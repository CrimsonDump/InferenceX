#!/usr/bin/env bash
# ============================================================================
# glm52_arch_bypass.sh — GLM-5.2 本地 checkpoint 的 architectures 旁路
#   DeepseekV32ForCausalLM  ->  GlmMoeDsaForCausalLM
# ============================================================================
# 本地权重 /tilert/zqk/.../glm-5p2-fp8/{model,mtp}/config.json 把 GLM-5.2 标成
# architectures=["DeepseekV32ForCausalLM"] (model_type=deepseek_v32), 而官方
# zai-org/GLM-5.2-FP8 标的是 ["GlmMoeDsaForCausalLM"]. 两个名字指向【同一个实现】
# (DeepseekV32ForCausalLM(DeepseekV2ForCausalLM): pass; GlmMoeDsaForCausalLM 只 override
# determine_num_fused_shared_experts), weight loader / 参数名 / module 树完全一致 ——
# 但各引擎的**按名路由**只认后者, 用前者会静默走到另一条(更慢/更旧的)代码路径:
#
#   sglang: server_args.py::_handle_model_specific_adjustments 的外层白名单不含
#     DeepseekV32ForCausalLM -> 整个 DSA 自动配置块被跳过: attention_backend 留
#     flashinfer(而非 nsa)、page_size 留 1(而非 64)、kv_cache_dtype 留 auto=bf16
#     (而非 fp8_e4m3)、NSA prefill/decode 后端不设. 即"不指定 --kv-cache-dtype"
#     并不等于 cookbook 的 fp8, 根本不是同一条路径, 数字不可比.
#     另: model_config.py::_config_draft_model 的 NextN 改写白名单在上游 build 里
#     也不含 V32 -> draft 会命中整模型类(78 层)而非 NextN, 必崩/OOM. 故【主模型与
#     draft(mtp/) 都要改名】.
#   tokenspeed: 只按 GlmMoeDsaForCausalLM 注册 GLM-5.2 (registry + DSA attention
#     family + NextN 检测 + get_config 的 _restore_raw_glm_dsa_fields).
#   vllm: glm52 镜像两个名字都注册, 无此问题 (故 vllm 侧不需要旁路).
#
# 修法(隔离): 造一个"只重写 config.json、其余文件全 symlink"的旁路目录, --model /
# draft 指向它. 【不改 /tilert 共享权重】—— 那会连带影响别的 backend 与别人的 run.
# model_type 不改 (无按 glm_moe_dsa 分流的代码; 且这份 config 靠 auto_map 走 DS
# remote code 解析, 改 model_type 会让 AutoConfig 找不到 config 类).
#
# 用法 (容器内脚本 source 本文件后):
#   MODEL="$(apply_arch_bypass "$MODEL")"
#   MTP_DRAFT_PATH="$(apply_arch_bypass "$MTP_DRAFT_PATH")"
# 幂等: 源已是目标 arch 时原样返回, 不建目录. 日志走 stderr (stdout 只有路径).
#
# 落点: 默认 ${src}_glmarch (与源同级); 源在只读/禁写区时用 ARCH_BASE 指到可写区
# (symlink 仍指向源权重, 读权重不受影响).
# ============================================================================

GLM_TARGET_ARCH="${GLM_TARGET_ARCH:-GlmMoeDsaForCausalLM}"

# 读 config.json 的 architectures[0] (读不到则空)
_arch_name_of() {
    python3 -c "
import json,sys
try:
    print((json.load(open(sys.argv[1]+'/config.json')).get('architectures') or [''])[0])
except Exception:
    print('')
" "$1" 2>/dev/null
}

# 旁路目录落点: ARCH_BASE 优先(可写区), 否则与源同级
arch_bypass_dir() {
    local src="${1%/}" name
    name="$(basename "$src")_glmarch"
    if [[ -n "${ARCH_BASE:-${TS_ARCH_BASE:-}}" ]]; then
        printf '%s\n' "${ARCH_BASE:-$TS_ARCH_BASE}/$name"
    else
        printf '%s\n' "${src}_glmarch"
    fi
}

# 建旁路目录: 除 config.json 外全 symlink; config.json 改写 architectures 后写入
# (用 find 而非 glob 枚举: 要带上点文件, 且 `.[!.]*` 在 zsh 下无匹配即报错)
make_arch_bypass() {
    local src="${1%/}" dst="${2%/}" arch="${3:-$GLM_TARGET_ARCH}" f bn
    mkdir -p "$dst" || return 1
    while IFS= read -r -d '' f; do
        bn="$(basename "$f")"
        [[ "$bn" == "config.json" ]] && continue
        ln -sfn "$f" "$dst/$bn" || return 1
    done < <(find "$src" -mindepth 1 -maxdepth 1 -print0)
    python3 -c "
import json,sys
src,dst,arch = sys.argv[1],sys.argv[2],sys.argv[3]
d = json.load(open(src+'/config.json'))
d['architectures'] = [arch]
json.dump(d, open(dst+'/config.json','w'), indent=2)
" "$src" "$dst" "$arch" || return 1
}

# 幂等入口: stdout = 实际该用的模型目录
apply_arch_bypass() {
    local src="${1%/}" dst cur
    [[ -d "$src" ]] || { echo "!! arch 旁路: 源目录不存在: $src" >&2; return 1; }
    cur="$(_arch_name_of "$src")"
    if [[ "$cur" == "$GLM_TARGET_ARCH" ]]; then
        echo "arch 旁路: $src 已是 $GLM_TARGET_ARCH, 直接用" >&2
        printf '%s\n' "$src"; return 0
    fi
    dst="$(arch_bypass_dir "$src")"
    [[ -n "${ARCH_BASE:-${TS_ARCH_BASE:-}}" ]] && mkdir -p "${ARCH_BASE:-$TS_ARCH_BASE}"
    make_arch_bypass "$src" "$dst" || { echo "!! arch 旁路失败: $src -> $dst" >&2; return 1; }
    echo "arch 旁路: $src ($cur) -> $dst ($GLM_TARGET_ARCH)" >&2
    printf '%s\n' "$dst"
}
