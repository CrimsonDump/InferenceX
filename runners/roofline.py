#!/usr/bin/env python3
"""decode 的 roofline: 每个 decode step 的 FLOPs / HBM 字节 -> MFU / MBU.

为什么不用 sglang 自带的 `--enable-mfu-metrics`: 那是通用 dense 估算器
(MLP 按 intermediate_size、KV 按 num_kv_heads*head_dim、字节按 model dtype=bf16),
对 GLM-5.2 的 MoE / MLA / DSA 稀疏注意力 / fp8 权重全都不成立, 数量级都不对。

三处让教科书公式失效、这里必须建模的地方:

1. **MTP** —— 一个 decode step 不产 1 个 token。EAGLE(N, topk=1, draft=N+1) 每步做
   1 次 target verify(N+1 个位置) + N 次 draft forward, 只收 acc 个 token。所以分母
   是 step 不是 accepted token; step 时长 = acc / UTPS。
2. **MoE 的字节数随并发变** —— 一步里真正被读进 HBM 的专家数
   `E = n_exp * (1 - (1-topk/n_exp)^T)`, T = 进入该层的 token 数 = batch*(N+1)。
   低并发只读到一小部分专家权重, E 饱和后再加并发字节就不涨了 —— 这是 MBU 曲线
   先升后降的直接原因。
3. **DSA 稀疏注意力** —— 主注意力每个 query 只读 top-k(=index_topk) 个 KV, 不是全上下文;
   而 indexer 要扫全上下文, 但只存在于一部分层上(index_topk_freq>1 时其余层复用上一层的
   top-k, 连 indexer 权重都没有), 层集合由 dsa_layer_skips_topk 决定。

口径与假设(读数时必须知道):
  * 只对稳态 decode 窗口有效的点算(ERR 点没有 step 时长可言)。
  * verify 的 N+1 个位置共享一次 KV 读(它们的 top-k 高度重叠, kernel 读并集)。
    若实际按位置重读, KV 项最多 x(N+1) —— KV 在高并发下约占总字节 8%, 影响有限。
  * MoE 路由按均匀假设算被碰到的专家数; 真实路由有热点, 低并发时会略少。
  * 只算 HBM 流量。TP all-reduce / EP a2a 走 NVLink, 不计入 MBU。
  * 峰值用 spec 值(B200: FP8 dense 4.5 PFLOPS / HBM 8.0 TB/s)。实测 copy 带宽约
    6.5-7 TB/s, 若按 achievable 报, MBU 要再乘 ~1.15。
"""
import json
import os

# ---- 每卡峰值(dense). key 用小写设备名 ----
PEAKS = {
    # name: (fp8_dense_flops, hbm_bytes_per_s)
    "b200": (4.5e15, 8.0e12),
    "h200": (1.979e15, 4.8e12),
    "h100": (1.979e15, 3.35e12),
}
DEFAULT_DEVICE = "b200"

# GLM-5.2 FP8 的 HF config 摘要 —— 读不到权重目录时的兜底(与
# /tilert/zqk/20260612/glm-5p2-fp8/model/config.json 一致)。
DEFAULT_HF_CONFIG = {
    "hidden_size": 6144, "num_hidden_layers": 78, "first_k_dense_replace": 3,
    "intermediate_size": 12288, "moe_intermediate_size": 2048,
    "n_routed_experts": 256, "num_experts_per_tok": 8, "n_shared_experts": 1,
    "vocab_size": 154880, "num_attention_heads": 64,
    "q_lora_rank": 2048, "kv_lora_rank": 512,
    "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256,
    "index_topk": 2048, "index_head_dim": 128, "index_n_heads": 32,
    "index_topk_freq": 4, "index_skip_topk_offset": 3, "index_topk_pattern": None,
    "num_nextn_predict_layers": 1,
    "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
}


def _skips_topk(cfg, layer_id):
    """与 sglang configs/model_config.py::dsa_layer_skips_topk 同逻辑 ——
    该层是否复用上一层的 top-k(复用 = 该层没有 indexer)。"""
    pat = cfg.get("index_topk_pattern")
    if pat is not None:
        return layer_id < len(pat) and pat[layer_id] == "S"
    freq = cfg.get("index_topk_freq") or 1
    off = cfg.get("index_skip_topk_offset")
    if off is not None:
        return max(layer_id - off + 1, 0) % freq != 0
    return max(layer_id - 1, 0) % freq != 0


class Shape(object):
    """从 HF config 推出的、算 roofline 需要的全部标量。"""

    def __init__(self, cfg):
        self.cfg = cfg
        g = cfg.get
        h = self.h = g("hidden_size")
        self.n_layers = g("num_hidden_layers")
        self.n_dense = g("first_k_dense_replace", 0) or 0
        self.n_moe_layers = self.n_layers - self.n_dense
        self.n_exp = g("n_routed_experts") or 0
        self.topk = g("num_experts_per_tok") or 0
        self.d_exp = g("moe_intermediate_size") or 0
        self.n_shared = g("n_shared_experts", 0) or 0
        self.d_dense = g("intermediate_size") or 0
        self.vocab = g("vocab_size")
        nh = self.nh = g("num_attention_heads")
        q_lora = self.q_lora = g("q_lora_rank") or 0
        kv_lora = self.kv_lora = g("kv_lora_rank")
        nope, rope, vd = g("qk_nope_head_dim"), g("qk_rope_head_dim"), g("v_head_dim")
        self.rope = rope
        self.dsa_topk = g("index_topk") or 0
        self.idx_d = g("index_head_dim") or 0
        self.idx_nh = g("index_n_heads") or 0
        self.n_mtp = g("num_nextn_predict_layers", 0) or 0

        # 带 indexer 的层(其余层复用上一层 top-k)
        self.idx_layers = sum(0 if _skips_topk(cfg, i) else 1
                              for i in range(self.n_layers))

        qc = cfg.get("quantization_config") or {}
        self.w_bytes = 1 if str(qc.get("quant_method", "")).startswith("fp8") else 2
        blk = (qc.get("weight_block_size") or [128, 128])[0]

        # ---- 每层参数量 ----
        # MLA: q_a, q_b, kv_a(+mqa), kv_b, o
        self.p_attn_l = (q_lora * h + nh * (nope + rope) * q_lora
                         + (kv_lora + rope) * h + nh * (nope + vd) * kv_lora
                         + nh * vd * h)
        # DSA indexer: wq_b, wk, weights_proj
        self.p_idx_l = self.idx_nh * self.idx_d * q_lora + self.idx_d * h + self.idx_nh * h
        self.p_exp = 3 * h * self.d_exp                      # 单个专家单层
        self.p_dense_l = 3 * h * self.d_dense
        self.p_gate_l = self.n_exp * h
        self.p_lmhead = self.vocab * h
        self.p_ehproj = h * 2 * h                            # MTP eh_proj

        # ---- 权重字节(全模型, 未分片) ----
        wb, W2 = self.w_bytes, 2
        self.b_attn = self.p_attn_l * self.n_layers * wb
        self.b_idx = self.p_idx_l * self.idx_layers * wb
        self.b_moe = self.p_exp * self.n_exp * self.n_moe_layers * wb
        self.b_shared = self.p_exp * self.n_shared * self.n_moe_layers * wb
        self.b_dense = self.p_dense_l * self.n_dense * wb
        self.b_gate = self.p_gate_l * self.n_moe_layers * W2
        self.b_lmhead = self.p_lmhead * W2
        self.b_d_attn = self.p_attn_l * wb
        self.b_d_idx = self.p_idx_l * wb
        self.b_d_moe = self.p_exp * self.n_exp * wb
        self.b_d_shared = self.p_exp * self.n_shared * wb
        self.b_d_eh = self.p_ehproj * W2
        self.b_total = (self.b_attn + self.b_idx + self.b_moe + self.b_shared
                        + self.b_dense + self.b_gate + self.b_lmhead)

        # ---- 每 token 激活参数(参与 GEMM 的) -> FLOPs = 2 * P ----
        self.p_act_target = (self.p_attn_l * self.n_layers
                             + self.p_idx_l * self.idx_layers
                             + self.topk * self.p_exp * self.n_moe_layers
                             + self.n_shared * self.p_exp * self.n_moe_layers
                             + self.p_dense_l * self.n_dense
                             + self.p_gate_l * self.n_moe_layers
                             + self.p_lmhead)
        self.p_act_draft = (self.p_attn_l + self.p_idx_l + self.topk * self.p_exp
                            + self.n_shared * self.p_exp + self.p_gate_l
                            + self.p_ehproj + self.p_lmhead)

        # ---- KV cache 每 token 每层字节 (sglang DSATokenToKVPool 布局) ----
        # 主 KV: nope 潜向量 fp8 + 每 128 一个 fp32 scale + rope 恒 bf16
        self.kv_main_pt = kv_lora * 1 + (kv_lora // blk) * 4 + rope * 2
        # indexer K: fp8 + 每 128 一个 fp32 scale
        self.kv_idx_pt = self.idx_d * 1 + (self.idx_d // blk) * 4

        # ---- 注意力点积 FLOPs 系数(每 query 位置) ----
        # 吸收式 MLA decode: score = q·kv_latent(kv_lora+rope), out = p·v_latent(kv_lora)
        self.attn_coef_main_l = 2 * nh * (kv_lora + rope + kv_lora)
        self.attn_coef_idx_l = 2 * self.idx_nh * self.idx_d

    # ------------------------------------------------------------------
    def summary(self):
        return {
            "总参数(B)": round(self.b_total / self.w_bytes / 1e9, 1),
            "权重字节(GB)": round(self.b_total / 1e9, 1),
            "每token激活参数(B)": round(self.p_act_target / 1e9, 2),
            "MoE专家权重(GB)": round(self.b_moe / 1e9, 1),
            "带indexer的层": "%d / %d" % (self.idx_layers, self.n_layers),
            "DSA top-k": self.dsa_topk,
            "KV字节/token/层": self.kv_main_pt,
            "indexerK字节/token/层": self.kv_idx_pt,
        }

    def experts_touched(self, tokens_per_layer):
        """一层里被至少一个 token 路由到的专家数(均匀路由假设)。"""
        if tokens_per_layer <= 0 or not self.n_exp:
            return 0.0
        return self.n_exp * (1.0 - (1.0 - float(self.topk) / self.n_exp) ** tokens_per_layer)


def load_shape(model_dir=None):
    """优先读权重目录的 config.json; 读不到就用内置的 GLM-5.2 摘要。"""
    if model_dir:
        p = os.path.join(model_dir, "config.json")
        try:
            with open(p) as f:
                return Shape(json.load(f)), p
        except Exception:
            pass
    return Shape(DEFAULT_HF_CONFIG), "(内置 GLM-5.2 默认值)"


DP_ATTN_PARALLELS = ("dep", "dpa-tp")   # attn 走 DP: attn 权重复制, KV 按卡分摊


def analyze(shape, batch, ctx_len, mtp_n, t_step_s, gpus, parallel,
            device=DEFAULT_DEVICE):
    """一个测点的 roofline.

    batch    全局并发(稳态窗口内在跑的请求数)
    ctx_len  稳态窗口内的平均上下文长度(≈ ISL + OSL/2)
    mtp_n    MTP 的 speculative steps; <=0 表示没开
    t_step_s 一个 decode step 的秒数(= acc / UTPS)
    parallel tp / dep / tep / dpa-tp
    """
    s = shape
    if not (t_step_s and t_step_s > 0 and batch and gpus):
        return None
    peak_flops, peak_bw = PEAKS.get(device, PEAKS[DEFAULT_DEVICE])
    dp_attn = parallel in DP_ATTN_PARALLELS
    n_pos_v = mtp_n + 1 if mtp_n > 0 else 1     # verify 一次过的位置数
    n_draft = mtp_n if mtp_n > 0 else 0
    lsel = min(s.dsa_topk, ctx_len) if s.dsa_topk else ctx_len
    seqs_pg = float(batch) / gpus if dp_attn else float(batch)
    sh_attn = 1.0 if dp_attn else 1.0 / gpus    # attn 权重: DP 下每卡一整份
    sh = 1.0 / gpus                             # 其余一律 TP/EP 分片

    # ---------------- FLOPs ----------------
    f_tgt = (2 * s.p_act_target
             + s.attn_coef_main_l * s.n_layers * lsel
             + s.attn_coef_idx_l * s.idx_layers * ctx_len)
    f_drf = (2 * s.p_act_draft
             + s.attn_coef_main_l * lsel
             + s.attn_coef_idx_l * ctx_len) if n_draft else 0.0
    flops_pg = (batch * n_pos_v * f_tgt + batch * n_draft * f_drf) / gpus

    # ---------------- HBM 字节 ----------------
    e_v = s.experts_touched(batch * n_pos_v)
    w_moe = s.b_moe * (e_v / s.n_exp if s.n_exp else 0) * sh
    w_other = ((s.b_attn + s.b_idx) * sh_attn
               + (s.b_shared + s.b_dense + s.b_gate + s.b_lmhead) * sh)
    kv_read = seqs_pg * (s.n_layers * lsel * s.kv_main_pt
                         + s.idx_layers * ctx_len * s.kv_idx_pt)
    kv_write = seqs_pg * n_pos_v * (s.n_layers * s.kv_main_pt
                                    + s.idx_layers * s.kv_idx_pt)
    if n_draft:
        e_d = s.experts_touched(batch)
        w_draft = n_draft * ((s.b_d_attn + s.b_d_idx) * sh_attn
                             + s.b_d_moe * (e_d / s.n_exp if s.n_exp else 0) * sh
                             + (s.b_d_shared + s.b_d_eh + s.p_gate_l * 2
                                + s.b_lmhead) * sh)
        kv_draft = n_draft * seqs_pg * (lsel * s.kv_main_pt + ctx_len * s.kv_idx_pt)
    else:
        w_draft = kv_draft = 0.0
    bytes_pg = w_moe + w_other + kv_read + kv_write + w_draft + kv_draft

    return {
        "mfu": flops_pg / t_step_s / peak_flops,
        "mbu": bytes_pg / t_step_s / peak_bw,
        "tflops_per_gpu": flops_pg / t_step_s / 1e12,
        "hbm_tbs_per_gpu": bytes_pg / t_step_s / 1e12,
        "gflop_per_step_gpu": flops_pg / 1e9,
        "gb_per_step_gpu": bytes_pg / 1e9,
        "experts_touched": e_v,
        "gb_moe": (w_moe + w_draft) / 1e9,
        "gb_wother": w_other / 1e9,
        "gb_kv": (kv_read + kv_write + kv_draft) / 1e9,
        "step_ms": t_step_s * 1e3,
        "ctx_len": ctx_len,
        "peak_tflops": peak_flops / 1e12,
        "peak_tbs": peak_bw / 1e12,
    }


def step_seconds(utps, mtp_n, acc):
    """一个 decode step 的秒数。一步吐 acc 个 token(MTP 关则 1 个), UTPS = 每用户 tok/s。"""
    if not utps or utps <= 0:
        return None
    return (float(acc) if (mtp_n and mtp_n > 0 and acc) else 1.0) / float(utps)


if __name__ == "__main__":
    import argparse
    import re

    ap = argparse.ArgumentParser(description="从结果 JSON 算 decode MFU / MBU")
    ap.add_argument("paths", nargs="+", help="结果 JSON 或包含它们的目录")
    ap.add_argument("--model-dir", default=None, help="权重目录(读 config.json 推结构)")
    ap.add_argument("--device", default=DEFAULT_DEVICE, choices=sorted(PEAKS))
    a = ap.parse_args()

    shape, src = load_shape(a.model_dir)
    print("结构来源: %s" % src)
    for k, v in shape.summary().items():
        print("  %-22s %s" % (k, v))
    print()

    fre = re.compile(r"glm52_([A-Za-z0-9.+-]+)_s(\d+)_c(\d+)(?:_r(\d+))?\.json$")
    files = []
    for p in a.paths:
        if os.path.isdir(p):
            files += [os.path.join(p, f) for f in sorted(os.listdir(p))
                      if f.startswith("glm52_") and f.endswith(".json")]
        else:
            files.append(p)
    cols = ("series", "c", "step_ms", "UTPS", "TFLOPS", "MFU%", "GB/step",
            "TB/s", "MBU%", "expts")
    hdr = "%-38s %4s %8s %7s %8s %6s %8s %6s %6s %6s" % cols
    print(hdr)
    print("-" * len(hdr))
    out = []
    for f in files:
        m = fre.search(os.path.basename(f))
        if not m:
            continue
        d = json.load(open(f))
        utps = d.get("steady_utps_per_user")
        if not utps:
            continue
        series, isl = m.group(1), int(m.group(2))
        par = "tp"
        for p in ("dpa-tp", "tep", "dep", "tp"):
            if "-%s-" % p in series:
                par = p
                break
        mm = re.search(r"mtpN(\d+)A([\d.]+)", series)
        n, acc = (int(mm.group(1)), float(mm.group(2))) if mm else (0, 1.0)
        osl = (d.get("output_lens") or [0])[0]
        r = analyze(shape, d.get("steady_num_reqs") or d.get("max_concurrency"),
                    isl + osl / 2.0, n, step_seconds(utps, n, acc),
                    d.get("gpu_count") or 8, par, a.device)
        if r:
            out.append((series, int(m.group(3)), utps, r))
    for series, c, utps, r in sorted(out, key=lambda x: (x[0], x[1])):
        print("%-38s %4d %8.2f %7.1f %8.1f %6.2f %8.1f %6.2f %6.1f %6.0f" % (
            series, c, r["step_ms"], utps, r["tflops_per_gpu"], r["mfu"] * 100,
            r["gb_per_step_gpu"], r["hbm_tbs_per_gpu"], r["mbu"] * 100,
            r["experts_touched"]))
