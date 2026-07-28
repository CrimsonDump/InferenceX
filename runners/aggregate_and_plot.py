#!/usr/bin/env python3
"""汇总 sweep 产生的结果 JSON -> CSV + SVG 曲线 + 自包含 HTML 报告.

每个 JSON 文件名形如 glm52_<series>_s<seqlen>_c<batch>_r<rep>.json, 内含稳态指标
(steady_utps_per_user / steady_stps_per_gpu, 由 benchmark_serving.py 写入).
命名分三层, 各层只放本层统一的字段:
  图     = 一份报告/一个目录: <model>_<quant>_i<ISL>o<OSL>_c<最小>-<最大>
  曲线   = <series>: <backend>-<commit>-<parallel>-mtpN<N>A<acc> (如 sglang-9844dd05-tp-mtpN5A3.5)
  点     = concurrency(batch), 即文件名的 _c<batch> 段 —— 曲线上的横轴
所以同一 outdir 里可以同时放多 backend / 多 commit / 多并行 / 多 MTP 的曲线, 画在一张图上.

输出(在 <results-dir>):
  <--report-name>        自包含报告(默认 report.html; bench.sh 传上面的图名 + .html):
                         测试配置 + 内联SVG曲线图 + 均值表 + 全量原始数据 (浏览器直接打开)
  tps_raw.csv            全部原始数据 (每次重复一行)
  tps_curve.csv          每点多次取均值
  tps_curve_s<seqlen>.svg 每个 seqlen 一张曲线图
  tps_mbu_s<seqlen>.svg   每个 seqlen 一张 MBU 曲线图
横轴 = interactivity (tok/s/user), 纵轴 = 每GPU吞吐 (tok/s/gpu).
配置区来自 <results-dir>/run_config.json (由 bench.sh 写入).

每个有效点还额外算 roofline 的 MFU / MBU (见 roofline.py —— 建模了 MTP 的
"一步 verify N+1 位置 + N 次 draft"、MoE 被碰到的专家数随并发变、DSA top-k 稀疏
注意力; 不能用 sglang 自带的 --enable-mfu-metrics, 那是通用 dense 估算器)。
"""
import argparse
import csv
import glob
import json
import os
import re

import roofline

# 文件名: glm52_<series>_s<seqlen>_c<batch>[_r<rep>].json  (rep 可选, 向后兼容)
# series 允许 字母/数字/'-'/'.'/'+' (含小数 accept_len 如 mtpN5A3.5), 【不含下划线】——
# 靠下划线切出 _s<ISL>_c<batch>_r<rep>.
SERIES_PAT = r"[A-Za-z0-9.+-]+"
FNAME_RE = re.compile(
    r"glm52_(?P<series>" + SERIES_PAT +
    r")_s(?P<seqlen>\d+)_c(?P<batch>\d+)(?:_r(?P<rep>\d+))?\.json$")


# series = <backend>-<commit>-<parallel>-mtp… ; parallel 可能含 '-'(dpa-tp), 故两头锚定
SERIES_RE = re.compile(r"^(?P<backend>[^-]+)-(?P<commit>[^-]+)-(?P<par>.+)-"
                       r"(?P<mtp>mtpoff|mtpN\d+A[0-9.]+)$")


def parse_series(sname):
    """series 名 -> (parallel, mtp_n, mtp_acc). MTP 关时 n=0/acc=1."""
    m = SERIES_RE.match(str(sname))
    if not m:
        return "tp", 0, 1.0
    par, mtp = m.group("par"), m.group("mtp")
    mm = re.match(r"mtpN(\d+)A([0-9.]+)$", mtp)
    if not mm:
        return par, 0, 1.0
    return par, int(mm.group(1)), float(mm.group(2))


def collect(results_dir, gpu_count, shape=None, device=roofline.DEFAULT_DEVICE):
    """读所有原始 JSON, 每个文件一行 (含 rep). 保留全部原始数据.

    shape 给出时, 每个稳态窗口有效的点额外算 roofline 的 MFU / MBU。
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "glm52_*_c*.json"))):
        m = FNAME_RE.search(os.path.basename(path))
        if not m:
            continue
        try:
            d = json.load(open(path))
        except Exception as e:
            print(f"跳过无法解析的 {path}: {e}")
            continue
        utps = d.get("steady_utps_per_user")
        stps_gpu = d.get("steady_stps_per_gpu")
        stps_sys = d.get("steady_stps_system")
        # 只认稳态 decode 窗口. 窗口无效(空窗口 / steady_* 缺失)时【绝不兜底】到全程聚合
        # (output_throughput、1000/mean_tpot —— 它们混入 prefill 和排队, 不是 decode 口径),
        # 那样等于用两个假数 + 一个 0 掩盖问题. 改为保持缺失 + 显式 status, 让无效点暴露在结果里.
        completed = d.get("completed", 0) or 0
        if completed <= 0:
            status, note = "ERR", "completed=0 (server 崩 / 该点无完成请求)"
        elif utps is None or stps_gpu is None:
            status, note = "ERR", "稳态窗口无效 — %s" % (d.get("steady_note") or "无 steady_* 字段")
        else:
            status, note = "ok", ""
        ok = (status == "ok")
        # cookbook 口径(与稳态口径并列, 不是兜底): cookbook 表的 tokens/s/user = 1000/mean_TPOT,
        # tokens/s/GPU = (ISL+OSL)*完成数/总时长/GPU数 (= total_token_throughput/gpu, 【含 prefill
        # token】). 它不需要稳态窗口, 所以 ERR 点(如 c=1024)在这个口径下依然有数 —— 正因如此
        # 这些字段对【所有】点都算, 而 utps/stps 只在窗口有效时才有值.
        _tpot = d.get("mean_tpot_ms") or 0
        _g = d.get("gpu_count") or gpu_count or 1
        # ---- roofline: MFU / MBU. 只在稳态窗口有效时算 —— 无效点没有"每 step 时长"可言,
        # 硬凑一个数只会污染曲线(和 utps/stps 同样的原则: 不用假数兜底).
        rf = None
        if ok and shape is not None:
            par, mtp_n, mtp_acc = parse_series(m.group("series"))
            osl = (d.get("output_lens") or [0])[0]
            rf = roofline.analyze(
                shape,
                batch=d.get("steady_num_reqs") or d.get("max_concurrency") or int(m.group("batch")),
                ctx_len=int(m.group("seqlen")) + osl / 2.0,   # 稳态窗口内的平均上下文
                mtp_n=mtp_n,
                t_step_s=roofline.step_seconds(utps, mtp_n, mtp_acc),
                gpus=_g, parallel=par, device=device)
        rows.append({
            # series = 曲线名(backend-commit-parallel-mtp…); backend 单独留一列便于过滤
            "series": m.group("series"),
            "backend": m.group("series").split("-")[0],
            "seqlen": int(m.group("seqlen")),
            "batch_size": int(m.group("batch")),
            "rep": int(m.group("rep")) if m.group("rep") else 1,
            "status": status,
            "utps_per_user": round(utps, 3) if ok else "",
            "stps_per_gpu": round(stps_gpu, 3) if ok else "",
            "stps_system": round(stps_sys, 2) if (ok and stps_sys is not None) else "",
            "cb_ttft_ms": round(d.get("mean_ttft_ms") or 0, 1),
            "cb_tpot_ms": round(_tpot, 3),
            "cb_utps": round(1000.0 / _tpot, 2) if _tpot else "",
            "cb_tps_per_gpu": round((d.get("total_token_throughput") or 0) / _g, 1),
            "cb_out_tps_per_gpu": round((d.get("output_throughput") or 0) / _g, 1),
            "steady_median_itl_ms": round(d.get("steady_median_itl_ms", 0), 4),
            # roofline (见 roofline.py 的口径与假设); 无效点留空
            "mfu_pct": round(rf["mfu"] * 100, 3) if rf else "",
            "mbu_pct": round(rf["mbu"] * 100, 2) if rf else "",
            "step_ms": round(rf["step_ms"], 3) if rf else "",
            "tflops_per_gpu": round(rf["tflops_per_gpu"], 2) if rf else "",
            "hbm_tbs_per_gpu": round(rf["hbm_tbs_per_gpu"], 3) if rf else "",
            "gb_per_step_gpu": round(rf["gb_per_step_gpu"], 2) if rf else "",
            "experts_touched": round(rf["experts_touched"], 1) if rf else "",
            "gb_moe": round(rf["gb_moe"], 2) if rf else "",
            "gb_wother": round(rf["gb_wother"], 2) if rf else "",
            "gb_kv": round(rf["gb_kv"], 3) if rf else "",
            "completed": d.get("completed", ""),
            "note": note,
        })
    rows.sort(key=lambda r: (r["seqlen"], r["series"], r["batch_size"], r["rep"]))
    return rows


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


# roofline 派生列 (列名, 保留小数位) —— collect() 写进 raw, aggregate() 逐点取均值
RF_KEYS = (("mfu_pct", 3), ("mbu_pct", 2), ("step_ms", 3), ("tflops_per_gpu", 2),
           ("hbm_tbs_per_gpu", 3), ("gb_per_step_gpu", 2), ("experts_touched", 1),
           ("gb_moe", 2), ("gb_wother", 2), ("gb_kv", 3))


def aggregate(raw):
    """按 (series, seqlen, batch) 对多次重复取均值, 附 min/max/n."""
    groups = {}
    for r in raw:
        if r["utps_per_user"] == "" or r["stps_per_gpu"] == "":
            continue
        groups.setdefault((r["series"], r["seqlen"], r["batch_size"]), []).append(r)
    out = []
    for (bk, sl, b), rs in groups.items():
        u = [x["utps_per_user"] for x in rs]
        s = [x["stps_per_gpu"] for x in rs]
        row = {
            "series": bk, "backend": bk.split("-")[0],
            "seqlen": sl, "batch_size": b, "n_reps": len(rs),
            "utps_per_user": round(_mean(u), 3),
            "stps_per_gpu": round(_mean(s), 3),
            "utps_min": round(min(u), 3), "utps_max": round(max(u), 3),
            "stps_min": round(min(s), 3), "stps_max": round(max(s), 3),
            "stps_system_mean": round(_mean([x["stps_system"] for x in rs]) or 0, 1),
        }
        # roofline 各列同样取均值(逐 rep 的 UTPS 有波动 -> step 时长与 MFU/MBU 也有)
        for k, nd in RF_KEYS:
            m = _mean([x.get(k) for x in rs])
            row[k] = round(m, nd) if m is not None else ""
        out.append(row)
    out.sort(key=lambda r: (r["seqlen"], r["series"], r["batch_size"]))
    return out


CB_KEYS = ("cb_ttft_ms", "cb_tpot_ms", "cb_utps", "cb_tps_per_gpu", "cb_out_tps_per_gpu")


def aggregate_cb(raw):
    """cookbook 口径的逐点均值. 与 aggregate() 的区别: 【不要求稳态窗口有效】——
    只要 completed>0 就算, 因为 cookbook 口径本身不依赖窗口(高并发点靠它才有数)."""
    groups = {}
    for r in raw:
        if not r.get("completed"):
            continue
        groups.setdefault((r["series"], r["seqlen"], r["batch_size"]), []).append(r)
    out = []
    for (bk, sl, b), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        row = {"series": bk, "seqlen": sl, "batch_size": b, "n_reps": len(rs)}
        for k in CB_KEYS:
            m = _mean([x.get(k) for x in rs])
            row[k] = round(m, 3) if m is not None else ""
        out.append(row)
    return out


def cookbook_refs(config):
    """{series: {batch: {…参考值}}} —— 取自 config.json 每条 curve 的 cookbook_ref.

    curve 里没有 series 名(它是推导出来的), 故按 (backend, parallel, mtp 标签) 反配: 这三者
    正是 series 名里除 commit 外的全部字段, 而 commit 只决定引擎版本、不决定该曲线对标
    cookbook 的哪一格. 于是同图 A/B 两个 commit 的曲线共用同一份参考值 —— 本就该如此.
    """
    curves = ((config or {}).get("config") or {}).get("curves") or []
    by_key = {}
    for c in curves:
        ref = c.get("cookbook_ref")
        if not ref:
            continue
        nn = c.get("mtp_n", 0)
        mtp = ("mtpN%sA%s" % (nn, c.get("mtp_acc"))) if (isinstance(nn, int) and nn > 0) else "mtpoff"
        by_key[(c.get("backend"), c.get("parallel"), mtp)] = {int(k): v for k, v in ref.items()}
    out = {}
    for sname in ((config or {}).get("series") or []):
        m = SERIES_RE.match(str(sname))
        if not m:
            continue
        ref = by_key.get((m.group("backend"), m.group("par"), m.group("mtp")))
        if ref:
            out[str(sname)] = ref
    return out


def write_csv(rows, out_csv):
    if not rows:
        print("没有可汇总的结果 JSON.")
        return
    cols = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"CSV 写出: {out_csv}  ({len(rows)} 行)")


PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]


def _series_by_seqlen(rows):
    """{seqlen: {series: [row, ...按batch排序]}} — row 为均值行(含 min/max)."""
    out = {}
    for r in rows:
        if r["utps_per_user"] == "" or r["stps_per_gpu"] == "":
            continue
        out.setdefault(r["seqlen"], {}).setdefault(r["series"], []).append(r)
    for sl in out:
        for bk in out[sl]:
            out[sl][bk].sort(key=lambda r: r["batch_size"])
    return out


def dropped_by_seqlen(raw, avg):
    """{seqlen: {series: [被判 ERR 而整点丢弃的 batch, ...]}}.

    一个 (series, seqlen, batch) 的【全部 rep 都无效】才算这个点丢了 —— 只挂了一次重复、
    另一次有效的话曲线上仍有该点, 不该在图上标 ✗.
    """
    have = {(r["series"], r["seqlen"], r["batch_size"]) for r in avg}
    out = {}
    for r in raw:
        if r.get("status", "ok") == "ok":
            continue
        key = (r["series"], r["seqlen"], r["batch_size"])
        if key in have:
            continue
        bs = out.setdefault(r["seqlen"], {}).setdefault(r["series"], set())
        bs.add(r["batch_size"])
    return {sl: {bk: sorted(v) for bk, v in d.items()} for sl, d in out.items()}


def build_svg(sl, series, dropped=None, osl=None,
              ykey="stps_per_gpu", ylo="stps_min", yhi="stps_max",
              ylabel="Per-GPU throughput  STPS (tok/s/gpu)",
              title="STPS/gpu vs UTPS", yfmt="%.0f"):
    """为某个 seqlen 生成 SVG 曲线图字符串 (零依赖, 白底). 横轴恒为 UTPS.

    ykey/ylo/yhi: 纵轴取哪一列(以及误差棒的上下界列, 缺列则不画棒) —— 同一套画法既出
                  主图(STPS/gpu), 也出 MBU 图。
    dropped: {series: [batch,...]} 该 seqlen 下被判 ERR 丢掉的并发点 —— 标进图例并加脚注。
             不标的话曲线会在某个并发"莫名其妙地断掉"(如 tp 到 64 就没了), 看图的人无法
             分辨"没测"与"测了但塞不下(稳态窗口空)"。
    osl:     进标题, 让单独看 SVG 时也知道这是什么口径的图。
    """
    dropped = dropped or {}
    W, H = 1040, 620
    # 右侧留图例: series 名较长(backend-commit-parallel-mtpN…A…), 故 mr 给足
    ml, mr, mt, mb = 80, 320, 60, 78
    pw, ph = W - ml - mr, H - mt - mb
    series = {bk: [r for r in rs if isinstance(r.get(ykey), (int, float))]
              for bk, rs in series.items()}
    series = {bk: rs for bk, rs in series.items() if rs}
    allx = [r["utps_per_user"] for bk in series for r in series[bk]]
    ally = [r.get(yhi, r[ykey]) if isinstance(r.get(yhi, r[ykey]), (int, float))
            else r[ykey] for bk in series for r in series[bk]]
    if not allx:
        return None
    xmin, xmax = 0, max(allx) * 1.08
    ymin, ymax = 0, max(ally) * 1.10

    def sx(x): return ml + (x - xmin) / (xmax - xmin) * pw
    def sy(y): return mt + ph - (y - ymin) / (ymax - ymin) * ph

    # ★必须带 viewBox★: HTML 里的 CSS 是 svg{max-width:100%;height:auto}, 没有 viewBox 时
    # 浏览器【不缩放内容】、只把元素框压窄 -> 超出容器宽度的部分被直接裁掉(踩过: 右侧图例
    # 被切、窗口再窄连右边的横轴刻度也没了)。有了 viewBox 才会整体等比缩放。
    s = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
         'width="%d" height="%d" preserveAspectRatio="xMidYMid meet" '
         'font-family="sans-serif" font-size="13">' % (W, H, W, H)]
    s.append('<rect width="%d" height="%d" fill="white"/>' % (W, H))
    s.append('<text x="%d" y="28" font-size="17" font-weight="bold" text-anchor="middle">'
             'GLM-5.2 FP8 B200 — %s (ISL=%d%s, MTP 固定 N/accept_len)</text>'
             % (W // 2, _esc(title), sl, (", OSL=%s" % osl) if osl else ""))
    for i in range(6):
        gx = ml + pw * i / 5
        xv = xmin + (xmax - xmin) * i / 5
        s.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#eee"/>' % (gx, mt, gx, mt + ph))
        s.append('<text x="%.1f" y="%d" text-anchor="middle" fill="#555">%.0f</text>' % (gx, mt + ph + 20, xv))
        gy = mt + ph * i / 5
        yv = ymax - (ymax - ymin) * i / 5
        s.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#eee"/>' % (ml, gy, ml + pw, gy))
        s.append(('<text x="%d" y="%.1f" text-anchor="end" fill="#555">' + yfmt + '</text>')
                 % (ml - 8, gy + 4, yv))
    s.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333"/>' % (ml, mt + ph, ml + pw, mt + ph))
    s.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333"/>' % (ml, mt, ml, mt + ph))
    s.append('<text x="%d" y="%d" text-anchor="middle">Interactivity  UTPS (tok/s/user)</text>'
             % (ml + pw // 2, H - 20))
    s.append('<text transform="translate(22,%d) rotate(-90)" text-anchor="middle">'
             '%s</text>' % (mt + ph // 2, _esc(ylabel)))
    # 图例的 y 用【累加】而不是 idx*固定行高: 带 ✗ 注解的条目要占两行, 否则注解会和下一条
    # 图例贴在一起(踩过: 注解在 ly+17、下一条在 ly+26, 只差 9px 就糊住了).
    ly = mt + 10
    for idx, bk in enumerate(sorted(series)):
        col = PALETTE[idx % len(PALETTE)]
        pts = series[bk]
        path = " ".join("%.1f,%.1f" % (sx(r["utps_per_user"]), sy(r[ykey])) for r in pts)
        s.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2"/>' % (path, col))
        for r in pts:
            x, y, b = r["utps_per_user"], r[ykey], r["batch_size"]
            cx, cy = sx(x), sy(y)
            _lo, _hi = r.get(ylo), r.get(yhi)
            if isinstance(_lo, (int, float)) and isinstance(_hi, (int, float)) and _hi != _lo:
                s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1" opacity="0.5"/>'
                         % (cx, sy(_lo), cx, sy(_hi), col))
            if "utps_min" in r and r["utps_max"] != r["utps_min"]:
                s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1" opacity="0.5"/>'
                         % (sx(r["utps_min"]), cy, sx(r["utps_max"]), cy, col))
            s.append('<circle cx="%.1f" cy="%.1f" r="4" fill="%s"/>' % (cx, cy, col))
            s.append('<text x="%.1f" y="%.1f" font-size="10" fill="#666">b%d</text>'
                     % (cx + 6, cy - 6, b))
        s.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="3"/>'
                 % (ml + pw + 16, ly, ml + pw + 44, ly, col))
        s.append('<text x="%d" y="%d" font-size="11">%s</text>'
                 % (ml + pw + 50, ly + 4, _esc(bk)))
        drop = dropped.get(bk) or []
        if drop:
            s.append('<text x="%d" y="%d" font-size="10" fill="#b91c1c">✗ c%s (稳态窗口无效)</text>'
                     % (ml + pw + 50, ly + 18, ",".join(str(b) for b in drop)))
            ly += 36
        else:
            ly += 24
    if any(dropped.get(bk) for bk in series):
        s.append('<text x="%d" y="%d" text-anchor="middle" font-size="11" fill="#b91c1c">'
                 '✗ 标注的并发点无有效稳态 decode 窗口(请求排队, offered 超出显存可真正并发的容量), '
                 '已排除出曲线 — 不用全程聚合的假数兜底</text>' % (ml + pw // 2, H - 6))
    s.append('</svg>')
    return "\n".join(s)


# MBU 图的 build_svg 参数 (纵轴换成 MBU%, 无误差棒列名冲突)
MBU_SVG_KW = dict(ykey="mbu_pct", ylo=None, yhi=None,
                  ylabel="MBU (% of 8 TB/s HBM peak)",
                  title="MBU vs UTPS", yfmt="%.0f%%")


def plot_svg(rows, results_dir, dropped=None, osl=None):
    """把每个 seqlen 的 SVG 单独写文件 (零依赖): 主图 + MBU 图."""
    dropped = dropped or {}
    for sl, series in _series_by_seqlen(rows).items():
        for fname, kw in (("tps_curve_s%d.svg" % sl, {}),
                          ("tps_mbu_s%d.svg" % sl, MBU_SVG_KW)):
            svg = build_svg(sl, series, dropped.get(sl), osl, **kw)
            if not svg:
                continue
            out_svg = os.path.join(results_dir, fname)
            with open(out_svg, "w") as f:
                f.write(svg)
            print("SVG 图写出: %s" % out_svg)


def _esc(v):
    return (str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _html_table(headers, rows_of_cells):
    h = "".join("<th>%s</th>" % _esc(x) for x in headers)
    body = "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % _esc(c) for c in r)
                   for r in rows_of_cells)
    return '<table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>' % (h, body)


def _pct(ours, ref):
    """(我们 - cookbook) / cookbook, 供对比表用. 缺参考值就留空."""
    if not (isinstance(ours, (int, float)) and isinstance(ref, (int, float)) and ref):
        return ""
    return "%+.1f%%" % ((ours - ref) / ref * 100.0)


def _curve_of_series(cfgin, sname):
    """由 series 名反查 config.json 里的那条 curve (见 cookbook_refs 的同款反配规则)."""
    m = SERIES_RE.match(str(sname))
    if not m:
        return None
    for c in (cfgin.get("curves") or []):
        nn = c.get("mtp_n", 0)
        mtp = ("mtpN%sA%s" % (nn, c.get("mtp_acc"))) if (isinstance(nn, int) and nn > 0) else "mtpoff"
        if (c.get("backend"), c.get("parallel"), mtp) == \
           (m.group("backend"), m.group("par"), m.group("mtp")):
            return c
    return None


def _roofline_section(avg, dropped, osl, shape, shape_src):
    """MFU / MBU 一节: 口径说明 + MBU 曲线图 + 逐点明细(含字节构成)."""
    have = [r for r in avg if isinstance(r.get("mbu_pct"), (int, float))]
    if not have or shape is None:
        return []
    p = ['<h2>Roofline — MFU / MBU</h2><div class="card">']
    p.append(
        '<p class="sub"><b>MFU</b> = 每卡实际 FLOPs / %.0f TFLOPS (B200 FP8 dense)，'
        '<b>MBU</b> = 每卡实际 HBM 流量 / %.1f TB/s。分母是<b>一个 decode step</b>，'
        '不是一个 token —— 开 MTP 时一步做 <code>1 次 target verify(N+1 个位置) + N 次 '
        'draft forward</code> 却只收 acc 个 token，step 时长 = acc / UTPS。</p>'
        % (roofline.PEAKS[roofline.DEFAULT_DEVICE][0] / 1e12,
           roofline.PEAKS[roofline.DEFAULT_DEVICE][1] / 1e12))
    p.append(
        '<p class="sub">⚠ <b>不要用 sglang 的 <code>--enable-mfu-metrics</code></b>：'
        '那是通用 dense 估算器(MLP 按 <code>intermediate_size</code>、KV 按 '
        '<code>num_kv_heads×head_dim</code>、字节按 bf16)，对本模型的 MoE / MLA / DSA / fp8 '
        '全不成立。这里按真实结构算，三处关键建模：'
        '<b>①MoE 字节随并发变</b> —— 一步真正读进 HBM 的专家数 '
        '<code>E = %d·(1-(1-%d/%d)^(batch·(N+1)))</code>，E 饱和后再加并发字节就不涨了；'
        '<b>②DSA 稀疏注意力</b> —— 主注意力每 query 只读 top-%d 个 KV，indexer 扫全上下文但'
        '只存在于 <b>%d/%d</b> 层(其余层复用上一层 top-k)；'
        '<b>③MTP</b> 如上。</p>'
        % (shape.n_exp, shape.topk, shape.n_exp, shape.dsa_topk,
           shape.idx_layers, shape.n_layers))
    sm = shape.summary()
    p.append('<p class="sub">结构来源 <code>%s</code>：总参 %s B / 权重 %s GB，'
             '每 token 激活 %s B，MoE 专家权重 %s GB，KV %s B/token/层 + indexer K %s B/token/层。</p>'
             % (_esc(shape_src), sm["总参数(B)"], sm["权重字节(GB)"], sm["每token激活参数(B)"],
                sm["MoE专家权重(GB)"], sm["KV字节/token/层"], sm["indexerK字节/token/层"]))
    for sl, series in sorted(_series_by_seqlen(avg).items()):
        svg = build_svg(sl, series, (dropped or {}).get(sl), osl, **MBU_SVG_KW)
        if svg:
            p.append(svg)
    cells = []
    for r in sorted(have, key=lambda x: (x["seqlen"], x["series"], x["batch_size"])):
        cells.append([r["series"], r["seqlen"], r["batch_size"], r.get("step_ms", ""),
                      r["utps_per_user"], r.get("tflops_per_gpu", ""), r.get("mfu_pct", ""),
                      r.get("gb_per_step_gpu", ""), r.get("hbm_tbs_per_gpu", ""),
                      r.get("mbu_pct", ""), r.get("experts_touched", ""),
                      r.get("gb_moe", ""), r.get("gb_wother", ""), r.get("gb_kv", "")])
    p.append(_html_table(
        ["series", "seqlen", "c", "step ms", "UTPS", "TFLOPS/gpu", "MFU%",
         "GB/step/gpu", "TB/s", "MBU%", "命中专家数", "└MoE GB", "└其余权重 GB", "└KV GB"],
        cells))
    p.append(
        '<p class="sub" style="margin-top:10px">读法：decode 是彻底的 memory-bound，'
        '<b>MFU 基本没有信息量，该看 MBU</b>。MBU 通常不单调 —— 低并发时只读到一小部分专家'
        '(见「命中专家数」)且 GEMM 太瘦，高并发时专家全被碰到、字节封顶而 step 时长还在涨'
        '(多出的时间花在 HBM 之外: TP all-reduce / EP a2a / 调度)，所以峰值在中间。<br>'
        '<b>假设</b>：verify 的 N+1 个位置共享一次 KV 读(top-k 高度重叠)；MoE 路由按均匀'
        '假设算命中专家数(真实路由有热点，低并发会略少)；只算 HBM 流量，NVLink 上的 '
        'all-reduce / a2a 不计入。峰值用 spec 值，若按实测 achievable 带宽(~6.5–7 TB/s)'
        '报，MBU 要再乘 ~1.15。无有效稳态窗口的点不算(没有 step 时长可言)。</p>')
    p.append('</div>')
    return p


def write_html(raw, avg, results_dir, config, report_name="report.html",
               shape=None, shape_src=""):
    """自包含 HTML 报告: 测试配置 + 内联SVG曲线图 + 均值表 + 全量原始数据表. 无外部依赖."""
    stem = re.sub(r"\.html?$", "", report_name)
    parts = ['<!doctype html><html lang="zh"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>%s</title><style>' % _esc(stem),
             'body{font-family:system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;',
             'margin:0;padding:32px 24px 64px;color:#16202e;background:#f7f8fa;line-height:1.5}',
             '.wrap{max-width:960px;margin:0 auto}h1{font-size:24px;margin:0 0 4px}',
             'h2{font-size:17px;margin:28px 0 10px;color:#334}',
             '.sub{color:#667;font-size:14px;margin:0 0 20px}',
             '.card{background:#fff;border:1px solid #e4e8ef;border-radius:12px;padding:16px 18px;',
             'box-shadow:0 1px 3px rgba(16,32,54,.06);margin-bottom:20px;overflow-x:auto}',
             'table{border-collapse:collapse;width:100%;font-size:12.5px;',
             'font-variant-numeric:tabular-nums}',
             'th,td{text-align:right;padding:6px 10px;border-bottom:1px solid #eef1f6;white-space:nowrap}',
             'th:first-child,td:first-child{text-align:left}',
             'thead th{color:#556;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.03em}',
             'td{font-family:ui-monospace,Menlo,Consolas,monospace}',
             '.cfg td,.cfg th{text-align:left}svg{max-width:100%;height:auto}',
             'td:last-child{white-space:normal;font-family:inherit}',
             'td.num{font-family:ui-monospace,Menlo,Consolas,monospace}',
             '.cbref{color:#8a94a6}.up{color:#166534}.dn{color:#b91c1c}',
             '.cmdhdr{font-weight:700;color:#334;margin-bottom:6px;text-transform:uppercase;font-size:12px}',
             'pre{white-space:pre-wrap;word-break:break-word;margin:0;font-family:ui-monospace,',
             'Menlo,Consolas,monospace;font-size:12px;line-height:1.45;background:#0f172a;',
             'color:#e2e8f0;padding:12px 14px;border-radius:8px;overflow-x:auto}',
             '</style></head><body><div class="wrap">']
    # 标题里的 backend 列表: 从 run_config.json 的 series 列表推(series = backend-commit-…),
    # 兼容老格式的 backends/framework 字段.
    _series = (config or {}).get("series") or []
    if _series:
        _bks = ",".join(sorted({str(s).split("-")[0] for s in _series}))
    else:
        _bks = (config or {}).get("backends", "") or (config or {}).get("framework", "")
    parts.append('<h1>GLM-5.2 B200 · 每GPU吞吐(STPS) vs 交互速度(UTPS)%s</h1>'
                 % ((' · [%s]' % _esc(_bks)) if _bks else ''))
    parts.append('<p class="sub" style="font-family:ui-monospace,Menlo,Consolas,monospace">%s</p>'
                 % _esc(stem))
    parts.append('<p class="sub">开 MTP(EAGLE, 固定 N/accept_len)、等长请求齐发、'
                 'max-TTFT 稳态窗口(窗内无 prefill)、每点多次取均值。横轴 UTPS(tok/s/user),纵轴 STPS/gpu(tok/s/gpu)。</p>')

    # ---- 测试配置: 【只列与本图相关的字段】 ----
    # 刻意不贴 config.json 全文: 全文已逐字存在同目录 run_config.json 里(可直接 --config 重跑),
    # 贴进报告只会糊成一堵墙。这里只留三块: 图级坐标轴 / 本图用到的 backend 的镜像+commit /
    # 逐曲线的并行·MTP·覆盖项 —— 恰好对应命名的三层.
    cfgin = (config or {}).get("config") or {}
    dflt = cfgin.get("defaults") or {}
    parts.append('<h2>测试配置 (本图相关; config.json 全文见同目录 run_config.json)</h2>'
                 '<div class="card">')
    if config:
        kv = [("时间", config.get("date", "")), ("配置文件", config.get("config_path", "")),
              ("模型 / 量化", "%s / %s" % (config.get("model_name", ""), config.get("quant", ""))),
              ("权重", dflt.get("model", ""))]
        if dflt.get("mtp_draft_path"):
            kv.append(("MTP draft", dflt["mtp_draft_path"]))
        kv += [("ISL / OSL", "%s / %s" % (dflt.get("seqlens", ""), dflt.get("osl", ""))),
               ("并发点(默认)", dflt.get("batches", "")),
               ("每点重复 / GPU 数", "%s / %s" % (dflt.get("reps", ""), dflt.get("gpu_count", ""))),
               ("稳态口径", config.get("metric", ""))]
        _env = dflt.get("env") or {}
        if _env:
            kv.append(("公共 env", ", ".join("%s=%s" % (k, v) for k, v in _env.items())))
        parts.append('<table class="cfg"><tbody>')
        for k, v in kv:
            parts.append("<tr><th>%s</th><td>%s</td></tr>" % (_esc(k), _esc(v)))
        parts.append('</tbody></table>')
        # 本图用到的 backend (由 series 前缀反查, 不列 config 里没启用的)
        brows = []
        for b in sorted({str(x).split("-")[0] for x in _series}):
            bc = (cfgin.get("backends") or {}).get(b) or {}
            brows.append([b, bc.get("image", ""),
                          bc.get("commit") or "(用镜像自带引擎)"])
        if brows:
            parts.append(_html_table(["backend", "镜像", "commit (从源码构建)"], brows))
        # 逐曲线
        crows = []
        for sname in _series:
            cur = _curve_of_series(cfgin, sname) or {}
            nn = cur.get("mtp_n", 0)
            crows.append([
                sname, cur.get("parallel", ""),
                ("N=%s acc=%s" % (nn, cur.get("mtp_acc"))) if (isinstance(nn, int) and nn > 0) else "off",
                cur.get("batches", dflt.get("batches", "")),
                ", ".join("%s=%s" % (k, v) for k, v in (cur.get("env") or {}).items()) or "—",
                cur.get("note", "")])
        if crows:
            parts.append(_html_table(
                ["series (曲线)", "并行", "MTP", "并发点", "env 覆盖", "note"], crows))
    else:
        parts.append('<p class="sub">(无 run_config.json, 配置未记录)</p>')
    parts.append('</div>')

    # ---- 每个 seqlen: 图 + 均值表 (放在启服务命令之前 —— 图才是报告的主角) ----
    _dropped = dropped_by_seqlen(raw, avg)
    _osl = ((config or {}).get("config") or {}).get("defaults", {}).get("osl")
    for sl, series in sorted(_series_by_seqlen(avg).items()):
        parts.append('<h2>seqlen = %d</h2><div class="card">' % sl)
        svg = build_svg(sl, series, _dropped.get(sl), _osl)
        if svg:
            parts.append(svg)
        # 均值表 (MFU/MBU 两列即可, 详细 roofline 分解不再单列一节)
        rows_cells = []
        for bk in sorted(series):
            for r in series[bk]:
                rows_cells.append([bk, r["batch_size"], r["n_reps"],
                                   r["utps_per_user"], "%s–%s" % (r["utps_min"], r["utps_max"]),
                                   r["stps_per_gpu"], "%s–%s" % (r["stps_min"], r["stps_max"]),
                                   r["stps_system_mean"],
                                   r.get("mfu_pct", ""), r.get("mbu_pct", "")])
        parts.append(_html_table(
            ["series", "batch", "n_reps", "UTPS均值", "UTPS范围", "STPS/gpu均值", "STPS范围",
             "系统STPS", "MFU%", "MBU%"],
            rows_cells))
        parts.append('</div>')

    # ---- 各 series 的完整启服务命令 (含 hack env / 固定 accept 的实现), 从 .cmd 读取 ----
    seen_cmd = {}
    for cf in sorted(glob.glob(os.path.join(results_dir, "glm52_*.cmd"))):
        m = re.match(r"glm52_(" + SERIES_PAT + r")_s\d+\.cmd$", os.path.basename(cf))
        bk = m.group(1) if m else os.path.basename(cf)
        if bk in seen_cmd:
            continue                       # 同 series 多 seqlen 命令基本一致, 取一条
        try:
            seen_cmd[bk] = open(cf).read().strip()
        except Exception:
            pass
    if seen_cmd:
        parts.append('<h2>启服务命令 (各 series · 含 hack env · 固定 MTP accept 的实现)</h2>')
        for bk, txt in seen_cmd.items():
            parts.append('<div class="card"><div class="cmdhdr">%s</div><pre>%s</pre></div>'
                         % (_esc(bk), _esc(txt)))

    # ---- 无效测点警示: 稳态 decode 窗口为空 / server 崩. 显式列出, 不塞进曲线, 不用假数掩盖 ----
    bad = [r for r in raw if r.get("status", "ok") != "ok"]
    if bad:
        parts.append('<h2 style="color:#b91c1c">⚠ 无效测点 (%d) — 无有效稳态 decode 窗口, 已排除出曲线</h2>'
                     % len(bad))
        parts.append('<div class="card" style="border-color:#f3c2c2;background:#fef6f6">')
        parts.append(_html_table(
            ["series", "seqlen", "batch", "rep", "status", "completed",
             "cookbook口径 tok/s/user", "cookbook口径 tok/s/gpu", "原因"],
            [[r["series"], r["seqlen"], r["batch_size"], r["rep"], r["status"],
              r.get("completed", ""), r.get("cb_utps", ""), r.get("cb_tps_per_gpu", ""),
              r.get("note", "")] for r in bad]))
        parts.append('<p class="sub" style="margin-top:8px">注: 这里只是<b>稳态 decode 口径</b>无效 ——'
                     'cookbook 口径(含 prefill 与排队, 见下节)对这些点依然有数, 故照样能与 cookbook 比。'
                     '窗口为空有两种成因: ①该并发真的超出显存可并发容量(请求在排队), ②OSL 太短, '
                     '解码时长盖不住 TTFT 铺开 —— 看 completed 是否等于并发数即可区分。</p>')
        parts.append('</div>')

    # ---- 全量原始数据 ----
    parts.append('<h2>全部原始数据 (%d 条, 每次重复一行)</h2><div class="card">' % len(raw))
    rcols = ["series", "seqlen", "batch_size", "rep", "status", "utps_per_user", "stps_per_gpu",
             "stps_system", "cb_ttft_ms", "cb_tpot_ms", "cb_utps", "cb_tps_per_gpu",
             "step_ms", "mfu_pct", "mbu_pct", "gb_per_step_gpu", "experts_touched",
             "completed", "note"]
    parts.append(_html_table(rcols, [[r.get(c, "") for c in rcols] for r in raw]))
    parts.append('</div>')

    parts.append('</div></body></html>')
    out_html = os.path.join(results_dir, report_name)
    with open(out_html, "w") as f:
        f.write("".join(parts))
    print("HTML 报告写出: %s" % out_html)


def plot(rows, results_dir, dropped=None, osl=None):
    # 始终生成零依赖 SVG; matplotlib 存在时额外出 PNG.
    plot_svg(rows, results_dir, dropped, osl)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(无 matplotlib, 仅 SVG: {e})")
        return
    seqlens = sorted({r["seqlen"] for r in rows})
    for sl in seqlens:
        fig, ax = plt.subplots(figsize=(9, 6))
        backends = sorted({r["series"] for r in rows if r["seqlen"] == sl})
        for bk in backends:
            pts = [r for r in rows if r["seqlen"] == sl and r["series"] == bk
                   and r["utps_per_user"] != "" and r["stps_per_gpu"] != ""]
            pts.sort(key=lambda r: r["batch_size"])
            xs = [r["utps_per_user"] for r in pts]
            ys = [r["stps_per_gpu"] for r in pts]
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", label=bk)
            for r in pts:
                ax.annotate(f"b{r['batch_size']}", (r["utps_per_user"], r["stps_per_gpu"]),
                            fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Interactivity  UTPS (tok/s/user)")
        ax.set_ylabel("Per-GPU throughput  STPS (tok/s/gpu)")
        ax.set_title(f"GLM-5.2 FP8 B200  UTPS vs STPS  (seqlen={sl}, MTP)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        out_png = os.path.join(results_dir, f"tps_curve_s{sl}.png")
        fig.tight_layout()
        fig.savefig(out_png, dpi=130)
        plt.close(fig)
        print(f"图写出: {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--gpu-count", type=int, default=8)
    # 报告文件名. bench.sh 传 <backend>_<commit>_<model>_<quant>_i<ISL>o<OSL>.html;
    # 手动调用时不给就退化成 report.html.
    ap.add_argument("--report-name", default="report.html")
    # roofline(MFU/MBU) 的模型结构来源. 不给就用 run_config.json 里的 defaults.model,
    # 再读不到就退到 roofline.py 内置的 GLM-5.2 摘要.
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--device", default=roofline.DEFAULT_DEVICE, choices=sorted(roofline.PEAKS))
    args = ap.parse_args()
    report_name = os.path.basename(args.report_name) or "report.html"
    if not report_name.endswith(".html"):
        report_name += ".html"
    # 测试配置 (bench.sh 写, 含 config.json 全文) —— 先读, roofline 要用里面的权重路径
    config = {}
    cfg_path = os.path.join(args.results_dir, "run_config.json")
    if os.path.exists(cfg_path):
        try:
            config = json.load(open(cfg_path))
        except Exception as e:
            print(f"run_config.json 读取失败: {e}")
    model_dir = args.model_dir or (config.get("config") or {}).get("defaults", {}).get("model")
    shape, shape_src = roofline.load_shape(model_dir)
    print("roofline 结构来源: %s (%s, indexer 层 %d/%d)"
          % (shape_src, args.device, shape.idx_layers, shape.n_layers))
    raw = collect(args.results_dir, args.gpu_count, shape, args.device)
    write_csv(raw, os.path.join(args.results_dir, "tps_raw.csv"))       # 全部原始数据
    avg = aggregate(raw)
    write_csv(avg, os.path.join(args.results_dir, "tps_curve.csv"))     # 每点取均值
    # 图上要标出"测了但无效(ERR)"的并发点, 故 plot 也要看 raw 与 osl
    plot(avg, args.results_dir, dropped_by_seqlen(raw, avg),
         (config.get("config") or {}).get("defaults", {}).get("osl"))
    write_html(raw, avg, args.results_dir, config, report_name,        # 自包含 HTML 报告
               shape, shape_src)


if __name__ == "__main__":
    main()
