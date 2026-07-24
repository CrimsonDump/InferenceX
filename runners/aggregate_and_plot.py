#!/usr/bin/env python3
"""汇总 sweep 产生的结果 JSON -> CSV + SVG 曲线 + 自包含 HTML 报告.

每个 JSON 文件名形如 glm52_<backend>_s<seqlen>_c<batch>_r<rep>.json, 内含稳态指标
(steady_utps_per_user / steady_stps_per_gpu, 由 benchmark_serving.py 写入).

输出(在 <results-dir>):
  report.html            自包含报告: 测试配置 + 内联SVG曲线图 + 均值表 + 全量原始数据 (浏览器直接打开)
  tps_raw.csv            全部原始数据 (每次重复一行)
  tps_curve.csv          每点多次取均值
  tps_curve_s<seqlen>.svg 每个 seqlen 一张曲线图
横轴 = interactivity (tok/s/user), 纵轴 = 每GPU吞吐 (tok/s/gpu).
配置区来自 <results-dir>/run_config.json (由 bench_sglang.sh 写入).
"""
import argparse
import csv
import glob
import json
import os
import re

# 文件名: glm52_<backend>_s<seqlen>_c<batch>[_r<rep>].json  (rep 可选, 向后兼容)
FNAME_RE = re.compile(
    r"glm52_(?P<backend>[a-zA-Z0-9]+)_s(?P<seqlen>\d+)_c(?P<batch>\d+)(?:_r(?P<rep>\d+))?\.json$")


def collect(results_dir, gpu_count):
    """读所有原始 JSON, 每个文件一行 (含 rep). 保留全部原始数据."""
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
            status = "ERR: completed=0 (server 崩 / 该点无完成请求)"
        elif utps is None or stps_gpu is None:
            status = "ERR: 稳态窗口无效 — %s" % (d.get("steady_note") or "无 steady_* 字段")
        else:
            status = "ok"
        ok = (status == "ok")
        rows.append({
            "backend": m.group("backend"),
            "seqlen": int(m.group("seqlen")),
            "batch_size": int(m.group("batch")),
            "rep": int(m.group("rep")) if m.group("rep") else 1,
            "status": status,
            "utps_per_user": round(utps, 3) if ok else "",
            "stps_per_gpu": round(stps_gpu, 3) if ok else "",
            "stps_system": round(stps_sys, 2) if (ok and stps_sys is not None) else "",
            "full_output_throughput": round(d.get("output_throughput", 0), 2),
            "full_mean_tpot_ms": round(d.get("mean_tpot_ms", 0), 3),
            "steady_median_itl_ms": round(d.get("steady_median_itl_ms", 0), 4),
            "steady_window_tokens_per_req_median": d.get("steady_window_tokens_per_req_median", ""),
            "completed": d.get("completed", ""),
        })
    rows.sort(key=lambda r: (r["seqlen"], r["backend"], r["batch_size"], r["rep"]))
    return rows


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def aggregate(raw):
    """按 (backend, seqlen, batch) 对多次重复取均值, 附 min/max/n."""
    groups = {}
    for r in raw:
        if r["utps_per_user"] == "" or r["stps_per_gpu"] == "":
            continue
        groups.setdefault((r["backend"], r["seqlen"], r["batch_size"]), []).append(r)
    out = []
    for (bk, sl, b), rs in groups.items():
        u = [x["utps_per_user"] for x in rs]
        s = [x["stps_per_gpu"] for x in rs]
        out.append({
            "backend": bk, "seqlen": sl, "batch_size": b, "n_reps": len(rs),
            "utps_per_user": round(_mean(u), 3),
            "stps_per_gpu": round(_mean(s), 3),
            "utps_min": round(min(u), 3), "utps_max": round(max(u), 3),
            "stps_min": round(min(s), 3), "stps_max": round(max(s), 3),
            "stps_system_mean": round(_mean([x["stps_system"] for x in rs]) or 0, 1),
        })
    out.sort(key=lambda r: (r["seqlen"], r["backend"], r["batch_size"]))
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
    """{seqlen: {backend: [row, ...按batch排序]}} — row 为均值行(含 min/max)."""
    out = {}
    for r in rows:
        if r["utps_per_user"] == "" or r["stps_per_gpu"] == "":
            continue
        out.setdefault(r["seqlen"], {}).setdefault(r["backend"], []).append(r)
    for sl in out:
        for bk in out[sl]:
            out[sl][bk].sort(key=lambda r: r["batch_size"])
    return out


def build_svg(sl, series):
    """为某个 seqlen 生成 SVG 曲线图字符串 (零依赖, 白底). 横轴 UTPS, 纵轴 STPS/gpu."""
    W, H = 860, 620
    ml, mr, mt, mb = 80, 180, 60, 70   # 边距 (右侧留图例)
    pw, ph = W - ml - mr, H - mt - mb
    allx = [r["utps_per_user"] for bk in series for r in series[bk]]
    ally = [r.get("stps_max", r["stps_per_gpu"]) for bk in series for r in series[bk]]
    if not allx:
        return None
    xmin, xmax = 0, max(allx) * 1.08
    ymin, ymax = 0, max(ally) * 1.10

    def sx(x): return ml + (x - xmin) / (xmax - xmin) * pw
    def sy(y): return mt + ph - (y - ymin) / (ymax - ymin) * ph

    s = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
         'font-family="sans-serif" font-size="13">' % (W, H)]
    s.append('<rect width="%d" height="%d" fill="white"/>' % (W, H))
    s.append('<text x="%d" y="28" font-size="17" font-weight="bold" text-anchor="middle">'
             'GLM-5.2 FP8 B200 — UTPS vs STPS (seqlen=%d, MTP)</text>' % (W // 2, sl))
    for i in range(6):
        gx = ml + pw * i / 5
        xv = xmin + (xmax - xmin) * i / 5
        s.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#eee"/>' % (gx, mt, gx, mt + ph))
        s.append('<text x="%.1f" y="%d" text-anchor="middle" fill="#555">%.0f</text>' % (gx, mt + ph + 20, xv))
        gy = mt + ph * i / 5
        yv = ymax - (ymax - ymin) * i / 5
        s.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#eee"/>' % (ml, gy, ml + pw, gy))
        s.append('<text x="%d" y="%.1f" text-anchor="end" fill="#555">%.0f</text>' % (ml - 8, gy + 4, yv))
    s.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333"/>' % (ml, mt + ph, ml + pw, mt + ph))
    s.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333"/>' % (ml, mt, ml, mt + ph))
    s.append('<text x="%d" y="%d" text-anchor="middle">Interactivity  UTPS (tok/s/user)</text>'
             % (ml + pw // 2, H - 20))
    s.append('<text transform="translate(22,%d) rotate(-90)" text-anchor="middle">'
             'Per-GPU throughput  STPS (tok/s/gpu)</text>' % (mt + ph // 2))
    for idx, bk in enumerate(sorted(series)):
        col = PALETTE[idx % len(PALETTE)]
        pts = series[bk]
        path = " ".join("%.1f,%.1f" % (sx(r["utps_per_user"]), sy(r["stps_per_gpu"])) for r in pts)
        s.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2"/>' % (path, col))
        for r in pts:
            x, y, b = r["utps_per_user"], r["stps_per_gpu"], r["batch_size"]
            cx, cy = sx(x), sy(y)
            if "stps_min" in r and r["stps_max"] != r["stps_min"]:
                s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1" opacity="0.5"/>'
                         % (cx, sy(r["stps_min"]), cx, sy(r["stps_max"]), col))
            if "utps_min" in r and r["utps_max"] != r["utps_min"]:
                s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1" opacity="0.5"/>'
                         % (sx(r["utps_min"]), cy, sx(r["utps_max"]), cy, col))
            s.append('<circle cx="%.1f" cy="%.1f" r="4" fill="%s"/>' % (cx, cy, col))
            s.append('<text x="%.1f" y="%.1f" font-size="10" fill="#666">b%d</text>'
                     % (cx + 6, cy - 6, b))
        ly = mt + 10 + idx * 22
        s.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="3"/>'
                 % (ml + pw + 20, ly, ml + pw + 50, ly, col))
        s.append('<text x="%d" y="%d">%s</text>' % (ml + pw + 56, ly + 4, bk))
    s.append('</svg>')
    return "\n".join(s)


def plot_svg(rows, results_dir):
    """把每个 seqlen 的 SVG 单独写文件 (零依赖)."""
    for sl, series in _series_by_seqlen(rows).items():
        svg = build_svg(sl, series)
        if not svg:
            continue
        out_svg = os.path.join(results_dir, "tps_curve_s%d.svg" % sl)
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


def write_html(raw, avg, results_dir, config):
    """自包含 HTML 报告: 测试配置 + 内联SVG曲线图 + 均值表 + 全量原始数据表. 无外部依赖."""
    parts = ['<!doctype html><html lang="zh"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>GLM-5.2 TPS 对比报告</title><style>',
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
             '.cmdhdr{font-weight:700;color:#334;margin-bottom:6px;text-transform:uppercase;font-size:12px}',
             'pre{white-space:pre-wrap;word-break:break-word;margin:0;font-family:ui-monospace,',
             'Menlo,Consolas,monospace;font-size:12px;line-height:1.45;background:#0f172a;',
             'color:#e2e8f0;padding:12px 14px;border-radius:8px;overflow-x:auto}',
             '</style></head><body><div class="wrap">']
    _bks = (config or {}).get("backends", "") or (config or {}).get("framework", "")
    parts.append('<h1>GLM-5.2 FP8 B200 · 每GPU吞吐(STPS) vs 交互速度(UTPS)%s</h1>'
                 % ((' · [%s]' % _esc(_bks)) if _bks else ''))
    parts.append('<p class="sub">开 MTP(EAGLE, 固定 N/accept_len)、等长请求齐发、'
                 'max-TTFT 稳态窗口(窗内无 prefill)、每点多次取均值。横轴 UTPS(tok/s/user),纵轴 STPS/gpu(tok/s/gpu)。</p>')

    # ---- 测试配置 ----
    parts.append('<h2>测试配置 (启服务命令之外的信息)</h2><div class="card"><table class="cfg"><tbody>')
    if config:
        for k, v in config.items():
            parts.append("<tr><th>%s</th><td>%s</td></tr>" % (_esc(k), _esc(v)))
    else:
        parts.append('<tr><td>(无 run_config.json, 配置未记录)</td></tr>')
    parts.append('</tbody></table></div>')

    # ---- 各 backend 的完整启服务命令 (含 hack env / 固定 accept 的实现), 从 .cmd 读取 ----
    seen_cmd = {}
    for cf in sorted(glob.glob(os.path.join(results_dir, "glm52_*.cmd"))):
        m = re.match(r"glm52_([a-zA-Z0-9]+)_s\d+\.cmd$", os.path.basename(cf))
        bk = m.group(1) if m else os.path.basename(cf)
        if bk in seen_cmd:
            continue                       # 同 backend 多 seqlen 命令基本一致, 取一条
        try:
            seen_cmd[bk] = open(cf).read().strip()
        except Exception:
            pass
    if seen_cmd:
        parts.append('<h2>启服务命令 (各 backend · 含 hack env · 固定 MTP accept 的实现)</h2>')
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
            ["backend", "seqlen", "batch", "rep", "原因 (status)", "completed",
             "参考·全程吞吐(tok/s)", "参考·全程TPOT(ms)"],
            [[r["backend"], r["seqlen"], r["batch_size"], r["rep"], r["status"],
              r.get("completed", ""), r.get("full_output_throughput", ""),
              r.get("full_mean_tpot_ms", "")] for r in bad]))
        parts.append('<p class="sub" style="margin-top:8px">注:"全程吞吐/TPOT"含 prefill 与排队,'
                     '<b>不是</b>稳态 decode 口径,仅作参考。此类点通常因请求排队(TTFT 铺开 ≫ OSL '
                     '解码时长)使稳态窗口为空 —— 说明该并发已超出显存可真正并发的容量。</p>')
        parts.append('</div>')

    # ---- 每个 seqlen: 图 + 均值表 ----
    for sl, series in sorted(_series_by_seqlen(avg).items()):
        parts.append('<h2>seqlen = %d</h2><div class="card">' % sl)
        svg = build_svg(sl, series)
        if svg:
            parts.append(svg)
        # 均值表
        rows_cells = []
        for bk in sorted(series):
            for r in series[bk]:
                rows_cells.append([bk, r["batch_size"], r["n_reps"],
                                   r["utps_per_user"], "%s–%s" % (r["utps_min"], r["utps_max"]),
                                   r["stps_per_gpu"], "%s–%s" % (r["stps_min"], r["stps_max"]),
                                   r["stps_system_mean"]])
        parts.append(_html_table(
            ["backend", "batch", "n_reps", "UTPS均值", "UTPS范围", "STPS/gpu均值", "STPS范围", "系统STPS"],
            rows_cells))
        parts.append('</div>')

    # ---- 全量原始数据 ----
    parts.append('<h2>全部原始数据 (%d 条, 每次重复一行)</h2><div class="card">' % len(raw))
    rcols = ["backend", "seqlen", "batch_size", "rep", "status", "utps_per_user", "stps_per_gpu",
             "stps_system", "full_output_throughput", "full_mean_tpot_ms", "completed"]
    parts.append(_html_table(rcols, [[r.get(c, "") for c in rcols] for r in raw]))
    parts.append('</div>')

    parts.append('</div></body></html>')
    out_html = os.path.join(results_dir, "report.html")
    with open(out_html, "w") as f:
        f.write("".join(parts))
    print("HTML 报告写出: %s" % out_html)


def plot(rows, results_dir):
    # 始终生成零依赖 SVG; matplotlib 存在时额外出 PNG.
    plot_svg(rows, results_dir)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(无 matplotlib, 仅 SVG: {e})")
        return
    seqlens = sorted({r["seqlen"] for r in rows})
    for sl in seqlens:
        fig, ax = plt.subplots(figsize=(8, 6))
        backends = sorted({r["backend"] for r in rows if r["seqlen"] == sl})
        for bk in backends:
            pts = [r for r in rows if r["seqlen"] == sl and r["backend"] == bk
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
    args = ap.parse_args()
    raw = collect(args.results_dir, args.gpu_count)
    write_csv(raw, os.path.join(args.results_dir, "tps_raw.csv"))       # 全部原始数据
    avg = aggregate(raw)
    write_csv(avg, os.path.join(args.results_dir, "tps_curve.csv"))     # 每点取均值
    plot(avg, args.results_dir)                                         # 单独的 .svg (+PNG if matplotlib)
    # 测试配置 (由 bench_sglang.sh 写入)
    config = {}
    cfg_path = os.path.join(args.results_dir, "run_config.json")
    if os.path.exists(cfg_path):
        try:
            config = json.load(open(cfg_path))
        except Exception as e:
            print(f"run_config.json 读取失败: {e}")
    write_html(raw, avg, args.results_dir, config)                     # 自包含 HTML 报告


if __name__ == "__main__":
    main()
