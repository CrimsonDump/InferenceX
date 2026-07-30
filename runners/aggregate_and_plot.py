#!/usr/bin/env python3
"""汇总 sweep 产生的结果 JSON -> CSV + SVG 曲线 + 自包含 HTML 报告.

每个 JSON 文件名形如 glm52_<series>_s<seqlen>_c<batch>_r<rep>.json, 由 benchmark_serving.py
写入, 里面【两套口径的原始量都有】: 稳态 decode (steady_*) 与全程 (median_tpot_ms /
total_token_throughput). 曲线用哪套由 run_config.json 的 metric_mode 决定(= config.json 的
defaults.metric_mode, 全局项) —— 所以换口径【不用重跑】, 重跑本脚本即可.
命名分三层, 各层只放本层统一的字段:
  图     = 一份报告/一个目录: <model>_<quant>_i<ISL>o<OSL>_c<最小>-<最大>
  曲线   = <series>: <backend>-<commit>-<parallel>-mtpN<N>A<acc> (如 sglang-fdebc938-tp-mtpN5A3.5)
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
注意力; 不能用 sglang 自带的 --enable-mfu-metrics, 那是通用 dense 估算器)。MFU/MBU
【与 metric_mode 同口径】(step 时长由该口径的 UTPS 反推), 所以曲线上画得出的点就有 MFU/MBU.
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


METRIC_MODES = ("steady-decode", "whole-run")


def metric_desc(metric_mode):
    """报告里"测量口径"那一行的文字 —— 【单一真相源就在这里】.

    刻意不读 run_config.json 的 metric 字段: 那是跑的时候 bench.sh 写下的措辞, 口径定义
    一改就和报告实际算出来的东西对不上(老结果目录里全是旧措辞). 报告必须描述它自己算的量.
    """
    if metric_mode == "whole-run":
        return ("UTPS = interactivity = 1000/TPOT_P50, "
                "STPS/gpu = 每GPU吞吐 = (ISL+OSL)×完成数/总时长/gpu, "
                "包含 prefill")
    return ("UTPS = interactivity = 各请求稳态窗内速率的中位数, "
            "STPS/gpu = 每GPU吞吐 = 窗内 decode token/窗长/gpu, "
            "不包含 prefill (max-TTFT 稳态窗口, 窗内物理上无 prefill forward)")


def collect(results_dir, gpu_count, shape=None, device=roofline.DEFAULT_DEVICE,
            metric_mode="steady-decode"):
    """读所有原始 JSON, 每个文件一行 (含 rep). 保留全部原始数据.

    metric_mode = 曲线两个轴用哪套口径 (全局, 来自 config.json 的 defaults.metric_mode).
    两套口径是【同一批原始数据的两种事后算法】, 每个结果 JSON 里都有, 所以换口径不用重跑:

      steady-decode  刨去 prefill: max-TTFT 稳态窗口(窗内物理上无 prefill forward)。
                     UTPS = 各请求窗内速率的中位数; STPS/gpu = 窗内 decode token/窗长/gpu
                     (【只有 output token】)。窗口空 -> 该点 ERR, 排除出曲线。
      whole-run      不刨 prefill: 口径覆盖请求全程, 与 sglang cookbook 表的两列一致。
                     UTPS = 1000/TPOT_P50 (★p50 不是 mean★, 见下; 被其它请求的 prefill 抢步拖慢);
                     STPS/gpu = (ISL+OSL)×完成数/总时长/gpu (【含 input token】)。
                     不需要窗口, 所以高并发点(稳态窗口空的那些)在这个口径下照样有数。

    shape 给出时额外算 roofline 的 MFU / MBU, 【与 metric_mode 同口径】: step 时长 =
    acc / 该口径的 UTPS, batch 也取该口径对应的并发数。所以曲线上画得出的点就有 MFU/MBU,
    不再因"稳态窗口无效"而留空。
    """
    if metric_mode not in METRIC_MODES:
        raise SystemExit("metric_mode 非法: %r (只认 %s)" % (metric_mode, list(METRIC_MODES)))
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
        # ---- 两套口径都算出来(见 docstring), 再按 metric_mode 决定曲线用哪套 ----
        s_utps = d.get("steady_utps_per_user")          # 稳态 decode: 窗内速率中位数
        s_stps_gpu = d.get("steady_stps_per_gpu")
        s_stps_sys = d.get("steady_stps_system")
        _tpot = d.get("mean_tpot_ms") or 0
        _tpot_p50 = d.get("median_tpot_ms") or 0
        _g = d.get("gpu_count") or gpu_count or 1
        # ★全程口径的 UTPS 按 p50 算, 不按 mean★: sglang cookbook 的 interactivity 列就是
        # 1000/TPOT_P50, 拿 1000/mean_TPOT 去跟它比是不同口径 —— 等长齐发时各请求的 TPOT
        # 铺得开, 尾部把 mean 拖高 => UTPS 系统性偏低. mean 仍在 tpot_ms 列里留作对照.
        w_utps = (1000.0 / _tpot_p50) if _tpot_p50 else None
        w_stps_sys = d.get("total_token_throughput")    # 含 input token (client 就是这么算的)
        w_stps_gpu = (w_stps_sys / _g) if w_stps_sys else None
        completed = d.get("completed", 0) or 0
        if metric_mode == "whole-run":
            utps, stps_gpu, stps_sys = w_utps, w_stps_gpu, w_stps_sys
        else:
            utps, stps_gpu, stps_sys = s_utps, s_stps_gpu, s_stps_sys
        # 无效点【绝不用另一套口径兜底】——两套口径的数差 20~40%(高并发更大), 混着填等于
        # 用假数掩盖问题. 保持缺失 + 显式 status/note, 让无效点暴露在结果里.
        if completed <= 0:
            status, note = "ERR", "completed=0 (server 崩 / 该点无完成请求)"
        elif utps is None or stps_gpu is None:
            if metric_mode == "whole-run":
                status, note = "ERR", "全程口径无数 — 缺 median_tpot_ms / total_token_throughput"
            else:
                status, note = "ERR", "稳态窗口无效 — %s" % (d.get("steady_note") or "无 steady_* 字段")
        else:
            status, note = "ok", ""
        ok = (status == "ok")
        # ---- roofline: MFU / MBU, 【与 metric_mode 同口径】: step 时长由选定口径的 UTPS
        # 反推(step = acc/UTPS), batch 也取该口径对应的并发数. 这样"曲线上有的点就有
        # MFU/MBU" —— 以前恒按稳态算, 于是 whole-run 曲线里稳态窗口空的高并发点画在图上
        # 却没有 MBU, 看报告的人对不上号.
        rf = None
        if utps is not None and shape is not None:
            par, mtp_n, mtp_acc = parse_series(m.group("series"))
            osl = (d.get("output_lens") or [0])[0]
            # 稳态口径: 窗内真正在跑的请求数; 全程口径: 整轮 offered 的并发数.
            _b = (d.get("steady_num_reqs") if metric_mode == "steady-decode"
                  else d.get("max_concurrency"))
            rf = roofline.analyze(
                shape,
                batch=_b or d.get("max_concurrency") or int(m.group("batch")),
                ctx_len=int(m.group("seqlen")) + osl / 2.0,   # 平均上下文 ≈ ISL + OSL/2
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
            # 选定口径(= 曲线的两个轴). 下面 sd_*/wr_* 把【两套口径都】原样留着, 便于核对
            # "这张图用的是哪套"以及直接跟 cookbook 表比(cookbook = wr_*).
            "utps_per_user": round(utps, 3) if ok else "",
            "stps_per_gpu": round(stps_gpu, 3) if ok else "",
            "stps_system": round(stps_sys, 2) if (ok and stps_sys is not None) else "",
            # 口径无关的 client 指标。★mean 与 p50 都列★: cookbook 表的 TTFT / TPOT 两列都是
            # p50(不是 mean), 等长齐发时它们在各请求间铺得很开 -> 拿 mean 去跟它比是不公平的
            # 比法。whole-run 的 UTPS 因此按 tpot_p50 定义(见上), mean 只留作对照。
            "ttft_ms": round(d.get("mean_ttft_ms") or 0, 1),
            "ttft_p50_ms": round(d.get("median_ttft_ms") or 0, 1),
            "tpot_ms": round(_tpot, 3),
            "tpot_p50_ms": round(d.get("median_tpot_ms") or 0, 3),
            "sd_utps": round(s_utps, 3) if s_utps is not None else "",          # 稳态 decode
            "sd_stps_per_gpu": round(s_stps_gpu, 3) if s_stps_gpu is not None else "",
            "wr_utps": round(w_utps, 3) if w_utps is not None else "",           # 全程(含 prefill)
            "wr_stps_per_gpu": round(w_stps_gpu, 3) if w_stps_gpu is not None else "",
            # 全程口径但【只算 output token】的每卡吞吐 (wr_stps_per_gpu 是含 input 的那个)
            "wr_out_stps_per_gpu": round((d.get("output_throughput") or 0) / _g, 1),
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


# 10 色 (tab10 配色, 两两之间在白底上足够分得开) + 4 种线型.
# ★为什么要 10 色不是 6 色★: 原来是 6 色 + PALETTE[idx % 6], 而这类图常有 8~9 条曲线 ——
# 于是第 7、8 条与第 1、2 条【颜色完全相同】, 不是"相近"而是一模一样, 根本没法认。
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
           "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
DASHES = ["", "7,3", "2,3", "9,3,2,3"]      # "" = 实线


def series_style(snames):
    """{series 名: (颜色, 线型)} —— 颜色按【全图 series 全集】分配, 不按某张图里的下标.

    这样同一条曲线在总图和它所属 backend 的分图里【颜色一致】, 两张图能对着看; 若按各图
    内下标分配, 分图里 sglang 的第一条会变成总图里 vllm 那条的颜色, 看图的人必然对错。
    颜色用满 10 种后再叠线型, 所以要到 40 条曲线才会出现重复样式.
    """
    out = {}
    for i, s in enumerate(sorted(snames)):
        out[s] = (PALETTE[i % len(PALETTE)],
                  DASHES[(i // len(PALETTE)) % len(DASHES)])
    return out


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


def build_svg(sl, series, dropped=None, osl=None, metric_mode="steady-decode",
              ykey="stps_per_gpu", ylo="stps_min", yhi="stps_max",
              ylabel="Per-GPU throughput  STPS (tok/s/gpu)",
              title="STPS/gpu vs UTPS", yfmt="%.0f", style=None, subtitle=None):
    """为某个 seqlen 生成 SVG 曲线图字符串 (零依赖, 白底). 横轴恒为 UTPS.

    ykey/ylo/yhi: 纵轴取哪一列(以及误差棒的上下界列, 缺列则不画棒) —— 同一套画法既出
                  主图(STPS/gpu), 也出 MBU 图。
    dropped: {series: [batch,...]} 该 seqlen 下被判 ERR 丢掉的并发点 —— 标进图例并加脚注。
             不标的话曲线会在某个并发"莫名其妙地断掉"(如 tp 到 64 就没了), 看图的人无法
             分辨"没测"与"测了但无效"。
    osl / metric_mode: 进标题与脚注, 让单独看 SVG 时也知道这是什么口径的图。
    style: {series: (颜色, 线型)}, 见 series_style()。画 backend 分图时【必须把总图那份
           传进来】, 否则同一条曲线在两张图里颜色会不一样。不给就按本图的 series 现算。
    subtitle: 主标题下的小字(如"轴自适应"的提醒) —— 单独看 SVG 时也能看到这个前提。
    """
    dropped = dropped or {}
    style = style or series_style(series.keys())
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
    _mtag = ("全程口径, 含 prefill 与 input token" if metric_mode == "whole-run"
             else "稳态 decode 口径, 刨去 prefill")
    s.append('<text x="%d" y="26" font-size="17" font-weight="bold" text-anchor="middle">'
             'GLM-5.2 FP8 B200 — %s (ISL=%d%s; %s)</text>'
             % (W // 2, _esc(title), sl, (", OSL=%s" % osl) if osl else "", _mtag))
    if subtitle:
        s.append('<text x="%d" y="44" font-size="11" text-anchor="middle" fill="#b45309">'
                 '%s</text>' % (W // 2, _esc(subtitle)))
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
    for bk in sorted(series):
        col, dash = style.get(bk, (PALETTE[0], ""))
        _d = (' stroke-dasharray="%s"' % dash) if dash else ""
        pts = series[bk]
        path = " ".join("%.1f,%.1f" % (sx(r["utps_per_user"]), sy(r[ykey])) for r in pts)
        s.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2"%s/>'
                 % (path, col, _d))
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
        s.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="3"%s/>'
                 % (ml + pw + 16, ly, ml + pw + 44, ly, col, _d))
        s.append('<text x="%d" y="%d" font-size="11">%s</text>'
                 % (ml + pw + 50, ly + 4, _esc(bk)))
        drop = dropped.get(bk) or []
        if drop:
            s.append('<text x="%d" y="%d" font-size="10" fill="#b91c1c">✗ c%s (无效)</text>'
                     % (ml + pw + 50, ly + 18, ",".join(str(b) for b in drop)))
            ly += 36
        else:
            ly += 24
    if any(dropped.get(bk) for bk in series):
        _why = ('该点无完成请求(server 崩)' if metric_mode == "whole-run" else
                '无有效稳态 decode 窗口(请求排队, offered 超出显存可真正并发的容量)')
        s.append('<text x="%d" y="%d" text-anchor="middle" font-size="11" fill="#b91c1c">'
                 '✗ 标注的并发点%s, 已排除出曲线 — 不用另一套口径的数兜底</text>'
                 % (ml + pw // 2, H - 6, _why))
    s.append('</svg>')
    return "\n".join(s)


# MBU 图的 build_svg 参数 (纵轴换成 MBU%, 无误差棒列名冲突)
MBU_SVG_KW = dict(ykey="mbu_pct", ylo=None, yhi=None,
                  ylabel="MBU (% of 8 TB/s HBM peak)",
                  title="MBU vs UTPS", yfmt="%.0f%%")

# backend 分图的提醒: 分图两轴各自自适应(否则曲线只占总图范围的一角, 分图就白分了),
# 代价是【两张分图之间不能目视对比】—— 必须写在图上, 不能只指望看图的人自己注意到.
SUBPLOT_NOTE = ("本图只含该 backend 的曲线, 两轴按本图数据自适应 —— "
                "跨 backend 比较请看总图, 勿拿两张分图直接目视对比")


def backends_of(snames):
    """series 名里的 backend 前缀集合 (series = <backend>-<commit>-<parallel>-<mtp>)."""
    return sorted({str(s).split("-")[0] for s in snames})


def per_backend_series(series):
    """[(backend, {该 backend 的 series: 点}), ...]; 只有一个 backend 时返回空.

    单 backend 时分图与总图逐像素相同, 出它只是让报告变长, 故不出 —— 也正是"如果存在
    多个 backend 才加分图"这条要求.
    """
    bks = backends_of(series)
    if len(bks) <= 1:
        return []
    return [(b, {s: rs for s, rs in series.items() if str(s).split("-")[0] == b})
            for b in bks]


def subplot_kw(backend):
    """backend 分图的标题/小字参数."""
    return dict(title="STPS/gpu vs UTPS · 仅 %s" % backend, subtitle=SUBPLOT_NOTE)


def plot_svg(rows, results_dir, dropped=None, osl=None, metric_mode="steady-decode"):
    """把每个 seqlen 的 SVG 单独写文件 (零依赖): 总图 + MBU 图 + 各 backend 分图."""
    dropped = dropped or {}
    for sl, series in _series_by_seqlen(rows).items():
        st = series_style(series.keys())     # ★总图与各分图共用同一套颜色★
        jobs = [("tps_curve_s%d.svg" % sl, series, {}),
                ("tps_mbu_s%d.svg" % sl, series, MBU_SVG_KW)]
        for b, sub in per_backend_series(series):
            jobs.append(("tps_curve_s%d_%s.svg" % (sl, b), sub, subplot_kw(b)))
        for fname, ser, kw in jobs:
            svg = build_svg(sl, ser, dropped.get(sl), osl, metric_mode, style=st, **kw)
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


def series_provenance(results_dir, config):
    """给报告里的每条曲线找回【真正生成它的那次 run 的那条 curve 定义】.

    返回 (合并后的 defaults, 各次 run 的 (date, config_path) 列表, {series 名: 出处}),
    出处 = {"curve": curve 定义, "image": 镜像, "commit": commit, "run": (date, cfg_path)}.

    为什么不能只读本目录的 run_config.json: 一张图常常是多次 bench.sh 分批攒出来的(补曲线、
    换引擎版本、重跑单点), 而每次 run 都把 run_config.json 【整个覆盖】成"本次这几条曲线".
    于是末次只跑 1 条时, 配置区就只剩那 1 条 —— 数据里 8 条曲线配置区却说 1 条, 而且那 1 条
    还可能正是【没跑出来的】那条(本目录实测就是如此). 历次 run_config 都被 bench.sh 归档进
    了 _prev_*/, 全读进来即可.

    为什么按 zip(run 的 series 列表, 该 run 里 enabled 的 curves) 配对, 而不按
    (backend, parallel, mtp) 反查: config 的 curves 是一个【库】, 绝大多数 enabled=false,
    并且同一个 (backend, parallel, mtp) 在库里有好几条(裸的 / cookbook 三档 / A-B 引擎臂 /
    三并行对照…), 旋钮和 note 都不一样。按字段反查只能碰运气取到"第一条", 实测就会把
    sglang 的三条曲线错配成 cookbook A/B 臂的定义(batches、note 全是别的图的)。
    bench.sh 写 run_config 的 series 列表时, 就是照 enabled 的 curves 【同序】展开的
    (见 bench.sh 里 `for cl in "${CURVE_LINES[@]}"`), 所以 zip 是精确配对, 不是启发式。
    """
    cfgs = []
    for p in glob.glob(os.path.join(results_dir, "_prev_*", "run_config.json")):
        try:
            cfgs.append(json.load(open(p)))
        except Exception as e:
            print("跳过无法解析的 %s: %s" % (p, e))
    if config:
        cfgs.append(config)
    # 按 run_config 里的 date 排序 —— 【不能按 _prev_ 目录名排】: 目录名是"归档时刻",
    # 里面那份 run_config 却是它上一轮 run 的, 两者顺序并不一致.
    cfgs.sort(key=lambda c: str(c.get("date") or ""))
    dflt, runs, smap = {}, [], {}
    for c in cfgs:
        inner = c.get("config") or {}
        dflt.update(inner.get("defaults") or {})
        bks = inner.get("backends") or {}
        snames = [str(s) for s in (c.get("series") or [])]
        enabled = [x for x in (inner.get("curves") or []) if x.get("enabled")]
        run = (c.get("date", ""), c.get("config_path", ""))
        runs.append(run)
        if len(snames) != len(enabled):
            # 对不上就不猜(宁可留空也不给错定义). 正常情况下两者恒等长.
            print("⚠ %s: series 数(%d) != enabled curves 数(%d), 该 run 的曲线定义跳过"
                  % (run[1] or run[0], len(snames), len(enabled)))
            continue
        for sname, cur in zip(snames, enabled):
            bc = bks.get(cur.get("backend")) or {}
            # 后跑的 run 覆盖先跑的 —— 同名 series 被重跑时, 最后那次才是数据的来源
            smap[sname] = {"curve": cur, "run": run,
                           "image": cur.get("image") or bc.get("image", ""),
                           "commit": cur.get("commit") or bc.get("commit")}
    return dflt, runs, smap


def write_html(raw, avg, results_dir, config, report_name="report.html",
               metric_mode="steady-decode"):
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
    # ★series 列表从【实际收到的数据】里推, 不读 run_config.json 的 series 字段★:
    # 后者只是"末次 bench.sh 跑了哪几条", 分批攒图时会漏掉之前几批的曲线(见
    # merged_run_configs 的说明). 数据才是这张报告里到底有哪些曲线的唯一真相.
    _series = sorted({r["series"] for r in raw})
    # 标题就一行, 不带 backend 列表 / 图名 / 口径说明 —— 那些各有归宿: backend 与镜像在
    # 下面的 backend 表, 图名是浏览器标题(<title>=报告文件名), 口径在"测试配置"的一行里.
    parts.append('<h1>GLM-5.2 B200 STPS vs UTPS</h1>')

    # ---- 测试配置: 【只列与本图相关的字段】 ----
    # 刻意不贴 config.json 全文: 全文已逐字存在同目录 run_config.json 里(可直接 --config 重跑),
    # 贴进报告只会糊成一堵墙。这里只留三块: 图级坐标轴 / 本图用到的 backend 的镜像+commit /
    # 逐曲线的并行·MTP·覆盖项 —— 恰好对应命名的三层.
    # 逐曲线的定义按【生成它的那次 run】反查(见 series_provenance): 分批攒出来的图才能把全部
    # backend / 曲线列全, 且不会错配到 config 库里同 (backend,parallel,mtp) 的别的定义上.
    dflt, _runs, _prov = series_provenance(results_dir, config)
    parts.append('<h2>测试配置</h2><div class="card">')
    if config:
        # 分批攒图时"时间/配置文件"不是一个值: 如实列出【每一次 run】, 而不是只显示末次
        # (末次常常只是补一条曲线, 拿它当整张图的时间与配置会误导).
        if len(_runs) > 1:
            _when = "%s ~ %s (%d 次 run 攒成)" % (_runs[0][0], _runs[-1][0], len(_runs))
            _cfgp = "; ".join(sorted({p for _, p in _runs if p}))
        else:
            _when, _cfgp = config.get("date", ""), config.get("config_path", "")
        kv = [("时间", _when), ("配置文件", _cfgp),
              ("模型 / 量化", "%s / %s" % (config.get("model_name", ""), config.get("quant", ""))),
              ("权重", dflt.get("model", ""))]
        if dflt.get("mtp_draft_path"):
            kv.append(("MTP draft", dflt["mtp_draft_path"]))
        kv += [("ISL / OSL", "%s / %s" % (dflt.get("seqlens", ""), dflt.get("osl", ""))),
               ("每点重复 / GPU 数", "%s / %s" % (dflt.get("reps", ""), dflt.get("gpu_count", ""))),
               # num_warmups 是一等字段; 老结果目录里它还在 env 里, 那就照实显示那个值
               ("client warmup 请求数",
                dflt.get("num_warmups", (dflt.get("env") or {}).get("NUM_WARMUPS", 0))),
               # 口径文字的单一真相源是 metric_desc(), 不是 run_config.json 的 metric 字段
               ("测量口径", "%s — %s" % (metric_mode, metric_desc(metric_mode)))]
        # env 只该剩调试/逃生阀(常用旋钮都有一等字段了); 非空时必须显示 —— 它会改变结果
        _env = dflt.get("env") or {}
        if _env:
            kv.append(("调试 env (逃生阀)", ", ".join("%s=%s" % (k, v) for k, v in _env.items())))
        parts.append('<table class="cfg"><tbody>')
        for k, v in kv:
            parts.append("<tr><th>%s</th><td>%s</td></tr>" % (_esc(k), _esc(v)))
        parts.append('</tbody></table>')
        # ★这里【不再列逐曲线的表】★(镜像/commit 表、并行/MTP/并发点 表都删了):
        # 全是冗余 —— commit 与并行/MTP 本来就编码在 series 名里(名字就是由这些字段推导的),
        # 镜像与全部 server 旋钮逐字出现在下面每条曲线的 .cmd 段, 并发点直接看数据表。
        # 冗余不只是啰嗦: 它们靠 provenance 反查, 查不到时会印出与事实相反的值
        # (实测把 mtpN5A5 的曲线印成 "MTP off"), 多一处表就多一处会说错话的地方。
    else:
        parts.append('<p class="sub">(无 run_config.json, 配置未记录)</p>')
    parts.append('</div>')

    # ---- NOTES.md 【不进报告】 ----
    # 结果目录里的 NOTES.md 是给人读的长文(动辄上百行, 含表格/链接/引用), 贴进 HTML 只会
    # 把图和数据表挤到看不见。它就在同目录, 要看直接打开; 报告只负责数据与复现命令。
    # ---- 每个 seqlen: 图 + 均值表 (放在启服务命令之前 —— 图才是报告的主角) ----
    _dropped = dropped_by_seqlen(raw, avg)
    _osl = ((config or {}).get("config") or {}).get("defaults", {}).get("osl")
    for sl, series in sorted(_series_by_seqlen(avg).items()):
        _st = series_style(series.keys())   # ★总图与下面各 backend 分图共用同一套颜色★
        _subs = per_backend_series(series)
        parts.append('<h2>seqlen = %d%s</h2><div class="card">'
                     % (sl, " · 总图 (全部 backend)" if _subs else ""))
        svg = build_svg(sl, series, _dropped.get(sl), _osl, metric_mode, style=_st)
        if svg:
            parts.append(svg)
        if _subs:
            parts.append('<p class="sub" style="margin:6px 0 0">曲线多时总图不好认色 —— '
                         '下面每个 backend 另有一张只含它自己曲线的分图, '
                         '<b>同一条曲线在总图与分图里颜色一致</b>, 可对着看。</p>')
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
        parts.append('<p class="sub" style="margin:10px 0 0">MFU/MBU <b>与上面的测量口径同口径</b>'
                     '(step 时长 = acc / 该口径的 UTPS, batch 取该口径的并发数; 开 MTP 时一步 = '
                     '1 次 target verify(N+1 个位置) + N 次 draft, 只收 acc 个 token)。'
                     'decode 是彻底 memory-bound —— MFU 基本没有信息量, 该看 MBU。'
                     '逐点字节构成见 tps_raw.csv 的 gb_* 列。</p>')
        parts.append('</div>')
        # ---- 多 backend 时: 每个 backend 一张只含它自己曲线的分图 ----
        # 只出图不再重复出表 —— 数值上面那张总表已经按 series 排好序、天然按 backend 成组,
        # 再抄一遍只会让报告更长(而"图太挤"本来就是要解决的问题).
        for b, sub in _subs:
            svg = build_svg(sl, sub, _dropped.get(sl), _osl, metric_mode,
                            style=_st, **subplot_kw(b))
            if not svg:
                continue
            parts.append('<h2>seqlen = %d · 仅 %s (%d 条曲线)</h2><div class="card">%s</div>'
                         % (sl, _esc(b), len(sub), svg))

    # ---- 各 series 的完整启服务命令 (含 hack env / 固定 accept 的实现), 从 .cmd 读取 ----
    # 也要翻 _prev_*/: 分批攒图时, 先跑那几条曲线的 .cmd 被 bench.sh 归档进 _prev_ 了, 只
    # 扫本目录会让报告里 9 条曲线只剩末次那几条的启服务命令(实测只剩 4 条). 就近优先:
    # 本目录 > _prev_ 从新到旧(目录名是归档时刻, 降序 = 从新到旧).
    _cmd_dirs = [results_dir] + sorted(
        glob.glob(os.path.join(results_dir, "_prev_*")), reverse=True)
    seen_cmd = {}
    for d in _cmd_dirs:
        for cf in sorted(glob.glob(os.path.join(d, "glm52_*.cmd"))):
            m = re.match(r"glm52_(" + SERIES_PAT + r")_s\d+\.cmd$", os.path.basename(cf))
            bk = m.group(1) if m else os.path.basename(cf)
            if bk in seen_cmd:
                continue                   # 同 series 多 seqlen / 多次 run: 取最近那条
            try:
                seen_cmd[bk] = open(cf).read().strip()
            except Exception:
                pass
    # 只列本报告里真有数据的曲线, 顺序与上面两张配置表一致(否则会混进别的图残留的 .cmd,
    # 以及本次没跑出任何点的曲线 —— 后者的情况请写进 NOTES.md 说明, 而不是留一条没数的命令)
    _cmds = [(s, seen_cmd[s]) for s in _series if s in seen_cmd]
    if _cmds:
        parts.append('<h2>启服务命令 (各 series · 含 hack env · 固定 MTP accept 的实现)</h2>')
        for bk, txt in _cmds:
            parts.append('<div class="card"><div class="cmdhdr">%s</div><pre>%s</pre></div>'
                         % (_esc(bk), _esc(txt)))
        _miss = [s for s in _series if s not in seen_cmd]
        if _miss:
            parts.append('<p class="sub">⚠ 找不到 .cmd 的曲线: %s</p>'
                         % _esc(", ".join(_miss)))

    # ---- 无效测点警示: 稳态 decode 窗口为空 / server 崩. 显式列出, 不塞进曲线, 不用假数掩盖 ----
    bad = [r for r in raw if r.get("status", "ok") != "ok"]
    if bad:
        parts.append('<h2 style="color:#b91c1c">⚠ 无效测点 (%d) — 已排除出曲线</h2>' % len(bad))
        parts.append('<div class="card" style="border-color:#f3c2c2;background:#fef6f6">')
        parts.append(_html_table(
            ["series", "seqlen", "batch", "rep", "status", "completed",
             "全程口径 tok/s/user", "全程口径 tok/s/gpu", "原因"],
            [[r["series"], r["seqlen"], r["batch_size"], r["rep"], r["status"],
              r.get("completed", ""), r.get("wr_utps", ""), r.get("wr_stps_per_gpu", ""),
              r.get("note", "")] for r in bad]))
        if metric_mode == "whole-run":
            parts.append('<p class="sub" style="margin-top:8px">注: 全程口径不需要稳态窗口, '
                         '所以这里只会剩<b>真正跑崩</b>的点(completed=0)。</p>')
        else:
            parts.append('<p class="sub" style="margin-top:8px">注: 这里只是<b>稳态 decode 口径</b>无效'
                         '—— 上面两列(全程口径, 含 prefill 与排队)对这些点依然有数, 换 '
                         '<code>metric_mode=whole-run</code> 重跑聚合就能把它们画进曲线。'
                         '稳态窗口为空有两种成因: ①该并发真的超出显存可并发容量(请求在排队, '
                         '去 .serverlog 里 grep <code>retract</code> 确认), ②OSL 太短, 解码时长'
                         '盖不住 TTFT 铺开(此时 completed 等于并发数)。</p>')
        parts.append('</div>')

    # ---- 全量原始数据 ----
    parts.append('<h2>全部原始数据 (%d 条, 每次重复一行)</h2><div class="card">' % len(raw))
    # utps_per_user/stps_per_gpu = 选定口径(上面画的那套); 另一套口径的两列也列出来便于核对
    # (metric_mode 选中的那套与它逐格相等 —— 这正是"这张图用的是哪套口径"的自证).
    _other = (["sd_utps", "sd_stps_per_gpu"] if metric_mode == "whole-run"
              else ["wr_utps", "wr_stps_per_gpu"])
    rcols = (["series", "seqlen", "batch_size", "rep", "status", "utps_per_user", "stps_per_gpu",
              "stps_system", "ttft_ms", "ttft_p50_ms", "tpot_ms", "tpot_p50_ms"] + _other +
             ["step_ms", "mfu_pct", "mbu_pct", "gb_per_step_gpu", "experts_touched",
              "completed", "note"])
    parts.append(_html_table(rcols, [[r.get(c, "") for c in rcols] for r in raw]))
    parts.append('</div>')

    parts.append('</div></body></html>')
    out_html = os.path.join(results_dir, report_name)
    with open(out_html, "w") as f:
        f.write("".join(parts))
    print("HTML 报告写出: %s" % out_html)


def plot(rows, results_dir, dropped=None, osl=None, metric_mode="steady-decode"):
    """出零依赖 SVG (主图 + MBU 图)。

    刻意【只出 SVG】: 以前还有一条 matplotlib PNG 分支, 但宿主机上从来没装 matplotlib
    (每次都打印"无 matplotlib"), 而且它画的图不带口径标注 —— 口径可配之后, 一张没标
    口径的图是会误导人的。要 PNG 就拿 SVG 去转。
    """
    plot_svg(rows, results_dir, dropped, osl, metric_mode)


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
    _dflt = (config.get("config") or {}).get("defaults", {})
    model_dir = args.model_dir or _dflt.get("model")
    shape, shape_src = roofline.load_shape(model_dir)
    print("roofline 结构来源: %s (%s, indexer 层 %d/%d)"
          % (shape_src, args.device, shape.idx_layers, shape.n_layers))
    # 测量口径【只从 run_config.json 读】(= config.json 的 defaults.metric_mode), 刻意不给
    # 命令行开关: 同一项两处可写会让"这张图到底是哪套口径"无从判断.
    metric_mode = _dflt.get("metric_mode") or config.get("metric_mode") or "steady-decode"
    print("测量口径: %s — %s" % (metric_mode, metric_desc(metric_mode)))
    raw = collect(args.results_dir, args.gpu_count, shape, args.device, metric_mode)
    write_csv(raw, os.path.join(args.results_dir, "tps_raw.csv"))       # 全部原始数据
    avg = aggregate(raw)
    write_csv(avg, os.path.join(args.results_dir, "tps_curve.csv"))     # 每点取均值
    # 图上要标出"测了但无效(ERR)"的并发点, 故 plot 也要看 raw 与 osl
    plot(avg, args.results_dir, dropped_by_seqlen(raw, avg), _dflt.get("osl"), metric_mode)
    write_html(raw, avg, args.results_dir, config, report_name, metric_mode)


if __name__ == "__main__":
    main()
