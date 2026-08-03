# GLM-5.2 decode 吞吐基准 (bench.sh)

在 8×B200 单节点上测 **每GPU吞吐 (STPS, tok/s/gpu) vs 交互速度 (UTPS, tok/s/user)** 曲线，
支持 sglang / vllm / tokenspeed 三个 backend 同图对比。

> 上游 InferenceX 的原始 README 见 [README.inferencex.md](./README.inferencex.md)。
> 本文件只讲**本仓库这套 `runners/bench.sh` 编排**：怎么跑、怎么配、怎么看输出。
> 集群/权重/踩坑/当前进展见仓库根的 `CLAUDE.md` 与 `pitfalls/`。

---

## 1. 用法

`bench.sh` 在**容器外**跑（宿主机需 8×B200 + docker），它自己起容器、起 server、压测、聚合出图。

```bash
RUN=/tilert/xbj/tps_bench/InferenceX/runners/bench.sh

bash $RUN                    # 跑 runners/config.json 里所有 enabled 的曲线
bash $RUN --dry-run          # 只打印解析出的计划(曲线/命名/镜像/源码树), 不起容器
bash $RUN --config my.json   # 换一份配置
bash $RUN --help             # 打印脚本头部的完整说明
```

**只有 `--config` / `--dry-run` / `--help` 三个开关，没有任何配置类命令行参数。**
模型、并行、MTP、并发、ISL/OSL、重复次数、口径、输出目录……全部只能在 config.json 里改 ——
同一项两处可写（命令行 + 配置文件）会让"这次到底跑的是什么"无从判断，也让 `run_config.json` 记录失真。

临时试跑就**复制一份配置**（配置文件本身就是配置的单位）：

```bash
cp runners/config.json /tilert/xbj/smoke.json   # 改小 batches / reps, 或给个单独的 outdir
bash $RUN --config /tilert/xbj/smoke.json
```

一次 run 的流程：解析配置 → 算图名/目录 → 按 (backend, commit) 准备源码树或探测镜像版本 →
**逐条 enabled 曲线各起一次容器**（容器内起 server 一次，在该曲线的并发列表上循环压测，
崩了自动重启重试）→ 全部结果写进**同一个 outdir** → 调 `aggregate_and_plot.py` 聚合成一份
含全部曲线的自包含 HTML。


### 只换口径 / 只重新出图，不重跑

两套测量口径是同一批原始数据的**事后算法**，client 无条件把两边的原始量都写进了每个点的
结果 JSON。所以改完 `metric_mode` 只需重跑聚合：

```bash
python3 runners/aggregate_and_plot.py --results-dir <结果目录> --gpu-count 8 \
                                      --report-name <图名>.html
```

⚠️ 它读的是**结果目录里的 `run_config.json`**（刻意不给命令行口径开关），所以要么把新口径同步进
那份文件，要么直接用新 config 重跑 bench。

---

## 2. 怎么配置输入

**唯一入口 = `runners/config.json`**。字段逐条说明就写在该文件自己的 `_readme` 里（改字段时同步改那儿）；
只有 `defaults` / `backends` / `curves` 三个键会影响运行，`_` 开头的键 loader 一律忽略。
**字段名写错 → 直接报错**（不会静默忽略）；**一等字段与 `env` 里同名变量双写 → 也直接报错**。

三层结构，按字段各自的"作用域"放：

```jsonc
{
  "defaults": { "model": …, "seqlens": [8192], "batches": [1,16,64,256],
                "osl": 1024, "reps": 1, "gpu_count": 8,
                "metric_mode": "whole-run", "num_warmups": 64, "outroot": … },
  "backends": { "sglang": { "image": "lmsysorg/sglang:latest", "repo": …, "commit": "fdebc938" },
                "vllm":   { "image": "vllm/vllm-openai:v0.26.0", "repo": …, "commit": null } },
  "curves":   [ { "backend": "sglang", "parallel": "tp", "mtp_n": 5, "mtp_acc": 3.5,
                  "mem_frac": 0.85, "enabled": true, "note": "…" } ]
}
```

### `defaults` — 整次 run 共享的坐标轴与口径

| 字段 | 含义 |
| --- | --- |
| `model` / `mtp_draft_path` | 权重目录（容器内可见）。**图名的 model/quant 从 `model` 的目录名拆出来**；指本地 tmpfs 比共享盘快 3 倍（只影响起服务耗时） |
| `quantization` | `--quantization`。不给 = 引擎按权重的 `quantization_config` 自动识别（fp8 就这样）；NVFP4 必须显式给 `modelopt_fp4` |
| `seqlens` / `osl` | ISL 列表 / 输出长度，都进图名 |
| `batches` | 并发列表 = **曲线的横轴采样点**，`curves[].batches` 可覆盖 |
| `reps` | 每点重复次数（全部原始 JSON 都留）。=1 时图上无误差棒 |
| `gpu_count` | GPU 数 = 并行预设的 world size，也是每卡吞吐的除数 |
| **`metric_mode`** | **全局测量口径**，见下面 §2.1 |
| `num_warmups` | client 正式测点前的 warmup 请求数（0=关；cookbook 口径是 64） |
| `arch_base` | `architectures` 旁路目录落点（须可写；权重在只读区时必须给） |
| `outroot` / `outdir` | 输出根目录 / 显式指定输出目录（让两次 run 并存） |
| `src_cache` / `fetch_proxy` | build from source 的源码树缓存根 / 取 tarball 的代理 |
| `env` | 追加给所有曲线容器的环境变量 —— **只放调试/逃生阀**（`AR_FUSION` / `ARCH_RENAME` / `SERVER_WARMUP` / `FLUSH_CACHE` 之类）。常用量都有一等字段，别往这里塞 |

#### 2.1 `metric_mode`：口径是配置项，两个取值不可比

同一批原始数据能算出两套差 20~40% 的数，**跨口径的数绝不能混比**，所以它是**全局项**
（一张图的两个轴不允许一半稳态一半全程）：

| | `steady-decode` | `whole-run`（= sglang cookbook 表的两列） |
| --- | --- | --- |
| prefill | **刨**：只取 max-TTFT 稳态窗口（窗内物理上无 prefill forward） | **不刨**：覆盖请求全程 |
| UTPS | 窗内各请求速率的**中位数** | `1000 / TPOT_p50` |
| STPS/gpu | 窗内 decode token/窗长/gpu（**只有 output token**） | `(ISL+OSL)×完成数/总时长/gpu`（**含 input token**） |
| 高并发点 | 窗口可能为空 → 该点判 `ERR` 排除出曲线 | 不需要窗口，**照样有数** |

`c=1` 时两者等价，并发越高越分叉。**MFU/MBU 不随本字段变**（恒按稳态 decode 的 step 时长算），
所以稳态窗口无效的点这两列留空，即使该点在 `whole-run` 下有 UTPS/STPS。

### `backends` — 每个 backend 的**构建来源默认值**

| 字段 | 含义 |
| --- | --- |
| `image` | 容器镜像。`commit` 非 null 时它只是**运行时基座**（CUDA/torch/flashinfer/deepep/sgl-kernel/deep_gemm），自带引擎会被源码那份 editable 覆盖 |
| `repo` | build from source 的仓库（从 codeload 取该 commit 的 **tarball**，不是 clone） |
| `commit` | `null` = 不从源码构建、用镜像自带版本（series 名里的 commit 位由镜像探测填）；非 null = build from source |

这里的值只是默认，单条 curve 可各自覆盖 → 同一张图里放 A/B 两个版本。
构建按 (backend, commit) **去重**：同一次 run 里相同组合只准备一次源码树，用到它的曲线共用。

### `curves` — 图上每条线（配置的单位就是曲线）

必填：`backend` / `parallel` / `mtp_n` / `mtp_acc`。

| 字段 | 含义 |
| --- | --- |
| `parallel` | `tp` 纯TP ｜ `dep` attn DP + MoE EP(deepep) ｜ `tep` attn TP + MoE EP ｜ `dpa-tp` attn DP + MoE TP。以 vllm 语义 world=tp×dp 为准，由 bench.sh 翻译成各 backend 的 flag |
| `mtp_n` | = `--speculative-num-steps`。`<=0` = 不开 spec（series 记 `mtpoff`）。draft-tokens 不用填（sglang 默认 = steps+1） |
| `mtp_acc` | **固定接受长度**（sglang 用 `SGLANG_SIMULATE_ACC_LEN` + `match-expected`，接受小数）。`<=0` = 走自然接受。★**UTPS 与它成正比**★，所以不同 `acc` 的曲线之间不能比引擎快慢 |
| `commit` / `image` | 覆盖 `backends.<bk>.*`。★换 commit 常常得连镜像一起换★（sgl-kernel / deep_gemm 只在镜像里、又被源码硬性依赖） |
| `enabled` | `false` 则跳过（定义留着备查 —— config 里那一堆 disabled 曲线就是历史台账） |
| `batches` / `num_prompts` | 覆盖 `defaults.batches` / 逐点的 client `--num-prompts`（不给 = `num_prompts==并发` = **等长齐发一轮**，本项目一直的口径） |
| server 旋钮 | `mem_frac` / `chunked_prefill_size` / `max_running_requests` / `context_length` / `cuda_graph_max_bs` / `kv_cache_dtype` / `all2all_backend` / `attention_backend` / `cudagraph_mode` / `moe_backend` / `max_num_batched_tokens` / `quantization` —— **都支持 `"off"` = 不下发该 flag** |
| `env` | 该曲线容器的额外环境变量（覆盖 `defaults.env` 同名键） |
| `note` | ★唯一会被读取的说明字段★：进 `run_config.json` 与报告的曲线表。只写一行标签 |

两个容易踩的点：
- **`max_running_requests` 正常一律不要给** —— 不给时 bench.sh 用"本曲线最大并发"，即
  max-running == 并发、**不排队**；给一个小于并发的值就是人为制造排队。
- `kv_cache_dtype` / `all2all_backend` 的**取值是各 backend 原生的名字、不通用**
  （vllm `fp8_e4m3` / tokenspeed `fp8` / sglang 让 DSA 自动配置选），所以跨 backend 同图时
  别写在 `defaults` 级，要写就逐曲线写。

### 命名三层：图 / 曲线 / 点

唯一规则：**一个名字里只允许出现"该层内统一"的字段**。

| 层 | 名字 | 含哪些字段 |
| --- | --- | --- |
| **图**（= 一份 html + 一个目录） | `<model>_<quant>_i<ISL>o<OSL>_c<最小>-<最大>.html` | 全图统一的：model、quant、ISL/OSL、并发**范围** |
| **曲线**（= series，进结果文件名与图例） | `<backend>-<commit>-<parallel>-mtpN<N>A<acc>` | 逐曲线变的：backend、commit、parallel、MTP N/acc |
| **点** | 结果文件名的 `_c<b>_r<rep>` 段 | concurrency（横轴）、重复序号 |

**曲线没有手写 id**，全由字段推导。**backend / commit / parallel / MTP 不进图名** ——
一张图里可以同时有多 backend、多 commit 的曲线，所以"换 commit 做 A/B"的正确做法是
**加一条 curve**（同 parallel、`commit` 填不同值），而不是另起一张图。

**目录** = `defaults.outdir`，否则 `<outroot>/<图名去掉 .html>`（目录与 html 同名）。
推论：同 model/quant/ISL/OSL/并发范围 的两次 run 会落进同一目录，**旧结果自动移进
`_prev_<时间戳>/`**（原始数据、`.cmd`、`.serverlog` 一起搬走，证据不丢）。要两份并存就给 `defaults.outdir`。

---

## 3. 怎么看输出

产物全在 outdir。**先看 HTML，需要细查再看 CSV，怀疑测点不对就去 `.serverlog`。**

| 产物 | 内容 |
| --- | --- |
| **`<图名>.html`** | ★主产物★ 自包含报告（无外部依赖，浏览器直接打开）：测试配置 + 内联 SVG 曲线图 + 均值表 + 逐曲线启服务命令 + **无效测点清单** + 全量原始数据表 |
| `tps_curve.csv` | 每个 (series, seqlen, batch) 取均值的**有效点**，附 min/max/n |
| `tps_raw.csv` | **全部原始点**（每次重复一行），含 `status` / 末列 `note`（无效原因）/ **两套口径并列**的列 |
| `tps_curve_s<ISL>.svg` | 曲线图：横轴 UTPS、纵轴 STPS/gpu（另有 `tps_curve_s<ISL>_<backend>.svg` 单 backend 版） |
| `tps_mbu_s<ISL>.svg` | MBU vs UTPS |
| `glm52_<series>_s<ISL>.cmd` | **实际启服务命令** + 需 export 的环境变量 + 引擎版本（commit / `sglang.__file__`）。逐曲线一份 |
| `glm52_<series>_s<ISL>.serverlog` | **逐曲线**的 server 日志 —— 事后查 `#running-req` / `token usage` / `retract` / `#cached-token` 全靠它 |
| `glm52_<series>_s<ISL>.gpu_metrics.csv` | GPU 采样 |
| `glm52_<series>_s<ISL>_c<b>_r<rep>.json` | 逐点原始结果（两套口径的原始量都在里面） |
| `run_config.json` | 图级配置 + `metric_mode` + series 列表 + **config 全文**（`json.load` 后重新序列化，注释与键序已丢） |
| **`config.json`** | 输入 config 的**逐字副本** —— 复现就用它：`bash bench.sh --config <结果目录>/config.json` |

### CSV 关键列

`tps_raw.csv` / `tps_curve.csv` 里：

- **`utps_per_user` / `stps_per_gpu`** = 图的两个轴 = **当前 `metric_mode` 选中的那套**。
- **`sd_*`（稳态）与 `wr_*`（全程）两套并列** —— 选中的那套与上面两列逐格相等，
  这就是"这张图是哪套口径"的**自证**。跟 cookbook 表比要用 `wr_*`。
- 口径无关的 client 指标：`ttft_ms`(mean) / **`ttft_p50_ms`** / `tpot_ms`(mean) / **`tpot_p50_ms`**
  —— ★跟 cookbook 比要用 **p50** 那两列★（那张表的 TTFT/TPOT 列是 p50；等长齐发时 mean 会系统性不公平）。
- roofline：`mfu_pct` / `mbu_pct` / `step_ms` / `gb_per_step_gpu` / `experts_touched` / 字节构成
  （`gb_moe` / `gb_wother` / `gb_kv`）—— 见 `runners/roofline.py` 与 `pitfalls/roofline-mbu.md`。
- **`status` / `note`**：`ok` 或 `ERR` + 无效原因。`completed`：**必须等于该点的并发数**，不等就先怀疑
  `ulimit -n` / 请求静默失败。

### 无效点（`ERR`）怎么读

`ERR` 的点**被排除出曲线**（图上单独标出），**绝不用另一套口径的数兜底掩盖**。三种成因：

| `note` | 含义 | 处置 |
| --- | --- | --- |
| `completed=0` | server 崩 / 该点无完成请求 | 看 `.serverlog` |
| 稳态窗口无效（只在 `steady-decode` 下） | ① 真塞不下（KV 池装不下，`.serverlog` 里 grep **`retract`**）② **OSL 太短**（请求都跑完了，只是解码时长盖不住 prefill 铺开） | ① 该并发超容量，ERR 就是结论；② 加长 OSL **另出一张图**（OSL 是图级字段），或整张图换 `whole-run` |
| 全程口径无数 | 缺 `median_tpot_ms` / `total_token_throughput` | 看 client 输出 |

窗口非空的充要条件是 `OSL × TPOT > max_i(ttft) - min_i(ttft)`。
⚠️ **反直觉**：引擎越快越容易触发 —— TPOT 压下去后解码时长变短，而 prefill 铺开不变。
所以"新引擎下某个高并发点变 ERR"**不一定是回退**。

---

## 4. 流水线（一次 sweep 经过哪些文件）

| # | 文件 | 职责 |
| --- | --- | --- |
| 1 | `runners/config.json` | 唯一配置入口（字段说明在文件内的 `_readme`） |
| 2 | `runners/bench.sh` | **容器外编排**：解析+校验配置、算图名/目录、准备源码树或探测镜像 commit、逐曲线起容器、跑完调聚合 |
| 3 | `benchmarks/single_node/fixed_seq_len/glm5.2_{fp8,fp4}_b200_{sglang,vllm,tokenspeed}_mtp.sh` | **容器内**：单节点合一（一个 engine 同时 prefill+decode，client 同容器）。起 server 一次、在并发列表上循环、崩了重启重试；写 `.cmd`。sglang 那份还负责 `architectures` 旁路、源码 editable 安装 + **"生效版本"硬校验**、flashinfer cubin 预热 |
| 4 | `benchmarks/benchmark_lib.sh` | 共享库：`run_benchmark_serving` / `wait_for_server_ready` / `start_gpu_monitor` / `emit_env` |
| 5 | `utils/bench_serving/benchmark_serving.py` | 压测 client：等长齐发 + `--ignore-eos` + `--flush-cache`（warmup 之后、正式测点之前），**无条件**把两套口径的原始量都写进结果 JSON |
| 6 | `runners/aggregate_and_plot.py` | 聚合出图：按 (series, seqlen, batch) 取均值，出 CSV/SVG/HTML。**口径只从 `run_config.json` 读** |
| 7 | `runners/roofline.py` | MFU/MBU 模型（从权重 `config.json` 推结构）。也能单独跑：`python3 roofline.py <结果目录>` |

辅助：`runners/probe_server_args.py <权重目录>` —— **不起服务**、秒级打印引擎自动推出来的
`kv_cache_dtype` / `page_size` / `attention_backend` / spec 等字段（换镜像或换 commit 后都该复跑）。
