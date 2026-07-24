#!/usr/bin/env bash
# ============================================================================
# bench.sh — GLM-5.2 FP8 多 backend (sglang / vllm) 的 每GPU吞吐(STPS) vs 交互速度(UTPS)
#            对比曲线. 合并自 bench_sglang.sh + bench_vllm.sh, 一张图画多条曲线.
# ============================================================================
# 在 node071 (8×B200, docker) 上运行. --backend 逗号列表(默认 vllm,sglang), 逐 backend
# 逐 seqlen 起对应容器跑同一套口径(max-TTFT 稳态窗口 + 固定 MTP N/accept_len +
# ), 结果全写同一 outdir,
# 最后统一聚合出【含所有 backend 曲线 + 全部原始数据】的 report.html.
#
# 配置项(命令行):
#   --backend LIST 逗号分隔, 默认 "vllm,sglang"
#   -m PATH        模型本地路径(容器内可见), 默认 /mnt/ramweights/glm-5p2-fp8/model
#   --parallel M   并行预设(vllm 语义), M ∈ {tp, dpa-tp, dep, tep}, 默认 dep:
#                    tp     = 纯TP:      attn TP,  MoE TP  (vllm tp=G,dp=1,ep0)
#                    dpa-tp = attn纯DP + MoE纯TP  (vllm tp=1,dp=G,ep0)  ← 本次
#                    dep    = attn DP  + MoE EP   (vllm tp=1,dp=G,ep1)
#                    tep    = attn TP  + MoE EP   (vllm tp=G,dp=1,ep1)
#                  脚本按 vllm(world=tp*dp)为准, 自动翻译成 sglang(卡数=tp, dp/ep 叠加):
#                    sglang_tp=PTP*PDP(=G), sglang_dp=PDP(>1则开dp-attn), sglang_ep=(G if EP else 1)
#   --tp/--dp/--ep 覆盖预设(vllm 语义: tp=切头, dp=attn数据并行, ep=0/1 专家并行开关)
#   -n / -a        MTP 的 N / accept_len (默认 5 / 6)
#   -s LIST        seqlen 列表 (默认 8192)
#   -c LIST        batch 列表 (默认 1,4,16,64,256)
#   -o N           OSL (默认 1024)
#   -r N           每点重复 (默认 3)
#   -g N           GPU 数 (默认 8)
#   --outdir DIR   输出目录 (默认 /tilert/xbj/tps_results_cmp)
#   --repo DIR     InferenceX 仓库路径 (默认 = 本 runner 的上一级)
#   --sglang-image / --vllm-image  覆盖镜像
#
# 产物(在 --outdir): report.html(多曲线) / tps_curve.csv / tps_raw.csv /
#   tps_curve_s<seqlen>.svg / run_config.json / glm52_<backend>_s<sl>_c<b>_r<rep>.json
#
# 运行实例见文件末尾 USAGE.
# ============================================================================
set -uo pipefail

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${MODEL:-/mnt/ramweights/glm-5p2-fp8/model}"
BACKENDS="vllm,sglang"
# 并行按 vllm 语义为准(world = tp*dp); --parallel 预设展开成 canonical(PTP,PDP,PEP),
# 再由本脚本逐 backend 翻译成各自 flag(见 loop 里的翻译). --tp/--dp/--ep 可覆盖预设.
PARALLEL="dep"      # tp | dpa-tp | dep | tep  (见 usage)
TP=""; DP=""; EP=""; MTP_N=5; MTP_ACC=6
SEQLENS="8192"; BATCHES="1,4,16,64,256"; OSL=1024; REPS=1; GPU_COUNT=8
OUTDIR="/tilert/xbj/tps_results_cmp"
SGLANG_IMAGE="zcr.zhipuai-infra.cn/infra/sglang:v0.5.10-prerelease-9844dd05"
VLLM_IMAGE="vllm/vllm-openai:glm52"
REPO="$(cd "$SELF_DIR/.." && pwd)"
AGG="$SELF_DIR/aggregate_and_plot.py"

usage() { sed -n '2,48p' "$0"; exit "${1:-0}"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKENDS="$2"; shift 2;;
    --parallel) PARALLEL="$2"; shift 2;;
    -m) MODEL="$2"; shift 2;;
    --tp) TP="$2"; shift 2;;
    --dp) DP="$2"; shift 2;;
    --ep) EP="$2"; shift 2;;
    -n) MTP_N="$2"; shift 2;;
    -a) MTP_ACC="$2"; shift 2;;
    -s) SEQLENS="$2"; shift 2;;
    -c) BATCHES="$2"; shift 2;;
    -o) OSL="$2"; shift 2;;
    -r) REPS="$2"; shift 2;;
    -g) GPU_COUNT="$2"; shift 2;;
    --outdir) OUTDIR="$2"; shift 2;;
    --repo) REPO="$2"; shift 2;;
    --sglang-image) SGLANG_IMAGE="$2"; shift 2;;
    --vllm-image) VLLM_IMAGE="$2"; shift 2;;
    -h|--help) usage 0;;
    *) echo "未知参数: $1" >&2; usage 1;;
  esac
done

[[ -z "$MODEL" ]] && { echo "错误: 必须用 -m 指定模型路径" >&2; usage 1; }
# 并行预设 -> canonical (vllm 语义): PTP=切头, PDP=attn 数据并行, PEP=0/1 专家并行开关
case "$PARALLEL" in
  tp)      PTP=$GPU_COUNT; PDP=1;          PEP=0;;
  dpa-tp)  PTP=1;          PDP=$GPU_COUNT; PEP=0;;
  dep)     PTP=1;          PDP=$GPU_COUNT; PEP=1;;
  tep)     PTP=$GPU_COUNT; PDP=1;          PEP=1;;
  *) echo "未知 --parallel: $PARALLEL (应为 tp|dpa-tp|dep|tep)" >&2; usage 1;;
esac
[[ -n "$TP" ]] && PTP="$TP"; [[ -n "$DP" ]] && PDP="$DP"; [[ -n "$EP" ]] && PEP="$EP"
if [[ $((PTP * PDP)) -ne "$GPU_COUNT" ]]; then
  echo "⚠️  注意: vllm 语义 tp($PTP)*dp($PDP)=$((PTP*PDP)) != GPU 数($GPU_COUNT)" >&2
fi
GPU_DEV="$(seq -s, 0 $((GPU_COUNT - 1)))"
[[ -f "$AGG" ]] || { echo "错误: 找不到 $AGG" >&2; exit 1; }
command -v docker >/dev/null || { echo "错误: 需要 docker (请在 node071 上运行)" >&2; exit 1; }

# GLM-5.2 硬约束预检: FP8 704G 需 ≥8 卡 (无论怎么切, 总卡数不足会 OOM)
if [[ "$GPU_COUNT" -lt 8 ]]; then
  echo "⚠️  GPU 数=$GPU_COUNT < 8, GLM-5.2 FP8(704G) 每卡权重≈$((704 / GPU_COUNT))G > B200 可用(~178G) -> 会 OOM." >&2
fi

mkdir -p "$OUTDIR"
# 解析成绝对路径: 容器内脚本用 RESULT_DIR 直接把结果写到这里(在 /tilert 挂载下可写),
# 否则相对路径(如 --outdir .)在容器内会指向 /workspace(仓库目录)造成混淆.
OUTDIR="$(readlink -f "$OUTDIR")"
# 隔离本次运行: 旧结果移入 _prev_<时间戳>/ (一次, 对全部 backend)
if ls "$OUTDIR"/glm52_*_c*.json >/dev/null 2>&1 || ls "$OUTDIR"/tps_curve.csv >/dev/null 2>&1; then
  PREV="$OUTDIR/_prev_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$PREV"
  mv -f "$OUTDIR"/glm52_*_c*.json "$OUTDIR"/tps_curve.csv "$OUTDIR"/tps_raw.csv \
        "$OUTDIR"/tps_curve_s*.svg "$OUTDIR"/report.html "$PREV/" 2>/dev/null || true
  echo "注意: OUTDIR 已有旧结果, 已移入 $PREV/ 隔离本次运行(原始数据保留)"
fi

echo "=========================================================="
echo " GLM-5.2 FP8 多 backend benchmark: [$BACKENDS]"
echo " MODEL=$MODEL   MTP N=$MTP_N accept=$MTP_ACC"
echo " 并行(vllm语义 --parallel=$PARALLEL): tp=$PTP dp=$PDP ep=$PEP  (world=$((PTP*PDP)))"
echo " SEQLENS=$SEQLENS  BATCHES=$BATCHES  OSL=$OSL  REPS=$REPS  GPU=$GPU_COUNT"
echo " 对齐: cudagraph-max-bs=max(batch)  "
echo " 输出目录: $OUTDIR"
echo "=========================================================="

cat > "$OUTDIR/run_config.json" <<JSON
{
  "date": "$(date '+%Y-%m-%d %H:%M:%S')",
  "model": "$MODEL",
  "backends": "$BACKENDS",
  "sglang_image": "$SGLANG_IMAGE",
  "vllm_image": "$VLLM_IMAGE",
  "gpu_count": $GPU_COUNT,
  "parallel_preset": "$PARALLEL",
  "isl_seqlens": "$SEQLENS", "osl": $OSL, "batches": "$BATCHES", "reps": $REPS,
  "metric": "steady-state max-TTFT window (decode-only, 刨 prefill/ramp); UTPS=各请求窗内速率中位数, STPS/gpu=窗内总token/窗长/gpu",
  "note": "并行/MTP/chunk/kernel 等启服务参数见下方各 backend 完整命令(含 hack env)"
}
JSON

# 逐 backend × 逐 seqlen 起容器
IFS=',' read -ra BK_ARR <<< "$BACKENDS"
IFS=',' read -ra SL_ARR <<< "$SEQLENS"
for backend in "${BK_ARR[@]}"; do
  backend="${backend// /}"; [[ -z "$backend" ]] && continue
  case "$backend" in
    sglang)
      IMAGE="$SGLANG_IMAGE"
      BENCH_REL="benchmarks/single_node/fixed_seq_len/glm5.2_fp8_b200_sglang_mtp.sh"
      # 翻译 vllm-canonical -> sglang(卡数=tp, dp/ep 叠加): tp=ptp*pdp, dp=pdp(>1开dp-attn), ep=全卡 if EP else 1
      BK_TP=$((PTP * PDP)); BK_DP=$PDP
      if [[ "$PEP" == "1" ]]; then BK_EP=$((PTP * PDP)); else BK_EP=1; fi
      # sglang 专属 docker 参数
      BK_DEVICE=( --device /dev/infiniband )
      BK_ENV=( -e NCCL_SHM_DISABLE=1 -e GLM_XGRAMMAR_BACKEND_CACHE_MAX_MB=76800
               -e GLM_GRAMMAR_OBJECT_CACHE_MAX_COUNT=1920 )
      BK_MOUNT=( -v /mnt/ramweights/jitp/cache:/root/.cache
                 -v /mnt/ramweights/jitp/triton:/root/.triton
                 -v /mnt/ramweights/jitp/tilelang:/root/.tilelang
                 -v /mnt/ramweights/jitp/deep_ep:/root/.deep_ep )
      ;;
    vllm)
      IMAGE="$VLLM_IMAGE"
      BENCH_REL="benchmarks/single_node/fixed_seq_len/glm5.2_fp8_b200_vllm_mtp.sh"
      # vllm 直接用 canonical: tp=切头, dp=attn 数据并行, ep=0/1 专家并行开关
      BK_TP=$PTP; BK_DP=$PDP; BK_EP=$PEP
      # vllm 专属: 关对称内存死锁两处 + 关 inductor max-autotune + 跳过 deepgemm warmup
      BK_DEVICE=()
      BK_ENV=( -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 -e VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=0
               -e VLLM_DEEP_GEMM_WARMUP=skip -e PYTHONFAULTHANDLER=1 )
      BK_MOUNT=( -v /mnt/ramweights/jitp/cache:/root/.cache
                 -v /mnt/ramweights/jitp/triton:/root/.triton )
      ;;
    *) echo "⚠️  未知 backend: $backend, 跳过" >&2; continue;;
  esac
  [[ -f "$REPO/$BENCH_REL" ]] || { echo "错误: 找不到 $REPO/$BENCH_REL" >&2; exit 1; }

  for seqlen in "${SL_ARR[@]}"; do
    seqlen="${seqlen// /}"; [[ -z "$seqlen" ]] && continue
    prefix="glm52_${backend}_s${seqlen}"
    CNAME="ix-glm52-${backend}-s${seqlen}"
    echo ">>> [$backend] seqlen=$seqlen  (容器 $CNAME)"
    docker rm -f "$CNAME" >/dev/null 2>&1 || true
    docker run --rm --name "$CNAME" \
      --gpus "\"device=$GPU_DEV\"" --network host --ipc host --shm-size 64g \
      --cap-add CAP_IPC_LOCK "${BK_DEVICE[@]}" \
      -e PYTHONUNBUFFERED=1 -e PYTHONNOUSERSITE=1 \
      -e TORCH_CUDA_ARCH_LIST=10.0 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e PORT=30000 \
      "${BK_ENV[@]}" \
      -e MODEL="$MODEL" -e TP="$BK_TP" -e DP="$BK_DP" -e EP="$BK_EP" \
      -e CONC="$BATCHES" -e ISL="$seqlen" -e OSL="$OSL" -e RANDOM_RANGE_RATIO=1.0 \
      -e REPS="$REPS" -e RESULT_FILENAME="$prefix" \
      -e MTP="${MTP:-1}" -e SPEC_NUM_STEPS="$MTP_N" -e MTP_ACC="$MTP_ACC" \
      -e MAX_RETRY="${MAX_RETRY:-2}" -e MEM_FRAC="${MEM_FRAC:-}" \
      -e RESULT_DIR="$OUTDIR" \
      -v "$REPO":/workspace -v /tilert:/tilert -v /mnt/ramweights:/mnt/ramweights \
      "${BK_MOUNT[@]}" \
      -w /workspace --entrypoint /bin/bash \
      "$IMAGE" "$BENCH_REL" \
      || echo "⚠️  [$backend] seqlen=$seqlen 容器异常退出, 已产出的原始数据仍会被收集"
    # 结果/.cmd/.serverlog 由容器内脚本经 RESULT_DIR 直接写到 $OUTDIR, 无需再 mv.
    # 兜底: 万一有旧版脚本把产物写到了仓库根, 顺手收一下(正常情况下匹配为空).
    mv -f "$REPO/${prefix}"_c*.json "$REPO/${prefix}.cmd" "$OUTDIR/" 2>/dev/null || true
  done
done

echo ">>> 汇总 CSV + 画图 + HTML 报告 (所有 backend 合并)"
python3 "$AGG" --results-dir "$OUTDIR" --gpu-count "$GPU_COUNT"
echo "=========================================================="
echo " 完成. 产物在 $OUTDIR :"
echo "   - report.html          (多 backend 曲线 + 全部原始数据)"
echo "   - tps_curve.csv / tps_raw.csv / tps_curve_s*.svg / run_config.json"
echo "=========================================================="

# ============================================================================
# USAGE (在 node071 上):
#   RUN=/tilert/xbj/tps_bench/InferenceX/runners/bench.sh
#   # 默认 vllm+sglang, seqlen1024 全扫, 出双曲线图
#   bash $RUN -m /mnt/ramweights/glm-5p2-fp8/model
#   # 只跑一个 backend / 快速试跑
#   bash $RUN --backend vllm -c 1,8,64 -r 1
# ============================================================================
