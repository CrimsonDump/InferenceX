#!/usr/bin/env python3
"""探针: 看 sglang 对某个权重目录【自动推导出什么 server args】(不起服务, 秒级).

用途: 判断 DSA 自动配置(kv fp8 / page_size 64 / dsa|nsa attention backend)是否真的生效
—— 本地 checkpoint 的 architectures 名曾把整条 DSA 路径挡掉(见 CLAUDE.md「DSA 路径」一节)。
换镜像 / 换源码 commit 后都应复跑, 因为白名单与字段名在版本间会变。

用法(容器内):
  docker run --rm --gpus "device=0" -v /tilert:/tilert -v /mnt/ramweights:/mnt/ramweights \
    --entrypoint python3 <image> /workspace/runners/probe_server_args.py <权重目录> [更多 flag...]
(要探"源码那份"就先 pip install -e <src>/python, 再跑本脚本; 脚本会打印实际生效的 sglang 路径)
"""
import sys

import sglang
from sglang.srt.server_args import prepare_server_args

# 字段名在版本间变过(如 attention_backend 的取值 nsa -> dsa; nsa_*_backend 后来没了),
# 所以用 getattr 探而不是直接取, 缺了就打 "-", 免得探针自己先崩.
FIELDS = [
    "kv_cache_dtype",
    "page_size",
    "attention_backend",
    "prefill_attention_backend",
    "decode_attention_backend",
    "nsa_prefill_backend",
    "nsa_decode_backend",
    "moe_runner_backend",
    "moe_a2a_backend",
    "speculative_algorithm",
    "speculative_draft_model_path",
    "disable_shared_experts_fusion",
]


def main(argv):
    if not argv:
        sys.exit(__doc__)
    args = prepare_server_args(
        ["--model-path", argv[0], "--trust-remote-code", "--tensor-parallel-size", "8"]
        + list(argv[1:])
    )
    print("PROBE sglang         =", sglang.__version__, sglang.__file__)
    print("PROBE model_path     =", argv[0])
    for f in FIELDS:
        v = getattr(args, f, "<无此字段>")
        print("PROBE %-24s = %s" % (f, v))


if __name__ == "__main__":
    main(sys.argv[1:])
