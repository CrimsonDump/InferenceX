# SPDX-License-Identifier: Apache-2.0
r"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    vLLM OpenAI API server
    vllm serve <your_model> \
        --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_tgi_server.sh <your_model> <max_batch_total_tokens>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name sharegpt \
        --dataset-path <path to dataset> \
        --request-rate <request_rate> \ # By default <request_rate> is inf
        --num-prompts <num_prompts> # By default <num_prompts> is 1000

    when using tgi backend, add
        --endpoint /generate_stream
    to the end of the command above.
"""
import argparse
import asyncio
import base64
import contextlib
import gc
import io
import json
import os
import random
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Any, AsyncGenerator, Collection, Dict, List, Optional, Tuple

import numpy as np
from backend_request_func import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                  RequestFuncOutput)
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

try:
    from backend_request_func import get_tokenizer
except ImportError:
    # vLLM moved get_tokenizer from vllm.transformers_utils.tokenizer to
    # vllm.tokenizers; the old path was a deprecation alias in v0.20 and
    # was removed by v0.23. Try the new location first, fall back to old.
    try:
        from vllm.tokenizers import get_tokenizer
    except ImportError:
        from vllm.transformers_utils.tokenizer import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

from benchmark_utils import convert_to_pytorch_benchmark_format
from encoding_dsv4 import encode_messages as dsv4_encode_messages

MILLISECONDS_TO_SECONDS_CONVERSION = 1000


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: List[Tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: List[Tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: List[Tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: List[Tuple[float, float]]


# --- Multiprocessing helpers for sample_random_requests ---
_worker_tokenizer = None


def _load_tokenizer(tokenizer_id, tokenizer_mode, trust_remote_code):
    """Load tokenizer for random-prompt generation.

    vLLM's get_tokenizer can raise AttributeError when transformers removes
    LlamaTokenizer.all_special_tokens_extended (e.g. Qwen3.5 with newer
    transformers). Prefer backend_request_func.get_tokenizer on fallback so
    client tokenization stays aligned with the sglang server (#1381, #1428).
    """
    if tokenizer_mode == "deepseek_v4":
        # HF AutoTokenizer may not recognize deepseek_v4; use vLLM's loader.
        try:
            from vllm.tokenizers import get_tokenizer as _vllm_get_tokenizer
        except ImportError:
            from vllm.transformers_utils.tokenizer import (
                get_tokenizer as _vllm_get_tokenizer,
            )
        return _vllm_get_tokenizer(
            tokenizer_id,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
        )
    try:
        return get_tokenizer(
            tokenizer_id,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
        )
    except AttributeError as exc:
        if "all_special_tokens_extended" not in str(exc):
            raise
        try:
            from backend_request_func import get_tokenizer as _backend_get_tokenizer
            return _backend_get_tokenizer(
                tokenizer_id,
                tokenizer_mode=tokenizer_mode,
                trust_remote_code=trust_remote_code,
            )
        except ImportError:
            from transformers import AutoTokenizer
            use_fast = tokenizer_mode != "slow"
            return AutoTokenizer.from_pretrained(
                tokenizer_id,
                trust_remote_code=trust_remote_code,
                use_fast=use_fast,
            )


def _init_tokenizer_worker(tokenizer_id, tokenizer_mode, trust_remote_code):
    """Initialize tokenizer once per worker process."""
    global _worker_tokenizer
    _worker_tokenizer = _load_tokenizer(
        tokenizer_id,
        tokenizer_mode=tokenizer_mode,
        trust_remote_code=trust_remote_code,
    )


def _apply_chat_template(prompt, tokenizer, dsv4):
    """Render a single user message into the appropriate chat-template prompt.

    When `dsv4` is True we use the self-contained DeepSeek-V4 encoder
    (encoding_dsv4.encode_messages) which emits the
    <bos><User>...<Assistant><think> framing the model expects. Otherwise we
    fall back to the tokenizer's built-in jinja chat template.
    """
    if dsv4:
        return dsv4_encode_messages(
            [{"role": "user", "content": prompt}],
            thinking_mode="thinking",
        )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )


def _process_prompt_chunk(chunk_args):
    """Generate a chunk of random prompts in a worker process."""
    (indices, prefix_token_ids, input_lens, output_lens, offsets,
     prefix_len, vocab_size, use_chat_template, dsv4, seed) = chunk_args

    rng = np.random.RandomState(seed)
    tokenizer = _worker_tokenizer

    results = []
    for local_idx, global_idx in enumerate(indices):
        tgt_prompt_len = prefix_len + input_lens[local_idx]
        prompt_token_ids = prefix_token_ids + [
            (offsets[local_idx] + global_idx + j) % vocab_size
            for j in range(input_lens[local_idx])
        ]
        prompt = tokenizer.decode(prompt_token_ids)

        max_retries = 10
        for _ in range(max_retries):
            prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(prompt_token_ids) < tgt_prompt_len:
                num_extras = tgt_prompt_len - len(prompt_token_ids)
                prompt_token_ids.extend(
                    rng.randint(0, vocab_size, size=num_extras).tolist())
            elif len(prompt_token_ids) > tgt_prompt_len:
                prompt_token_ids = prompt_token_ids[:tgt_prompt_len]
            else:
                break
            prompt = tokenizer.decode(prompt_token_ids)

        if use_chat_template:
            prompt = _apply_chat_template(prompt, tokenizer, dsv4)

        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        mismatch = prompt_len - tgt_prompt_len
        results.append((prompt, prompt_len, output_lens[local_idx], None, mismatch))

    return results


def sample_random_requests(
    prefix_len: int,
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    use_chat_template: bool = False,
    dsv4: bool = False,
    tokenizer_id: Optional[str] = None,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    num_workers: int = 0,
) -> List[Tuple[str, int, int]]:
    vocab_size = tokenizer.vocab_size
    prefix_token_ids = np.random.randint(0, vocab_size, size=prefix_len).tolist()

    if dsv4 and not use_chat_template:
        raise ValueError("--dsv4 requires --use-chat-template to be set.")

    if use_chat_template:
        chat_template_dummy = _apply_chat_template("a", tokenizer, dsv4)
        tokenized_chat_template_dummy = tokenizer.encode(chat_template_dummy, add_special_tokens=False)
        chat_template_len = len(tokenized_chat_template_dummy) - 1
        input_len = input_len - chat_template_len

    def sample_uniform(seq_len):
        lower = int(seq_len * range_ratio)
        upper = seq_len
        seq_lens = np.random.randint(lower, upper+1, size=num_prompts).tolist()
        return seq_lens

    input_lens = sample_uniform(input_len)
    output_lens = sample_uniform(output_len)
    offsets = np.random.randint(0, vocab_size, size=num_prompts)

    # Create a local RNG for retry-loop padding so that neither serial nor
    # parallel path consumes global np.random draws beyond this point.
    # This ensures downstream code (e.g. gamma draws for inter-arrival times)
    # sees identical global RNG state regardless of num_workers.
    local_rng = np.random.RandomState(
        np.random.get_state()[1][:4].tolist()  # derive seed from current state without advancing it
    )

    # Decide whether to use multiprocessing
    if num_workers <= 0:
        num_workers = min(cpu_count() or 1, 8)
    use_parallel = num_workers > 1 and tokenizer_id is not None

    if use_parallel:
        # Split work into chunks, one per worker
        chunk_size = (num_prompts + num_workers - 1) // num_workers
        chunk_args_list = []
        for w in range(num_workers):
            start = w * chunk_size
            end = min(start + chunk_size, num_prompts)
            if start >= num_prompts:
                break
            chunk_args_list.append((
                list(range(start, end)),
                prefix_token_ids,
                input_lens[start:end],
                output_lens[start:end],
                offsets[start:end].tolist(),
                prefix_len,
                vocab_size,
                use_chat_template,
                dsv4,
                int(local_rng.randint(0, 2**31)),
            ))

        actual_workers = len(chunk_args_list)
        print(f"Generating {num_prompts} prompts using {actual_workers} worker processes...")
        t0 = time.perf_counter()
        with Pool(
            processes=actual_workers,
            initializer=_init_tokenizer_worker,
            initargs=(tokenizer_id, tokenizer_mode, trust_remote_code),
        ) as pool:
            chunk_results = pool.map(_process_prompt_chunk, chunk_args_list)

        input_requests = []
        mismatches = []
        for chunk in chunk_results:
            for prompt, prompt_len, out_len, mm_content, mismatch in chunk:
                input_requests.append((prompt, prompt_len, out_len, mm_content))
                mismatches.append(mismatch)
        elapsed = time.perf_counter() - t0
        print(f"Prompt generation completed in {elapsed:.1f}s")
    else:
        # Original serial path — also uses local_rng for retry-loop padding
        # to keep global RNG consumption identical to the parallel path.
        if tokenizer_id is None and num_workers > 1:
            print("Warning: tokenizer_id not provided, falling back to serial prompt generation.")
        input_requests = []
        mismatches = []
        for i in range(num_prompts):
            tgt_prompt_len = prefix_len + input_lens[i]
            prompt_token_ids = prefix_token_ids + [(offsets[i] + i + j) % vocab_size for j in range(input_lens[i])]
            prompt = tokenizer.decode(prompt_token_ids)

            max_retries = 10
            for _ in range(max_retries):
                prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
                if len(prompt_token_ids) < tgt_prompt_len:
                    num_extras = tgt_prompt_len - len(prompt_token_ids)
                    prompt_token_ids.extend(local_rng.randint(0, vocab_size, size=num_extras).tolist())
                elif len(prompt_token_ids) > tgt_prompt_len:
                    prompt_token_ids = prompt_token_ids[:tgt_prompt_len]
                else:
                    break
                prompt = tokenizer.decode(prompt_token_ids)

            if use_chat_template:
                prompt = _apply_chat_template(prompt, tokenizer, dsv4)

            prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
            mismatches.append(prompt_len - tgt_prompt_len)
            input_requests.append((prompt, prompt_len, output_lens[i], None))

    header_str = f'{"-"*19}  Input/Output Length Statistics  {"-"*19}'
    print(header_str)
    print(
        f' input_lens : '
        f'min={min(r[1] for r in input_requests):<4d}  '
        f'max={max(r[1] for r in input_requests):<4d}  '
        f'mean={np.mean([r[1] for r in input_requests]):<7.2f}  '
        f'avg_token_mismatch={np.mean(mismatches):<5.2f} '
    )
    print(
        f' output_lens: '
        f'min={min(r[2] for r in input_requests):<4d}  '
        f'max={max(r[2] for r in input_requests):<4d}  '
        f'mean={np.mean([r[2] for r in input_requests]):<7.2f} '
    )
    print('-' * len(header_str), '\n')

    return input_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    """
    Asynchronously generates requests at a specified rate
    with OPTIONAL burstiness.

    Args:
        input_requests:
            A list of input requests, each represented as a tuple.
        request_rate:
            The rate at which requests are generated (requests/s).
        burstiness (optional):
            The burstiness factor of the request generation.
            Only takes effect when request_rate is not inf.
            Default value is 1, which follows a Poisson process.
            Otherwise, the request intervals follow a gamma distribution.
            A lower burstiness value (0 < burstiness < 1) results
            in more bursty requests, while a higher burstiness value
            (burstiness > 1) results in a more uniform arrival of requests.
    """
    input_requests = iter(input_requests)

    # Calculate scale parameter theta to maintain the desired request_rate.
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}.")
    theta = 1.0 / (request_rate * burstiness)

    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the gamma distribution.
        # If burstiness is 1, it follows exponential distribution.
        interval = np.random.gamma(shape=burstiness, scale=theta)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[float],
    goodput_config_dict: Dict[str, float],
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    all_tpots: List[float] = []
    ttfts: List[float] = []
    e2els: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_tokens

            if output_len is None:
                # We use the tokenizer to count the number of output tokens
                # for some serving backends instead of looking at
                # len(outputs[i].itl) since multiple output tokens may be
                # bundled together
                # Note : this may inflate the output token count slightly
                output_len = len(
                    tokenizer(outputs[i].generated_text,
                              add_special_tokens=False).input_ids)
            actual_output_lens.append(output_len)
            total_input += input_requests[i][1]
            tpot = 0
            if output_len > 1:
                latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if goodput_config_dict:
        valid_metrics = []
        slo_values = []

        if "ttft" in goodput_config_dict:
            valid_metrics.append(ttfts)
            slo_values.append(goodput_config_dict["ttft"] /
                              MILLISECONDS_TO_SECONDS_CONVERSION)
        if "tpot" in goodput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(goodput_config_dict["tpot"] /
                              MILLISECONDS_TO_SECONDS_CONVERSION)
        if "e2el" in goodput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(goodput_config_dict["e2el"] /
                              MILLISECONDS_TO_SECONDS_CONVERSION)

        for req_metric in zip(*valid_metrics):
            is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
            if is_good_req:
                good_completed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) *
        1000,  # ttfts is empty if streaming is not supported by backend
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[(p, np.percentile(ttfts or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[(p, np.percentile(tpots or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[(p, np.percentile(itls or 0, p) * 1000)
                            for p in selected_percentiles],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[(p, np.percentile(e2els or 0, p) * 1000)
                             for p in selected_percentiles],
    )

    return metrics, actual_output_lens


def compute_steady_state_metrics(
    per_req_itls: List[List[float]],
    ttfts: List[float],
    output_lens: List[int],
    gpu_count: int,
) -> Dict[str, Any]:
    """纯 decode 稳态采样 (max-TTFT 窗口), 物理保证窗口内无 prefill (见 CLAUDE.md "完全刨去 prefill").

      - 所有请求等长、一次性发出 (num_prompts=concurrency, rate=inf), 近似同时起跑,
        故用 ttft 作为各请求"首 token 到达时刻"(相对共同原点; 派发抖动 <ms 可忽略).
      - token j 绝对到达时刻 = ttft + cumsum(itl[:j+1]).
      - 窗口起点 = max_i(ttft_i): 最后一个请求都出了首 token => 所有请求都已过 prefill;
        又因 num_prompts 有限且无新请求进来 + enable_mixed_chunk=False,
        => 此刻起服务端不再有任何 prefill forward, 是纯 decode.
      - 窗口终点 = min_i(请求结束时刻): 在此之前 batch 恒满 (无请求退出).
      - 窗内: STPS = 窗内总 token / 窗长; UTPS = 各请求窗内 token 率的中位数.
    """
    reqs = []
    for itl, ttft, olen in zip(per_req_itls, ttfts, output_lens):
        if not itl or ttft is None or olen <= 1:
            continue
        times, t = [], float(ttft)
        for d in itl:
            t += d
            times.append(t)          # 第 k 个流式 chunk 的绝对到达时刻
        # 每个 ITL 对应一次流式 chunk, 一个 chunk 可能携带多个 token: 例如 MTP 一个
        # decode step 接受 accept_len 个 token, vllm 把它们放进同一个 SSE chunk =>
        # len(itl) = chunk 数 < 实际 token 数 (sglang 逐 token 流式则 chunk≈token).
        # 用 olen/chunk数 折算每 chunk 的 token 数, 使窗内按真实 token 计数, 口径与
        # 逐 token 后端一致 (否则 MTP 后端 UTPS/STPS 会被低估 accept_len 倍).
        tpc = float(olen) / len(itl)
        reqs.append({"ttft": float(ttft), "times": times, "end": times[-1], "tpc": tpc})

    out: Dict[str, Any] = {"gpu_count": gpu_count, "steady_num_reqs": len(reqs)}
    if not reqs:
        out["steady_note"] = "no steady-state samples"
        return out

    win_start = max(r["ttft"] for r in reqs)     # 所有请求都过了 prefill 的时刻
    win_end = min(r["end"] for r in reqs)        # 第一个请求结束的时刻 (之前 batch 恒满)
    if win_end <= win_start:
        out["steady_note"] = "max-TTFT window empty (OSL too short vs prefill spread)"
        return out

    dur = win_end - win_start
    rates, total = [], 0.0
    for r in reqs:
        # 窗内 chunk 数 × 每 chunk token 数 = 窗内真实 token 数
        n = sum(1 for tt in r["times"] if win_start <= tt <= win_end) * r["tpc"]
        total += n
        rates.append(n / dur)
    stps = total / dur
    out.update({
        "steady_utps_per_user": float(np.median(rates)),   # 曲线横轴 (tok/s/user)
        "steady_stps_system": float(stps),                 # 系统 tok/s
        "steady_stps_per_gpu": float(stps / gpu_count) if gpu_count else float(stps),
        "steady_window_start_s": round(win_start, 4),      # =max(ttft), prefill 全部结束点
        "steady_window_dur_s": round(dur, 4),
        "steady_window_tokens_total": round(total, 1),
        "steady_ttft_min_s": round(min(r["ttft"] for r in reqs), 4),
        "steady_ttft_max_s": round(win_start, 4),
    })
    return out


async def flush_server_cache(base_url: str, attempts: int = 15,
                             wait_s: float = 2.0) -> bool:
    """清 server 端 prefix/radix cache —— cookbook 口径的 `--flush-cache`.

    cookbook 的速度数字是带 `--flush-cache` 测的: 不清的话同一批 prompt 在重复测点
    (REPS>1) 之间会命中 prefix cache, prefill 变快 -> 稳态窗口起点(max-TTFT)提前,
    与 cookbook 不可比. 每个测点开测前调一次.

    端点各家不同, 依次试: sglang `/flush_cache`, vllm `/reset_prefix_cache`.

    ★必须真清掉, 所以要重试★: sglang 在【还有请求在跑】时会拒绝清缓存 ——
    `HTTP 400 "Cache not flushed because there are pending requests. #running-req: 16"`。
    warmup 的最后一批常常还没 drain 完就撞上这个(实测撞到过: POST 400 -> 1s 后 GET 才成功,
    纯属侥幸)。所以这里【重试到真成功】, 而不是换个 method/端点赌一次: 赌输了就是带着热
    prefix cache 开测 —— prefill 变快、TTFT 偏低、稳态窗口起点提前, 与 cookbook 口径不一致,
    而且事后极难发现(要去 serverlog 里数 `#cached-token` 才看得出来)。
    只在见到【可重试】的失败(HTTP 400 = 有请求在跑)时才等待重试; 若只见到 404/405
    (该 backend 压根没这个端点)就立刻放弃, 免得每个测点白等几十秒.
    """
    import aiohttp

    endpoints = ("/flush_cache", "/reset_prefix_cache")
    last = ""
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=120)
    ) as session:
        for i in range(attempts):
            retryable = False
            for ep in endpoints:
                url = base_url + ep
                for method in ("post", "get"):
                    try:
                        async with getattr(session, method)(url) as resp:
                            if resp.status < 400:
                                print("Flushed server cache: %s %s%s"
                                      % (method.upper(), url,
                                         (" (第 %d 次尝试)" % (i + 1)) if i else ""))
                                return True
                            body = (await resp.text())[:120].replace("\n", " ")
                            last = "%s %s -> HTTP %d: %s" % (method.upper(), url,
                                                             resp.status, body)
                            # 400 = "有请求在跑, 清不了" -> 等一下就能成; 404/405 = 没这端点
                            if resp.status == 400:
                                retryable = True
                    except Exception as e:  # noqa: BLE001 - 端点不存在/连接问题都试下一个
                        last = "%s %s -> %s" % (method.upper(), url, type(e).__name__)
                        continue
            if not retryable:
                break
            if i + 1 < attempts:
                print("flush cache 暂时清不掉(%s), %.1fs 后重试 %d/%d"
                      % (last, wait_s, i + 2, attempts))
                await asyncio.sleep(wait_s)
    print(f"WARNING: flush cache failed (tried {endpoints} on {base_url}; 末次: {last}); "
          "结果可能受 prefix cache 影响, 与 cookbook 口径不一致")
    return False


async def benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    model_name: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    logprobs: Optional[int],
    best_of: int,
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    num_warmups: int,
    profile: bool,
    flush_cache: bool,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[str],
    ignore_eos: bool,
    goodput_config_dict: Dict[str, float],
    max_concurrency: Optional[int],
    lora_modules: Optional[List[str]],
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print("Starting initial single prompt test run...")
    test_prompt, test_prompt_len, test_output_len, test_mm_content = (
        input_requests[0])
    if backend != "openai-chat" and test_mm_content is not None:
        # multi-modal benchmark is only available on OpenAI Chat backend.
        raise ValueError(
            "Multi-modal content is only supported on 'openai-chat' backend.")
    test_input = RequestFuncInput(
        model=model_id,
        model_name=model_name,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        logprobs=logprobs,
        best_of=best_of,
        multi_modal_content=test_mm_content,
        ignore_eos=ignore_eos,
    )

    if num_warmups > 0:
        print(f"Warming up with {num_warmups} requests...")
        warmup_pbar = None if disable_tqdm else tqdm(total=num_warmups)
        warmup_semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

        async def warmup_limited_req_fn():
            if warmup_semaphore is None:
                return await request_func(request_func_input=test_input, pbar=warmup_pbar)
            async with warmup_semaphore:
                return await request_func(request_func_input=test_input, pbar=warmup_pbar)

        warmup_tasks = []
        for _ in range(num_warmups):
            task = asyncio.create_task(warmup_limited_req_fn())
            warmup_tasks.append(task)
        _ = await asyncio.gather(*warmup_tasks)

        if warmup_pbar is not None:
            warmup_pbar.close()
        print("Warmup completed.")

    # 清缓存放在 warmup 之后、正式测点之前: warmup 本身会把 prompt 灌进 prefix cache,
    # 顺序反了等于没清.
    if flush_cache:
        await flush_server_cache(base_url)

    if lora_modules:
        # For each input request, choose a LoRA module at random.
        lora_modules = iter(
            [random.choice(lora_modules) for _ in range(len(input_requests))])

    if profile:
        print("Starting profiler...")
        profile_input = RequestFuncInput(model=model_id,
                                         model_name=model_name,
                                         prompt=test_prompt,
                                         api_url=base_url + "/start_profile",
                                         prompt_len=test_prompt_len,
                                         output_len=test_output_len,
                                         extra_body={"num_steps": 1, "merge_profiles": True, "profile_by_stage": True},
                                         logprobs=logprobs,
                                         best_of=best_of,
                                         multi_modal_content=test_mm_content,
                                         ignore_eos=ignore_eos)
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler started")

    if burstiness == 1.0:
        distribution = "Poisson process"
    else:
        distribution = "Gamma distribution"

    print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    semaphore = (asyncio.Semaphore(max_concurrency)
                 if max_concurrency else None)

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input,
                                      pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input,
                                      pbar=pbar)

    print("Starting main benchmark run...")

    benchmark_start_time = time.perf_counter()
    benchmark_start_time_unix = time.time()
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate, burstiness):
        prompt, prompt_len, output_len, mm_content = request
        req_model_id, req_model_name = model_id, model_name
        if lora_modules:
            req_lora_module = next(lora_modules)
            req_model_id, req_model_name = req_lora_module, req_lora_module

        request_func_input = RequestFuncInput(model=req_model_id,
                                              model_name=req_model_name,
                                              prompt=prompt,
                                              api_url=api_url,
                                              prompt_len=prompt_len,
                                              output_len=output_len,
                                              logprobs=logprobs,
                                              best_of=best_of,
                                              multi_modal_content=mm_content,
                                              ignore_eos=ignore_eos)
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input,
                                     pbar=pbar)))
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if profile:
        print("Stopping profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/stop_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
            best_of=best_of,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler stopped")

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time
    benchmark_end_time_unix = time.time()

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
        goodput_config_dict=goodput_config_dict,
    )

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:",
                                 metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                    metrics.request_throughput))
    if goodput_config_dict:
        print("{:<40} {:<10.2f}".format("Request goodput (req/s):",
                                        metrics.request_goodput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                    metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):",
                                    metrics.total_token_throughput))

    result = {
        "duration": benchmark_duration,
        "benchmark_start_time_unix": benchmark_start_time_unix,
        "benchmark_end_time_unix": benchmark_end_time_unix,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "request_goodput:":
        metrics.request_goodput if goodput_config_dict else None,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c='-'))
        print("{:<40} {:<10.2f}".format(
            f"Mean {metric_name} (ms):",
            getattr(metrics, f"mean_{metric_attribute_name}_ms")))
        print("{:<40} {:<10.2f}".format(
            f"Median {metric_name} (ms):",
            getattr(metrics, f"median_{metric_attribute_name}_ms")))
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms")
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms")
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms")
        for p, value in getattr(metrics,
                                f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):",
                                            value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT",
                       "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    return result


def check_goodput_args(args):
    # Check and parse goodput arguments
    goodput_config_dict = {}
    VALID_NAMES = ["ttft", "tpot", "e2el"]
    if args.goodput:
        goodput_config_dict = parse_goodput(args.goodput)
        for slo_name, slo_val in goodput_config_dict.items():
            if slo_name not in VALID_NAMES:
                raise ValueError(
                    f"Invalid metric name found, {slo_name}: {slo_val}. "
                    "The service level objective name should be one of "
                    f"{str(VALID_NAMES)}. ")
            if slo_val < 0:
                raise ValueError(
                    f"Invalid value found, {slo_name}: {slo_val}. "
                    "The service level objective value should be "
                    "non-negative.")
    return goodput_config_dict


def parse_goodput(slo_pairs):
    goodput_config_dict = {}
    try:
        for slo_pair in slo_pairs:
            slo_name, slo_val = slo_pair.split(":")
            goodput_config_dict[slo_name] = float(slo_val)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "Invalid format found for service level objectives. "
            "Specify service level objectives for goodput as \"KEY:VALUE\" "
            "pairs, where the key is a metric name, and the value is a "
            "number in milliseconds.") from err
    return goodput_config_dict


def save_to_pytorch_benchmark_format(args: argparse.Namespace,
                                     results: Dict[str, Any],
                                     file_name: str) -> None:
    metrics = [
        "median_ttft_ms", "mean_ttft_ms", "std_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "median_tpot_ms", "std_tpot_ms", "p99_tpot_ms",
        "median_itl_ms", "mean_itl_ms", "std_itl_ms", "p99_itl_ms"
    ]
    # These raw data might be useful, but they are rather big. They can be added
    # later if needed
    ignored_metrics = ["ttfts", "itls", "generated_texts", "errors"]
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={k: [results[k]]
                 for k in metrics},
        extra_info={
            k: results[k]
            for k in results if k not in metrics and k not in ignored_metrics
        })
    if pt_records:
        # Don't use json suffix here as we don't want CI to pick it up
        pt_file = f"{os.path.splitext(file_name)[0]}.pytorch.json"
        with open(pt_file, "w") as f:
            json.dump(pt_records, f)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    model_name = args.served_model_name
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer_mode = args.tokenizer_mode

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"

    tokenizer = _load_tokenizer(
        tokenizer_id,
        tokenizer_mode=tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
    )


    if args.dataset_name == "random":
        input_requests = sample_random_requests(
            prefix_len=args.random_prefix_len,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_prompts=args.num_prompts,
            range_ratio=args.random_range_ratio,
            tokenizer=tokenizer,
            use_chat_template=args.use_chat_template,
            dsv4=args.dsv4,
            tokenizer_id=tokenizer_id,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=args.trust_remote_code,
            num_workers=args.random_num_workers,
        )

    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    goodput_config_dict = check_goodput_args(args)

    # Avoid GC processing "static" data - reduce pause times.
    gc.collect()
    gc.freeze()

    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            model_name=model_name,
            tokenizer=tokenizer,
            input_requests=input_requests,
            logprobs=args.logprobs,
            best_of=args.best_of,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            disable_tqdm=args.disable_tqdm,
            num_warmups=args.num_warmups,
            profile=args.profile,
            flush_cache=args.flush_cache,
            selected_percentile_metrics=args.percentile_metrics.split(","),
            selected_percentiles=[
                float(p) for p in args.metric_percentiles.split(",")
            ],
            ignore_eos=args.ignore_eos,
            goodput_config_dict=goodput_config_dict,
            max_concurrency=args.max_concurrency,
            lora_modules=args.lora_modules,
        ))

    # 稳态 decode 指标: max-TTFT 窗口 (物理保证无 prefill), 注入结果 JSON.
    # 【无条件算】—— 它和全程口径(mean_tpot / total_token_throughput, 上面已在 JSON 里)是同一批
    # 原始数据的两种事后算法, 都留在结果 JSON 里, 由报告端的 metric_mode 决定画哪套(见
    # runners/config.json 的 defaults.metric_mode). 客户端不该也不需要知道要报哪个口径.
    steady = compute_steady_state_metrics(
        per_req_itls=benchmark_result.get("itls", []),
        ttfts=benchmark_result.get("ttfts", []),
        output_lens=benchmark_result.get("output_lens", []),
        gpu_count=args.gpu_count,
    )
    benchmark_result.update(steady)
    print("{s:{c}^{n}}".format(
        s=' Steady-State (max-TTFT window, prefill-excluded) ', n=56, c='='))
    for _k in ("steady_utps_per_user", "steady_stps_system",
               "steady_stps_per_gpu", "steady_window_dur_s",
               "steady_window_tokens_total", "steady_ttft_min_s",
               "steady_ttft_max_s", "steady_num_reqs"):
        if _k in steady:
            print("{:<40} {:<12.4f}".format(_k + ":", steady[_k]))

    # Save config and results to json
    if args.save_result:
        result_json: Dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["best_of"] = args.best_of
        result_json["num_prompts"] = args.num_prompts

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        # Traffic
        result_json["request_rate"] = (args.request_rate if args.request_rate
                                       < float("inf") else "inf")
        result_json["burstiness"] = args.burstiness
        result_json["max_concurrency"] = args.max_concurrency

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}
        
        if not args.save_detailed:
            # Remove fields with too many data points
            for field in [
                "ttfts",
                "itls",
                "generated_texts",
                "errors",
            ]:
                if field in result_json:
                    del result_json[field]
                if field in benchmark_result:
                    del benchmark_result[field]

        # Save to file
        base_model_id = model_id.split("/")[-1]
        max_concurrency_str = (f"-concurrency{args.max_concurrency}"
                               if args.max_concurrency is not None else "")
        file_name = f"{backend}-{args.request_rate}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"  #noqa
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
        with open(file_name, "w", encoding='utf-8') as outfile:
            json.dump(result_json, outfile)
        save_to_pytorch_benchmark_format(args, result_json, file_name)

    max_failure_rate = 0.05
    completed = benchmark_result["completed"]
    failure_rate = 1 - completed / args.num_prompts
    if failure_rate > max_failure_rate:
        raise SystemExit(
            f"FAIL: request failure rate {failure_rate:.1%} exceeds "
            f"{max_failure_rate:.0%} threshold "
            f"({completed}/{args.num_prompts} completed)"
        )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    # Use 127.0.0.1 here instead of localhost to force the use of ipv4
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["random"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        help="Path to the sharegpt/sonnet dataset. "
                        "Or the huggingface dataset ID if using HF dataset.")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and "
        "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help=("Number of logprobs-per-token to compute & return as part of "
              "the request. If unspecified, then either (1) if beam search "
              "is disabled, no logprobs are computed & a single dummy "
              "logprob is returned for each token; or (2) if beam search "
              "is enabled 1 logprob per token is computed"),
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process or gamma distribution "
        "to synthesize the request arrival times.",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor of the request generation. "
        "Only take effect when request_rate is not inf. "
        "Default value is 1, which follows Poisson process. "
        "Otherwise, the request intervals follow a gamma distribution. "
        "A lower burstiness value (0 < burstiness < 1) results in more "
        "bursty requests. A higher burstiness value (burstiness > 1) "
        "results in a more uniform arrival of requests.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "VLLM_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--flush-cache",
        action="store_true",
        help="Flush the server's prefix/radix cache right before the measured "
        "requests (cookbook 口径; sglang /flush_cache, vllm /reset_prefix_cache). "
        "Without it, repeated measurements of the same prompts hit the prefix "
        "cache and prefill gets faster, which shifts the steady-state window.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--save-detailed",
        action="store_true",
        default=False,
        help="When saving results, include detailed per-request data "
        "(input_lens, output_lens, ttfts, itls, generated_texts, errors). "
        "By default, only aggregated metrics are saved to reduce file size.",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        " format.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag when sending the benchmark request."
        "Warning: ignore_eos is not supported in deepspeed_mii and tgi.")
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl,e2el",
        help="Comma-seperated list of selected metrics to report percentils. "
        "This argument specifies the metrics to report percentiles. "
        "Allowed metric names are \"ttft\", \"tpot\", \"itl\", \"e2el\". "
        "Default value is \"ttft,tpot,itl,e2el\".")
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="90,99,99.9",
        help="Comma-seperated list of percentiles for selected metrics. "
        "To report 25-th, 50-th, and 75-th percentiles, use \"25,50,75\". "
        "Default value is \"90,99,99.9\". "
        "Use \"--percentile-metrics\" to select metrics.",
    )
    parser.add_argument(
        "--goodput",
        nargs="+",
        required=False,
        help="Specify service level objectives for goodput as \"KEY:VALUE\" "
        "pairs, where the key is a metric name, and the value is in "
        "milliseconds. Multiple \"KEY:VALUE\" pairs can be provided, "
        "separated by spaces. Allowed request level metric names are "
        "\"ttft\", \"tpot\", \"e2el\". For more context on the definition of "
        "goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 "
        "and the blog: https://hao-ai-lab.github.io/blogs/distserve")

    # group for dataset specific arguments
    sonnet_group = parser.add_argument_group("sonnet dataset options")
    sonnet_group.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help=
        "Number of input tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-output-len",
        type=int,
        default=150,
        help=
        "Number of output tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help=
        "Number of prefix tokens per request, used only for sonnet dataset.",
    )

    sharegpt_group = parser.add_argument_group("sharegpt dataset options")
    sharegpt_group.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.")

    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help=
        "Number of input tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help=
        "Number of output tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=1.0,
        help="Range of sampled ratio of input/output length, "
        "used only for random sampling.",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help="Number of fixed prefix tokens before random "
        " context. The length range of context in a random "
        " request is [random-prefix-len, "
        " random-prefix-len + random-prefix-len * random-range-ratio).")
    random_group.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Use chat template to format the prompt.",
    )
    random_group.add_argument(
        '--random-num-workers',
        type=int,
        default=0,
        help="Number of worker processes for parallel random prompt generation. "
        "Only used with --dataset-name random. "
        "0 (default) = auto (min(cpu_count, 8)). 1 = serial (no multiprocessing).",
    )

    dsv4_group = parser.add_argument_group("DeepSeek-V4 chat template options")
    dsv4_group.add_argument(
        "--dsv4",
        action="store_true",
        help="Use the DeepSeek-V4 chat template (encoding_dsv4.py) instead of "
        "the tokenizer's built-in jinja chat template. Requires "
        "--use-chat-template to also be set. Applies to the random dataset.",
    )

    hf_group = parser.add_argument_group("hf dataset options")
    hf_group.add_argument("--hf-subset",
                          type=str,
                          default=None,
                          help="Subset of the HF dataset.")
    hf_group.add_argument("--hf-split",
                          type=str,
                          default=None,
                          help="Split of the HF dataset.")
    hf_group.add_argument(
        "--hf-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output lengths "
        "from the sampled HF dataset.",
    )

    parser.add_argument(
        '--tokenizer-mode',
        type=str,
        default="auto",
        choices=['auto', 'slow', 'mistral', 'custom', 'deepseek_v4'],
        help='The tokenizer mode.\n\n* "auto" will use the '
        'fast tokenizer if available.\n* "slow" will '
        'always use the slow tokenizer. \n* '
        '"mistral" will always use the `mistral_common` tokenizer. \n*'
        '"custom" will use --tokenizer to select the preregistered tokenizer.')

    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The model name used in the API. "
                        "If not specified, the model name will be the "
                        "same as the ``--model`` argument. ")

    parser.add_argument("--lora-modules",
                        nargs='+',
                        default=None,
                        help="A subset of LoRA module names passed in when "
                        "launching the server. For each request, the "
                        "script chooses a LoRA module at random.")

    parser.add_argument('--num-warmups', type=int, default=0)

    # 稳态 decode 采样 (max-TTFT 窗口, 物理保证窗内无 prefill) 恒开, 无开关 ——
    # 报哪套口径是【报告端】的事(config.json 的 defaults.metric_mode), 不是压测端的事.
    parser.add_argument('--gpu-count', type=int, default=8,
                        help="用于把系统吞吐换算成每GPU吞吐(STPS/gpu).")

    args = parser.parse_args()
    main(args)
