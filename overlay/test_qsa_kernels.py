import math

import pytest
import torch

from vllm.models.qwen4_exp.amd.ops import qsa as qsa_ops
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
requires_qsa_kernels = pytest.mark.skipif(not HAS_TRITON, reason="Triton")

def _qsa_mqa_paged_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    visible_lengths: torch.Tensor,
) -> torch.Tensor:
    pages = page_table.index_select(0, token_to_req.long()).long()
    keys = k_cache[pages, :, 0, :].flatten(1, 2)
    scores = torch.einsum("rhd,rnd->rnh", q.float(), keys.float())
    logits = torch.relu(scores).sum(dim=-1) / math.sqrt(q.shape[-1])
    positions = torch.arange(keys.shape[1], device=q.device).unsqueeze(0)
    return logits.masked_fill(positions >= visible_lengths.unsqueeze(1), -torch.inf)


def _indexer_tensor(shape: tuple[int, ...], dtype: str) -> torch.Tensor:
    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp8": current_platform.fp8_dtype(),
    }[dtype]
    return torch.randn(shape, device="cuda").to(torch_dtype)


@requires_qsa_kernels
@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp8"])
def test_qsa_mqa_paged_matches_reference_on_visible_columns(
    dtype: str,
) -> None:
    # A prefill chunk, a decode request, and rows with nothing visible, so one
    # row block spans three requests. Top-k reads only [0, visible_blocks).
    torch.manual_seed(0)
    page_size, compress_ratio, table_width, num_pages = 16, 4, 24, 80
    positions = [range(1160, 1203), range(159, 162), range(3)]
    lengths = torch.tensor([1203, 162, 3], device="cuda", dtype=torch.int32)
    token_to_req = torch.tensor(
        [r for r, p in enumerate(positions) for _ in p],
        device="cuda",
        dtype=torch.int32,
    )
    query_positions = torch.tensor(
        [i for p in positions for i in p], device="cuda", dtype=torch.int64
    )
    q = _indexer_tensor((token_to_req.numel(), 2, 32), dtype)
    k_cache = _indexer_tensor((num_pages, page_size, 1, 32), dtype)
    page_table = torch.randperm(num_pages, device="cuda", dtype=torch.int32)[
        : 3 * table_width
    ].view(3, table_width)

    logits, visible_blocks = qsa_ops.qsa_mqa_paged(
        q, k_cache, page_table, token_to_req, query_positions, lengths, compress_ratio
    )

    expected_visible = torch.minimum(
        (query_positions + 1) // compress_ratio,
        lengths[token_to_req.long()] // compress_ratio,
    ).to(torch.int32)
    torch.testing.assert_close(visible_blocks, expected_visible, rtol=0, atol=0)
    columns = torch.arange(logits.shape[1], device="cuda")
    logits = logits.masked_fill(columns >= expected_visible[:, None], -torch.inf)
    expected = _qsa_mqa_paged_reference(
        q, k_cache, page_table, token_to_req, expected_visible
    )
    torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-5)


@requires_qsa_kernels
@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp8"])
@pytest.mark.parametrize("num_requests", [1, 5, 32, 33])
def test_qsa_mqa_paged_decode_rows_match_reference(
    monkeypatch: pytest.MonkeyPatch, dtype: str, num_requests: int
) -> None:
    # One decode row per request, with lengths that end mid-page. Up to 32 rows
    # are scored row by row on gfx942, 33 use the tiled kernel.
    spy = _force_rows_kernel(monkeypatch)
    torch.manual_seed(num_requests)
    page_size, compress_ratio, table_width = 16, 4, 24
    num_pages = num_requests * table_width + 8
    lengths = torch.randint(1, 1500, (num_requests,), device="cuda", dtype=torch.int32)
    token_to_req = torch.arange(num_requests, device="cuda", dtype=torch.int32)
    query_positions = (lengths - 1).to(torch.int64)
    q = _indexer_tensor((num_requests, 2, 32), dtype)
    k_cache = _indexer_tensor((num_pages, page_size, 1, 32), dtype)
    page_table = torch.randperm(num_pages, device="cuda", dtype=torch.int32)[
        : num_requests * table_width
    ].view(num_requests, table_width)

    logits, visible_blocks = qsa_ops.qsa_mqa_paged(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        lengths,
        compress_ratio,
        single_row_decode=True,
    )

    assert spy.launches == (num_requests <= 32)
    expected_visible = (lengths // compress_ratio).to(torch.int32)
    torch.testing.assert_close(visible_blocks, expected_visible, rtol=0, atol=0)
    columns = torch.arange(logits.shape[1], device="cuda")
    logits = logits.masked_fill(columns >= expected_visible[:, None], -torch.inf)
    expected = _qsa_mqa_paged_reference(
        q, k_cache, page_table, token_to_req, expected_visible
    )
    torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-5)


class _KernelSpy:
    """Counts launches of a Triton kernel."""

    def __init__(self, kernel) -> None:
        self.kernel = kernel
        self.launches = 0

    def __getitem__(self, grid):
        self.launches += 1
        return self.kernel[grid]


def _force_rows_kernel(monkeypatch: pytest.MonkeyPatch) -> _KernelSpy:
    # The per-row path is gfx942 only, but the kernel runs on any ROCm arch.
    monkeypatch.setattr(qsa_ops, "_on_gfx942", lambda: True)
    spy = _KernelSpy(qsa_ops._qsa_mqa_paged_rows_kernel)
    monkeypatch.setattr(qsa_ops, "_qsa_mqa_paged_rows_kernel", spy)
    return spy


def _decode_select_case(
    num_requests: int, dtype: str
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
]:
    # Positional arguments of qsa_select_paged_tokens, with token_topk=16.
    torch.manual_seed(num_requests)
    page_size, compress_ratio, table_width = 16, 4, 24
    num_pages = num_requests * table_width + 8
    lengths = torch.randint(8, 1500, (num_requests,), device="cuda", dtype=torch.int32)
    return (
        _indexer_tensor((num_requests, 2, 32), dtype),
        _indexer_tensor((num_pages, page_size, 1, 32), dtype),
        torch.randperm(num_pages, device="cuda", dtype=torch.int32)[
            : num_requests * table_width
        ].view(num_requests, table_width),
        torch.arange(num_requests, device="cuda", dtype=torch.int32),
        (lengths - 1).to(torch.int64),
        lengths,
        16,
        compress_ratio,
    )


@requires_qsa_kernels
@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp8"])
def test_qsa_select_decode_rows_ignores_unwritten_logits(
    monkeypatch: pytest.MonkeyPatch, dtype: str
) -> None:
    # Neither kernel writes logits past visible_blocks, so top-k must not read
    # them: fill every fresh logits buffer with NaN and compare both paths.
    spy = _force_rows_kernel(monkeypatch)
    case = _decode_select_case(7, dtype)
    expected = qsa_ops.qsa_select_paged_tokens(*case, single_row_decode=False)

    real_empty = torch.empty

    def empty_nan(*args, **kwargs):
        out = real_empty(*args, **kwargs)
        if out.dtype == torch.float32 and out.dim() == 2 and out.is_cuda:
            out.fill_(float("nan"))
        return out

    monkeypatch.setattr(torch, "empty", empty_nan)
    spy.launches = 0
    actual = qsa_ops.qsa_select_paged_tokens(*case, single_row_decode=True)
    assert spy.launches == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_qsa_kernels
def test_qsa_select_decode_rows_replays_in_cuda_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _force_rows_kernel(monkeypatch)
    (
        q,
        k_cache,
        page_table,
        token_to_req,
        positions,
        lengths,
        token_topk,
        compress_ratio,
    ) = _decode_select_case(8, "bf16")

    def select():
        return qsa_ops.qsa_select_paged_tokens(
            q,
            k_cache,
            page_table,
            token_to_req,
            positions,
            lengths,
            token_topk,
            compress_ratio,
            single_row_decode=True,
        )

    select()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = select()

    # Replay on new inputs written in place, as the runner does between steps.
    torch.manual_seed(1)
    q.copy_(torch.randn_like(q))
    new_lengths = torch.randint(
        8, 1500, lengths.shape, device="cuda", dtype=lengths.dtype
    )
    lengths.copy_(new_lengths)
    positions.copy_((new_lengths - 1).to(torch.int64))
    graph.replay()
    torch.accelerator.synchronize()
    expected = qsa_ops.qsa_select_paged_tokens(
        q,
        k_cache,
        page_table,
        token_to_req,
        positions,
        lengths,
        token_topk,
        compress_ratio,
        single_row_decode=False,
    )
    # Warmup and capture launched it; the replay does not go through Python.
    assert spy.launches == 2
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)

