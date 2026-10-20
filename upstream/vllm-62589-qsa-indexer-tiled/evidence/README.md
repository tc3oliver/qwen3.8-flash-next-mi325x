# Evidence for vllm#62589

`../qsa.py` is the final branch file, mounted for the end-to-end run `fin2-tiled-tp2`
(`results/experiments/`, vs `pr-stock-tp2`).

| dir | content |
|---|---|
| `review3/` | final revision: pytest (18 passed), the three mutations the new test catches, logits bf16/fp16/fp8, microbenchmarks main vs final, pre-commit output, scripts |
| `review2/` | review round 2: dot dtype (fp16 inputs), MTP-off and mixed decode, per-request skip candidates, top-k with poisoned columns past the visible range, pre-commit, scripts |
| `pr2-*`, `mutants/`, `scripts/` | first revision (one shared early exit per row block). Numbers in the original PR text came from here |

Scripts run inside the vLLM v0.32.0 image. They load the files to compare from `/work` (`stock.py` = the v0.32.0
`qsa.py`, variants by name) and `harness.py` from `/work` or `/patch`. Branch variants and mutants of revisions 2 and 3
(~40 KB copies of `qsa.py` each) are not included.
