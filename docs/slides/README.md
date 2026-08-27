# GPU Core Principles slides

This directory contains a three-slide, 16:9 visual primer on why GPUs deliver
high throughput:

1. CPU versus GPU architecture, using the sports-car-versus-bus analogy and
   simplified hardware layouts.
2. Parallel execution, showing how a large concurrency window supports memory
   bandwidth and aggregate FLOP throughput.
3. Warp scheduling, showing how a ready resident warp can issue while another
   waits on memory or a dependency.

## Files

- [`gpu-core-principles.pdf`](gpu-core-principles.pdf) is the primary,
  ready-to-present three-page deck.
- [`gpu-core-principles.html`](gpu-core-principles.html) is the editable,
  vector-first source. Open it in a
  browser to present it or print it directly to PDF with backgrounds enabled.
- [`previews/slide-1.png`](previews/slide-1.png),
  [`slide-2.png`](previews/slide-2.png), and
  [`slide-3.png`](previews/slide-3.png) are 1280×720 previews suitable for
  Markdown or MkDocs pages.
- [`render_gpu_core_principles.py`](render_gpu_core_principles.py)
  deterministically rebuilds the PDF and PNG previews with a local
  Chromium/Chrome executable.

Rebuild from the repository root:

```bash
python3 docs/slides/render_gpu_core_principles.py
```

If Chromium is not on `PATH`, set `CHROME_BIN` to its executable before running
the command.

## Speaker notes

### 1. CPU minimizes latency. GPU maximizes throughput.

The car and bus describe different optimization goals, not a claim that one
processor is universally faster. CPUs use a few sophisticated cores, large
caches, branch prediction, out-of-order execution, and SIMD to finish diverse
threads quickly. GPUs repeat many throughput-oriented streaming
multiprocessors and arithmetic lanes so far more total work can be completed
per second. Both architectures contain caches, control logic, and parallel
execution; the diagrams are deliberately conceptual and not to scale.

### 2. Many threads in flight keep memory and math busy.

Modern CPUs are parallel, but a GPU exposes a much larger pool of independent
threads and memory requests. Hardware groups CUDA threads into 32-thread warps
and schedules many resident warps across finite execution lanes. Adjacent
accesses can coalesce into efficient wide memory transactions, while SIMT
execution applies an instruction across many arithmetic lanes. Parallelism
helps approach peak bandwidth and FLOP/s only when there is enough independent
work, access patterns are efficient, and divergence does not leave lanes idle.

### 3. When one warp waits, the GPU issues another.

The memory delay is not removed. When Warp A cannot issue because it is waiting
for data or a dependency, its state remains resident and the scheduler can
select a ready Warp B, C, or D. This is hardware latency hiding, not an
operating-system-style swap. It works when enough independent warps are
resident and ready; register and shared-memory use limit residency, and an SM
can still idle if every warp is waiting.

## Five review passes

1. **Story pass:** reduced the material to one claim, one dominant diagram, and
   one takeaway per slide.
2. **Accuracy pass:** replaced “CPUs serialize” with a smaller-concurrency-window
   comparison; replaced “swap out” with selecting another resident ready warp;
   added coalescing, divergence, and residency qualifications.
3. **Hierarchy pass:** reviewed true 1280×720 renders at full size and thumbnail
   scale; strengthened the title/diagram/takeaway reading order and corrected
   tight diagram labels.
4. **Consistency and accessibility pass:** normalized margins, color semantics,
   labels, and line weights; paired the stall color with hatching; enlarged
   supporting text; marked CPU work rounds as conceptual rather than cycles.
5. **PDF QA pass:** verified three pages, a 960×540-point 16:9 MediaBox, embedded
   DejaVu Sans fonts, complete 1280×720 previews, and no visible clipping or
   overlap.

## Technical references

- [CUDA Programming Guide: Introduction](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html)
- [CUDA Programming Guide: Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
