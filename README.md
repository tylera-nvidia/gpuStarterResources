# GPU Engineering Starter Resources

A compact index for engineers building, profiling, and optimizing CUDA software. This repository links to maintained primary material instead of reproducing it.

## Start here
- **[NVIDIA Training / Deep Learning Institute (DLI)](https://www.nvidia.com/en-us/training/find-training/?topics=accelerated+computing)** — hands-on, GPU-backed courses and workshops in CUDA and accelerated computing.
- **[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)** — programming model, CUDA C++, CUDA Python, execution, memory, and advanced features.
- **[NVIDIA Accelerated Computing Hub](https://github.com/NVIDIA/accelerated-computing-hub)** — open tutorials and user guides for GPU programming.


## Core references
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/) — current toolkit documentation and architecture tuning guides.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — performance and correctness guidance for CUDA C++.
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples) — maintained feature examples; not a validation or benchmarking suite.
- [CUDA Library Samples](https://github.com/NVIDIA/CUDALibrarySamples) — examples for CUDA-X math, image, signal-processing, and data-processing libraries.
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/) — memory, race, initialization, and synchronization checking.


## Performance workflow
1. Define correctness checks and a representative workload.
2. Prefer maintained libraries or primitives before writing a custom kernel.
3. Establish a reproducible baseline; record the GPU, driver, toolkit, problem size, and build configuration.
4. Use [Nsight Systems](https://docs.nvidia.com/nsight-systems/) to locate application-level bottlenecks.
5. Use [Nsight Compute](https://docs.nvidia.com/nsight-compute/) only on kernels that materially affect the workload; make one change and remeasure.


## Introductory examples
- [Example 1: vector multiplication and launch parallelism](examples/example_1/README.md)
- [Example 2: window operations and shared-memory exploration](examples/example_2/README.md)


## Detailed guides
- [CUDA Performance and Optimization Guide](docs/cuda-performance-guide.md) — GPU hardware, bottleneck analysis, optimization techniques, profiling workflows, and worked examples.

The documentation site is built from `docs/` with MkDocs. For a local preview:

```bash
python -m pip install --requirement requirements-docs.txt
mkdocs serve
```


## Hardware and architecture
- [Compute capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) — supported features and hardware limits by compute capability.
- Use the architecture tuning guides from the [CUDA documentation](https://docs.nvidia.com/cuda/) for architecture-specific behavior. Treat whitepapers as architectural overviews, not programming contracts.

### Architecture whitepapers and technical briefs

Public NVIDIA architecture papers for CUDA-capable GPUs, newest first:

- **Blackwell:** [Architecture technical overview](https://resources.nvidia.com/en-us-blackwell-architecture)
- **Hopper:** [H100 / GH100](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c)
- **Ada Lovelace:** [AD102](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
- **Ampere:** [A100 / GA100](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf) · [GA102](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf)
- **Turing:** [TU102](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf)
- **Volta:** [V100 / GV100](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)
- **Pascal:** [P100 / GP100](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper-v1.2.pdf) · [GTX 1080 / GP104](https://international.download.nvidia.com/geforce-com/international/pdfs/GeForce_GTX_1080_Whitepaper_FINAL.pdf)
- **Maxwell:** [GTX 750 Ti / GM107](https://www.nvidia.com/en-us/geforce/graphics-cards/geforce-gtx-750-ti/) · [GTX 980 / GM204](https://international.download.nvidia.com/geforce-com/international/pdfs/GeForce_GTX_980_Whitepaper_FINAL.PDF)
- **Kepler:** [GTX 680 / GK104](https://www.nvidia.com/content/PDF/product-specifications/GeForce_GTX_680_Whitepaper_FINAL.pdf) · [GK110/GK210](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/NVIDIA-Kepler-GK110-GK210-Architecture-Whitepaper.pdf)
- **Fermi:** [GF100](https://www.nvidia.com/content/PDF/fermi_white_papers/NVIDIAFermiComputeArchitectureWhitepaper.pdf)
- **Tesla:** [GeForce 8800 / G80 technical brief](https://www.nvidia.com/content/PDF/Geforce_8800/GeForce_8800_GPU_Architecture_Technical_Brief.pdf)

**Rubin:** NVIDIA has published an [official GPU architecture deep dive](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/), but not an architecture whitepaper.


## Discovery
These are useful for techniques and recorded talks. Confirm version-sensitive behavior against current documentation.
- [CUDA technical blog](https://developer.nvidia.com/blog/tag/cuda/)
- [NVIDIA On-Demand](https://www.nvidia.com/en-us/on-demand/)



## System configuration
- [Real-time GPU optimization notes](docs/guides/RealTimeTips.md) — prescriptive PCIe, NUMA, power, and kernel configuration guidance for RHEL, CentOS, and Rocky Linux systems.
- [GPU System Diagnostics](docs/guides/system-diagnostics.md) — short, read-only PCIe and NUMA checks.
