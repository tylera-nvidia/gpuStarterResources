# Introduction

## Table of Contents:
- [Educational Resources](#educational-resources)
- [GPU HW Overview](#gpu-hw-overview)
- [Key Performance Drivers and Bottlenecks](#key-performance-drivers-and-bottlenecks)
- [Techniques for Accelerating Code](#techniques-for-accelerating-code)
- [CUDA Profiling Tools Overview](#cuda-profiling-tools-overview)
- [Optimization Examples](#optimization-examples)


<!----------------------------  Resources Section  ---------------------------->
# Educational Resources
## CUDA Programming Guide
The CUDA Documentation contains best practices and a programming guide that establishs many of the patterns that result in optimal CUDA performance. This is a critical reference and daily resource for any level of developer.

[CUDA Documentation](https://docs.nvidia.com/cuda/)

[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)


## NVIDIA On Demand
NVIDIA On Demand is a great resource for video format learning.
This contains recordings of all past GTC (GPU Technology Conference) talks, covering topics from begginers to the most advanced.  

[NVIDIA On Demand](https://www.nvidia.com/en-us/on-demand/)

### Suggested Talks

## NVIDIA Blogs
NVIDIA Blogs is the written format equivilant of our GTC talks. most blogs focus on a single technical topic and provide a detailed tutorial, write-up, or demonstration of the given technology. Simiarly to the GTC talks, topics can range from deeply technincal to introductory, on topics from CUDA to precise applications in their scientific field.

[NVIDIA Blog Front page](https://developer.nvidia.com/blog/)

### Suggested Blogs
- [Asychronous CUDA Memory](https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/)
- [cuBLASLt Optimization](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/)

## Architecture White Papers
An Architecture white paper is released with every new GPU architecture, and is the "ground truth" for HW changes and it's associated capability changes. 

- [Blackwell White Paper](https://resources.nvidia.com/en-us-blackwell-architecture)
- [Hopper White Paper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper)
- [Ampere 102 White Paper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)
- [Ampere 100 White Paper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)

<!----------
# Accelerate Computing Basics & Goals
## What is Accelerated Computing
## When is my program a candidate for acceleration
----------->

<!----------------------------  HW Section  ----------------------------------->
# GPU HW Overview
## Types of HW 
comparison of HW diagrams from whitepapers
### Compute HW Resources
| GA100                                |                                 GA102   |
| --------                             |                                 ------- |
| ![image](images/GA100_SM_BLOCK.png)  |  ![image](images/GA102_SM_BLOCK.png)    |

[Ampere Compute Capability Core Description](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-8-x)
- CUDA Cores 
- Tensor Cores
- Special Function Unit
- Raytracing Cores
   - [Optix Documentation](https://raytracing-docs.nvidia.com/optix8/index.html)
### Memory HW Resources

- HW Memory Types and Hierarchy
  
  [Technical Blog on Memory](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/)
  - Global(Device) Memory
  - L1/L2 Cache
  - Shared Memory
  - Registers
    
- Specialty Memory Types
  
  [CUDA Documentation on Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)
  - Constant Memory
  - Textures
  - Local Memory

- Memory Alignment and Optimization

  [CUDA Documentation on Memory Alignment](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)


### Operation Specific Dedicated HW
- Copy Engines
- Codecs

## CUDA work breakdown to HW
[HOW GPU COMPUTING WORKS Stephen Jones, GTC 2021: 64-66](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31151/)
## Balancing HW to Maximize Performance
[HOW CUDA PROGRAMMING WORKS Stephen Jones, GTC 2022: 49-84](https://www.nvidia.com/en-us/on-demand/session/gtcfall22-a41101/)
## Generational Changes in GPU HW
Compute Capabilites Doc: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

<!----------------------------  SoL Section  ---------------------------------->
# Key Performance Drivers and Bottlenecks
## Speed of Light
NVIDIA Concept of "How fast can something possibly be done". Said another way, "what is the absolute limit of performance for a given problem on a given device".

Very useful to use SoL as a metric to understand your current solution. To calcualte SoL, you need to understand the ype of compute you have, the type and size of data it requries, and the relatiosnhip of the data and compute relative to each data. This, combined with the possible FLOPS/bandwidth of a system can establish a possible SoL for a given problem. This SoL can then be used as a metric for your own solutions. 

Often we use CUB or other highly optimized CUDA libraries as a default "SoL", however in many cases those may still be some percent off true SoL.

## Memory Bound vs Compute Bound
Fundimentally defines: is my performance limited by the speed of my memory, or the speed of my processor? 
- Compute Bound 
  
  When the compute HW can not compute results for a given set of memory faster than the memory can retrieve new memory

- Memory Bound:

  When the Memory bandwidth cannot provide new memory input within the time it takes for compute to complete a computation for a given segment of memory
  
  
### Example Calcuation FP32 10x10 convolution kernel

```
4096*4096 image with a 10x10 kernel 
H100
51200 GFLOPS FP32
2.04TB/s

each pixel requires 100multiples + 99 adds, 199 opers per pixel 

16.77M pixels in image

3.355 GFLOP per image

GPU can do 51,200 GFLOPS/s

GPU can process 15,250 images per second

each images is 16.77M FP32 numbers (4 Bytes) 
67MB per image

need 1.023 TB/s of bandwidth (< 2TB/s HW can do)

Compute Limited!

```

### Example Calcuation 5x5 convolution kernel FFT

```
4096*4096 image with a 5x5 kernel 
H100
51,200 GFLOPS FP32
2.04TB/s

each pixel requires 25 multiples + 24 adds, 49 opers per pixel 

16.77M pixels in image

0.822 GFLOP per image

GPU can do 51200 GFLOPS/s

GPU can process 62,287 images per second

each images is 16.77M FP32 numbers (4 Bytes) 
67MB per image

need 4.18 TB/s of bandwidth (>2TB/s HW can do)

Bandwidth Limited!
```

This example is very "macroscopic" and real-world performance is further limited by global memory latencies, efficiency of workbreakdown, or other factors of the GPU execution. This is still a useful baseline for understanding the performance limiters we can expect in the real kernel, as we will see scaled down versions of these bottlenecks in practice. 

## Latency vs Throughput
[HOW GPU COMPUTING WORKS Stephen Jones, GTC 2021: 41-55](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31151/)

## GPU Occupancy
[How To Write A CUDA Program:THE NINJA EDITION Stephen Jones, NVIDIA | GTC 2024 : 12-28](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62401/)

<!---------------------------  Acceleration Section  -------------------------->
# Techniques for Accelerating Code
## Optimizing Data Allocations 
- Using the correct type of memory
- Creating Plans and Data allocations ahead of time
## Optimizing Data Movement
- Conscientious Copies 
- Pinned vs un-Pinned copies
- Impact of Types of CUDA Memory

## Host APIs vs Device APIs
most CUDA libraries have versions that you can call directly from the host ([cuFFT](https://docs.nvidia.com/cuda/cufft/index.html)), and a version you can call while on the device ([cufftDx](https://docs.nvidia.com/cuda/cufftdx/index.html))

In General, the Host API is intended as the easier to use, Good-to-SoL performance entrypoint for a developer. for "good" sizes and sufficiently large work, the host API can provide the best performance. 

In many scenarios however, you may have problem sizes that are too small to saturate a GPU, have awkward and non-performant sizes, or simply be a non-optimized corner case for the host API. In these situations, the device API enables developers to create a more precise launch configuration that appropriately matches the problem. This also allows developers to fuse other operations with the core mathmatical operation, providing additional speed up. 

## CUDA Streams 
  - [Streams Blog](https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/)
  - [Stream Docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)

## CUDA Graphs
  - [Graphs Blog](https://developer.nvidia.com/blog/cuda-graphs/)
  - [Graphs Docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
  
## Kernel Stalls 
Kernel stalls are the distinct, instruction-level impact of memory, compute, or other resource contention that slows down a kernel. By evaluating (and eliminating) kernel stalls, we can further accelerlate a well-designed kernel.


  **disclaimer: This level of analysis should only be conducted once you have refined a good kernel-level solution. If you have poorly thought-out or optimized data accesss or algorithmic patterns, optimizing at the instruction level will result in mediocre speedups on a poor design. This level of optimization cannot fundimentally correct those higher level issues.**
  
  
### types of stalls
  [Table of Stalls from NCU Documentation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#id31)
### Most common stalls
- Short/Long Scoreboard Stall
- Stall MIO  Throttle
- Stall Math Throttle
- Stall Wait / not selected

## Kernel Expensive Operations
- Atomics
- Trig
- Syncrhonization 


<!----------------------------  Profiling Section  ---------------------------->
# CUDA Profiling Tools Overview

## Nsight Systems
[Nsight Systems Landing Page](https://developer.nvidia.com/nsight-systems)

[Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/index.html)

Two Functionality to be familiar with:
1. Command Line collection (nsys)
2. In-App Review (Full Windows App)

### Interpretation Guidance
(In Person Demonstration)

## Nsight Compute 
[Nsight Compute Landing Page](https://developer.nvidia.com/nsight-compute)

[Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

Two Functionality to be familiar with:
1. Command Line collection (ncu)
2. In-App Review (Full Windows App)

### Interpretation Guidance
(In Person Demonstration)
## ~~Nsight Graphics~~
### ~~Interpretation Guidance~~

## Profiling & Optimizing CUDA Math Libraries

### CUTLASS Profiler
- [Documentation on Profiling CUTLASS](https://github.com/NVIDIA/cutlass/wiki/Performance-Profiling)
### Suggested use patterns and scripting
<!--- Make a Flow Diagram later-->
- Start with Host API
- move plan and setup code out of critical loop as much as possible
- Evaluate performance relative to Speed of Light
- determine if 

## nsys/Nsight Systems Examples
Below are a set of common, useful command options. They can be combined and enabled all in a single report.

| Command                                         | Notes                                                                     |
| ------------------------------------------------| --------------------------------------------------------------------------|
| `nsys profile -o outputName ./myExec`           | basic command. will not automaticall overwrite, requires `-f` flag        |
| `nsys profile --gpu-metrics-device=0 ./myExec`  | Adds the GPU Metrics section to the report. requires Elevated Permissions |
| `nsys profile --cuda-graph-trace=node ./myExec` | Shows Kernel information internal to CUDA Graph Node                      |
| `nsys profile --cuda-memory-usage=true ./myExec`| Adds GPU memory usage section to report                                   |

## ncu/Nsight Compute Examples
Below are a set of common, useful command options. They can be combined and enabled all in a single report. I suggest collecting the "full" set, unless you know you want a specific subset of the report.

| Command                                         | Notes                                                                                  |
| ------------------------------------------------| ---------------------------------------------------------------------------------------|
| `ncu --set=full  -o outputName ./myExec`                     | basic command. will not automaticall overwrite, requires `-f` flag        |
| `ncu --set=full --import-source=true -o outputName ./myExec` | adds source collection. requires `-lineinfo` in compilation               |
| `ncu --set=full -k kernelName -o outputName ./myExec`        | Only collects profile for a specific kernel in the exection               |


<!----------
### System Configuration for full profiling
`Notes on System setup`
----------->

<!----------------------------  Profiling Section  ---------------------------->
# Optimization Examples
## Example 1: Low Occupancy and Increasing Parallelism
- [Example 1](examples/example_1/README.md)

## Example 2: Reducing Register Pressure with Shared Memory 
- [Example 2](examples/example_2/README.md)

## Example 3: Optimizing Plan Creation for cuFFT
- [FFT Sizing Benchmark](https://github.com/tylera-nvidia/fftSizing)

## Example 4: using cuBlasLT Auto Tuning
- [cuBLASLt Optimization](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/)
- [CUDA Library Samples](https://github.com/tylera-nvidia/CUDALibrarySamples/tree/master/cuBLASLt/LtSgemmSimpleAutoTuning)

<!---
## Example 5: Memory Transfer Optimizations
-->