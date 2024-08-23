# Introduction

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

## Architecture White Papers
An Architecture white paper is released with every new GPU architecture, and is the "ground truth" for HW changes and it's associated capability changes. 

- [Blackwell White Paper](https://resources.nvidia.com/en-us-blackwell-architecture)
- [Hopper White Paper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper)
- [Ampere 102 White Paper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)
- [Ampere 100 White Paper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
# Accelerate Computing Basics & Goals

## What is Accelerated Computing
## When is my program a candidate for acceleration

# GPU HW Overview
## Types of HW 
comparison of HW diagrams from whitepapers
### Compute HW Resources
- CUDA cores
- Tensor cores
- RT Cores
- Special Function Unit
### Memory HW Resources
- Global Memory
- L1/L2 Cache
- Shared Memory
- Constant Memory
- Registers
- Textures
### Operation Specific Dedicated HW
- Copy Engines
- Codecs
## CUDA work breakdown to HW
HOW GPU COMPUTING WORKS Stephen Jones, GTC 2021: 64-66
## Balancing HW to Maximize Performance
HOW CUDA PROGRAMMING WORKS Stephen Jones, GTC 2022: 49-84
## Generational Changes in GPU HW
Compute Capabilites Doc: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

# CUDA Programming Overview
## The CUDA Programming Guide
## Host Vs Device Programming
### CUDA Host API
### CUDA Device API
### CUDA Math Libraries 
## Writing Software to Maximize HW
### GPU Resource Management
### Concurrency  


# Key Performance Drivers and Bottlenecks
# Speed of Light
## Memory Bound vs Compute Bound
## GPU Occupancy
How To Write A CUDA Program:THE NINJA EDITION Stephen Jones, NVIDIA | GTC 2024 : 12-28
## Stalls / Expensive Operations
### types of stalls
  https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference
### Most common stalls
- Short/Long Scoreboard Stall
- Stall MIO  Throttle
- Stall Math Throttle
- Stall Wait / not selected
## Latency vs Throughput

# CUDA Profiling Tools Overview
## Nsight Systems
### Interpretation Guidance
## Nsight Compute 
### Interpretation Guidance
## Nsight Graphics
### Interpretation Guidance
## Suggested use patterns and scripting

