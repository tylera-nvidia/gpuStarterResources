# GPU System Diagnostics

Use these read-only checks to characterize a system before tuning it. Run performance measurements with a representative workload and compare against a recorded baseline.

## GPU and software

```bash
nvidia-smi --query-gpu=index,name,driver_version,pci.bus_id,compute_cap,pstate --format=csv
nvcc --version
```

## PCIe and topology

```bash
nvidia-smi topo -m
lspci -t
lspci -vv -s <GPU_BDF>
```

Compare `LnkCap` with `LnkSta` for the GPU and every bridge on its path. Link speed may downshift while idle, so confirm unexpected results under load.

## NUMA placement

```bash
numactl --hardware
lscpu
```

Keep latency-sensitive CPU threads and their memory on the NUMA node nearest the GPU, then verify the effect with the application workload.

## Change discipline

- Change one variable at a time and retain before/after measurements.
- Record firmware, driver, kernel, GPU, CPU, and benchmark configuration.
- Use system-vendor documentation for BIOS, power, IOMMU, and PCIe settings.
- Do not apply blanket kernel parameters, relaxed memory-access settings, disabled power management, or privileged containers as performance defaults.
