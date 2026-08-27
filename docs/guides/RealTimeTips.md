# Optimization Tips for Real time Data Processing Processing
This is a repository to store lessons-learned in configuring and optimizing systems for real-time data processing, specifically with GPUs. There is no warranty or expectation that this information will ensure any reader's best performance or a definitive source of guidance, but more a starting point to prevent the authors from losing past knowledge over time. This guide is Linux-centric, specifically for RHEL/Centos/Rocky.


# System Configuration Tips

## PCIe Configuration & Optimization
This section outlines guidance on how to ensure you maximize throughput and reduce latency for host to device transfers. Inefficiencies can spawn from both host (system) configurations, as well as the GPU itself. 
### Link Training and Performance
The system will determine the Link capability of the system at power on through link training. see [this](https://www.ti.com/lit/an/snla415/snla415.pdf?ts=1702067722917&ref_url=https%253A%252F%252Fwww.google.com%252F) TI paper on the process details.

Link training occurs are part of the boot sequence, with the determined link capability viewable from the system.

Users can query link capability at:
- nvidia-smi -q
    * Below is an example output from nvidia-smi showing a Gen4 system with all 16 links working 
        ```
        GPU Link Info
            PCIe Generation
                Max                       : 4
                Current                   : 4
                Device Current            : 4
                Device Max                : 4
                Host Max                  : 4
            Link Width
                Max                       : 16x
                Current                   : 16x
        ```
- lscpi -vvv (requires sudo to see PCIe info)

### Check PCIe Path 
Excessive switches, or a poor path to the PCIe root complex can negatively impact performance. In situations where PCIe devices may be hosted from expansion chassis or be a part of a converged device, endpoint PCIe capability may exceed intermediate capability, so it is useful to check the total path of a given transfer to verify the total path's link speed.

- lspci -t -v shows the device name, PCIe address, and an ASCII graph of the path to the root.


## System Memory Configuration & Optimization
System memory can bottleneck transfers out to PCIe if poorly configured or aligned to operations. 
### Memory Channels
The utility ``dmidecode -t memory ``  shows how many memory channels are populated. If memory is poorly configured in the system, this may significantly impact both latency and total bandwidth. Check the Speed and Rank of each memory to ensure they're operating at the expected HW limit.

### NUMA / Core Affinity
If the system is based on AMD EPYC, NPS != 1 may cause memory bandwidth tp be limited to the number of channels affine to a NUMA domain.

If the system has multiple NUMA nodes (multi socket or NPS != 1) and the application does not run on the domain affine to the GPU, you could see a performance hit. In that case you can use taskset or numactl to pin the processes to the correct domain

On Supermicro, several BIOS settings effect performance:
- Intel Single Root I/O Virtualization Enable/Disable

You can find NUMA option at BIOS menu \ACPI Settings. <-- defaults is enabled.
If you enable the BIOS menu \PCIe/PCI/PnP Configuration \ASPM Support then it will enable the Intel SR-I/O, but there is a limitation. The PCIe slot link to PCH doesn't support ACS feature. The PCIe slot from CPUs does support ACS feature.

## Power Modes
 In SuperMicro Chassis BIOS under CPU Configuration :: Advance Power Management Configuration,  set `Power Technology=Disable` 

## Kernel Configuration & Optimization

### Kernel Command Line
The kernel command line ``/proc/cmdline`` has options that effect on performance. specific options are:
 - ``pci=pcie_bus_perf``
 - ``iommu=pt``
 - ``iommu.passthrough=1``
 - ``iomem=[strict|relaxed]``
 
Kernel documentation [here](https://www.kernel.org/doc/html/v6.2/admin-guide/kernel-parameters.html) for more info and options.

IOMMU specific documentation can he found [here](https://www.kernel.org/doc/html/latest/driver-api/vfio.html#groups-devices-and-iommus).

On AMD64 systems you need to pass ``iommu=pt`` in the kernel command line to enable IOMMU passthrough

On ARM64 systems you need to pass ``iommu.passthrough=1`` in the kernel command line to enable IOMMU passthrough

Before enabled  ``iomem=relaxed``, you can check if the PCIe devices support relaxed through ```lspci -vvv```
- RlxdOrd+ means enabled
- RlxdOrd- means disabled





