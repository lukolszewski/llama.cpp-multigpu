# machine-03-6xv100 — Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz + 8 x Tesla V100-SXM2-32GB (rented, Vast.ai)

Read from inside the benchmark container on **2026-10-07** by `scripts/multigpu/bench/vast/collect-hardware.py`.
`TBD` = not readable from a container without root (PCIe capability registers, motherboard, RAM modules).
Everything else is the driver's own report. Vast.ai machine/instance ids are in `hardware.json`.

## System

| field | value | source |
| --- | --- | --- |
| Host | Vast.ai rental, machine id TBD, instance TBD | environment |
| Container OS | Ubuntu 24.04.2 LTS (host OS not visible) | `/etc/os-release` |
| Kernel | `7.0.0-30-generic` | `uname -r` |
| CPU | Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz, 80 logical CPUs, 2 socket(s), AVX-512 yes | `lscpu` |
| NUMA | 2 node(s) | `lscpu` |
| RAM | 754.5 GiB visible (cgroup limit 777764470784); modules/speed TBD | `/proc/meminfo` |
| Motherboard | TBD (no DMI access in the container) | |
| NVIDIA driver | 580.173.02 | `nvidia-smi` |
| Driver-reported CUDA | 13.0 | `nvidia-smi` |
| Binaries | `BUILD_INFO.json` in the run directory (CUDA toolkit, SASS/PTX list, commits) | image |

## GPUs

| index | model | VRAM | compute cap. | PCI bus id | PCIe gen (max/current) | PCIe width (max/current) | power limit | ECC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Tesla V100-SXM2-32GB | 32768 MiB | 7.0 | `00000000:1A:00.0` | 3/3 | 16/16 | 275.00 W | Enabled |
| 1 | Tesla V100-SXM2-32GB | 32768 MiB | 7.0 | `00000000:1B:00.0` | 3/3 | 16/16 | 275.00 W | Enabled |
| 2 | Tesla V100-SXM2-32GB | 32768 MiB | 7.0 | `00000000:3D:00.0` | 3/3 | 16/16 | 275.00 W | Enabled |
| 3 | Tesla V100-SXM2-32GB | 32768 MiB | 7.0 | `00000000:3E:00.0` | 3/3 | 16/16 | 275.00 W | Enabled |
| 4 | Tesla V100-SXM2-32GB | 32768 MiB | 7.0 | `00000000:88:00.0` | 3/3 | 16/16 | 275.00 W | Enabled |
| 5 | Tesla V100-SXM2-32GB | 32768 MiB | 7.0 | `00000000:89:00.0` | 3/3 | 16/16 | 275.00 W | Enabled |
| 6 | Tesla V100-SXM2-32GB | 32768 MiB | 7.0 | `00000000:B2:00.0` | 3/3 | 16/16 | 275.00 W | Enabled |
| 7 | Tesla V100-SXM2-32GB | 32768 MiB | 7.0 | `00000000:B3:00.0` | 3/3 | 16/16 | 275.00 W | Enabled |

**Aggregate VRAM: 262144 MiB (256 GiB).** PCIe gen/width "current" is what the driver
reported at collection time (idle GPUs often sit at Gen1 in P8; the "max" column is the link capability).

### Interconnect

`nvidia-smi topo -m`:

```
	[4mGPU0	GPU1	GPU2	GPU3	GPU4	GPU5	GPU6	GPU7	NIC0	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	NV1	NV1	NV2	NV2	SYS	SYS	SYS	NODE	0-19,40-59	0		N/A
GPU1	NV1	 X 	NV2	NV1	SYS	NV2	SYS	SYS	NODE	0-19,40-59	0		N/A
GPU2	NV1	NV2	 X 	NV2	SYS	SYS	NV1	SYS	NODE	0-19,40-59	0		N/A
GPU3	NV2	NV1	NV2	 X 	SYS	SYS	SYS	NV1	NODE	0-19,40-59	0		N/A
GPU4	NV2	SYS	SYS	SYS	 X 	NV1	NV1	NV2	SYS	20-39,60-79	1		N/A
GPU5	SYS	NV2	SYS	SYS	NV1	 X 	NV2	NV1	SYS	20-39,60-79	1		N/A
GPU6	SYS	SYS	NV1	SYS	NV1	NV2	 X 	NV2	SYS	20-39,60-79	1		N/A
GPU7	SYS	SYS	SYS	NV1	NV2	NV1	NV2	 X 	SYS	20-39,60-79	1		N/A
NIC0	NODE	NODE	NODE	NODE	SYS	SYS	SYS	SYS	 X 				

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_2
```

`nvidia-smi nvlink -s`:

```
GPU 0: Tesla V100-SXM2-32GB (UUID: GPU-2b5fbbe0-802e-d2fd-c501-69ec52b11bb7)
	 Link 0: 25.781 GB/s
	 Link 1: 25.781 GB/s
	 Link 2: 25.781 GB/s
	 Link 3: 25.781 GB/s
	 Link 4: 25.781 GB/s
	 Link 5: 25.781 GB/s
GPU 1: Tesla V100-SXM2-32GB (UUID: GPU-14f80677-5867-613f-c2c9-c7226182bbfe)
	 Link 0: 25.781 GB/s
	 Link 1: 25.781 GB/s
	 Link 2: 25.781 GB/s
	 Link 3: 25.781 GB/s
	 Link 4: 25.781 GB/s
	 Link 5: 25.781 GB/s
GPU 2: Tesla V100-SXM2-32GB (UUID: GPU-a1583f16-043d-1e74-7cf9-f7382ef8e78d)
	 Link 0: 25.781 GB/s
	 Link 1: 25.781 GB/s
	 Link 2: 25.781 GB/s
	 Link 3: 25.781 GB/s
	 Link 4: 25.781 GB/s
	 Link 5: 25.781 GB/s
GPU 3: Tesla V100-SXM2-32GB (UUID: GPU-4c7a4f3c-97ba-1058-f449-87e058101e32)
	 Link 0: 25.781 GB/s
	 Link 1: 25.781 GB/s
	 Link 2: 25.781 GB/s
	 Link 3: 25.781 GB/s
	 Link 4: 25.781 GB/s
	 Link 5: 25.781 GB/s
GPU 4: Tesla V100-SXM2-32GB (UUID: GPU-da78fed9-7297-75b5-cbfe-ae1a756e3872)
	 Link 0: 25.781 GB/s
	 Link 1: 25.781 GB/s
	 Link 2: 25.781 GB/s
	 Link 3: 25.781 GB/s
	 Link 4: 25.781 GB/s
	 Link 5: 25.781 GB/s
GPU 5: Tesla V100-SXM2-32GB (UUID: GPU-72ed3bc4-b1e1-12c5-3709-2b564db7ce0b)
	 Link 0: 25.781 GB/s
	 Link 1: 25.781 GB/s
	 Link 2: 25.781 GB/s
	 Link 3: 25.781 GB/s
	 Link 4: 25.781 GB/s
	 Link 5: 25.781 GB/s
GPU 6: Tesla V100-SXM2-32GB (UUID: GPU-152dfe6e-ca09-ee82-637b-9dc9651b4c47)
	 Link 0: 25.781 GB/s
	 Link 1: 25.781 GB/s
	 Link 2: 25.781 GB/s
	 Link 3: 25.781 GB/s
	 Link 4: 25.781 GB/s
	 Link 5: 25.781 GB/s
GPU 7: Tesla V100-SXM2-32GB (UUID: GPU-b0d45f0b-6922-8af3-6b29-8826de10b235)
	 Link 0: 25.781 GB/s
	 Link 1: 25.781 GB/s
	 Link 2: 25.781 GB/s
	 Link 3: 25.781 GB/s
	 Link 4: 25.781 GB/s
	 Link 5: 25.781 GB/s
```
