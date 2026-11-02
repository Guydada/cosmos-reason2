# Jetson AGX Orin

Cosmos-Reason2 installs on Jetson AGX Orin with JetPack 7 (L4T R39, Ubuntu 24.04, CUDA 13) through the standard `cu130` extra. No Orin-specific extra, package index or system library is needed.

> **JetPack 6 is not covered.** Its CUDA 12.6 PyTorch wheels come from the [Jetson AI Lab index](https://pypi.jetson-ai-lab.io/jp6/cu126) and need extra setup that this repository does not maintain. Upgrade to JetPack 7.

## Requirements

| Component | Version |
| --- | --- |
| Hardware | Jetson AGX Orin |
| JetPack | 7.x (L4T R39.x) |
| CUDA | 13.0 or newer, from the JetPack driver |
| Python | 3.12 (system `python3` on Ubuntu 24.04) |

Check your system:

```shell
cat /etc/nv_tegra_release  # R39
nvidia-smi                 # CUDA Version: 13.x
```

## Install

Complete the [Setup](../README.md#setup) prerequisites (clone, uv, Hugging Face login), then:

```shell
uv sync --extra cu130
source .venv/bin/activate
```

The PyTorch CUDA 13 aarch64 wheels ship `sm_80` kernels, which run on Orin (`sm_87`). A system CUDA toolkit is not required. Triton uses its bundled `ptxas`, which supports `sm_87`, so do not set `TRITON_PTXAS_PATH`.

## Verify

```shell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 2.9.0+cu130 True Orin
```

Then follow [Inference](../README.md#inference). Model inference on Orin has not been validated yet.

## Troubleshooting

| Error | Solution |
| --- | --- |
| `no kernel image is available for execution on the device` | The installed PyTorch has no Orin-compatible kernels, for example a CUDA 12 wheel from another index. Recreate the environment: `rm -rf .venv && uv sync --extra cu130`. |
| `Free memory less than desired` | Orin shares memory between CPU and GPU. Close other GPU processes or lower `--gpu-memory-utilization`. |
