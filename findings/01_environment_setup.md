# Environment Setup

**Date:** 2026-06-20

## System

| Component | Value |
|-----------|-------|
| OS | Ubuntu 24.04 LTS |
| Python | 3.14.5 (system), 3.12.13 (project venv) |
| GPU | AMD Radeon RX 7900 GRE (gfx1100) |
| ROCm | 7.2.4 |
| Package manager | `uv` |
| Project root | `/home/lightdesk/Downloads/Projects/Percepta` |

## Virtual Environment

```bash
uv venv --python 3.12.13
source .venv/bin/activate
```

Python 3.12 was chosen over the system 3.14.5 because PyTorch has broader compatibility with 3.12.

## Dependencies

**PyTorch for ROCm** — installed from AMD's official WHL repo, NOT PyTorch nightly:

```
https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/
  torch-2.9.1+rocm7.2.4.lw.git39497456-cp312-cp312-linux_x86_64.whl
  torchvision-0.24.0+rocm7.2.4.gitb919bd0c-cp312-cp312-linux_x86_64.whl
  triton-3.5.1+rocm7.2.4.gita272dfa8-cp312-cp312-linux_x86_64.whl
```

Standard dependencies installed via `uv pip install`:
- scipy, matplotlib, numpy

## Verification

```python
import torch
print(torch.cuda.is_available())       # True
print(torch.cuda.get_device_name(0))   # Radeon RX 7900 GRE
print(torch.__version__)               # 2.9.1+rocm7.2.4.git39497456
```
