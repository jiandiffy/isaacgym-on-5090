# Isaac Gym on RTX 5090（Blackwell / sm_120）使用指南

## 1. 背景

**GeForce RTX 5090 的 CUDA Compute Capability 为 12.0（对应 `sm_120`）**。([NVIDIA Developer][1])
很多 PyTorch 的稳定版二进制（wheel）在编译时未包含 `sm_120`，因此在 5090 上会出现典型报错，例如：

* `NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation`
* `RuntimeError: no kernel image is available for execution on the device`

5090 上使用 Isaac Gym 的主要障碍通常不是 Isaac Gym 本身，而是 **PyTorch wheel 与 GPU 架构 / CUDA 版本不匹配**。只要把 PyTorch 换成“包含 `sm_120` 的构建”，Isaac Gym 大多可正常工作。

---

## 2. 前置条件

* Linux（Ubuntu 20.04/22.04/24.04 皆可）
* NVIDIA Driver 已正确安装，`nvidia-smi` 可识别 RTX 5090
* 建议使用 **Conda** 单独管理一个 Python 环境（避免 `conda + venv` 叠加）

> Isaac Gym 的 Python 绑定通常是 `gym_38`，建议使用 **Python 3.8**。

---

## 3. 创建环境与基础依赖

```bash
conda create -n isaacgym python=3.8 -y
conda activate isaacgym

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
```

## 4. 安装 PyTorch

### 路线 A：直接安装仓库提供的 sm_120 PyTorch wheel

1. 安装仓库内提供的 `.whl`（示例路径请按你的仓库实际位置修改）：

```bash
python -m pip install ./wheels/torch-2.3.0a0+git63d5e92-cp38-cp38-linux_x86_64.whl
# 如果你也提供了 torchvision/torchaudio 的匹配 whl，则同样安装对应文件
```

2. 验证：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))  # 期望 (12, 0)
PY
```

只要 `capability` 输出为 `(12, 0)` 且 `cuda available: True`，说明你当前 PyTorch 已可在 5090 上正确运行。
---

### 路线 B：从 PyTorch 官方仓库源码编译（最可控）

> 适用于：你需要完全可复现的构建、或仓库未提供可用 wheel、或你需要特定 commit / patch。

1. 获取 PyTorch 源码：

```bash
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch
```

2. 关键：指定 `sm_120` 架构（否则编出来的 PyTorch 仍可能不含 5090 支持）：

```bash
export TORCH_CUDA_ARCH_LIST="sm_120"
```

PyTorch 社区明确给出：通过源码编译并设置 `TORCH_CUDA_ARCH_LIST` 可启用 `sm_120` 支持。([PyTorch Forums][2])

3. 编译并安装（在当前 conda 环境）：

```bash
python -m pip install -r requirements.txt
python setup.py develop
```

4. 验证：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY
```

---

## 5. 安装与验证 Isaac Gym

假设你的 Isaac Gym 路径为 `<ISAACGYM_ROOT>`（包含 `python/isaacgym`）：

```bash
cd <ISAACGYM_ROOT>/python
python -c "from isaacgym import gymapi; print('isaacgym import ok')"
```

运行一个示例验证 viewer：

```bash
python examples/simple_viewer.py
```

