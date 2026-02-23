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

下面是**仅保留“方案 B：源码编译 PyTorch（面向 RTX 5090 / sm_120）”**的 `README.md` 版本，并严格按你给出的三步组织（在不改变要点的前提下，对个别细节做了更可执行的表述，如“按关键代码块定位而非固定行号”）。

---

### 4.1 克隆 PyTorch 源码到本地

```bash
git clone https://github.com/pytorch/pytorch    # 1) 克隆主仓库
cd pytorch
git checkout v2.3.1                             # 2) 切换到 v2.3.1
git submodule update --init --recursive         # 3) 初始化子模块
```

---


### 4.2 修改源码并编译

#### 4.2.1 修改 `select_compute_arch.cmake`

打开：

`pytorch/cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake`

定位到类似逻辑：

```cmake
elseif(${arch_name} STREQUAL "Hopper")
  set(arch_bin 9.0)
  set(arch_ptx 9.0)
else()
  message(SEND_ERROR "Unknown CUDA Architecture Name ${arch_name} in CUDA_SELECT_NVCC_ARCH_FLAGS")
endif()
```

将 `else()` 分支改为对 `12.0` ：

```cmake
elseif(${arch_name} STREQUAL "Hopper")
  set(arch_bin 9.0)
  set(arch_ptx 9.0)
else()
  set(arch_bin 12.0)
  set(arch_ptx 12.0)
endif()
```

---

#### 4.2.2 修改 `pytorch/Dockerfile`

打开：

`pytorch/Dockerfile`

找到包含 `TORCH_CUDA_ARCH_LIST=...` 的内容：

```bash
TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX 8.0" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
```

在末尾追加 `12.0`：

```bash
TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX 8.0 12.0" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
```

> 说明：即使你不在 Docker 内编译，这一步按你的要求保留；实际本机编译仍以第 3.3 节的环境变量为准。

---

### 4.3 开始编译

```bash
conda activate isaacgym
cd pytorch

export USE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
export MAX_JOBS=5

python setup.py bdist_wheel
```

---

### 4.4 编译产物安装与验证

编译成功后，wheel 通常在 `dist/` 下：

```bash
ls -lh dist/*.whl
python -m pip install --force-reinstall dist/torch-*.whl
```

验证 CUDA 可用与设备能力：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY
```


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

