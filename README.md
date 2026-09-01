# ECQCO: Encrypted-State Quantum Compilation

[English](#english) | [中文](#中文)

<a id="english"></a>

## English

This repository contains the research prototype for **ECQCO**, the encrypted-state quantum compilation scheme presented in:

> C. Zhang, T. Shang, X. Guo, and Y. Zhang, "Encrypted-State Quantum Compilation Scheme Based on Quantum Circuit Obfuscation for Quantum Cloud Platforms," *IEEE Transactions on Quantum Engineering*, vol. 7, pp. 1-18, 2026. [https://doi.org/10.1109/TQE.2026.3659096](https://doi.org/10.1109/TQE.2026.3659096)

ECQCO combines quantum circuit output obfuscation (QCOO) with quantum circuit structure obfuscation (QCSO). The released implementation uses PyQPanda3 for QASM processing and simulation, and MindQuantum for circuit scheduling and adaptive decoupling obfuscation.

### Artifact scope

The public prototype includes:

- QOTP-based circuit output obfuscation and key updates in `QCOO.py`;
- circuit scheduling and structure obfuscation in `QCSO.py`;
- the adaptive decoupling obfuscation algorithm (ADOA) in `scripts/circuit_od.py`;
- QASM benchmarks, result-processing utilities, and implementation tests.

For local simulation, the decryption gates are appended before the circuit is passed to QCSO. This ordering simplifies the local experimental workflow and does not change the encryption/decryption relation underlying the ECQCO construction described in the paper.

The production quantum-cloud integration code, including a separated client/cloud execution workflow, is part of ongoing follow-up research and is not provided at this stage. The complete PTD, normGED, and paper-figure reproduction pipeline is also outside the scope of this public release. 

### Repository structure

- `ECQCO.py`: main local-simulation entry point and evaluation workflow.
- `QCOO.py`: encrypted-state generation, QOTP key generation, and key updates.
- `QCSO.py`: QCSO scheduling, execution, and result generation.
- `scripts/circuit_od.py`: ADOA and decoupling-gate insertion.
- `scripts/utils.py`: QASM, instruction-table, simulation, and evaluation utilities.
- `benchmarks/`: QASM inputs used by the default smoke run.
- `benchmarks_all/`: additional benchmark circuits.
- `other/`: implementations of candidate comparison schemes.
- `result/`: generated QASM, tables, counts, and intermediate artifacts.
- `figure/` and `draw_fig.py`: experimental figures and plotting utilities.
- `tests/test_ecqco.py`: implementation tests for the released code.

### Environment

The current prototype was tested on Windows with:

- Python 3.11.14
- PyQPanda3 0.3.2
- MindQuantum 0.12.0
- SymPy 1.14.0
- NumPy 2.4.0
- pandas 2.3.3
- Matplotlib 3.10.8

Create and activate a conda environment:

```powershell
conda create -n ecqco python=3.11 -y
conda activate ecqco
python -m pip install pyqpanda3==0.3.2 mindquantum==0.12.0 sympy==1.14.0 numpy==2.4.0 pandas==2.3.3 matplotlib==3.10.8
```

On Windows, MindQuantum requires a compatible `win_amd64` wheel. If importing MindQuantum reports that `mindquantum.mqbackend` is missing, reinstall the CPython 3.11 Windows wheel from the [MindQuantum 0.12.0 files](https://pypi.org/project/mindquantum/0.12.0/#files).

### Quick start

Run the local ECQCO simulation from the repository root:

```powershell
conda activate ecqco
python ECQCO.py
```

`ECQCO.py` processes every `.qasm` file under `benchmarks/`. The included configuration runs the Toffoli example with 10,000 shots and writes generated artifacts under `result/`. QOTP keys are generated randomly, so encrypted circuits and sampled counts can vary between runs.

Run the implementation tests with:

```powershell
conda activate ecqco
python -m unittest discover -s tests -p "test_ecqco.py" -v
```

The Toffoli smoke run and all six tests completed in the environment listed above.

### Citation

If this repository supports your research, please cite the paper:

```bibtex
@article{zhang2026ecqco,
  author  = {Chenyi Zhang and Tao Shang and Xueyi Guo and Yuanjing Zhang},
  title   = {Encrypted-State Quantum Compilation Scheme Based on Quantum Circuit Obfuscation for Quantum Cloud Platforms},
  journal = {IEEE Transactions on Quantum Engineering},
  volume  = {7},
  pages   = {1--18},
  year    = {2026},
  doi     = {10.1109/TQE.2026.3659096}
}
```

### Research-use note

This repository is a research artifact rather than a production quantum-cloud security library. Security and correctness claims should be interpreted under the system model and assumptions stated in the paper.

<a id="中文"></a>

## 中文

本仓库包含 **ECQCO** 的研究原型，对应论文：

> C. Zhang, T. Shang, X. Guo, and Y. Zhang, "Encrypted-State Quantum Compilation Scheme Based on Quantum Circuit Obfuscation for Quantum Cloud Platforms," *IEEE Transactions on Quantum Engineering*, vol. 7, pp. 1-18, 2026. [https://doi.org/10.1109/TQE.2026.3659096](https://doi.org/10.1109/TQE.2026.3659096)

ECQCO 将量子电路输出混淆（QCOO）与量子电路结构混淆（QCSO）相结合。公开实现使用 PyQPanda3 完成 QASM 处理与仿真，并使用 MindQuantum 完成电路调度和自适应解耦混淆。

### 公开代码范围

当前公开原型包括：

- `QCOO.py` 中基于 QOTP 的电路输出混淆与密钥更新；
- `QCSO.py` 中的电路调度与结构混淆；
- `scripts/circuit_od.py` 中的自适应解耦混淆算法（ADOA）；
- QASM 基准电路、结果处理工具和实现级测试。

在本地仿真中，解密门会在电路送入 QCSO 之前追加。这一处理用于简化本地实验流程，不改变论文所述 ECQCO 构造中的加密与解密关系。

真实量子云平台的集成代码，包括客户端与云端分离的执行流程，涉及后续研究，现阶段暂不提供。完整的 PTD、normGED 和论文图表复现流程也不在本次公开范围内。

### 仓库结构

- `ECQCO.py`：本地仿真的主入口与评估流程。
- `QCOO.py`：加密态生成、QOTP 密钥生成与密钥更新。
- `QCSO.py`：QCSO 调度、执行与结果生成。
- `scripts/circuit_od.py`：ADOA 与解耦门插入。
- `scripts/utils.py`：QASM、指令表、仿真与评估工具。
- `benchmarks/`：默认示例使用的 QASM 输入。
- `benchmarks_all/`：其他基准电路。
- `other/`：候选对比方案的实现。
- `result/`：生成的 QASM、指令表、计数结果和中间文件。
- `figure/` 与 `draw_fig.py`：实验图和绘图工具。
- `tests/test_ecqco.py`：公开代码的实现级测试。

### 运行环境

当前原型已在以下 Windows 环境中测试：

- Python 3.11.14
- PyQPanda3 0.3.2
- MindQuantum 0.12.0
- SymPy 1.14.0
- NumPy 2.4.0
- pandas 2.3.3
- Matplotlib 3.10.8

创建并激活 conda 环境：

```powershell
conda create -n ecqco python=3.11 -y
conda activate ecqco
python -m pip install pyqpanda3==0.3.2 mindquantum==0.12.0 sympy==1.14.0 numpy==2.4.0 pandas==2.3.3 matplotlib==3.10.8
```

在 Windows 中，MindQuantum 需要兼容的 `win_amd64` wheel。如果导入 MindQuantum 时提示缺少 `mindquantum.mqbackend`，请从 [MindQuantum 0.12.0 文件列表](https://pypi.org/project/mindquantum/0.12.0/#files)重新安装 CPython 3.11 对应的 Windows wheel。

### 快速开始

在仓库根目录运行本地 ECQCO 仿真：

```powershell
conda activate ecqco
python ECQCO.py
```

`ECQCO.py` 会处理 `benchmarks/` 中的所有 `.qasm` 文件。当前配置使用 10,000 shots 运行 Toffoli 示例，并将生成文件写入 `result/`。QOTP 密钥为随机生成，因此不同运行中的加密电路和采样计数可能不同。

运行实现级测试：

```powershell
conda activate ecqco
python -m unittest discover -s tests -p "test_ecqco.py" -v
```

在上述环境中，Toffoli 基础运行和全部六项测试均已完成。

### 引用

如果本仓库对你的研究有所帮助，请引用：

```bibtex
@article{zhang2026ecqco,
  author  = {Chenyi Zhang and Tao Shang and Xueyi Guo and Yuanjing Zhang},
  title   = {Encrypted-State Quantum Compilation Scheme Based on Quantum Circuit Obfuscation for Quantum Cloud Platforms},
  journal = {IEEE Transactions on Quantum Engineering},
  volume  = {7},
  pages   = {1--18},
  year    = {2026},
  doi     = {10.1109/TQE.2026.3659096}
}
```

### 研究使用说明

本仓库是研究代码，不是面向生产环境的量子云安全库。有关安全性和正确性的结论应结合论文中的系统模型与假设进行理解。
