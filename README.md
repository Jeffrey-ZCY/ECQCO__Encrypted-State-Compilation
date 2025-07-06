# Encrypted-State Compilation Scheme Based on Quantum Circuit Obfuscation

## 1. Framework description

This project uses **MindSpore Quantum** and **Pyqpanda** to implement a encrypted compilation framework ECQCO. This framework utilizes the concepts of quantum homomorphic encryption and quantum indistinguishability obfuscation. By encrypting the output of quantum circuits and obfuscating their structures, it achieves end - to - end security of quantum information in the quantum cloud, effectively protecting users' intellectual property rights.

## 2. Project file structure

- `ECQCO.py`：The main entrance of the project, including core logics such as initializing parameters, simulating circuits, and generating circuits.
- `QCOO.py`：Encrypt the quantum state after the action of the quantum circuit to achieve obfuscation of untrusted quantum clouds and third - party compilers.
- `QCSO.py`：The main entrance of the project, including core logics such as initializing parameters, simulating circuits, and generating circuits.
- `draw_fig.py`：Responsible for drawing images of experimental results.

- `benchmarks/`：Store QASM files for quantum circuit simulation.
- `figure/`：Store the images generated from the experimental results.
- `result/`：Store the output data and intermediate results of the experimental simulation run.
- `other/`：Implementation of quantum cryptography tools for other candidate solutions that are unrelated to the solutionscripts/
  - `__init__.py`：Initialization file.
  - `circuit_od.py`：The main processing logic of obfuscation decoupling.
  - `utils.py`：Some auxiliary tool functions for quantum circuit processing.
