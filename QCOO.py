from pyqpanda3.core import *
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
from pyqpanda3.quantum_info import Unitary
from pyqpanda3.intermediate_compiler import convert_qprog_to_qasm, convert_qasm_string_to_qprog


class RotationQHE:
    def __init__(self, n_qubits: int):
        """init QCOO"""
        self.n_qubits = n_qubits
        self.key = None  # Initial key
        self.final_key = None  # final key
        self.updated_gates = []
        self.init_state = None
        self.qvm = CPUQVM()

    def generate_key(self) -> Tuple[List[int], List[int]]:
        """Generate quantum one-time pad key (a, b)"""
        # a = [random.randint(0, 1) for _ in range(self.n_qubits)]
        # b = [random.randint(0, 1) for _ in range(self.n_qubits)]
        a = [1, 0, 1]
        b = [0, 1, 0]
        self.key = (a, b)
        return self.key

    def encrypt(self, circuit: QCircuit, initial_state_str) -> Tuple[
        QCircuit, QCircuit, Tuple[List[int], List[int]]]:
        """
        Use quantum one-time pad to encrypt the plaintext quantum state
        Args:
            circuit: The description of the original quantum circuit (QCircuit object)
            initial_state: The initial quantum state, default is |00...0>
        Returns:
            encrypted_circuit: The encrypted quantum circuit
            homomorphic_circuit: The homomorphic computation circuit
            initial_key: The initial encryption key
        """
        # 创建初始态
        self.init_state = initial_state_str
        # 构建量子虚拟机和量子电路
        homomorphic_circuit = QCircuit()

        # 初始化量子比特和经典比特,创建初始量子电路
        encrypted_circuit = QCircuit(self.n_qubits)
        # 若为空，则重置全0态为初始量子态
        qubit = range(self.n_qubits)
        # 制备初始态的量子线路，采用基态编码
        cir_encode = Encode()
        cir_encode.basic_encode(qubit, self.init_state)
        encrypted_circuit << cir_encode.get_circuit()

        # 应用量子一次一密加密 X^a Z^b
        a, b = self.key
        initial_key = (a.copy(), b.copy())
        current_key = (a.copy(), b.copy())

        for i in range(self.n_qubits):
            if a[i] == 1:
                encrypted_circuit << X(i)
            if b[i] == 1:
                encrypted_circuit << Z(i)
        operations = circuit.gate_operations()

        # Handle quantum gate replacement and key update
        for op in operations:
            gate_type = op.name()
            targets = op.target_qubits()
            isdagger = op.is_dagger()

            if gate_type == "T":
                if not isdagger:
                    j = targets[0]
                    angle = np.pi / 4 if current_key[0][j] == 0 else -np.pi / 4
                    homomorphic_circuit << RZ(j, angle)

                else:
                    j = targets[0]
                    angle = -np.pi / 4 if current_key[0][j] == 0 else np.pi / 4
                    homomorphic_circuit << RZ(j, angle)
                    # T/T†门 gates does not update the key.
            else:
                # Handle Clifford gates and update the key
                if gate_type == "H":
                    homomorphic_circuit << H(targets[0])
                elif gate_type == "X":
                    homomorphic_circuit << X(targets[0])
                elif gate_type == "Z":
                    homomorphic_circuit << Z(targets[0])
                elif gate_type == "S":
                    homomorphic_circuit << S(targets[0])
                elif gate_type == "CNOT":
                    homomorphic_circuit << CNOT(targets[0], targets[1])
                elif gate_type == "CZ":
                    homomorphic_circuit << CZ(targets[0], targets[1])
                elif gate_type.startswith("U3"):
                    # Extract rotation angle parameters
                    params = op.parameters()
                    if len(params) == 3:
                        theta, phi, lam = params
                        homomorphic_circuit << RZ(targets[0], lam)
                        homomorphic_circuit << RY(targets[0], theta)
                        homomorphic_circuit << RZ(targets[0], phi)

                current_key = self.update_key(gate_type, current_key, targets)

        self.final_key = current_key
        return encrypted_circuit, homomorphic_circuit, initial_key

    def update_key(self, gate_type: str, current_key: Tuple[List[int], List[int]],
                   targets: List[int]) -> Tuple[List[int], List[int]]:
        """
        Update the key according to the Clifford gate.
        Args:
            gate_type: The type of the gate.
            current_key: The current key (a, b).
            targets: The list of target qubit indices.
        Returns:
            The updated key (a, b).
        """
        a, b = current_key
        new_a = a.copy()
        new_b = b.copy()

        if gate_type in ["X", "Y", "Z"]:
            # X、Y、Z门不改变密钥
            pass
        elif gate_type == "H":
            # 哈德玛门交换a和b
            j = targets[0]
            new_a[j], new_b[j] = new_b[j], new_a[j]
        elif gate_type == "S":
            # S门: b[j] = a[j] XOR b[j]
            j = targets[0]
            new_b[j] = a[j] ^ b[j]
        elif gate_type == "CNOT":
            # CNOT门: 控制位i, 目标位j
            i, j = targets[0], targets[1]
            new_a[i], new_a[j] = new_a[i], new_a[i] ^ new_a[j]
            new_b[i], new_b[j] = new_b[i] ^ new_b[j], new_b[j]
        elif gate_type == "CZ":
            # CZ门: 控制位i, 目标位j
            i, j = targets[0], targets[1]
            new_a[i], new_a[j] = new_a[i], new_a[i] ^ new_a[j]
            new_b[i], new_b[j] = new_b[i] ^ new_b[j], new_b[j]

        return (new_a, new_b)

    def homomorphic_compute(self, encrypted_circuit: QCircuit,
                            homomorphic_circuit: QCircuit) -> QCircuit:
        """
        Perform homomorphic computation on the ciphertext quantum state
        Args:
            encrypted_circuit: The encrypted quantum circuit
            homomorphic_circuit: The homomorphic computation circuit
        Returns:
            The quantum circuit after homomorphic computation
        """
        qc = QCircuit()
        qc << encrypted_circuit
        qc << homomorphic_circuit
        prog = QProg()
        prog << qc
        qasm = convert_qprog_to_qasm(prog)
        print('qasm:\n', qasm)
        return qc

    def decrypt(self, computed_circuit: QCircuit) -> QCircuit:
        """
        Decrypt using the final key
        Args:
            computed_circuit: The quantum circuit after homomorphic computation
        Returns:
            The decrypted quantum circuit
        """
        qc = QCircuit()
        qc << computed_circuit

        a_final, b_final = self.final_key

        for i in range(self.n_qubits):
            if b_final[i] == 1:
                qc << Z(i)
            if a_final[i] == 1:
                qc << X(i)

        return qc

    def simulate(self, circuit: QCircuit, initial_state_str: str = None, shots: int = 1024) -> dict:
        """
        Simulate the entire quantum homomorphic encryption process
        Args:
            circuit: The original quantum circuit (QCircuit object)
            initial_state_str: The string of the initial quantum state, such as "001"
            shots: The number of simulation times
        Returns:
            The measurement result statistics (in string form)
        """
        # Generate a key
        self.generate_key()
        # Encrypt
        encrypted_circuit, updated_gates, _ = self.encrypt(circuit, initial_state_str)
        print('Enc:', draw_qprog(encrypted_circuit))
        # Homomorphic computation
        computed_circuit = self.homomorphic_compute(encrypted_circuit, updated_gates)
        print('QHE:', draw_qprog(computed_circuit))
        # Decrypt
        decrypted_circuit = self.decrypt(computed_circuit)
        print('Dec:', draw_qprog(decrypted_circuit))
        matrix = Unitary(encrypted_circuit)
        # Print result
        print(matrix)
        # Add measurement operations
        prog = QProg()
        prog << decrypted_circuit
        for i in range(self.n_qubits):
            prog << measure(i, i)
        # Execute simulation
        self.qvm.run(prog, 1000)
        result = self.qvm.result().get_counts()
        qasm = convert_qprog_to_qasm(prog)
        print('qasm:\n', qasm)
        # Format of conversion results
        counts = {}
        for key, value in result.items():
            # Reverse the bit order to match the regular representation
            reversed_key = key[::-1]
            counts[reversed_key] = value

        return counts

    def state_str_to_vector(self, state_str: str) -> List[complex]:
        """
        Convert the quantum state in string form to vector form.
        Args:
            state_str: The string of the quantum state, for example, "001" represents |001⟩
        Returns:
            The corresponding quantum state vector
        """
        # Check if the input string is legal
        if not all(c in ['0', '1'] for c in state_str):
            raise ValueError("量子态字符串只能包含'0'和'1'")

        n_qubits = len(state_str)
        index = int(state_str, 2)

        state_vector = [0.0] * (2 ** n_qubits)
        state_vector[index] = 1.0

        return state_vector


def create_toffoli_circuit1() -> QCircuit:
    """
    Create a circuit for the decomposition of the Toffoli gate
    return a QCircuit object
    """
    qc = QCircuit()
    qc << H(2) << T(0).dagger() << T(1) << T(2)
    qc << CNOT(0, 1) << CNOT(2, 0) << CNOT(1, 2)
    qc << T(0).dagger()
    qc << CNOT(1, 0)
    qc << T(0).dagger() << T(1).dagger() << T(2)
    qc << CNOT(2, 0)
    qc << S(0)
    qc << CNOT(1, 2)
    qc << H(2)
    qc << CNOT(0, 1)

    return qc


def create_toffoli_circuit() -> QCircuit:
    """
    Create a circuit for Toffoli gate decomposition
    return a QCircuit object
    """
    qc = QCircuit()
    qc << H(2) << T(0) << T(1)
    qc << CNOT(2, 0) << CNOT(1, 2)
    qc << T(0).dagger()
    qc << CNOT(1, 0)
    qc << T(2).dagger()
    qc << CNOT(1, 2)
    qc << T(0)
    qc << CNOT(2, 0)
    qc << T(0).dagger() << T(2) << H(2)
    qc << CNOT(1, 0)

    return qc


def QCOO():
    """Run the quantum homomorphic encryption example of the Toffoli gate"""
    n_qubits = 3
    qhe = RotationQHE(n_qubits)
    # 创建Toffoli门分解线路
    toffoli_circuit = create_toffoli_circuit()

    # 测试不同的明文量子态（字符串形式）
    states = ["011"]

    results = []
    for state_str in states:
        counts = qhe.simulate(toffoli_circuit, state_str)
        results.append(counts)

    for i, counts in enumerate(results):
        print(f"明文量子态 |{states[i]}>:")
        print(counts)
        print()

    # Visualize the results
    plt.figure(figsize=(15, 10))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    all_states = [f"{i:03b}" for i in range(8)]

    for i, counts in enumerate(results):
        plt.subplot(2, 2, i + 1)

        # Ensure that each sub-plot contains all 8 states
        plot_counts = {state: counts.get(state, 0) for state in all_states}
        keys = sorted(plot_counts.keys())
        values = [plot_counts[key] for key in keys]

        plt.bar(keys, values)
        plt.title(f"|{states[i]}> 加密解密结果")
        plt.xlabel("量子态")
        plt.ylabel("计数")
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    QCOO()
