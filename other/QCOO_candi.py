from pyqpanda3.core import *
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
from pyqpanda3.quantum_info import Unitary


class RotationQHE:
    def __init__(self, n_qubits: int):
        """初始化QCOO方案"""
        self.n_qubits = n_qubits
        self.key = None  # 初始密钥
        self.final_key = None  # 最终密钥
        self.updated_gates = []  # 更新后的量子门列表
        self.init_state = None  # 初始量子态字符串
        self.qvm = CPUQVM()  # 构建量子虚拟机和量子电路

    def generate_key(self) -> Tuple[List[int], List[int]]:
        """生成量子一次一密密钥(a, b)"""
        # a = [random.randint(0, 1) for _ in range(self.n_qubits)]
        # b = [random.randint(0, 1) for _ in range(self.n_qubits)]
        a = [0, 1, 1]
        b = [1, 1, 0]
        self.key = (a, b)
        return self.key

    def encrypt(self, circuit: QCircuit, initial_state_str) -> Tuple[
        QCircuit, QCircuit, Tuple[List[int], List[int]]]:
        """
        使用量子一次一密加密明文量子态

        Args:
            circuit: 原始量子线路描述(QCircuit对象)
            initial_state: 初始量子态，默认为|00...0>

        Returns:
            encrypted_circuit: 加密后的量子电路
            homomorphic_circuit: 同态计算电路
            initial_key: 初始加密密钥
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
        # 处理量子门替换和密钥更新
        for op in operations:
            gate_type = op.name()
            targets = op.target_qubits()
            isdagger = op.is_dagger()

            if gate_type == "T":
                if not isdagger:
                    # T门替换为 Rz(π/4)
                    j = targets[0]
                    angle = np.pi / 4 if current_key[0][j] == 0 else -np.pi / 4
                    homomorphic_circuit << RZ(j, angle)
                    # T门不更新密钥
                else:
                    # T†门替换为 Rz(-π/4)
                    j = targets[0]
                    angle = -np.pi / 4 if current_key[0][j] == 0 else np.pi / 4
                    homomorphic_circuit << RZ(j, angle)
                    # T†门不更新密钥
            else:
                # 处理Clifford门并更新密钥
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
                    # 提取旋转角度参数
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
        根据Clifford门更新密钥

        Args:
            gate_type: 门类型
            current_key: 当前密钥(a, b)
            targets: 目标量子比特索引列表

        Returns:
            更新后的密钥(a, b)
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
            new_b[i], new_b[j] = new_b[i] ^ new_b[j] , new_b[j]
        elif gate_type == "CZ":
            # CZ门: 控制位i, 目标位j
            i, j = targets[0], targets[1]
            new_a[i], new_a[j] = new_a[i], new_a[i] ^ new_a[j]
            new_b[i], new_b[j] = new_b[i] ^ new_b[j], new_b[j]

        return (new_a, new_b)

    def homomorphic_compute(self, encrypted_circuit: QCircuit,
                            homomorphic_circuit: QCircuit) -> QCircuit:
        """
        对密文量子态进行同态计算

        Args:
            encrypted_circuit: 加密后的量子电路
            homomorphic_circuit: 同态计算电路

        Returns:
            同态计算后的量子电路
        """
        qc = QCircuit()
        qc << encrypted_circuit
        qc << homomorphic_circuit
        return qc

    def decrypt(self, computed_circuit: QCircuit) -> QCircuit:
        """
        使用最终密钥解密

        Args:
            computed_circuit: 同态计算后的量子电路

        Returns:
            解密后的量子电路
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
        模拟整个量子同态加密过程

        Args:
            circuit: 原始量子线路(QCircuit对象)
            initial_state_str: 初始量子态字符串，例如"001"
            shots: 模拟次数

        Returns:
            测量结果统计（字符串形式）
        """
        # 生成密钥
        self.generate_key()

        # 加密
        encrypted_circuit, updated_gates, _ = self.encrypt(circuit, initial_state_str)
        print('Enc:', draw_qprog(encrypted_circuit))
        # 同态计算
        computed_circuit = self.homomorphic_compute(encrypted_circuit, updated_gates)
        print('QHE:', draw_qprog(computed_circuit))
        # 解密
        decrypted_circuit = self.decrypt(computed_circuit)
        print('Dec:', draw_qprog(decrypted_circuit))
        matrix = Unitary(encrypted_circuit)
        # Print result
        print(matrix)
        # 添加测量
        # 添加测量操作
        prog = QProg()
        prog << decrypted_circuit
        for i in range(self.n_qubits):
            prog << measure(i, i)
        # 执行模拟
        self.qvm.run(prog, 1000)
        result = self.qvm.result().get_counts()
        # 转换结果格式
        counts = {}
        for key, value in result.items():
            # 反转比特顺序以匹配常规表示
            reversed_key = key[::-1]
            counts[reversed_key] = value

        return counts

    def state_str_to_vector(self, state_str: str) -> List[complex]:
        """
        将字符串形式的量子态转换为向量形式

        Args:
            state_str: 量子态字符串，例如"001"表示|001⟩

        Returns:
            对应的量子态向量
        """
        # 检查输入字符串是否合法
        if not all(c in ['0', '1'] for c in state_str):
            raise ValueError("量子态字符串只能包含'0'和'1'")

        n_qubits = len(state_str)
        index = int(state_str, 2)  # 将二进制字符串转换为整数索引

        # 创建对应的量子态向量
        state_vector = [0.0] * (2 ** n_qubits)
        state_vector[index] = 1.0

        return state_vector


def create_toffoli_circuit() -> QCircuit:
    """创建Toffoli门分解线路（返回QCircuit对象）"""
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


def run_toffoli_example():
    """运行Toffoli门量子同态加密示例"""
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

    # 打印结果
    for i, counts in enumerate(results):
        print(f"明文量子态 |{states[i]}>:")
        print(counts)
        print()

    # 可视化结果
    plt.figure(figsize=(15, 10))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 生成所有可能的三比特量子态
    all_states = [f"{i:03b}" for i in range(8)]

    for i, counts in enumerate(results):
        plt.subplot(2, 2, i + 1)

        # 确保每个子图都包含所有8种状态
        plot_counts = {state: counts.get(state, 0) for state in all_states}
        keys = sorted(plot_counts.keys())
        values = [plot_counts[key] for key in keys]

        plt.bar(keys, values)
        plt.title(f"|{states[i]}> 加密解密结果")
        plt.xlabel("量子态")
        plt.ylabel("计数")
        plt.xticks(rotation=45)  # 旋转x轴标签，使其更易读

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_toffoli_example()
