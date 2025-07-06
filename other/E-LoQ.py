import numpy as np
from pyqpanda3.core import *
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional


class ELoQ:
    def __init__(self, seed: int = 42):
        """初始化E-LoQ锁定器，设置随机数种子"""
        self.rng = np.random.RandomState(seed)
        self.qvm = CPUQVM()

    def generate_key(self, key_bits: int) -> np.ndarray:
        """生成n位随机密钥"""
        return self.rng.randint(0, 2, key_bits)

    def encrypt_circuit(self, original_circuit: QCircuit, key: np.ndarray) -> QCircuit:
        """
        对原始电路进行E-LoQ加密
        Args:
            original_circuit: 原始量子电路
            key: 密钥数组，如np.array([1,0,1])表示3位密钥
        Returns:
            加密后的量子电路
        """
        n_qubits = original_circuit.number_of_qubits()
        key_qubit = self.qvm.allocate_qubit()  # 分配密钥量子比特
        locked_circuit = QCircuit()

        # 将原始电路添加到锁定电路
        locked_circuit << original_circuit

        n_key_bits = len(key)
        n_ones = np.sum(key)
        n_zeros = n_key_bits - n_ones

        # 检查密钥位数是否超过门数（文档中要求n_ones <= 门数）
        original_gates = original_circuit.get_operations()
        if n_ones > len(original_gates):
            raise ValueError("密钥中1的位数超过原始电路门数，需重新生成密钥")

        # 随机选择n_ones个门转换为受控门
        gates_to_convert = self.rng.choice(len(original_gates), n_ones, replace=False)
        gate_indices = sorted(gates_to_convert)

        # 插入哈达玛门掩码（H-masking）
        for i in range(n_key_bits):
            locked_circuit << H(key_qubit)

        # 转换原始门为受控门并插入虚拟门
        virtual_gate_positions = []
        current_pos = 0

        for i, is_one in enumerate(key):
            if is_one:
                # 转换原始门为受控门（由密钥量子比特控制）
                gate_idx = gate_indices[current_pos]
                original_gate = original_gates[gate_idx]
                controlled_gate = self._convert_to_controlled_gate(original_gate, key_qubit)
                locked_circuit.replace_operation(gate_idx + i, controlled_gate)
                current_pos += 1
            else:
                # 插入虚拟受控门（文档中红色标记的门）
                virtual_gate = self._create_dummy_controlled_gate(n_qubits, key_qubit)
                locked_circuit.insert_operation(i + current_pos, virtual_gate)
                virtual_gate_positions.append(i + current_pos)

        return locked_circuit, key_qubit, virtual_gate_positions

    def _convert_to_controlled_gate(self, gate: QGate, control_qubit: Qubit) -> QGate:
        """将单量子比特门转换为受密钥量子比特控制的受控门"""
        if isinstance(gate, XGate):
            return CNOT(control_qubit, gate.get_qubit_list()[0])
        elif isinstance(gate, HGate):
            # 文档中提到使用CNOT和H组合实现受控H门
            circ = QCircuit()
            circ << CNOT(control_qubit, gate.get_qubit_list()[0]) << H(gate.get_qubit_list()[0])
            return circ
        # 其他门类型的处理...
        return gate  # 默认为原门

    def _create_dummy_controlled_gate(self, n_qubits: int, control_qubit: Qubit) -> QGate:
        """创建虚拟受控门（文档中由|0⟩控制的恒等门）"""
        # 选择一个随机量子比特作为目标
        target_qubit = self.qvm.allocate_qubit()
        dummy_gate = CNOT(control_qubit, target_qubit)
        self.qvm.free_qubit(target_qubit)
        return dummy_gate

    def decrypt_circuit(self, locked_circuit: QCircuit, key: np.ndarray,
                        key_qubit: Qubit, virtual_gate_positions: List[int]) -> QCircuit:
        """
        对锁定电路进行解密
        Args:
            locked_circuit: 加密后的电路
            key: 正确密钥
            key_qubit: 密钥量子比特
            virtual_gate_positions: 虚拟门位置列表
        Returns:
            解密后的电路
        """
        decrypted_circuit = locked_circuit.copy()

        # 移除哈达玛门并插入Pauli-X门（文档中解密步骤）
        for i, is_one in enumerate(key):
            decrypted_circuit.remove_operation(i)  # 移除H门
            if i > 0 and key[i] != key[i - 1]:
                decrypted_circuit.insert_operation(i, X(key_qubit))

        # 移除虚拟门（文档中红色标记的门，由|0⟩控制时为恒等门）
        for pos in sorted(virtual_gate_positions, reverse=True):
            decrypted_circuit.remove_operation(pos)

        # 移除密钥量子比特相关操作
        key_gate_indices = [i for i, gate in enumerate(decrypted_circuit.get_operations())
                            if key_qubit in gate.get_qubit_list()]
        for pos in sorted(key_gate_indices, reverse=True):
            decrypted_circuit.remove_operation(pos)

        # 简化电路（文档中提到的受控门替换为独立门）
        simplified_circuit = self._simplify_circuit(decrypted_circuit)
        return simplified_circuit

    def _simplify_circuit(self, circuit: QCircuit) -> QCircuit:
        """简化电路，移除恒等操作"""
        simplified = QCircuit()
        for gate in circuit.get_operations():
            if not isinstance(gate, IGate):
                simplified << gate
        return simplified

    def simulate_circuit(self, circuit: QCircuit, shots: int = 1000) -> Dict[str, int]:
        """使用pyqpanda3仿真电路并返回测量结果"""
        prog = QProg(circuit)
        self.qvm.run(prog, 1000)
        result = self.qvm.result().get_counts()
        return result

    def calculate_tvd(self, result1: Dict[str, int], result2: Dict[str, int], shots: int) -> float:
        """计算总变异距离（TVD）"""
        all_keys = set(result1.keys()).union(set(result2.keys()))
        tvd = 0.0
        for key in all_keys:
            cnt1 = result1.get(key, 0)
            cnt2 = result2.get(key, 0)
            tvd += abs(cnt1 - cnt2)
        return tvd / (2 * shots)

    def calculate_hvd(self, result1: Dict[str, int], result2: Dict[str, int],
                      shots: int, n_bits: int) -> float:
        """计算汉明变异距离（HVD）"""
        hvd = 0.0
        for key1, cnt1 in result1.items():
            for key2, cnt2 in result2.items():
                if key1 == key2:
                    continue
                hamming_dist = sum(int(a) != int(b) for a, b in zip(key1, key2))
                hvd += hamming_dist * abs(cnt1 - cnt2)
        return hvd / (2 * shots)

    def calculate_dfc(self, result: Dict[str, int], correct_output: str, shots: int) -> float:
        """计算功能损坏度（DFC）"""
        correct_cnt = result.get(correct_output, 0)
        incorrect_cnt = shots - correct_cnt
        return (correct_cnt - incorrect_cnt) / shots

    def evaluate_security(self, locked_circuit: QCircuit, key: np.ndarray,
                          original_result: Dict[str, int], shots: int = 1000) -> float:
        """评估密钥猜测率"""
        n_key_bits = len(key)
        total_guesses = 2 ** n_key_bits
        guess_rate = 0.0

        for guess in range(total_guesses):
            guess_key = np.array([(guess >> i) & 1 for i in range(n_key_bits)])
            decrypted = self.decrypt_circuit(locked_circuit, guess_key, None, [])
            guess_result = self.simulate_circuit(decrypted, shots)
            tvd = self.calculate_tvd(guess_result, original_result, shots)
            guess_rate += tvd / total_guesses

        return guess_rate


# 使用示例
def demo_eloq():
    # 初始化E-LoQ
    eloq = ELoQ()

    # 创建原始电路（以1-bit加法器为例）
    original = QCircuit()
    original << H(0) << CNOT(0, 1) << CNOT(0, 2)

    # 生成3位密钥
    key = eloq.generate_key(3)
    print(f"生成的密钥: {key}")

    # 加密电路
    locked_circuit, key_qubit, virtual_gates = eloq.encrypt_circuit(original, key)

    # 仿真原始电路
    original_result = eloq.simulate_circuit(original)
    print("原始电路结果:", original_result)

    # 用正确密钥解密
    decrypted_circuit = eloq.decrypt_circuit(locked_circuit, key, key_qubit, virtual_gates)
    decrypted_result = eloq.simulate_circuit(decrypted_circuit)
    print("正确解密后结果:", decrypted_result)

    # 计算TVD
    tvd = eloq.calculate_tvd(original_result, decrypted_result, 1000)
    print(f"正确密钥TVD: {tvd:.4f}")

    # 用错误密钥解密（如翻转最后一位）
    wrong_key = key.copy()
    wrong_key[-1] = 1 - wrong_key[-1]
    wrong_decrypted = eloq.decrypt_circuit(locked_circuit, wrong_key, key_qubit, virtual_gates)
    wrong_result = eloq.simulate_circuit(wrong_decrypted)
    print("错误密钥解密结果:", wrong_result)

    # 计算错误密钥TVD
    wrong_tvd = eloq.calculate_tvd(original_result, wrong_result, 1000)
    print(f"错误密钥TVD: {wrong_tvd:.4f}")

    # 评估安全性（密钥猜测率）
    security_score = eloq.evaluate_security(locked_circuit, key, original_result)
    print(f"密钥猜测率: {security_score:.4f}")

    # 释放量子比特
    eloq.qvm.free_qubit(q0)
    eloq.qvm.free_qubit(q1)
    eloq.qvm.free_qubit(c)
    eloq.qvm.free_qubit(key_qubit)


if __name__ == "__main__":
    demo_eloq()