import os
import glob
import pandas as pd
from pyqpanda3.core import *
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
from pyqpanda3.quantum_info import Unitary
from pyqpanda3.intermediate_compiler import convert_qprog_to_qasm, convert_qasm_string_to_qprog
import re

pd.options.mode.chained_assignment = None  # default='warn'
from QCSO import QCSO
from QCOO import QCOOManager
from pyqpanda3.intermediate_compiler import convert_qasm_file_to_qprog
import time


class ECQCOManager:
    """
    A class to manage quantum circuits in ECQCO, including loading, parsing, and processing QASM files.

    Attributes:
        program_path (str): The path to the QASM file.
        QCOO_program_path (str): THe path to the QCOO QASM file.
        result_path (str): The path where results will be saved.
        circuit: The quantum circuit object (loaded from the QASM file).
    """

    def __init__(self, program_path, QCOO_program_path, new_program_path, init_states=None):
        self.program_path = program_path
        self.result_path = new_program_path
        self.circuit = QCOO_program_path
        self.init_state = init_states
        self.tvd = None
        self.cpu_duration = 0

    def parse_qasm_file(self, file_path):
        """
        解析QASM文件，返回量子比特数并检查格式

        参数:
        file_path (str): QASM文件路径

        返回:
        tuple: (量子比特数, 错误信息列表)
        """
        errors = []
        qubit_count = 0

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # 提取量子寄存器声明
            qreg_pattern = re.compile(r'qreg\s+(\w+)\[(\d+)\]\s*;')
            qregs = []

            for line in lines:
                line = line.strip()
                if not line or line.startswith('//'):
                    continue  # 跳过空行和注释

                # 匹配量子寄存器声明
                match = qreg_pattern.match(line)
                if match:
                    qreg_name, size = match.groups()
                    qregs.append((qreg_name, int(size)))
                    continue

                # 检查基本语法错误（非完整检查，仅示例）
                if '(' in line and ')' not in line:
                    errors.append(f"Syntax error: Unclosed parentheses - {line}")
                if '[' in line and ']' not in line:
                    errors.append(f"Syntax error: Unclosed square brackets - {line}")

            # 计算总量子比特数
            if not qregs:
                errors.append("No quantum register declaration found")
            else:
                qubit_count = sum(size for _, size in qregs)

            return qubit_count, errors

        except FileNotFoundError:
            return 0, [f"The file does not exist：{file_path}"]
        except Exception as e:
            return 0, [f"Parsing error：{str(e)}"]

    def total_variation_distance(self, p, q):
        """
        Calculate the Total Variation Distance (TVD) between two probability distributions P and Q.

        Args:
            p (array-like): Ideal probability distribution.
            q (array-like): Real experiment's probability distribution.

        Returns:
            float: Total Variation Distance.
        """
        # Get all possible quantum states
        all_keys = set(p.keys()).union(set(q.keys()))

        # Normalized probability distribution
        norm_dist1 = self.normalize_distribution(p)
        norm_dist2 = self.normalize_distribution(q)

        # Calculate the tvd
        tvd = 0
        for key in all_keys:
            p1 = norm_dist1.get(key, 0.0)
            p2 = norm_dist2.get(key, 0.0)
            tvd += abs(p1 - p2)

        return tvd / 2.0

    def normalize_distribution(self, distribution):
        """
        Normalize the values in a distribution so that they sum up to 1.

        Args:
            distribution (dict): A dictionary where keys are categories
            and values are numeric counts or frequencies.

        Returns:
            dict: A new dictionary where the values are normalized
        """
        total_count = sum(distribution.values())
        return {key: value / total_count for key, value in distribution.items()}

    def ECQCO(self, shots):
        n_qubits, errors = self.parse_qasm_file(self.program_path)
        if errors:
            print("发现错误：")
            for error in errors:
                print(f"- {error}")
        # get init program result
        init_prog = convert_qasm_file_to_qprog(self.program_path)
        machine = CPUQVM()
        machine.run(init_prog, shots)
        baseline_count = machine.result().get_counts()
        start = time.process_time()  # CPU时间（秒）
        # apply QCOO
        qcoo = QCOOManager(n_qubits, self.program_path, result_path)
        QCOO_path, final_key = qcoo.QCOO(self.init_state)
        # apply QCSO
        ecqco_circ, ECQCO_count, depth_without_single, depth = QCSO(QCOO_path, self.result_path, lamb_param=2,
                                                                    total_shots=shots)
        end = time.process_time()
        self.cpu_duration = end - start
        print(f"运行时长：{self.cpu_duration:.6f} 秒")
        # evaluation
        self.tvd = self.total_variation_distance(baseline_count, ECQCO_count)


if __name__ == '__main__':
    PREFIX_PATH = "benchmarks/"
    filelist = glob.glob(os.path.join(PREFIX_PATH, '*.qasm'))
    QCOO_PATH = 'result/QCOO/'
    result_PATH = 'result'

    for program_ in filelist:
        path, program_name = os.path.split(program_)
        input_path = os.path.join(PREFIX_PATH, program_name)
        QCOO_program_path = os.path.join(QCOO_PATH, program_name)
        result_path = os.path.join(result_PATH, program_name)
        ecqco = ECQCOManager(input_path, QCOO_program_path, result_path)
        ecqco.ECQCO(10000)
