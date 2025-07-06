import os
import glob
import pandas as pd
from mindquantum.algorithm.compiler import DAGCircuit
from scripts import QCManager
from scripts import DDManager
from pyqpanda3.core import *
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
from pyqpanda3.quantum_info import Unitary
from pyqpanda3.intermediate_compiler import convert_qprog_to_qasm, convert_qasm_string_to_qprog
pd.options.mode.chained_assignment = None  # default='warn'



class ECQCOManager:
    """
    A class to manage quantum circuits in ECQCO, including loading, parsing, and processing QASM files.

    Attributes:
        program_path (str): The path to the QASM file.
        result_path (str): The path where results will be saved.
        circuit: The quantum circuit object (loaded from the QASM file).
    """
    def __init__(self, program_path, new_program_path):
        self.program_path = program_path
        self.result_path = new_program_path
        self.circuit = None





if __name__ == '__main__':
    PREFIX_PATH = "benchmarks/"
    filelist = glob.glob(os.path.join(PREFIX_PATH, '*.qasm'))
    NEW_PATH = 'result/'

    for program_ in filelist:
        path, program_name = os.path.split(program_)
        each_program_path = os.path.join(NEW_PATH, program_name)
