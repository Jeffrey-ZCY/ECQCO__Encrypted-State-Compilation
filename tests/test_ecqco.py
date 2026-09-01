"""ECQCO 核心逻辑的回归测试。"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path, stub_modules):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stub_modules):
        spec.loader.exec_module(module)
    return module


def pyqpanda_stubs():
    core = types.ModuleType('pyqpanda3.core')
    core.QCircuit = type('QCircuit', (), {})
    quantum_info = types.ModuleType('pyqpanda3.quantum_info')
    quantum_info.Unitary = object
    compiler = types.ModuleType('pyqpanda3.intermediate_compiler')
    compiler.convert_qprog_to_qasm = lambda program: ''
    compiler.convert_qasm_string_to_qprog = lambda qasm: None
    compiler.convert_qasm_file_to_qprog = lambda path: None
    return {
        'pyqpanda3': types.ModuleType('pyqpanda3'),
        'pyqpanda3.core': core,
        'pyqpanda3.quantum_info': quantum_info,
        'pyqpanda3.intermediate_compiler': compiler,
    }


def mindquantum_stubs():
    compiler = types.ModuleType('mindquantum.algorithm.compiler')
    compiler.DAGCircuit = object
    io = types.ModuleType('mindquantum.io')
    io.OpenQASM = object
    simulator = types.ModuleType('mindquantum.simulator')
    simulator.Simulator = object
    simulator.decompose_stabilizer = lambda element: None
    circuit = types.ModuleType('mindquantum.core.circuit')
    circuit.Circuit = type('Circuit', (), {})
    gates = types.ModuleType('mindquantum.core.gates')
    for gate_name in ('X', 'Y', 'Z', 'RX', 'U3', 'Measure', 'I', 'RZ', 'H',
                      'PhaseDampingChannel', 'DepolarizingChannel'):
        setattr(gates, gate_name, object())
    error_mitigation = types.ModuleType('mindquantum.algorithm.error_mitigation')
    error_mitigation.query_single_qubit_clifford_elem = lambda index: None
    scripts = types.ModuleType('scripts')
    scripts.QCManager = object
    scripts.QCSOManager = object
    sympy = types.ModuleType('sympy')
    sympy.sympify = float
    return {
        'mindquantum': types.ModuleType('mindquantum'),
        'mindquantum.algorithm': types.ModuleType('mindquantum.algorithm'),
        'mindquantum.algorithm.compiler': compiler,
        'mindquantum.algorithm.error_mitigation': error_mitigation,
        'mindquantum.io': io,
        'mindquantum.simulator': simulator,
        'mindquantum.core': types.ModuleType('mindquantum.core'),
        'mindquantum.core.circuit': circuit,
        'mindquantum.core.gates': gates,
        'scripts': scripts,
        'sympy': sympy,
    }


class TestQCOOFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module('qcoo_under_test', ROOT / 'QCOO.py', pyqpanda_stubs())

    def test_generate_key_matches_number_of_qubits(self):
        manager = self.module.QCOOManager.__new__(self.module.QCOOManager)
        manager.n_qubits = 5
        randint = mock.Mock(side_effect=[0, 1, 0, 1, 1, 1, 0, 1, 0, 0])
        fake_random = types.SimpleNamespace(randint=randint)

        with mock.patch.object(self.module, 'random', fake_random, create=True):
            a, b = manager.generate_key()

        self.assertEqual(a, [0, 1, 0, 1, 1])
        self.assertEqual(b, [1, 0, 1, 0, 0])

    def test_cz_keeps_x_mask_and_cross_updates_z_mask(self):
        manager = self.module.QCOOManager.__new__(self.module.QCOOManager)
        cases = [
            (([1, 0], [0, 0]), ([1, 0], [0, 1])),
            (([0, 1], [0, 0]), ([0, 1], [1, 0])),
            (([1, 1], [1, 0]), ([1, 1], [0, 1])),
        ]

        for current_key, expected_key in cases:
            with self.subTest(current_key=current_key):
                updated = manager.update_key('CZ', current_key, [0, 1])
                self.assertEqual(updated, expected_key)


class DummyCircuit:
    def depth(self, with_single, with_barrier):
        return 11 if with_single else 7


class DummyCircuitPrep:
    result_path = 'result'

    def save_circuit_svg(self, circuit, name):
        return None


class DummyOpenQASM:
    def to_file(self, path, circuit):
        return None


class TestQCSOReturnContract(unittest.TestCase):
    def test_qcso_returns_counts_in_second_position(self):
        module = load_module('qcso_under_test', ROOT / 'QCSO.py', mindquantum_stubs())
        circuit = DummyCircuit()
        manager = types.SimpleNamespace(circ=circuit)
        counts = types.SimpleNamespace(data={'000': 4096})
        analog_frame = pd.DataFrame({0: ['Measure']}, index=[100])

        module.QCManager = lambda program, path: DummyCircuitPrep()
        module.initialize_params = lambda circ_prep: 3
        module.compile_circuit = lambda circ_prep, program: ('dag', 4)
        module.generate_gate_lengths_and_frames = (
            lambda circ_prep, dag, depth, path: ({}, analog_frame)
        )
        module.make_qubit_length_same = lambda frame: frame
        module.QCSOManager = lambda gate_lengths, qubits, lamb, depth: manager
        module.qcso_for_baseline_circ = (
            lambda qcso_manager, frame, circ_prep: (circuit, frame)
        )
        module.generate_baseline_circuit = (
            lambda circ_prep, qcso_manager, frame, qubits, shots: [counts]
        )
        module.OpenQASM = DummyOpenQASM

        result = module.QCSO('input.qasm', 'output.qasm', True, 4096)

        self.assertEqual(result[1], {'000': 4096})
        self.assertEqual(result[2], 7)
        self.assertEqual(result[3], 11)


class TestMeasurementAlignment(unittest.TestCase):
    def test_moves_all_measurements_to_latest_timestep(self):
        module = load_module('qcso_alignment_under_test', ROOT / 'QCSO.py', mindquantum_stubs())
        analog_frame = pd.DataFrame(
            {
                0: ['Measure', '', ''],
                1: ['', 'Measure', ''],
                2: ['', '', 'Measure'],
            },
            index=[100, 200, 300],
        )

        aligned_frame = module.make_qubit_length_same(analog_frame)

        self.assertEqual(aligned_frame.loc[300].tolist(), ['Measure', 'Measure', 'Measure'])
        self.assertNotIn('Measure', aligned_frame.loc[100].tolist())
        self.assertNotIn('Measure', aligned_frame.loc[200].tolist())


class TestADOAFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module(
            'circuit_od_under_test',
            ROOT / 'scripts' / 'circuit_od.py',
            mindquantum_stubs(),
        )

    def make_manager(self, lamb_param):
        manager = self.module.QCSOManager.__new__(self.module.QCSOManager)
        manager.gate_lengths = {
            'X': [84],
            'Y': [84],
            'RZ': [84],
            'H': [84],
            'Measure': [100],
        }
        manager.tab = [0]
        manager.lamb_param = lamb_param
        return manager

    def test_long_idle_prefers_xy8_sequence(self):
        manager = self.make_manager(True)
        analog_frame = pd.DataFrame({0: ['X', 'Measure']}, index=[84, 856])

        sequence = manager.Add_DD_gates_in_timestep(analog_frame, 0)
        gate_names = [next(iter(gate.values())) for gate in sequence]

        self.assertEqual(gate_names, ['X', 'Y', 'X', 'Y', 'Y', 'X', 'Y', 'X'])

    def test_false_lambda_disables_short_idle_z_sequence(self):
        manager = self.make_manager(False)
        analog_frame = pd.DataFrame({0: ['X', 'H']}, index=[84, 252])
        original_frame = analog_frame.copy(deep=True)

        sequence = manager.Add_DD_gates_in_timestep(analog_frame, 0)

        self.assertEqual(sequence, [])
        pd.testing.assert_frame_equal(analog_frame, original_frame)

        manager.lamb_param = True
        enabled_frame = original_frame.copy(deep=True)
        enabled_sequence = manager.Add_DD_gates_in_timestep(enabled_frame, 0)
        self.assertEqual(enabled_sequence, [{168: 'RZ (3.141592653589793)'}])


if __name__ == '__main__':
    unittest.main()
