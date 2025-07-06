import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


def draw_impact_DD_q0(current_dir):
    prefix_path = os.path.join(current_dir, "result")
    y_no = []
    y_with = []
    angle = [0, '0.3', '0.5', '0.8', '1']
    for i in range(len(angle)):
        program_name = 'Rotation-' + f'{angle[i]}-1.qasm'
        new_program_path = os.path.join(prefix_path, program_name)
        df = pd.read_csv(f'{new_program_path}_all_el_fidelity.txt')
        first_value = float(df.iloc[0, 0])
        last_value = float(df.iloc[-1, 0])
        y_no.append(first_value)
        y_with.append(last_value)

    x = ['0', 'Π/3', 'Π/2', '2Π/3', 'Π']
    # x = np.array([0, np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi])
    # plt.xticks(ticks=x, labels=[r'$0$', r'$\frac{\pi}{3}$', r'$\frac{\pi}{2}$', r'$\frac{2\pi}{3}$', r'$\pi$'],
    #            fontsize=12)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_no, label='No-DD', marker='o', linestyle='-', color='b', linewidth=2)
    plt.plot(x, y_with, label='With-DD', marker='o', linestyle='--', color='r', linewidth=2)
    plt.title('Fidelity of q[0] with free evolution and with DD', fontsize=14)
    plt.xlabel(r'Rotation Angle ($\theta$)', fontsize=14)
    plt.ylabel('Fidelity of q[0]', fontsize=14)
    plt.legend(fontsize=12)
    figure_path = os.path.join(current_dir, "figure")
    plt.savefig(os.path.join(figure_path, 'fig4_c.png'))


def draw_impact_DD_q2(current_dir):
    prefix_path = os.path.join(current_dir, "result")
    y_no = []
    y_with = []
    angle = [0, '0.3', '0.5', '0.8', '1']
    for i in range(len(angle)):
        program_name = 'Rotation-' + f'{angle[i]}-3.qasm'
        new_program_path = os.path.join(prefix_path, program_name)
        df = pd.read_csv(f'{new_program_path}_all_el_fidelity.txt')
        first_value = float(df.iloc[0, 0])
        last_value = float(df.iloc[-1, 0])
        y_no.append(first_value)
        y_with.append(last_value)

    x = ['0', 'Π/3', 'Π/2', '2Π/3', 'Π']
    # x = np.array([0, np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi])
    # plt.xticks(ticks=x, labels=[r'$0$', r'$\frac{\pi}{3}$', r'$\frac{\pi}{2}$', r'$\frac{2\pi}{3}$', r'$\pi$'],
    #            fontsize=12)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_no, label='No-DD', marker='o', linestyle='-', color='b', linewidth=2)
    plt.plot(x, y_with, label='With-DD', marker='o', linestyle='--', color='r', linewidth=2)
    plt.title('Fidelity of q[0] in the presence of crosstalk, with and without DD', fontsize=14)
    plt.xlabel(r'Rotation Angle ($\theta$)', fontsize=14)
    plt.ylabel('Fidelity of q[0]', fontsize=14)
    plt.legend(fontsize=12)
    figure_path = os.path.join(current_dir, "figure")
    plt.savefig(os.path.join(figure_path, 'fig4_f.png'))


def draw_DD_combinations(current_dir):
    prefix_path = os.path.join(current_dir, "result")
    y_no = []
    y_qft_dd = []
    program_name_1 = 'QFT-6-A.qasm'
    new_program_path = os.path.join(prefix_path, program_name_1)
    df = pd.read_csv(f'{new_program_path}_all_el_fidelity.txt')
    y_qft_dd = df.iloc[:, 0].values

    program_name_2 = 'BV-6.qasm'
    new_program_path = os.path.join(prefix_path, program_name_2)
    df = pd.read_csv(f'{new_program_path}_all_el_fidelity.txt')
    y_bv_dd = df.iloc[:, 0].values

    x = range(0, len(y_qft_dd))
    # x = np.array([0, np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi])
    # plt.xticks(ticks=x, labels=[r'$0$', r'$\frac{\pi}{3}$', r'$\frac{\pi}{2}$', r'$\frac{2\pi}{3}$', r'$\pi$'],
    #            fontsize=12)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_qft_dd, label='QFT-6', marker='o', linestyle='--', color='b', linewidth=2)
    plt.plot(x, y_bv_dd, label='BV-6', marker='o', linestyle='--', color='r', linewidth=2)
    plt.title('Fidelity of QFT and BV benchmarks with all possible DD sequences', fontsize=14)
    plt.xlabel('Qubit combinations on which DD pulses are applied', fontsize=14)
    plt.ylabel('Program Fidelity', fontsize=14)
    plt.legend(fontsize=12)
    figure_path = os.path.join(current_dir, "figure")
    plt.savefig(os.path.join(figure_path, 'fig8.png'))


def draw_diff_policy(current_dir):
    prefix_path = os.path.join(current_dir, "result")
    program_name_1 = 'BV-7.qasm'
    new_program_path = os.path.join(prefix_path, program_name_1)
    df = pd.read_csv(f'{new_program_path}_all_el_fidelity.txt')
    y_bv7_no = float(df.iloc[0, 0])
    y_bv7_all = float(df.iloc[-1, 0])
    y_bv7_best = float(df.iloc[:, 0].max())
    df_SDC = pd.read_csv(f'{new_program_path}_all_skeleton_el_fidelity.txt')
    max_index = df_SDC.iloc[:, 0].idxmax()
    y_bv7_adapt = float(df.iloc[max_index, 0])

    program_name_2 = 'BV-8.qasm'
    new_program_path = os.path.join(prefix_path, program_name_2)
    df = pd.read_csv(f'{new_program_path}_all_el_fidelity.txt')
    y_bv8_no = float(df.iloc[0, 0])
    y_bv8_all = float(df.iloc[-1, 0])
    y_bv8_best = float(df.iloc[:, 0].max())
    df_SDC = pd.read_csv(f'{new_program_path}_all_skeleton_el_fidelity.txt')
    max_index = df_SDC.iloc[:, 0].idxmax()
    y_bv8_adapt = float(df.iloc[max_index, 0])

    program_name_3 = 'QPEA-5.qasm'
    new_program_path = os.path.join(prefix_path, program_name_3)
    df = pd.read_csv(f'{new_program_path}_all_el_fidelity.txt')
    y_qpea_no = float(df.iloc[0, 0])
    y_qpea_all = float(df.iloc[-1, 0])
    y_qpea_best = float(df.iloc[:, 0].max())
    df_SDC = pd.read_csv(f'{new_program_path}_all_skeleton_el_fidelity.txt')
    max_index = df_SDC.iloc[:, 0].idxmax()
    y_qpea_adapt = float(df.iloc[max_index, 0].max())

    program_name_4 = 'QAOA-10-B.qasm'
    new_program_path = os.path.join(prefix_path, program_name_4)
    df = pd.read_csv(f'{new_program_path}_all_el_fidelity.txt')
    y_qaoa_no = float(df.iloc[0, 0])
    y_qaoa_all = float(df.iloc[-1, 0])
    y_qaoa_best = float(df.iloc[:, 0].max())
    df_SDC = pd.read_csv(f'{new_program_path}_all_skeleton_el_fidelity.txt')
    max_index = df_SDC.iloc[:, 0].idxmax()
    y_qaoa_adapt = float(df.iloc[max_index, 0].max())

    program_name_5 = 'QFT-6-B.qasm'
    new_program_path = os.path.join(prefix_path, program_name_5)
    df = pd.read_csv(f'{new_program_path}_all_el_fidelity.txt')
    y_qft8_no = float(df.iloc[0, 0])
    y_qft8_all = float(df.iloc[-1, 0])
    y_qft8_best = float(df.iloc[:, 0].max())
    df_SDC = pd.read_csv(f'{new_program_path}_all_skeleton_el_fidelity.txt')
    max_index = df_SDC.iloc[:, 0].idxmax()
    y_qft8_adapt = float(df.iloc[max_index, 0].max())

    labels = ['QPEA-5', 'BV-7', 'QFT-6B', 'BV-8', 'QAOA-10-B']
    group1 = [y_qpea_all/y_qpea_no, y_bv7_all/y_bv7_no, y_qft8_all/y_qft8_no, y_bv8_all/y_bv8_no, y_qaoa_all/y_qaoa_no]
    group2 = [y_qpea_adapt/y_qpea_no, y_bv7_adapt/y_bv7_no, y_qft8_adapt/y_qft8_no, y_bv8_adapt/y_bv8_no, y_qaoa_adapt/y_qaoa_no]
    group3 = [y_qpea_best/y_qpea_no, y_bv7_best/y_bv7_no, y_qft8_best/y_qft8_no, y_bv8_best/y_bv8_no, y_qaoa_best/y_qaoa_no]

    # 设置图形参数
    x = np.arange(len(labels))  # 横坐标位置
    width = 0.25  # 柱的宽度

    # 创建柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, group1, width, label='All-DD', color='skyblue')
    plt.bar(x, group2, width, label='ADAPT', color='lightgreen')
    plt.bar(x + width, group3, width, label='Runtime Best', color='lightcoral')

    # 添加标签、标题等
    plt.xlabel('Benchmarks using XY4', fontsize=12)
    plt.ylabel('Relative Fidelity', fontsize=12)
    plt.title('Relative Fidelity of different policies using XY4', fontsize=14)
    plt.xticks(x, labels, fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 显示图形
    plt.tight_layout()
    figure_path = os.path.join(current_dir, "figure")
    plt.savefig(os.path.join(figure_path, 'fig13.png'))


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    draw_impact_DD_q0(current_dir)
    draw_impact_DD_q2(current_dir)
    # draw_DD_combinations(current_dir)
    # draw_diff_policy(current_dir)
