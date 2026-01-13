import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.integrate import solve_ivp
from multi_arm_ode import my_ode

import sys
sys.setrecursionlimit(100000000)

import seaborn as sns


import matplotlib
params = {
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'axes.grid': False,
    'savefig.dpi': 600,  # to adjust notebook inline plot size
    'axes.labelsize': 14, # fontsize for x and y labels (was 10)
    'axes.titlesize': 14,
    'font.size': 12, # was 10
    'legend.fontsize': 12, # was 10
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    #'text.usetex': True,
    #'figure.figsize': [7, 4],
    'font.family': 'Helvetica',
}
matplotlib.rcParams.update(params)



bright_palette = sns.color_palette("Set2")  # 也可以换成上面任一自定义调色板

# === 基本设置 ===
sns.set_theme(context='paper',  # 'talk' 适合幻灯片，'paper' 适合出版
              style='white',    # 白色背景
              palette=bright_palette  # 色觉友好配色方案
             )   # 调整整体字体大小

# 设置背景 - 确保subplot背景是白色
plt.rcParams['figure.facecolor'] = 'none'  # 图表外区域透明
plt.rcParams['axes.facecolor'] = 'white'   # 图表区域保持白色
plt.rcParams['savefig.facecolor'] = 'none' # 保存时figure背景透明

palette = sns.color_palette("Set2")
my_colors = [palette[i] for i in [0, 1]]  # 比如青-蓝紫-苹果绿-灰褐
sns.set_palette(my_colors)


# 1. 读取实验数据
mat_path = 'Datas/TriplePendulum/TripleDataFreeSwing_3_Dt_0_0001.mat'
data = loadmat(mat_path)
theta1 = data['Theta1']
theta2 = data['Theta2']
theta3 = data['Theta3']
d_theta1 = data['dTheta1']
d_theta2 = data['dTheta2']
d_theta3 = data['dTheta3']

# 拼成 (N,6) 数组
my_data = np.concatenate([theta1, theta2, theta3, d_theta1, d_theta2, d_theta3], axis=1)

# 时间轴
dt = 0.01
down_sample_rate = 100
time = np.arange(0, 10, dt)

# 2. 仿真
x0 = my_data[0]  # 初始状态

def ode_func(t, x):
    return my_ode(t, x)

sol = solve_ivp(ode_func, [0, 10], x0, t_eval=time, method='RK45', rtol=1e-8, atol=1e-8)

# 3. 可视化对比
labels = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$', r'$\dot{\theta}_1$', r'$\dot{\theta}_2$', r'$\dot{\theta}_3$']

# Y轴刻度自定义设置 - 您可以在这里修改每个子图的y轴刻度
yticks_config = {
    0: [2.4, 3.0, 3.6, 4.2],     # θ1的y轴刻度
    1: [2.4, 3.0, 3.6, 4.2],     # θ2的y轴刻度  
    2: [1.2, 2.5, 3.8, 5.1],     # θ3的y轴刻度
    3: [-5,  0,  5],       # θ̇1的y轴刻度
    4: [-5,  0,  5],       # θ̇2的y轴刻度
    5: [-10,  0, 10],       # θ̇3的y轴刻
}

plt.figure(figsize=(6, 3), facecolor='none')
for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    ax.set_facecolor('white')  # 确保每个子图的背景是白色
    ax.patch.set_facecolor('white')  # 额外确保背景是白色
    plt.plot(time, my_data[:100000:100, i], label='Experiment')
    plt.plot(sol.t, sol.y[i], label='Identified Model', linestyle='--')
    plt.ylabel(labels[i])
    
    # 在第一个子图添加透明图例
    if i == 0:
        plt.legend(loc='upper right', framealpha=0, fontsize=6)
    
    # 设置自定义y轴刻度
    if i in yticks_config:
        plt.yticks(yticks_config[i])
    
    if i >= 3:  # 只在第二排（后三个子图）显示x轴标签
        plt.xlabel('Time [s]')
    else:  # 第一排（前三个子图）
        plt.xticks(ticks=[0,5,10], labels=['','',''])  # 设置刻度位置和空标签

# 调整子图间距
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
#plt.suptitle('Triple Pendulum: Simulation vs Experiment')
plt.savefig('triple_pendulum_comparison.png', dpi=600, bbox_inches='tight', transparent=True)
plt.show() 