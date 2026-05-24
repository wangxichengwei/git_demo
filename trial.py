import casadi as ca
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import expm
from scipy.interpolate import interp1d

try:
    import tkinter
    matplotlib.use('TkAgg')
except ImportError:
    matplotlib.use('Agg') 

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False  

# ========== 系统参数 ==========
M = 1.0      # 小车质量
m = 0.1      # 摆杆质量
l = 0.5      # 半长
g = 9.81
I = 1/3 * m * l**2
denom_const = I + m*l**2  # 常值

# 状态导数函数（CasADi 符号形式，用于优化）
def f_expr(z, u):
    x, theta, dx, dtheta = z[0], z[1], z[2], z[3]
    sin_theta = ca.sin(theta)
    cos_theta = ca.cos(theta)
    tmp = (I + m*l**2)
    denom = M + m - (m**2 * l**2 * cos_theta**2) / tmp
    num = u + m*l * dtheta**2 * sin_theta - (m**2 * l**2 * g * sin_theta * cos_theta) / tmp
    ddx = num / denom
    ddtheta = (m*l * (g * sin_theta - ddx * cos_theta)) / tmp
    return ca.vertcat(dx, dtheta, ddx, ddtheta)

# 动力学用于数值仿真（numpy）
def f_numeric(z, u):
    x, theta, dx, dtheta = z
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    tmp = I + m*l**2
    denom = M + m - (m**2 * l**2 * cos_theta**2) / tmp
    num = u + m*l * dtheta**2 * sin_theta - (m**2 * l**2 * g * sin_theta * cos_theta) / tmp
    ddx = num / denom
    ddtheta = (m*l * (g * sin_theta - ddx * cos_theta)) / tmp
    return np.array([dx, dtheta, ddx, ddtheta])

# ========== 轨迹优化 ==========
T = 3.0        
N = 50          
dt = T / N

opti = ca.Opti()

# 决策变量
Z = opti.variable(4, N+1)   
U = opti.variable(1, N)     

# 目标函数：最小化控制平方积分
cost = ca.sum2(U**2) * dt
opti.minimize(cost)

# 动力学约束（梯形配点）
for k in range(N):
    zk = Z[:, k]
    zk1 = Z[:, k+1]
    uk = U[:, k]
    fk = f_expr(zk, uk)
    fk1 = f_expr(zk1, uk)
    opti.subject_to( zk1 - zk == (dt/2) * (fk + fk1) )

# 边界条件
z_start = [0, 0, 0, 0]      
z_end   = [2.0, 0, 0, 0]    
opti.subject_to( Z[:, 0] == z_start )
opti.subject_to( Z[:, -1] == z_end )
# 终点加速度为零（真正的平衡态）
f_end = f_expr(Z[:, -1], U[:, -1])
opti.subject_to( f_end[2] == 0 ) 
opti.subject_to( f_end[3] == 0 )  

# 硬约束：位置限制
x_min, x_max = -1.5, 3.0
opti.subject_to( opti.bounded(x_min, Z[0,:], x_max) )
# 控制限制
u_max = 20.0
opti.subject_to( opti.bounded(-u_max, U, u_max) )

# 初始猜测
opti.set_initial(Z, ca.repmat([1.0,0,0,0],1,N+1))  
opti.set_initial(U, 0)

# 求解器选项
opti.solver('ipopt')
sol = opti.solve()

# 提取结果
Z_opt = sol.value(Z).T  
U_opt = sol.value(U).flatten()  
t_grid = np.linspace(0, T, N+1)
t_u    = np.linspace(0, T, N, endpoint=False)

# ========== 绘制优化结果（标称轨迹） ==========
fig0, axs0 = plt.subplots(3, 1, figsize=(12, 8))
fig0.suptitle('标称轨迹优化结果 (IPOPT)', fontsize=14)
axs0[0].plot(t_grid, Z_opt[:,0], 'b-', label='x (m)')
axs0[0].set_ylabel('x'); axs0[0].legend(); axs0[0].grid()
axs0[1].plot(t_grid, Z_opt[:,1], 'r-', label='theta (rad)')
axs0[1].set_ylabel('theta'); axs0[1].legend(); axs0[1].grid()
axs0[2].step(t_u, U_opt, 'g-', where='post', label='u (N)')
axs0[2].set_ylabel('u'); axs0[2].set_xlabel('Time (s)'); axs0[2].legend(); axs0[2].grid()
plt.tight_layout()

# ========== 时变 LQR ==========
dt_lqr = 0.01                
N_lqr = int(T / dt_lqr)
t_lqr = np.linspace(0, T, N_lqr)

# 插值标称轨迹
interp_z = interp1d(t_grid, Z_opt, axis=0, kind='cubic', fill_value="extrapolate")
interp_u = interp1d(t_u, U_opt, kind='linear', fill_value="extrapolate")

# 状态和控制代价
Q = np.diag([10, 100, 1, 10])   
R = np.array([[0.1]])
Qf = Q  # 终端代价

# 离散化 Riccati
def discretize_AB(Ac, Bc, dt):
    n = Ac.shape[0]
    M = np.zeros((n+1, n+1))
    M[:n, :n] = Ac
    M[:n, n] = Bc.flatten()
    expM = expm(M * dt)
    Ad = expM[:n, :n]
    Bd = expM[:n, n].reshape(-1,1)
    return Ad, Bd

# 前向获取所有离散系统矩阵
A_d_list = []
B_d_list = []
for k in range(N_lqr):
    tk = t_lqr[k]
    z_nom = interp_z(tk)
    u_nom = interp_u(tk)
    # 线性化（数值微分）
    eps = 1e-6
    Ac = np.zeros((4,4))
    f_nom = f_numeric(z_nom, u_nom)
    for i in range(4):
        z_pert = z_nom.copy()
        z_pert[i] += eps
        f_pert = f_numeric(z_pert, u_nom)
        Ac[:, i] = (f_pert - f_nom) / eps
    Bc = np.zeros((4,1))
    u_pert = u_nom + eps
    f_pert = f_numeric(z_nom, u_pert)
    Bc[:,0] = (f_pert - f_nom) / eps
    Ad, Bd = discretize_AB(Ac, Bc, dt_lqr)
    A_d_list.append(Ad)
    B_d_list.append(Bd)

# 逆向 Riccati
K_gains = [None] * N_lqr
P = Qf
for k in reversed(range(N_lqr)):
    Ad = A_d_list[k]
    Bd = B_d_list[k]
    K = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
    P = Q + Ad.T @ P @ Ad - Ad.T @ P @ Bd @ K
    K_gains[k] = K

# 仿真：初始扰动
np.random.seed(0)
z_sim = z_start.copy() + np.array([0, 0.05, 0, 0])  # 摆杆偏离0.05 rad
z_log = [z_sim.copy()]
u_log = []
t_log = [0.0]
for step in range(N_lqr):
    tk = t_lqr[step]
    z_nom = interp_z(tk)
    u_nom = interp_u(tk)
    K = K_gains[step]
    delta_z = z_sim - z_nom
    u_nom_val = u_nom.item() if hasattr(u_nom, 'item') else u_nom
    feedback = (K @ delta_z).item() if hasattr(K @ delta_z, 'item') else (K @ delta_z)
    u = u_nom_val - feedback
    u = np.clip(u, -u_max, u_max)
    u_log.append(u)
    # 欧拉积分
    dz = f_numeric(z_sim, u)
    z_sim = z_sim + dz * dt_lqr
    z_log.append(z_sim.copy())
    t_log.append(tk + dt_lqr)
z_log = np.array(z_log)
u_log = np.array(u_log).flatten()
t_log = np.array(t_log)

# ========== 对比实验：不同 R 矩阵的影响 ==========
R_values = [0.01, 0.1, 1.0]  # 小R(激进控制), 中R(基准), 大R(保守控制)
labels = ['R=0.01 (激进)', 'R=0.1 (基准)', 'R=1.0 (保守)']
colors = ['r', 'b', 'g']
sim_results = []

for idx, R_val in enumerate(R_values):
    R_current = np.array([[R_val]])
    
    # 1. 逆向 Riccati 求解新增益
    K_gains_new = [None] * N_lqr
    P = Qf
    for k in reversed(range(N_lqr)):
        Ad = A_d_list[k] 
        Bd = B_d_list[k]
      
        K = np.linalg.inv(R_current + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
        P = Q + Ad.T @ P @ Ad - Ad.T @ P @ Bd @ K
        K_gains_new[k] = K

    # 2. 重新仿真
    z_sim_comp = z_start.copy() + np.array([0, 0.05, 0, 0]) # 同样的初始扰动
    z_log_comp = [z_sim_comp.copy()]
    u_log_comp = []
    
    for step in range(N_lqr):
        tk = t_lqr[step]
        z_nom = interp_z(tk)
        u_nom = interp_u(tk)
        K = K_gains_new[step]
        delta_z = z_sim_comp - z_nom
        
        u_nom_val = u_nom.item() if hasattr(u_nom, 'item') else u_nom
        feedback = (K @ delta_z).item() if hasattr(K @ delta_z, 'item') else (K @ delta_z)
        u = u_nom_val - feedback
        u = np.clip(u, -u_max, u_max)
        u_log_comp.append(u)
        
        dz = f_numeric(z_sim_comp, u)
        z_sim_comp = z_sim_comp + dz * dt_lqr
        z_log_comp.append(z_sim_comp.copy())
        
    z_log_comp = np.array(z_log_comp)
    u_log_comp = np.array(u_log_comp).flatten()
    sim_results.append((z_log_comp, u_log_comp))

# ========== 绘制完整的 TVLQR 跟踪结果 ==========
# 创建标称控制的密集插值，用于时间对齐
fig1, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
axs[0].plot(t_log, z_log[:, 0], 'b-', linewidth=2, label='实际轨迹')
axs[0].plot(t_grid, Z_opt[:, 0], 'g--', linewidth=2, label='标称轨迹')
axs[0].set_ylabel('x (m)'); axs[0].legend(); axs[0].grid(True)
axs[0].set_title('TVLQR 闭环跟踪结果（初始扰动：θ=0.05 rad）')

axs[1].plot(t_log, z_log[:, 1], 'r-', linewidth=2, label='实际轨迹')
axs[1].plot(t_grid, Z_opt[:, 1], 'b--', linewidth=2, label='标称轨迹')
axs[1].set_ylabel('θ (rad)'); axs[1].legend(); axs[1].grid(True)

# 修正控制输入绘图的时间轴对齐
# u_log 对应的是 t_log[0] 到 t_log[-2] 的时刻
t_u_plot = t_log[:-1] 
axs[2].plot(t_u_plot, u_log, 'r-', linewidth=2, label='实际控制')
# 使用线性插值对比更平滑
interp_u_linear = interp1d(t_u, U_opt, kind='linear', fill_value="extrapolate")
axs[2].plot(t_u_plot, interp_u_linear(t_u_plot), 'g--', linewidth=2, label='标称控制')
axs[2].set_ylabel('u (N)'); axs[2].set_xlabel('时间 (s)'); axs[2].legend(); axs[2].grid(True)
plt.tight_layout()

# 3. 绘制误差
fig2, ax = plt.subplots(figsize=(12, 5))
z_nom_interp_log = interp_z(t_log)
error_x = z_log[:, 0] - z_nom_interp_log[:, 0]
error_theta = z_log[:, 1] - z_nom_interp_log[:, 1]
ax.plot(t_log, error_x, 'b-', linewidth=2, label='x 误差')
ax.plot(t_log, error_theta, 'r-', linewidth=2, label='θ 误差')
ax.set_ylabel('跟踪误差'); ax.set_xlabel('时间 (s)'); ax.legend(); ax.grid(True)
ax.set_title('TVLQR 跟踪误差曲线')
plt.tight_layout()

# ========== 绘制对比结果 ==========
fig3, axs_comp = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig3.suptitle('不同 R 值对 TVLQR 性能的影响对比', fontsize=16)

# 时间轴用于绘图 (u_log 长度比 z_log 少 1，对应 t_lqr)
t_plot = t_lqr 

# 1. 摆角跟踪对比 (Theta)
for idx, (z_res, u_res) in enumerate(sim_results):
    axs_comp[0].plot(t_plot, z_res[1:, 1], color=colors[idx], linewidth=2, label=labels[idx])
axs_comp[0].plot(t_grid, Z_opt[:, 1], 'k--', linewidth=1.5, label='标称轨迹')
axs_comp[0].set_ylabel('摆角 θ (rad)')
axs_comp[0].legend(loc='best')
axs_comp[0].grid(True)
axs_comp[0].set_title('摆角跟踪对比')

# 2. 推车位置跟踪对比 (X)
for idx, (z_res, u_res) in enumerate(sim_results):
    axs_comp[1].plot(t_plot, z_res[1:, 0], color=colors[idx], linewidth=2, label=labels[idx])
axs_comp[1].plot(t_grid, Z_opt[:, 0], 'k--', linewidth=1.5, label='标称轨迹')
axs_comp[1].set_ylabel('位置 x (m)')
axs_comp[1].legend(loc='best')
axs_comp[1].grid(True)
axs_comp[1].set_title('推车位置跟踪对比')

# 3. 控制输入对比 (U)
for idx, (z_res, u_res) in enumerate(sim_results):
    # u_res 长度与 t_lqr 一致
    axs_comp[2].plot(t_plot, u_res, color=colors[idx], linewidth=2, label=labels[idx])
axs_comp[2].set_ylabel('控制力 u (N)')
axs_comp[2].set_xlabel('时间 (s)')
axs_comp[2].legend(loc='best')
axs_comp[2].grid(True)
axs_comp[2].set_title('控制输入对比')

plt.tight_layout()


# 4. 动画
fig_anim, ax_anim = plt.subplots(figsize=(8,4))
ax_anim.set_xlim(-1, 3.5)
ax_anim.set_ylim(-0.8, 0.8)
ax_anim.set_aspect('equal')
ax_anim.grid()

cart_width = 0.3
cart_height = 0.2
pole_length = l * 2

cart_rect = plt.Rectangle((-cart_width/2, -cart_height/2), cart_width, cart_height, fc='blue')
pole_line, = ax_anim.plot([], [], 'r-', lw=3)
ax_anim.add_patch(cart_rect)
def init():
    cart_rect.set_xy((-cart_width/2, -cart_height/2))
    pole_line.set_data([], [])
    return cart_rect, pole_line

def animate(i):
    # 直接使用 i 作为索引，步长由 frames 控制，避免跳帧逻辑复杂化
    if i >= len(z_log):
        i = len(z_log) - 1
    x = z_log[i, 0]
    theta = z_log[i, 1]
    cart_rect.set_xy((x - cart_width/2, -cart_height/2))
    pole_x = [x, x + pole_length * np.sin(theta)]
    pole_y = [0, pole_length * np.cos(theta)]
    pole_line.set_data(pole_x, pole_y)
    return cart_rect, pole_line

# 每一帧都显示，interval 控制速度
ani = FuncAnimation(fig_anim, animate, frames=len(z_log), init_func=init,
                    blit=False, interval=20) # 20ms per frame ~ 50fps

plt.title('Cart-Pole TVLQR Tracking')

# 创建动画
ani = FuncAnimation(fig_anim, animate, frames=len(z_log), init_func=init,
                    blit=False, interval=20) 

plt.title('Cart-Pole TVLQR Tracking')

# 2. 保存为 GIF

try:
    ani.save('cartpole_tracking.gif', writer='pillow', fps=50, dpi=100)
    print("成功保存 GIF: cartpole_tracking.gif")
except Exception as e:
    print(f"保存 GIF 失败: {e}")
    print("提示: 请确保已安装 pillow 库 (pip install pillow)")
# 最后统一显示所有窗口
plt.show()