# 模型预测控制（MPC）原理与应用

## 引言：能预测未来的控制器

想象你在驾驶汽车：你不仅看当前位置，还会预测未来几秒的轨迹，提前转向、刹车、加速。这就是**模型预测控制（Model Predictive Control, MPC）**的核心思想——利用系统模型预测未来行为，通过优化求解最优控制序列。

MPC是过去30年控制理论最成功的成果之一，从化工过程到自动驾驶，从机器人到航天器，MPC凭借其独特的优势成为复杂系统控制的首选方案。

---

## 一、MPC的核心思想

### 1.1 为什么需要MPC？

**PID的局限：**
- 无法处理多输入多输出（MIMO）系统
- 无法显式处理约束（饱和、安全等）
- 无法最优化性能指标
- 缺乏预测能力

**MPC的优势：**
1. **预测未来**：利用模型预测系统行为
2. **优化决策**：在线求解最优控制问题
3. **处理约束**：显式考虑物理限制
4. **多变量协调**：自然处理MIMO系统
5. **滚动优化**：实时更新适应变化

### 1.2 MPC的三大核心

#### **1. 预测模型（Prediction Model）**

```
x(k+1) = f(x(k), u(k))  （状态方程）
y(k) = h(x(k))          （输出方程）
```

利用模型预测未来N步的系统行为。

#### **2. 滚动优化（Receding Horizon Optimization）**

在每个时刻求解优化问题：

```
min  Σ ||y(k+i) - r(k+i)||² + Σ ||u(k+i)||²
u    i=1...N                   i=0...N-1

s.t. 系统动力学
     u_min ≤ u(k+i) ≤ u_max
     y_min ≤ y(k+i) ≤ y_max
```

#### **3. 反馈校正（Feedback Correction）**

仅执行第一步控制u(k)，下一时刻重新优化（滚动窗口）。

```
时刻k：  |----预测窗口N----|
         k  k+1 k+2 ... k+N

时刻k+1:    |----预测窗口N----|
            k+1 k+2 ... k+N+1
```

### 1.3 MPC工作流程

```
1. 测量当前状态 x(k)
2. 预测未来轨迹 {x(k+1), ..., x(k+N)}
3. 优化控制序列 {u(k), u(k+1), ..., u(k+N-1)}
4. 执行第一步 u(k)
5. k ← k+1，回到步骤1
```

---

## 二、线性MPC

### 2.1 离散时间线性系统

**状态空间模型：**
```
x(k+1) = Ax(k) + Bu(k)
y(k) = Cx(k)
```

**预测方程：**

```
x(k+1) = Ax(k) + Bu(k)
x(k+2) = A²x(k) + ABu(k) + Bu(k+1)
...
x(k+N) = A^N x(k) + Σ A^(N-i-1) Bu(k+i)
```

**矩阵形式：**
```
X = Ψx(k) + ΘU

其中：
X = [x(k+1), x(k+2), ..., x(k+N)]^T
U = [u(k), u(k+1), ..., u(k+N-1)]^T

Ψ = [A, A², ..., A^N]^T
Θ = [B,   AB,  ... ;
     0,   B,   AB, ... ;
     ...]
```

### 2.2 二次规划（QP）形式

**目标函数：**
```
J = Σ ||y(k+i) - r(k+i)||²_Q + Σ ||u(k+i)||²_R + Σ ||Δu(k+i)||²_S

其中：
Q: 输出权重（跟踪误差）
R: 控制权重（能量消耗）
S: 控制增量权重（平滑性）
```

**约束：**
```
u_min ≤ u(k+i) ≤ u_max          （输入约束）
Δu_min ≤ u(k+i) - u(k+i-1) ≤ Δu_max  （变化率约束）
y_min ≤ y(k+i) ≤ y_max          （输出约束）
```

**标准QP形式：**
```
min  (1/2) U^T H U + f^T U
U

s.t. AU ≤ b
```

### 2.3 MPC求解算法

#### **内点法（Interior Point Method）**

**思想：**将不等式约束转化为对数障碍函数。

**优点：**
- 收敛快（多项式时间）
- 精度高

**缺点：**
- 计算量大
- 不适合实时

#### **有效集法（Active Set Method）**

**思想：**识别哪些约束是有效的（active），求解等式约束QP。

**优点：**
- 适合温启动（warm start）
- 迭代次数少

**缺点：**
- 最坏情况指数时间

#### **快速梯度法**

**FISTA（Fast Iterative Shrinkage-Thresholding Algorithm）：**

```python
def fista_mpc(H, f, A, b, max_iter=100):
    U = np.zeros(n)
    Z = U.copy()
    t = 1.0

    for k in range(max_iter):
        # 梯度步
        grad = H @ Z + f
        U_new = project_box(Z - alpha * grad, u_min, u_max)

        # Nesterov加速
        t_new = (1 + np.sqrt(1 + 4*t**2)) / 2
        Z = U_new + (t - 1) / t_new * (U_new - U)

        U = U_new
        t = t_new

    return U
```

### 2.4 Python实现示例

```python
import numpy as np
from scipy.optimize import minimize
import cvxpy as cp

class LinearMPC:
    def __init__(self, A, B, C, Q, R, N, u_min, u_max):
        """
        线性MPC控制器

        参数：
        A, B, C: 状态空间矩阵
        Q: 输出权重
        R: 控制权重
        N: 预测时域
        u_min, u_max: 控制输入约束
        """
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.N = N
        self.u_min = u_min
        self.u_max = u_max

        self.nx = A.shape[0]  # 状态维度
        self.nu = B.shape[1]  # 控制维度
        self.ny = C.shape[0]  # 输出维度

    def predict(self, x0, U):
        """预测未来轨迹"""
        X = [x0]
        Y = [self.C @ x0]

        for k in range(self.N):
            x_next = self.A @ X[-1] + self.B @ U[k]
            y_next = self.C @ x_next
            X.append(x_next)
            Y.append(y_next)

        return np.array(X), np.array(Y)

    def solve_qp_cvxpy(self, x0, r):
        """使用CVXPY求解QP问题"""

        # 定义优化变量
        U = cp.Variable((self.N, self.nu))

        # 预测轨迹
        X = [x0]
        for k in range(self.N):
            x_next = self.A @ X[-1] + self.B @ U[k]
            X.append(x_next)

        Y = [self.C @ x for x in X[1:]]

        # 目标函数
        cost = 0
        for k in range(self.N):
            cost += cp.quad_form(Y[k] - r[k], self.Q)
            cost += cp.quad_form(U[k], self.R)

        # 约束
        constraints = []
        for k in range(self.N):
            constraints += [U[k] >= self.u_min]
            constraints += [U[k] <= self.u_max]

        # 求解
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP, warm_start=True)

        if problem.status == cp.OPTIMAL:
            return U.value
        else:
            print(f"求解失败: {problem.status}")
            return np.zeros((self.N, self.nu))

    def control(self, x0, r):
        """计算控制输入（仅返回第一步）"""
        U_opt = self.solve_qp_cvxpy(x0, r)
        return U_opt[0]


# 示例：温度控制系统
if __name__ == "__main__":
    # 系统模型：dx/dt = -0.1x + 0.5u
    dt = 1.0  # 采样周期
    A = np.array([[0.9]])
    B = np.array([[0.5]])
    C = np.array([[1.0]])

    # MPC参数
    Q = np.array([[10.0]])  # 输出权重
    R = np.array([[0.1]])   # 控制权重
    N = 10                  # 预测时域
    u_min = np.array([0.0])
    u_max = np.array([1.0])

    # 创建控制器
    mpc = LinearMPC(A, B, C, Q, R, N, u_min, u_max)

    # 仿真
    x = np.array([0.0])  # 初始状态
    r = np.array([[1.0]] * N)  # 参考轨迹

    for t in range(50):
        u = mpc.control(x, r)
        x = A @ x + B @ u

        print(f"t={t}, x={x[0]:.3f}, u={u[0]:.3f}")
```

---

## 三、非线性MPC

### 3.1 非线性系统模型

**连续时间：**
```
ẋ = f(x, u)
y = h(x)
```

**离散时间：**
```
x(k+1) = f_d(x(k), u(k))
y(k) = h(x(k))
```

### 3.2 非线性优化问题

**目标函数：**
```
min  Σ L(x(k+i), u(k+i)) + F(x(k+N))
u

s.t. x(k+i+1) = f(x(k+i), u(k+i)), i=0,...,N-1
     u_min ≤ u(k+i) ≤ u_max
     g(x(k+i)) ≤ 0  （状态约束）
```

**终端约束/代价：**
- F(x(k+N)): 终端代价（保证稳定性）
- x(k+N) ∈ Ω: 终端约束集

### 3.3 求解方法

#### **序列二次规划（SQP）**

**思想：**迭代线性化，求解一系列QP子问题。

**步骤：**
1. 线性化动力学：x(k+i+1) ≈ f(x̄, ū) + A_i(x - x̄) + B_i(u - ū)
2. 二次化目标函数
3. 求解QP
4. 更新 x̄, ū

#### **直接法（Direct Method）**

**单次打靶（Single Shooting）：**
```
优化变量：U = [u(k), u(k+1), ..., u(k+N-1)]

通过积分预测：
x(k+1) = ∫ f(x, u(k)) dt
x(k+2) = ∫ f(x, u(k+1)) dt
...
```

**多次打靶（Multiple Shooting）：**
```
优化变量：[U, X] = [u₀, ..., u_{N-1}, x₁, ..., x_N]

约束：x_{i+1} = f(x_i, u_i)
```

**直接配置（Direct Collocation）：**

将状态和控制同时离散化，通过配置点满足动力学。

### 3.4 非线性MPC实现

```python
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint

class NonlinearMPC:
    def __init__(self, dynamics, cost_func, N, dt, u_min, u_max):
        """
        非线性MPC控制器

        参数：
        dynamics: 系统动力学 ẋ = f(x, u)
        cost_func: 阶段代价 L(x, u)
        N: 预测时域
        dt: 采样周期
        u_min, u_max: 控制约束
        """
        self.dynamics = dynamics
        self.cost_func = cost_func
        self.N = N
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max

    def predict(self, x0, U):
        """预测轨迹"""
        X = [x0]

        for u in U:
            # 数值积分
            x_next = odeint(
                lambda x, t: self.dynamics(x, u),
                X[-1],
                [0, self.dt]
            )[-1]
            X.append(x_next)

        return np.array(X)

    def objective(self, U_flat, x0, x_ref):
        """目标函数"""
        U = U_flat.reshape((self.N, -1))
        X = self.predict(x0, U)

        cost = 0
        for i in range(self.N):
            cost += self.cost_func(X[i], U[i], x_ref)

        # 终端代价
        cost += 10 * np.linalg.norm(X[-1] - x_ref)**2

        return cost

    def control(self, x0, x_ref, U_init=None):
        """求解最优控制"""

        nu = len(self.u_min)

        # 初始猜测
        if U_init is None:
            U_init = np.zeros((self.N, nu))

        # 约束
        bounds = [(self.u_min[i], self.u_max[i])
                  for _ in range(self.N)
                  for i in range(nu)]

        # 优化求解
        result = minimize(
            self.objective,
            U_init.flatten(),
            args=(x0, x_ref),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-6}
        )

        U_opt = result.x.reshape((self.N, nu))
        return U_opt[0], U_opt  # 返回第一步和完整序列


# 示例：倒立摆控制
def pendulum_dynamics(x, u):
    """倒立摆动力学: [θ, θ̇]"""
    theta, theta_dot = x
    g = 9.81
    L = 1.0
    m = 1.0
    b = 0.1

    theta_ddot = (g/L) * np.sin(theta) - (b/(m*L**2)) * theta_dot + u[0] / (m*L**2)

    return np.array([theta_dot, theta_ddot])

def pendulum_cost(x, u, x_ref):
    """代价函数"""
    Q = np.diag([10.0, 1.0])
    R = np.array([[0.1]])

    x_error = x - x_ref
    cost = x_error.T @ Q @ x_error + u.T @ R @ u

    return cost

if __name__ == "__main__":
    # 创建非线性MPC
    mpc = NonlinearMPC(
        dynamics=pendulum_dynamics,
        cost_func=pendulum_cost,
        N=10,
        dt=0.1,
        u_min=np.array([-10.0]),
        u_max=np.array([10.0])
    )

    # 仿真
    x = np.array([np.pi - 0.5, 0.0])  # 初始：接近倒立
    x_ref = np.array([np.pi, 0.0])    # 目标：倒立

    U_prev = None

    for t in range(100):
        u, U_prev = mpc.control(x, x_ref, U_prev[1:] if U_prev is not None else None)

        # 应用控制
        x = odeint(lambda x, t: pendulum_dynamics(x, u), x, [0, mpc.dt])[-1]

        print(f"t={t*mpc.dt:.2f}, θ={x[0]:.3f}, u={u[0]:.3f}")
```

---

## 四、MPC的稳定性

### 4.1 稳定性问题

**挑战：**MPC是有限时域优化，如何保证闭环稳定？

**反例：**
```
系统：x(k+1) = 2x(k) + u(k)
目标：x → 0

短时域MPC可能选择u=0（因为控制代价），导致x发散！
```

### 4.2 终端约束与终端代价

#### **终端等式约束**
```
x(k+N) = 0
```

**优点：**保证稳定性

**缺点：**
- 可行域小
- 计算困难

#### **终端集约束**
```
x(k+N) ∈ Ω

其中Ω是不变集（Invariant Set）
```

**设计：**
- 计算终端控制律 κ(x)
- 设计终端代价 V_f(x)
- 满足：V_f(f(x, κ(x))) - V_f(x) ≤ -L(x, κ(x))

#### **终端代价（无约束）**
```
J = Σ L(x(k+i), u(k+i)) + V_f(x(k+N))
```

**选择：**V_f 为LQR代价函数

**定理：**若N足够大，V_f选择恰当，则MPC闭环稳定。

### 4.3 名义稳定性与鲁棒稳定性

**名义稳定性：**
- 假设模型完美匹配
- 理论分析相对容易

**鲁棒稳定性：**
- 考虑模型不确定性
- 需要鲁棒MPC设计

---

## 五、MPC变种

### 5.1 显式MPC（Explicit MPC）

**思想：**离线求解所有可能的最优控制律，在线查表。

**方法：**多参数二次规划（mp-QP）

**结果：**分段仿射（PWA）控制律
```
u*(x) = F_i x + g_i,  if x ∈ Region_i
```

**优点：**
- 在线计算极快（查表）
- 实时性好

**缺点：**
- 仅适用于低维系统
- 存储空间指数增长

### 5.2 经济MPC（Economic MPC）

**目标：**直接优化经济指标，而非跟踪设定值。

**例子：**
```
min  Σ [电价(k+i) × 功率(k+i)]
```

而非：
```
min  Σ ||功率(k+i) - 功率参考||²
```

**应用：**
- 能源管理
- 化工过程优化
- 建筑空调控制

### 5.3 学习型MPC（Learning MPC）

**思想：**结合数据驱动与模型驱动。

**方法：**
1. **模型学习**：用神经网络学习f(x, u)
2. **MPC优化**：基于学习模型求解
3. **在线更新**：根据新数据更新模型

**优势：**
- 减少建模工作
- 适应性强
- 性能提升

### 5.4 鲁棒MPC

**考虑不确定性：**
```
x(k+1) = f(x(k), u(k), w(k))

其中w是不确定项
```

**方法：**

1. **最小最大MPC**
   ```
   min max  J(x, u, w)
   u    w
   ```

2. **管道MPC（Tube MPC）**
   - 将实际轨迹约束在名义轨迹附近的"管道"中
   - 分离名义控制与反馈校正

3. **随机MPC**
   - 建立概率约束
   - 期望值优化

---

## 六、实际应用案例

### 6.1 自动驾驶轨迹跟踪

**车辆运动学模型（自行车模型）：**
```
ẋ = v cos(ψ)
ẏ = v sin(ψ)
ψ̇ = v tan(δ) / L

控制输入：[加速度a, 转角δ]
```

**MPC设计：**
```python
class PathTrackingMPC:
    def __init__(self, L=2.5, dt=0.1, N=10):
        self.L = L    # 轴距
        self.dt = dt
        self.N = N

        # 权重
        self.Q = np.diag([1.0, 1.0, 0.5, 0.1])  # [x, y, ψ, v]
        self.R = np.diag([0.1, 0.1])            # [a, δ]

    def dynamics(self, state, control):
        x, y, psi, v = state
        a, delta = control

        x_dot = v * np.cos(psi)
        y_dot = v * np.sin(psi)
        psi_dot = v * np.tan(delta) / self.L
        v_dot = a

        return np.array([x_dot, y_dot, psi_dot, v_dot])

    def solve(self, state, ref_path):
        """求解MPC问题"""
        # 这里简化，实际需调用优化器
        pass
```

**约束：**
- 转角：|δ| ≤ 30°
- 加速度：-5 m/s² ≤ a ≤ 3 m/s²
- 速度：0 ≤ v ≤ 30 m/s
- 避障：与障碍物距离 > d_safe

### 6.2 四旋翼轨迹规划

**动力学模型：**
```
[简化] ẍ = (sin(θ)cos(φ) u_T) / m
       ÿ = (sin(φ) u_T) / m
       z̈ = (cos(θ)cos(φ) u_T - mg) / m

控制输入：[u_T, φ, θ, ψ̇]
```

**MPC目标：**
```
min  Σ ||p(k+i) - p_ref(k+i)||² + ||u(k+i)||²

s.t. 动力学约束
     ||u|| ≤ u_max
     ||φ||, ||θ|| ≤ angle_max
```

### 6.3 建筑能源管理

**系统：**
- 热动力学模型
- 电力系统模型
- 可再生能源（太阳能、风能）

**优化目标：**
```
min  Σ [电价(k+i) × 用电(k+i) + 不适度惩罚]

s.t. T_min ≤ 温度(k+i) ≤ T_max
     电池SOC约束
     功率平衡
```

**滚动优化：**
- 预测窗口：24小时
- 控制周期：15分钟
- 每小时更新预测

---

## 七、MPC实现优化

### 7.1 计算效率

**瓶颈：**在线优化求解

**加速方法：**

1. **温启动（Warm Start）**
   ```python
   U_init = np.vstack([U_prev[1:], U_prev[-1]])
   ```

2. **简化模型**
   - 线性化
   - 降维

3. **代码生成**
   - CVXGEN
   - ACADO
   - CasADi

4. **并行计算**
   - GPU加速
   - 多核并行

### 7.2 CasADi实现

```python
import casadi as ca

class CasADiMPC:
    def __init__(self, N, dt):
        self.N = N
        self.dt = dt

        # 符号变量
        self.x = ca.MX.sym('x', 4)  # 状态
        self.u = ca.MX.sym('u', 2)  # 控制

        # 动力学函数
        self.f = ca.Function('f', [self.x, self.u],
                            [self.dynamics(self.x, self.u)])

        # 构建NLP
        self.build_nlp()

    def dynamics(self, x, u):
        # 定义状态方程
        pass

    def build_nlp(self):
        # 优化变量
        U = ca.MX.sym('U', 2, self.N)
        X = ca.MX.sym('X', 4, self.N+1)

        # 目标函数
        J = 0
        g = []  # 约束

        for k in range(self.N):
            # 阶段代价
            J += ca.mtimes([X[:, k].T, self.Q, X[:, k]])
            J += ca.mtimes([U[:, k].T, self.R, U[:, k]])

            # 动力学约束
            x_next = X[:, k] + self.dt * self.f(X[:, k], U[:, k])
            g.append(X[:, k+1] - x_next)

        # 终端代价
        J += ca.mtimes([X[:, self.N].T, self.P, X[:, self.N]])

        # 创建NLP求解器
        nlp = {'x': ca.vertcat(ca.reshape(U, -1, 1),
                              ca.reshape(X, -1, 1)),
               'f': J,
               'g': ca.vertcat(*g)}

        self.solver = ca.nlpsol('solver', 'ipopt', nlp)
```

---

## 八、总结与展望

### 8.1 MPC的优势

1. **多变量协调**：自然处理MIMO
2. **约束处理**：显式优化约束
3. **最优性**：性能指标可调
4. **预测能力**：提前规划轨迹
5. **灵活性**：易于扩展

### 8.2 MPC的挑战

1. **计算负担**：实时优化求解
2. **模型依赖**：需要准确模型
3. **参数调节**：N, Q, R选择
4. **稳定性保证**：理论设计
5. **鲁棒性**：不确定性处理

### 8.3 未来方向

1. **学习与MPC融合**
   - 神经网络模型
   - 强化学习初始化
   - 在线自适应

2. **分布式MPC**
   - 多智能体协调
   - 大规模系统
   - 通信约束

3. **快速求解算法**
   - 实时迭代
   - 结构利用
   - 硬件加速

4. **安全关键系统**
   - 形式化验证
   - 安全约束
   - 故障处理

### 8.4 学习资源

**教材：**
- 《Model Predictive Control》- Camacho & Bordons
- 《Predictive Control for Linear and Hybrid Systems》- Borrelli et al.

**软件工具：**
- MATLAB MPC Toolbox
- CasADi
- ACADO Toolkit
- do-mpc

**在线课程：**
- Coursera: MPC Specialization
- ETH Zurich: Model Predictive Control

MPC是现代控制理论的明珠，从理论到应用都在快速发展。掌握MPC，你将拥有解决复杂控制问题的强大武器！
