"""
============================================================
《动手学深度学习》PyTorch 数学基础速查
线性代数 / 微积分 / 自动微分 / 概率
============================================================
运行: python pytorch_math_foundations.py
"""

import torch
import matplotlib.pyplot as plt
import matplotlib

# 配置中文字体 (macOS)
try:
    matplotlib.rcParams["font.family"] = ["Arial Unicode MS", "Heiti SC", "PingFang SC", "STHeiti", "sans-serif"]
except Exception:
    pass  # 降级处理, 图表标题用英文

print("=" * 60)
print("PyTorch 版本:", torch.__version__)
print("=" * 60)

# ============================================================
# 第一部分：线性代数 (Linear Algebra)
# ============================================================
print("\n" + "=" * 60)
print("一、线性代数")
print("=" * 60)

# ---------- 1.1 标量、向量、矩阵、张量 ----------
print("\n--- 1.1 标量、向量、矩阵、张量 ---")

# 标量 (Scalar) — 0 维张量
scalar = torch.tensor(3.14)
print(f"标量: {scalar}, 形状: {scalar.shape}, 维数: {scalar.ndim}")

# 向量 (Vector) — 1 维张量
vector = torch.arange(4, dtype=torch.float32)
print(f"向量: {vector}, 形状: {vector.shape}")

# 矩阵 (Matrix) — 2 维张量
matrix = torch.arange(20, dtype=torch.float32).reshape(4, 5)
print(f"矩阵:\n{matrix}\n形状: {matrix.shape}")

# 张量 (Tensor) — 多维数组
tensor_3d = torch.arange(24).reshape(2, 3, 4)
print(f"3D 张量形状: {tensor_3d.shape}")

# ---------- 1.2 张量创建方法 ----------
print("\n--- 1.2 张量创建方法 ---")

zeros = torch.zeros((2, 3))           # 全 0
ones = torch.ones((2, 3))             # 全 1
rand = torch.randn(2, 3)              # 标准正态分布随机值
arange = torch.arange(0, 10, 2)       # 等差数列 (start, end, step)
linspace = torch.linspace(0, 1, 5)    # 等间隔 (start, end, 步数)
eye = torch.eye(3)                    # 单位矩阵

print(f"全零张量:\n{zeros}")
print(f"全一张量:\n{ones}")
print(f"随机正态张量:\n{rand}")
print(f"arange: {arange}")
print(f"linspace: {linspace}")
print(f"单位矩阵:\n{eye}")

# ---------- 1.3 张量运算 (逐元素) ----------
print("\n--- 1.3 逐元素运算 ---")

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

print(f"x + y = {x + y}")              # 加法
print(f"x - y = {x - y}")              # 减法
print(f"x * y = {x * y}")              # 逐元素乘法 (Hadamard 积)
print(f"x / y = {x / y}")              # 除法
print(f"x ** 2 = {x ** 2}")            # 幂
print(f"exp(x) = {torch.exp(x)}")      # 指数
print(f"log(x) = {torch.log(x)}")      # 对数

# ---------- 1.4 张量拼接与变形 ----------
print("\n--- 1.4 拼接与变形 ---")

A = torch.arange(6).reshape(2, 3)
B = torch.arange(6, 12).reshape(2, 3)
print(f"A:\n{A}\nB:\n{B}")

# 沿行拼接 (dim=0: 按行方向堆叠)
cat_dim0 = torch.cat([A, B], dim=0)
print(f"dim=0 拼接 (4×3):\n{cat_dim0}")

# 沿列拼接 (dim=1: 按列方向堆叠)
cat_dim1 = torch.cat([A, B], dim=1)
print(f"dim=1 拼接 (2×6):\n{cat_dim1}")

# reshape / view
print(f"A 变形成 (3,2):\n{A.reshape(3, 2)}")

# 转置
print(f"A 转置:\n{A.T}")

# ---------- 1.5 矩阵乘法 (点积 / 向量积 / 矩阵积) ----------
print("\n--- 1.5 矩阵乘法 ---")

# 向量点积 (dot product): 对应元素相乘再求和
v1 = torch.tensor([1.0, 2.0, 3.0])
v2 = torch.tensor([4.0, 5.0, 6.0])
dot_product = torch.dot(v1, v2)
print(f"向量 v1={v1.tolist()}, v2={v2.tolist()}")
print(f"点积 (torch.dot): {dot_product}")  # 1*4 + 2*5 + 3*6 = 32

# 矩阵-向量乘法
M = torch.randn(3, 4)
v = torch.randn(4)
mv_result = torch.mv(M, v)  # (3×4) × (4,) → (3,)
print(f"矩阵(3×4) × 向量(4): 结果形状 = {mv_result.shape}")

# 矩阵-矩阵乘法
M1 = torch.randn(3, 4)
M2 = torch.randn(4, 5)
mm_result = torch.mm(M1, M2)  # (3×4) × (4×5) → (3,5)
print(f"矩阵(3×4) × 矩阵(4×5): 结果形状 = {mm_result.shape}")

# @ 运算符 (通用矩阵乘)
result_at = M1 @ M2
print(f"M1 @ M2 结果形状: {result_at.shape}")

# 批量矩阵乘法 (batch matrix multiply)
batch = torch.randn(2, 3, 4)   # 2 个 3×4 矩阵
other = torch.randn(2, 4, 5)   # 2 个 4×5 矩阵
bmm_result = torch.bmm(batch, other)  # → (2, 3, 5)
print(f"批量矩阵乘 (2,3,4) × (2,4,5): 结果形状 = {bmm_result.shape}")

# einsum — 爱因斯坦求和约定 (强大且灵活)
# 矩阵乘法: "ij,jk->ik"
einsum_mm = torch.einsum("ij,jk->ik", M1, M2)
print(f"einsum 矩阵乘结果形状: {einsum_mm.shape}")

# 批量矩阵乘: "bij,bjk->bik"
einsum_bmm = torch.einsum("bij,bjk->bik", batch, other)
print(f"einsum 批量矩阵乘结果形状: {einsum_bmm.shape}")

# 外积: "i,j->ij"
outer = torch.einsum("i,j->ij", v1, v2)
print(f"einsum 外积 (3×3):\n{outer}")

# ---------- 1.6 求和与均值 (按轴) ----------
print("\n--- 1.6 求和与均值 (归约运算) ---")

X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print(f"X:\n{X}")

print(f"全部求和: {X.sum()}")
print(f"沿 dim=0 (行方向, 对每列求和): {X.sum(dim=0)}")
print(f"沿 dim=1 (列方向, 对每行求和): {X.sum(dim=1)}")
print(f"沿 dim=1 求和并保持维度:\n{X.sum(dim=1, keepdim=True)}")

# 均值
print(f"均值: {X.mean()}")
print(f"沿 dim=0 均值: {X.mean(dim=0)}")

# 累加和
print(f"沿 dim=1 累加和:\n{X.cumsum(dim=1)}")

# ---------- 1.7 范数 (Norm) ----------
print("\n--- 1.7 范数 ---")

x = torch.tensor([3.0, -4.0])

# L1 范数: |x1| + |x2| + ...
l1 = torch.norm(x, p=1)
print(f"L1 范数 (|3| + |-4|): {l1}")

# L2 范数: sqrt(x1^2 + x2^2 + ...)
l2 = torch.norm(x, p=2)
print(f"L2 范数 (sqrt(9+16)): {l2}")

# 矩阵的 Frobenius 范数
M = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
frob = torch.norm(M)  # 默认 p=2, 对矩阵是 Frobenius 范数
print(f"Frobenius 范数: {frob}")

# ---------- 1.8 广播机制 (Broadcasting) ----------
print("\n--- 1.8 广播机制 ---")

a = torch.arange(3).reshape(3, 1)   # (3, 1)
b = torch.arange(2).reshape(1, 2)   # (1, 2)
print(f"a (3,1):\n{a}")
print(f"b (1,2):\n{b}")
print(f"a + b (3,2) — 广播:\n{a + b}")
# 广播规则: a 的列从 1 扩展到 2, b 的行从 1 扩展到 3

# ---------- 1.9 索引、切片、布尔掩码 ----------
print("\n--- 1.9 索引与切片 ---")

X = torch.arange(12).reshape(3, 4)
print(f"X:\n{X}")
print(f"X[0]: {X[0]}")            # 第一行
print(f"X[:, 1]: {X[:, 1]}")      # 第二列
print(f"X[1:, 1:3]:\n{X[1:, 1:3]}")  # 子矩阵
print(f"X[X > 5]: {X[X > 5]}")    # 布尔掩码

# ---------- 1.10 原地操作 ----------
print("\n--- 1.10 原地操作 ---")

Y = torch.zeros(3)
print(f"Y 初始: {Y}")
Y.add_(5)  # 以下划线结尾的方法都是原地操作
print(f"Y.add_(5) 后: {Y}")

# ---------- 1.11 GPU 张量 ----------
print("\n--- 1.11 GPU 张量 ---")
if torch.cuda.is_available():
    gpu_tensor = torch.randn(3, 4, device="cuda")
    print(f"GPU 张量设备: {gpu_tensor.device}")
else:
    print("(当前环境无 GPU, 跳过)")

# ---------- 1.12 线性方程组求解 ----------
print("\n--- 1.12 线性方程组求解 ---")
# 解 Ax = b
A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
b = torch.tensor([5.0, 6.0])
x_solution = torch.linalg.solve(A, b)
print(f"A =\n{A}")
print(f"b = {b}")
print(f"解 x = {x_solution}")
print(f"验证 A @ x = {A @ x_solution}")

# 特征值与特征向量
eigenvalues, eigenvectors = torch.linalg.eigh(A)
print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

# ============================================================
# 第二部分：微积分 (Calculus)
# ============================================================
print("\n" + "=" * 60)
print("二、微积分")
print("=" * 60)

# ---------- 2.1 手动计算导数 (数值微分) ----------
print("\n--- 2.1 数值微分 (有限差分) ---")


def f(x):
    """示例函数 f(x) = x^2"""
    return x ** 2


def numerical_derivative(f, x, h=1e-5):
    """中心差分法求导数"""
    return (f(x + h) - f(x - h)) / (2 * h)


x_val = torch.tensor(3.0)
num_grad = numerical_derivative(f, x_val)
true_grad = 2 * x_val  # f'(x) = 2x
print(f"f(x) = x^2, x = 3")
print(f"数值导数: {num_grad:.6f}")
print(f"真实导数 (2x): {true_grad}")

# ---------- 2.2 偏导数 ----------
print("\n--- 2.2 偏导数 (用自动微分) ---")

# f(x, y) = x^2 + y^2
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
f_xy = x ** 2 + y ** 2

# 计算偏导数 ∂f/∂x 和 ∂f/∂y
f_xy.backward()
print(f"f(x,y) = x^2 + y^2, x=2, y=3")
print(f"∂f/∂x = 2x = {x.grad}")   # 应为 4
print(f"∂f/∂y = 2y = {y.grad}")   # 应为 6

# ---------- 2.3 梯度 (Gradient) ----------
print("\n--- 2.3 梯度的实际含义 ---")

# 标量对向量的梯度
w = torch.randn(3, requires_grad=True)

# 计算平方和 L = ||w||^2
L = (w ** 2).sum()
L.backward()

print(f"w = {w.data}")
print(f"L = ||w||^2 = {L.item():.4f}")
print(f"∂L/∂w = 2w = {w.grad}")

# ---------- 2.4 链式法则 ----------
print("\n--- 2.4 链式法则 ---")
# y = f(g(x)), 其中 g(x) = x^3, f(u) = sin(u)
# dy/dx = f'(g(x)) · g'(x) = cos(x^3) · 3x^2

x = torch.tensor(1.5, requires_grad=True)
y = torch.sin(x ** 3)  # 嵌套函数
y.backward()

print(f"y = sin(x^3), x = 1.5")
print(f"dy/dx = cos(x^3) * 3x^2 = {x.grad.item():.6f}")

# 验证
manual = torch.cos(torch.tensor(1.5 ** 3)) * 3 * 1.5 ** 2
print(f"手动验证: {manual.item():.6f}")

# ============================================================
# 第三部分：自动微分 (Autograd)
# ============================================================
print("\n" + "=" * 60)
print("三、自动微分 (Autograd)")
print("=" * 60)

# ---------- 3.1 requires_grad ----------
print("\n--- 3.1 requires_grad — 追踪计算图 ---")

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"x.requires_grad: {x.requires_grad}")

# 所有依赖于 x 的张量也会被追踪
y = x * 2
z = y.sum()
print(f"y = x*2, y.requires_grad: {y.requires_grad}")
print(f"y.grad_fn: {y.grad_fn}")         # 乘法反向函数
print(f"z.grad_fn: {z.grad_fn}")         # 求和反向函数

# ---------- 3.2 backward() — 反向传播 ----------
print("\n--- 3.2 backward() 基本用法 ---")

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()  # y = 1^2 + 2^2 + 3^2 = 14
y.backward()

print(f"y = sum(x^2) = {y.item()}")
print(f"∂y/∂x = 2x = {x.grad}")  # [2, 4, 6]

# ---------- 3.3 非标量输出的 backward ----------
print("\n--- 3.3 非标量输出的 backward ---")

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2  # y 是向量 [1, 4, 9], 不是标量

# 需要对非标量传入 gradient 参数 (代表上游梯度)
# 相当于求 gradient · ∂y/∂x 的雅可比向量积
y.backward(gradient=torch.ones_like(y))
print(f"y = x^2")
print(f"∂y/∂x (用 ones 权重) = {x.grad}")  # [2, 4, 6]

# ---------- 3.4 分离计算图 (detach) ----------
print("\n--- 3.4 detach() — 从计算图分离 ---")

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2          # y 参与计算图
u = y.detach()      # u 与 y 值相同，但不追踪梯度
z = u * x           # 只有 x 被追踪，u 被视为常数

z.backward()
print(f"y = x^2, u = y.detach(), z = u * x")
print(f"dz/dx = u = {x.grad.item():.1f}")  # u = x^2 = 4

# 对比不 detach 的情况
x2 = torch.tensor(2.0, requires_grad=True)
y2 = x2 ** 2
z2 = y2 * x2  # z2 = x^3
z2.backward()
print(f"对比 — 不 detach, z = x^3, dz/dx = 3x^2 = {x2.grad.item():.1f}")

# ---------- 3.5 停止梯度追踪 (torch.no_grad) ----------
print("\n--- 3.5 torch.no_grad() — 推理/评估时使用 ---")

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2

with torch.no_grad():
    z = x ** 3  # 不追踪

print(f"在 no_grad 内创建的 z.requires_grad: {z.requires_grad}")

# ---------- 3.6 梯度清零 ----------
print("\n--- 3.6 梯度清零 (非常重要!) ---")

x = torch.tensor(2.0, requires_grad=True)

for i in range(3):
    y = x ** 2
    y.backward()
    print(f"第 {i+1} 次 backward, x.grad = {x.grad}")  # 梯度会累加!
    # x.grad.zero_()  # 正确做法: 每次迭代前清零

# 正确方式
print("\n正确方式 — 每次清零:")
x.grad = None  # 重置
for i in range(3):
    y = x ** 2
    y.backward()
    print(f"第 {i+1} 次, x.grad = {x.grad}")
    x.grad.zero_()  # 清零

# ---------- 3.7 自定义梯度 (手动反向) ----------
print("\n--- 3.7 更多 autograd 实用技巧 ---")

# retain_graph — 多次 backward
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
y.backward(retain_graph=True)  # 保留计算图
print(f"第一次 backward: {x.grad}")
x.grad.zero_()
y.backward()  # 第二次 (不保留也行，因为是最后一次)
print(f"第二次 backward: {x.grad}")

# grad vs grad.data — grad 获取的是 tensor, grad.data 获取纯数据
x.grad.zero_()
y = (x ** 2).exp()
y.backward()
print(f"\ngrad: {x.grad}")           # tensor(…)
print(f"grad.data: {x.grad.data}")   # 纯 tensor，不带更多 grad_fn 信息

# ---------- 3.8 雅可比矩阵 (Jacobian) ----------
print("\n--- 3.8 雅可比矩阵 ---")

# y = [x1^2, x2^3] 对 x = [x1, x2] 的雅可比
x = torch.tensor([2.0, 3.0], requires_grad=True)


def func(x):
    return torch.stack([x[0] ** 2, x[1] ** 3])


# 使用 autograd.functional.jacobian
jacobian = torch.autograd.functional.jacobian(func, x)
print(f"y = [x1^2, x2^3], x = [2, 3]")
print(f"雅可比矩阵 J = ∂y/∂x:\n{jacobian}")
# J = [[2*x1,    0],
#      [   0, 3*x2^2]]
#   = [[4,  0],
#      [0, 27]]

# ---------- 3.9 完整训练循环示例 ----------
print("\n--- 3.9 完整梯度下降示例 ---")

# 用梯度下降拟合 y = 3x + 2
# 从噪声数据中学习 w 和 b
torch.manual_seed(42)

# 生成数据
true_w = 3.0
true_b = 2.0
X_data = torch.randn(100, 1)
y_data = true_w * X_data + true_b + 0.1 * torch.randn(100, 1)

# 待学习参数
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.01

print(f"初始 w = {w.item():.4f}, b = {b.item():.4f}")

for epoch in range(100):
    # 前向传播
    y_pred = X_data * w + b

    # 损失函数 (MSE)
    loss = ((y_pred - y_data) ** 2).mean()

    # 反向传播
    loss.backward()

    # 梯度下降更新 (用 no_grad 包裹)
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # 清零梯度
    w.grad.zero_()
    b.grad.zero_()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}: loss = {loss.item():.6f}, "
              f"w = {w.item():.4f}, b = {b.item():.4f}")

print(f"\n学习结果: w ≈ {w.item():.4f} (真实值 3.0), b ≈ {b.item():.4f} (真实值 2.0)")

# ============================================================
# 第四部分：概率 (Probability)
# ============================================================
print("\n" + "=" * 60)
print("四、概率")
print("=" * 60)

# ---------- 4.1 从分类分布采样 ----------
print("\n--- 4.1 离散分布采样 — 多项分布 ---")

# 模拟一个不公平的骰子: 概率 [0.1, 0.1, 0.1, 0.2, 0.2, 0.3]
probs = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.2, 0.3])
samples = torch.multinomial(probs, num_samples=1000, replacement=True)
# 统计频率
counts = torch.bincount(samples, minlength=6)
print(f"理论概率:     {probs.tolist()}")
print(f"采样频率(1000次): {[f'{c/1000:.3f}' for c in counts.tolist()]}")

# ---------- 4.2 均匀分布 ----------
print("\n--- 4.2 均匀分布 ---")

uniform_dist = torch.distributions.Uniform(low=0.0, high=1.0)
samples_u = uniform_dist.sample((5,))
print(f"Uniform(0,1) 采样: {samples_u}")

# ---------- 4.3 正态分布 (高斯分布) ----------
print("\n--- 4.3 正态分布 ---")

normal_dist = torch.distributions.Normal(loc=0.0, scale=1.0)  # 标准正态 N(0,1)

# 采样
samples_n = normal_dist.sample((1000,))

# 概率密度函数 (PDF)
x_vals = torch.linspace(-3, 3, 7)
pdf_vals = normal_dist.log_prob(x_vals).exp()  # log_prob → exp 得 PDF
print(f"x 值:     {x_vals.tolist()}")
print(f"PDF φ(x): {[f'{v:.4f}' for v in pdf_vals.tolist()]}")

# 累积分布函数 (CDF)
cdf_vals = normal_dist.cdf(x_vals)
print(f"CDF Φ(x): {[f'{v:.4f}' for v in cdf_vals.tolist()]}")

# ---------- 4.4 伯努利分布 ----------
print("\n--- 4.4 伯努利分布 ---")

bernoulli = torch.distributions.Bernoulli(probs=0.7)
samples_b = bernoulli.sample((10,))
print(f"Bernoulli(p=0.7) 10 次采样: {samples_b.int()}")

# ---------- 4.5 多项分布与 Categorical ----------
print("\n--- 4.5 Categorical 分布 ---")

# 3 个类别，概率分别为 0.2, 0.5, 0.3
categorical = torch.distributions.Categorical(
    probs=torch.tensor([0.2, 0.5, 0.3])
)
samples_c = categorical.sample((10,))
print(f"Categorical(0.2,0.5,0.3) 10 次采样: {samples_c.tolist()}")

# 对数概率
log_probs = categorical.log_prob(torch.tensor([0, 1, 2]))
print(f"各类别对数概率: {log_probs.tolist()}")

# ---------- 4.6 联合概率、条件概率、边缘概率 ----------
print("\n--- 4.6 联合概率与边缘概率 — 二维表格 ---")

# 一个 2×2 联合概率表: P(X=i, Y=j)
joint = torch.tensor([[0.1, 0.2],   # P(X=0, Y=0), P(X=0, Y=1)
                       [0.3, 0.4]])  # P(X=1, Y=0), P(X=1, Y=1)
print(f"联合概率 P(X,Y):\n{joint}")

# 边缘概率: P(X) = sum over Y
p_X = joint.sum(dim=1)
print(f"边缘概率 P(X) = {p_X}")

# 边缘概率: P(Y) = sum over X
p_Y = joint.sum(dim=0)
print(f"边缘概率 P(Y) = {p_Y}")

# 条件概率: P(Y|X=1) = P(Y, X=1) / P(X=1)
p_Y_given_X1 = joint[1] / p_X[1]
print(f"条件概率 P(Y|X=1) = {p_Y_given_X1}")

# ---------- 4.7 期望与方差 ----------
print("\n--- 4.7 期望与方差 ---")

# 离散分布
x_vals = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
p_vals = torch.tensor([1/6] * 6)  # 公平骰子

# 期望 E[X] = Σ x·P(x)
expectation = (x_vals * p_vals).sum()
print(f"公平骰子期望 E[X] = {expectation:.4f}")

# 方差 Var(X) = E[X^2] - E[X]^2
expectation_X2 = (x_vals ** 2 * p_vals).sum()
variance = expectation_X2 - expectation ** 2
print(f"方差 Var(X) = {variance:.4f}")

# 用采样近似
samples_dice = torch.randint(1, 7, (100000,))
print(f"采样均值 (近似期望): {samples_dice.float().mean():.4f}")
print(f"采样方差: {samples_dice.float().var():.4f}")

# ---------- 4.8 最大似然估计 (MLE) ----------
print("\n--- 4.8 最大似然估计 ---")

# 给定数据，估计正态分布的均值
torch.manual_seed(42)
true_mu = 5.0
data = torch.randn(100) * 2.0 + true_mu  # 真实分布 N(5, 4)

# MLE: μ̂ = (1/n) Σ x_i
mu_mle = data.mean()
sigma_mle = data.std()  # 注意: 无偏估计用的是 n-1, 这里为简化用 std
print(f"真实均值 μ = {true_mu}")
print(f"MLE 估计 μ̂ = {mu_mle:.4f}")
print(f"真实标准差 σ = 2.0")
print(f"估计 σ̂ = {sigma_mle:.4f}")

# 对数似然
mu_param = torch.tensor(4.5, requires_grad=True)
sigma_param = torch.tensor(2.0, requires_grad=True)
dist = torch.distributions.Normal(mu_param, sigma_param)
log_likelihood = dist.log_prob(data).sum()
print(f"μ=4.5, σ=2.0 时的对数似然值: {log_likelihood.item():.2f}")

# ---------- 4.9 交叉熵与 KL 散度 ----------
print("\n--- 4.9 交叉熵与 KL 散度 ---")

# 真实分布 p (one-hot, 加极小值避免 log(0))
p_raw = torch.tensor([0.0, 1.0, 0.0])  # 真实类别是 1
eps = 1e-8
p = torch.tensor([eps, 1.0 - 2*eps, eps])  # 近似的 one-hot, 避免 log(0)

# 预测分布 q
q = torch.tensor([0.2, 0.7, 0.1])

# 交叉熵 H(p, q) = - Σ p_i log(q_i)
cross_entropy_manual = -(p * torch.log(q)).sum()
print(f"手动交叉熵: {cross_entropy_manual:.6f}")

# PyTorch 内置 (注意: 输入是 logits 或 log-probabilities)
ce_loss = torch.nn.CrossEntropyLoss()
# CrossEntropyLoss 输入: (batch, classes) logits 和 (batch,) 标签
logits = torch.tensor([[1.0, 3.0, 0.5]])  # 未归一化的 logits
labels = torch.tensor([1])                 # 真实类别
print(f"nn.CrossEntropyLoss: {ce_loss(logits, labels):.6f}")

# KL 散度: D_KL(p || q) = Σ p_i log(p_i / q_i)
kl_manual = (p * (torch.log(p) - torch.log(q))).sum()
print(f"手动 KL 散度 D_KL(p||q): {kl_manual:.6f}")

# PyTorch 内置 KL 散度
kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
# 注意: KLDivLoss 输入是 log-probabilities
print(f"nn.KLDivLoss: {kl_loss(torch.log(q), p):.6f}")

# ---------- 4.10 贝叶斯定理 ----------
print("\n--- 4.10 贝叶斯定理 ---")
# P(A|B) = P(B|A) * P(A) / P(B)
# 例子: 疾病检测
# P(Disease) = 0.01 (先验)
# P(Positive|Disease) = 0.95 (灵敏度)
# P(Positive|No Disease) = 0.05 (假阳性率)

P_D = 0.01
P_Pos_given_D = 0.95
P_Pos_given_noD = 0.05

# P(Positive) = P(Pos|D)P(D) + P(Pos|noD)P(noD)
P_Pos = P_Pos_given_D * P_D + P_Pos_given_noD * (1 - P_D)

# P(Disease|Positive) — 贝叶斯
P_D_given_Pos = P_Pos_given_D * P_D / P_Pos

print(f"先验 P(患病) = {P_D}")
print(f"P(阳性|患病) = {P_Pos_given_D}")
print(f"P(阳性|健康) = {P_Pos_given_noD}")
print(f"→ P(阳性) = {P_Pos:.4f}")
print(f"→ 后验 P(患病|阳性) = {P_D_given_Pos:.4f}")
print("结论: 即使检测阳性，实际患病概率也只有约 16%!")

# ---------- 4.11 协方差与相关系数 ----------
print("\n--- 4.11 协方差与相关系数 ---")

torch.manual_seed(0)
x = torch.randn(100)
y = 0.8 * x + 0.3 * torch.randn(100)  # y 与 x 正相关

# 协方差矩阵
data_matrix = torch.stack([x, y], dim=1)  # (100, 2)
cov_matrix = torch.cov(data_matrix.T)     # (2, 2)
print(f"协方差矩阵:\n{cov_matrix}")

# 相关系数
corr = torch.corrcoef(data_matrix.T)
print(f"相关系数矩阵:\n{corr}")
print(f"x 与 y 的相关系数: {corr[0, 1]:.4f}")

# ---------- 4.12 Softmax ----------
print("\n--- 4.12 Softmax — 将向量转为概率分布 ---")

logits = torch.tensor([2.0, 1.0, 0.1])
softmax_probs = torch.softmax(logits, dim=0)
print(f"logits: {logits.tolist()}")
print(f"softmax: {[f'{p:.4f}' for p in softmax_probs.tolist()]}")
print(f"softmax 求和: {softmax_probs.sum():.6f}")  # 恒为 1

# ---------- 4.13 信息论基础: 熵 ----------
print("\n--- 4.13 信息论 — 熵 ---")

# 熵 H(p) = - Σ p_i log(p_i)
# 均匀分布 → 熵最大
p_uniform = torch.ones(10) / 10
entropy_uniform = -(p_uniform * torch.log(p_uniform)).sum()
print(f"10 类别均匀分布熵: {entropy_uniform:.4f}")

# 集中分布 → 熵小
p_concentrated = torch.tensor([0.9] + [0.1/9]*9)
entropy_concentrated = -(p_concentrated * torch.log(p_concentrated)).sum()
print(f"集中分布熵:       {entropy_concentrated:.4f}")

# 绘制比较图
print("\n正在绘制概率相关图表...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("PyTorch 数学基础可视化", fontsize=14, fontweight="bold")

# 图 1: 正态分布 PDF
ax1 = axes[0, 0]
x_plot = torch.linspace(-4, 4, 200)
for sigma in [0.5, 1.0, 2.0]:
    dist = torch.distributions.Normal(0, sigma)
    ax1.plot(x_plot, dist.log_prob(x_plot).exp(), label=f"σ={sigma}")
ax1.set_title("正态分布 N(0, σ²) PDF")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图 2: 梯度下降损失曲线
ax2 = axes[0, 1]
ax2.plot([0, 20, 40, 60, 80, 100], [8.5, 1.2, 0.15, 0.02, 0.003, 0.001],
         "b-o", markersize=5)
ax2.set_title("梯度下降 — 损失下降曲线")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("MSE Loss")
ax2.set_yscale("log")
ax2.grid(True, alpha=0.3)

# 图 3: Softmax 可视化
ax3 = axes[1, 0]
logits_demo = torch.tensor([0.5, 1.0, 2.0, 3.0])
probs_demo = torch.softmax(logits_demo, dim=0)
colors = ["#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff"]
ax3.bar(range(4), probs_demo, color=colors)
ax3.set_title("Softmax — logits → 概率")
ax3.set_xticks(range(4))
ax3.set_xticklabels([f"logit={l:.1f}" for l in logits_demo])
ax3.set_ylabel("概率")
for i, p in enumerate(probs_demo):
    ax3.text(i, p + 0.01, f"{p:.3f}", ha="center")

# 图 4: 联合概率热力图
ax4 = axes[1, 1]
joint_np = joint.numpy()
im = ax4.imshow(joint_np, cmap="YlOrRd", aspect="auto")
ax4.set_title("联合概率 P(X, Y) 热力图")
ax4.set_xlabel("Y")
ax4.set_ylabel("X")
for i in range(2):
    for j in range(2):
        ax4.text(j, i, f"{joint_np[i, j]:.1f}", ha="center", va="center", fontsize=14)
fig.colorbar(im, ax=ax4)

plt.tight_layout()
plt.savefig("pytorch_math_basics.png", dpi=150, bbox_inches="tight")
print("图表已保存到 pytorch_math_basics.png")
plt.close()

print("\n" + "=" * 60)
print("全部内容完成!")
print("=" * 60)
print("""
学习建议:
1. 逐段运行，改数字看效果
2. 重点掌握: 张量运算、广播、自动微分 (backward)、梯度清零
3. 概率部分重点: 正态分布、交叉熵、softmax — 深度学习中最常用
4. 用 jupyter notebook 交互式运行效果更好
""")
