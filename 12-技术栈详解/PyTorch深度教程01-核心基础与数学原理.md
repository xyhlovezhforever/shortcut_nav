# PyTorch深度教程（一）：核心基础与数学原理

> **目标读者**：具有Python基础，追求十年经验级别的深度理解
> **学习路径**：从数学原理到工程实践的完整闭环

---

## 第一部分：深度学习的数学基础

### 1.1 线性代数核心概念

#### 1.1.1 向量空间与张量

**数学定义**
```
向量空间 V 上的张量是一个多线性映射
- T: V* × V* × ... × V* × V × V × ... × V → ℝ
- 其中 V* 是 V 的对偶空间
```

**纯数学例子：向量与矩阵的基本运算**

**例1：向量加法与数乘**
```
给定向量 v₁ = [2, 3, -1] 和 v₂ = [4, -2, 5]

向量加法：
v₁ + v₂ = [2+4, 3+(-2), -1+5] = [6, 1, 4]

数乘（标量乘法）：
3v₁ = [3×2, 3×3, 3×(-1)] = [6, 9, -3]

线性组合：
2v₁ + 3v₂ = 2[2, 3, -1] + 3[4, -2, 5]
          = [4, 6, -2] + [12, -6, 15]
          = [16, 0, 13]
```

**例2：向量点积（内积）**
```
给定 a = [1, 2, 3] 和 b = [4, 5, 6]

点积计算：
a · b = (1)(4) + (2)(5) + (3)(6)
      = 4 + 10 + 18
      = 32

几何意义：
a · b = ||a|| ||b|| cos(θ)
其中 ||a|| = √(1² + 2² + 3²) = √14 ≈ 3.742
    ||b|| = √(4² + 5² + 6²) = √77 ≈ 8.775

cos(θ) = 32 / (√14 × √77) ≈ 0.974
θ ≈ 13°（两向量夹角很小，几乎平行）
```

**例3：向量范数（长度）**
```
给定向量 v = [3, 4, 12]

L₁范数（曼哈顿距离）：
||v||₁ = |3| + |4| + |12| = 19

L₂范数（欧几里得距离）：
||v||₂ = √(3² + 4² + 12²) = √(9 + 16 + 144) = √169 = 13

L∞范数（最大绝对值）：
||v||∞ = max(|3|, |4|, |12|) = 12

单位向量（归一化）：
v̂ = v / ||v||₂ = [3/13, 4/13, 12/13] ≈ [0.231, 0.308, 0.923]
验证：||v̂||₂ = 1
```

**例4：矩阵-向量乘法**
```
矩阵 A = [ 1  2  3 ]    向量 x = [ 2 ]
         [ 4  5  6 ]            [ 1 ]
         [ 7  8  9 ]            [ 3 ]

计算 Ax：
Ax = [ (1×2 + 2×1 + 3×3) ]   [ 2 + 2 + 9  ]   [ 13 ]
     [ (4×2 + 5×1 + 6×3) ] = [ 8 + 5 + 18 ] = [ 31 ]
     [ (7×2 + 8×1 + 9×3) ]   [ 14 + 8 + 27]   [ 49 ]

几何意义：
- 第1行：A的第1行与x的点积 → 输出的第1个分量
- 第2行：A的第2行与x的点积 → 输出的第2个分量
- 第3行：A的第3行与x的点积 → 输出的第3个分量
```

**例5：矩阵乘法**
```
A = [ 1  2 ]    B = [ 5  6 ]
    [ 3  4 ]        [ 7  8 ]

计算 AB：
AB = [ (1×5 + 2×7)  (1×6 + 2×8) ]   [ 5+14   6+16  ]   [ 19  22 ]
     [ (3×5 + 4×7)  (3×6 + 4×8) ] = [ 15+28  18+32 ] = [ 43  50 ]

规则：第i行第j列 = A的第i行 与 B的第j列的点积

验证不可交换性（AB ≠ BA）：
BA = [ 5  6 ][ 1  2 ]   [ 5+18   10+24 ]   [ 23  34 ]
     [ 7  8 ][ 3  4 ] = [ 7+24   14+32 ] = [ 31  46 ]

AB ≠ BA ✓
```

**例6：矩阵转置**
```
A = [ 1  2  3 ]    Aᵀ = [ 1  4 ]
    [ 4  5  6 ]          [ 2  5 ]
                         [ 3  6 ]

性质验证：
1. (Aᵀ)ᵀ = A
2. (AB)ᵀ = BᵀAᵀ

例如，若 A = [1 2], B = [5]，则
              [3 4]      [6]

AB = [1×5+2×6]   [17]
     [3×5+4×6] = [39]

(AB)ᵀ = [17  39]

BᵀAᵀ = [5  6][1  3] = [5+18  15+24] = [23  39]... 等等
                [2  4]

等等，让我重新验证维度...
实际上 A(2×2) × B(2×1) = C(2×1)
所以 (AB)ᵀ = C ᵀ(1×2)
    BᵀAᵀ = (1×2)(2×2) = (1×2) ✓
```

**例7：矩阵的迹（Trace）**
```
A = [ 2  3  1 ]
    [ 0  5  4 ]
    [ 6  7  8 ]

trace(A) = 对角线元素之和 = 2 + 5 + 8 = 15

性质：
1. trace(A + B) = trace(A) + trace(B)
2. trace(cA) = c·trace(A)
3. trace(AB) = trace(BA)
4. trace(A) = 所有特征值之和
```

**例8：向量的外积（张量积）**
```
u = [1]    v = [4]
    [2]        [5]
    [3]        [6]

外积 u ⊗ v = uvᵀ：
[1]                [1×4  1×5  1×6]   [ 4   5   6]
[2] [4  5  6]  =   [2×4  2×5  2×6] = [ 8  10  12]
[3]                [3×4  3×5  3×6]   [12  15  18]

注意：rank(u ⊗ v) = 1（秩1矩阵）
```

**例9：向量投影**
```
将向量 a = [2, 1] 投影到向量 b = [3, 4] 上

投影公式：
proj_b(a) = (a·b / b·b) × b

计算：
a·b = 2×3 + 1×4 = 10
b·b = 3² + 4² = 25

proj_b(a) = (10/25) × [3, 4] = 0.4 × [3, 4] = [1.2, 1.6]

验证：投影向量与b平行
[1.2, 1.6] = 0.4 × [3, 4] ✓

垂直分量：
a - proj_b(a) = [2, 1] - [1.2, 1.6] = [0.8, -0.6]

验证垂直：
[0.8, -0.6] · [3, 4] = 2.4 - 2.4 = 0 ✓
```

**例10：三维空间中的叉积**
```
a = [1, 2, 3]  b = [4, 5, 6]

叉积 a × b：
a × b = | i   j   k  |
        | 1   2   3  |
        | 4   5   6  |

= i(2×6 - 3×5) - j(1×6 - 3×4) + k(1×5 - 2×4)
= i(12 - 15) - j(6 - 12) + k(5 - 8)
= -3i + 6j - 3k
= [-3, 6, -3]

验证垂直性：
(a × b) · a = (-3)(1) + (6)(2) + (-3)(3) = -3 + 12 - 9 = 0 ✓
(a × b) · b = (-3)(4) + (6)(5) + (-3)(6) = -12 + 30 - 18 = 0 ✓

叉积的模：
||a × b|| = ||a|| ||b|| sin(θ)
         = √14 × √77 × sin(θ)
         = √(9 + 36 + 9) = √54 ≈ 7.35
```

---

**PyTorch中的张量实现**
```python
import torch
import numpy as np

# 张量的本质：多维数组 + 计算图节点
class TensorInternals:
    """
    PyTorch张量的内部结构理解
    """
    def demonstrate_tensor_properties(self):
        # 1. 存储视图（Storage View）
        x = torch.tensor([[1, 2], [3, 4]])
        print(f"数据指针: {x.data_ptr()}")
        print(f"存储对象: {x.storage()}")
        print(f"步幅(stride): {x.stride()}")
        print(f"形状(shape): {x.shape}")

        # 2. 视图操作不复制数据
        y = x.view(4)  # 共享底层存储
        print(f"是否共享存储: {x.data_ptr() == y.data_ptr()}")

        # 3. 连续性（Contiguous）
        z = x.t()  # 转置改变步幅，可能不连续
        print(f"是否连续: {z.is_contiguous()}")
        z_cont = z.contiguous()  # 强制连续化

# 数学原理：向量的范数
def vector_norms_deep_dive():
    """
    向量范数的数学定义与计算
    """
    x = torch.tensor([3.0, 4.0])

    # L1范数：||x||₁ = Σ|xᵢ|
    l1_norm = torch.norm(x, p=1)
    print(f"L1范数: {l1_norm}")  # 7.0

    # L2范数（欧几里得范数）：||x||₂ = √(Σxᵢ²)
    l2_norm = torch.norm(x, p=2)
    print(f"L2范数: {l2_norm}")  # 5.0

    # L∞范数：||x||∞ = max|xᵢ|
    linf_norm = torch.norm(x, p=float('inf'))
    print(f"L∞范数: {linf_norm}")  # 4.0

    # Frobenius范数（矩阵）：||A||F = √(Σᵢⱼ aᵢⱼ²)
    A = torch.tensor([[1., 2.], [3., 4.]])
    frob_norm = torch.norm(A, p='fro')
    print(f"Frobenius范数: {frob_norm}")

# 手动实现以理解原理
def manual_l2_norm(x):
    """手动实现L2范数计算"""
    return torch.sqrt(torch.sum(x ** 2))

# 更多范数运算示例
def comprehensive_norm_examples():
    """
    全面的范数运算示例
    """
    # 1. 向量范数计算
    v = torch.tensor([1., 2., 3., 4., 5.])

    # 不同p值的Lp范数
    for p in [1, 2, 3, 4, float('inf')]:
        norm = torch.norm(v, p=p)
        print(f"L{p}范数: {norm:.4f}")

    # 2. 矩阵范数
    M = torch.randn(4, 5)

    # Frobenius范数（默认）
    frob = torch.norm(M)
    print(f"Frobenius范数: {frob:.4f}")

    # 核范数（奇异值之和）
    nuclear = torch.norm(M, p='nuc')
    print(f"核范数: {nuclear:.4f}")

    # 算子范数（最大奇异值）
    operator = torch.norm(M, p=2)
    print(f"算子范数: {operator:.4f}")

    # 3. 批量范数计算
    batch_vectors = torch.randn(32, 128)  # 32个128维向量
    batch_norms = torch.norm(batch_vectors, dim=1)  # 每个向量的L2范数
    print(f"批量范数形状: {batch_norms.shape}")  # (32,)

    # 4. 归一化向量
    normalized = F.normalize(batch_vectors, p=2, dim=1)
    print(f"归一化后的范数: {torch.norm(normalized, dim=1)}")  # 全为1

def advanced_vector_operations():
    """
    高级向量运算
    """
    # 1. 点积（内积）
    a = torch.tensor([1., 2., 3.])
    b = torch.tensor([4., 5., 6.])

    # 方法1：torch.dot
    dot_product = torch.dot(a, b)
    print(f"点积: {dot_product}")  # 32.0

    # 方法2：einsum
    dot_einsum = torch.einsum('i,i->', a, b)
    print(f"点积(einsum): {dot_einsum}")

    # 方法3：手动计算
    dot_manual = (a * b).sum()
    print(f"点积(手动): {dot_manual}")

    # 2. 外积
    outer = torch.outer(a, b)
    print(f"外积形状: {outer.shape}")  # (3, 3)
    print(f"外积:\n{outer}")

    # 3. 叉积（3维向量）
    cross = torch.cross(a, b)
    print(f"叉积: {cross}")

    # 验证叉积性质：c⊥a 且 c⊥b
    print(f"叉积与a正交: {torch.dot(cross, a):.6f}")  # ≈0
    print(f"叉积与b正交: {torch.dot(cross, b):.6f}")  # ≈0

    # 4. 投影
    def project(v, u):
        """将向量v投影到u上"""
        return (torch.dot(v, u) / torch.dot(u, u)) * u

    projection = project(a, b)
    print(f"a在b上的投影: {projection}")

    # 5. 格拉姆-施密特正交化
    def gram_schmidt(vectors):
        """
        对一组向量进行正交化
        """
        orthogonal = []
        for v in vectors:
            # 减去在之前向量上的投影
            for u in orthogonal:
                v = v - project(v, u)
            # 归一化
            v = v / torch.norm(v)
            orthogonal.append(v)
        return torch.stack(orthogonal)

    # 测试
    vectors = [
        torch.tensor([1., 2., 3.]),
        torch.tensor([4., 5., 6.]),
        torch.tensor([7., 8., 9.])
    ]
    ortho = gram_schmidt(vectors[:2])  # 只对前两个向量正交化
    print(f"正交向量组:\n{ortho}")

    # 验证正交性
    print(f"正交性验证: {torch.dot(ortho[0], ortho[1]):.6f}")  # ≈0

def matrix_vector_operations():
    """
    矩阵-向量运算详解
    """
    A = torch.tensor([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])
    x = torch.tensor([1., 2., 3.])

    # 1. 矩阵-向量乘法 y = Ax
    y = torch.mv(A, x)  # 或 A @ x
    print(f"Ax = {y}")

    # 手动计算验证
    y_manual = torch.tensor([
        A[0] @ x,
        A[1] @ x,
        A[2] @ x
    ])
    print(f"手动计算: {y_manual}")

    # 2. 向量-矩阵乘法 y = x^T A
    y_left = torch.mv(A.t(), x)  # 或 x @ A
    print(f"x^T A = {y_left}")

    # 3. 批量矩阵-向量乘法
    batch_A = torch.randn(32, 3, 3)  # 32个3x3矩阵
    batch_x = torch.randn(32, 3)     # 32个3维向量

    batch_y = torch.bmm(batch_A, batch_x.unsqueeze(-1)).squeeze(-1)
    print(f"批量矩阵-向量乘法结果形状: {batch_y.shape}")  # (32, 3)

    # 4. 二次型 x^T A x
    quadratic = x @ A @ x
    print(f"二次型 x^T A x = {quadratic}")

    # 使用einsum
    quadratic_einsum = torch.einsum('i,ij,j->', x, A, x)
    print(f"二次型(einsum) = {quadratic_einsum}")
```

#### 1.1.2 矩阵分解与特征值

**数学理论**
```
特征值分解（EVD）：
A = QΛQ⁻¹
其中：Q是特征向量矩阵，Λ是特征值对角矩阵

奇异值分解（SVD）：
A = UΣVᵀ
其中：U和V是正交矩阵，Σ是奇异值对角矩阵
```

**纯数学例子：矩阵分解与特征值计算**

**例1：特征值与特征向量的基本计算**
```
给定 2×2 矩阵：
A = [ 4  2 ]
    [ 1  3 ]

求特征值：det(A - λI) = 0
| 4-λ   2  |
|  1   3-λ | = (4-λ)(3-λ) - 2 = 0

展开：12 - 4λ - 3λ + λ² - 2 = 0
     λ² - 7λ + 10 = 0
     (λ - 5)(λ - 2) = 0

特征值：λ₁ = 5, λ₂ = 2

求特征向量（λ₁ = 5）：
(A - 5I)v = 0
[-1  2 ][v₁]   [0]
[ 1 -2 ][v₂] = [0]

-v₁ + 2v₂ = 0  →  v₁ = 2v₂
取 v₂ = 1，得 v₁ = [2, 1]ᵀ

归一化：v̂₁ = [2/√5, 1/√5]ᵀ ≈ [0.894, 0.447]ᵀ

求特征向量（λ₂ = 2）：
(A - 2I)v = 0
[ 2  2 ][v₁]   [0]
[ 1  1 ][v₂] = [0]

2v₁ + 2v₂ = 0  →  v₁ = -v₂
取 v₂ = 1，得 v₂ = [-1, 1]ᵀ

归一化：v̂₂ = [-1/√2, 1/√2]ᵀ ≈ [-0.707, 0.707]ᵀ

验证：Av₁ = λ₁v₁
[4 2][2]   [8+2]   [10]      [2]
[1 3][1] = [2+3] = [ 5] = 5 [1] ✓
```

**例2：对称矩阵的谱分解**
```
对称矩阵：
A = [ 3  1 ]
    [ 1  3 ]

特征值：det(A - λI) = 0
(3-λ)² - 1 = 0
9 - 6λ + λ² - 1 = 0
λ² - 6λ + 8 = 0
λ₁ = 4, λ₂ = 2

特征向量：
λ₁ = 4: v₁ = [1/√2, 1/√2]ᵀ
λ₂ = 2: v₂ = [-1/√2, 1/√2]ᵀ

谱分解：A = λ₁v₁v₁ᵀ + λ₂v₂v₂ᵀ

验证：
4[1/√2][1/√2  1/√2] + 2[-1/√2][-1/√2  1/√2]
  [1/√2]                [1/√2]

= 4[1/2  1/2] + 2[ 1/2  -1/2]
    [1/2  1/2]     [-1/2   1/2]

= [2  2] + [ 1  -1]   [3  1]
  [2  2]   [-1   1] = [1  3] = A ✓
```

**例3：2×2矩阵的奇异值分解（SVD）**
```
给定矩阵：
A = [ 3  0 ]
    [ 4  5 ]

步骤1：计算 AᵀA
AᵀA = [3  4][3  0]   [9+16   0+20]   [25  20]
      [0  5][4  5] = [0+20   0+25] = [20  25]

步骤2：求 AᵀA 的特征值
det(AᵀA - λI) = (25-λ)² - 400 = 0
λ² - 50λ + 625 - 400 = 0
λ² - 50λ + 225 = 0

λ = (50 ± √(2500-900))/2 = (50 ± 40)/2
λ₁ = 45, λ₂ = 5

奇异值：σ₁ = √45 ≈ 6.708, σ₂ = √5 ≈ 2.236

步骤3：求右奇异向量（AᵀA的特征向量）
对于 λ₁ = 45:
[25-45   20  ][v₁]   [-20  20][v₁]
[  20  25-45 ][v₂] = [ 20 -20][v₂] = 0

v₁ = v₂，归一化：v₁ = [1/√2, 1/√2]ᵀ

对于 λ₂ = 5:
v₂ = [-1/√2, 1/√2]ᵀ

步骤4：求左奇异向量
u₁ = Av₁/σ₁ = (1/6.708)[3  0][1/√2]   (1/6.708)[3/√2]
                        [4  5][1/√2] =          [9/√2]
     ≈ [0.316, 0.949]ᵀ

u₂ = Av₂/σ₂ = (1/2.236)[-3/√2]
                        [ 1/√2] ≈ [-0.949, 0.316]ᵀ

SVD分解：A = UΣVᵀ
U = [0.316  -0.949]  Σ = [6.708    0  ]  Vᵀ = [ 0.707  0.707]
    [0.949   0.316]      [   0   2.236]       [-0.707  0.707]
```

**例4：矩阵的秩与行列式**
```
矩阵：
A = [ 1  2  3 ]
    [ 2  4  6 ]
    [ 0  1  2 ]

观察：第2行 = 2×第1行（线性相关）

计算行列式：
det(A) = 1(4×2 - 6×1) - 2(2×2 - 6×0) + 3(2×1 - 4×0)
       = 1(8-6) - 2(4-0) + 3(2-0)
       = 2 - 8 + 6
       = 0

结论：det(A) = 0，矩阵奇异（不可逆）

秩的计算（行化简）：
[ 1  2  3 ]     [ 1  2  3 ]     [ 1  2  3 ]
[ 2  4  6 ]  →  [ 0  0  0 ]  →  [ 0  1  2 ]
[ 0  1  2 ]     [ 0  1  2 ]     [ 0  0  0 ]

rank(A) = 2（两个非零行）
```

**例5：QR分解（Gram-Schmidt正交化）**
```
给定矩阵：
A = [ 1  1 ]
    [ 1  0 ]
    [ 0  1 ]

列向量：a₁ = [1, 1, 0]ᵀ, a₂ = [1, 0, 1]ᵀ

Gram-Schmidt正交化：

步骤1：u₁ = a₁
u₁ = [1, 1, 0]ᵀ
||u₁|| = √(1+1+0) = √2
q₁ = u₁/||u₁|| = [1/√2, 1/√2, 0]ᵀ

步骤2：u₂ = a₂ - proj_{u₁}(a₂)
proj_{u₁}(a₂) = (a₂·u₁/u₁·u₁)u₁
a₂·u₁ = 1×1 + 0×1 + 1×0 = 1
u₁·u₁ = 2

proj_{u₁}(a₂) = (1/2)[1, 1, 0]ᵀ = [0.5, 0.5, 0]ᵀ

u₂ = [1, 0, 1]ᵀ - [0.5, 0.5, 0]ᵀ = [0.5, -0.5, 1]ᵀ
||u₂|| = √(0.25 + 0.25 + 1) = √1.5 ≈ 1.225
q₂ = u₂/||u₂|| ≈ [0.408, -0.408, 0.816]ᵀ

QR分解：
Q = [1/√2    0.408 ]    R = [√2   1/√2  ]
    [1/√2   -0.408 ]        [ 0   1.225 ]
    [ 0      0.816 ]

验证：QR = A
[1/√2    0.408 ][√2   1/√2  ]   [1  1]
[1/√2   -0.408 ][ 0   1.225 ] ≈ [1  0] ✓
[ 0      0.816 ]                 [0  1]
```

**例6：Cholesky分解（正定矩阵）**
```
正定对称矩阵：
A = [ 4  2 ]
    [ 2  3 ]

Cholesky分解：A = LLᵀ
其中L是下三角矩阵

设 L = [l₁₁   0 ]
       [l₂₁  l₂₂]

A = LLᵀ = [l₁₁   0 ][l₁₁  l₂₁]   [l₁₁²         l₁₁l₂₁    ]
          [l₂₁  l₂₂][ 0   l₂₂] = [l₁₁l₂₁  l₂₁² + l₂₂²]

匹配元素：
l₁₁² = 4          →  l₁₁ = 2
l₁₁l₂₁ = 2        →  l₂₁ = 2/2 = 1
l₂₁² + l₂₂² = 3   →  1 + l₂₂² = 3  →  l₂₂ = √2

因此：
L = [ 2     0  ]
    [ 1    √2  ]

验证：
LLᵀ = [2  0 ][2  1 ]   [4  2]
      [1  √2][ 0  √2] = [2  3] = A ✓

应用：求解线性方程组 Ax = b
如 b = [8, 7]ᵀ

先解 Ly = b:
[2   0 ][y₁]   [8]
[1  √2 ][y₂] = [7]

y₁ = 4
y₂ = (7-1×4)/√2 = 3/√2

再解 Lᵀx = y:
[2  1 ][x₁]   [ 4   ]
[0  √2][x₂] = [3/√2 ]

√2·x₂ = 3/√2  →  x₂ = 3/2
2x₁ + x₂ = 4  →  x₁ = (4 - 3/2)/2 = 5/4

解：x = [1.25, 1.5]ᵀ
```

**例7：矩阵的幂次方（通过特征值分解）**
```
给定对角化矩阵：
A = [ 3  1 ]
    [ 0  2 ]

虽然不是对称矩阵，但可以对角化
A = PDP⁻¹，其中D是特征值对角矩阵

特征值：λ₁ = 3, λ₂ = 2
特征向量：v₁ = [1, 0]ᵀ, v₂ = [1, -1]ᵀ

P = [1   1]    D = [3  0]    P⁻¹ = [ 1  1]
    [0  -1]        [0  2]           [ 0 -1]

计算 A¹⁰：
A¹⁰ = PD¹⁰P⁻¹

D¹⁰ = [3¹⁰   0  ]   [59049      0 ]
      [ 0   2¹⁰ ] = [    0   1024 ]

A¹⁰ = [1   1][59049      0][ 1  1]
      [0  -1][    0   1024][ 0 -1]

    = [59049   1024][ 1  1]
      [    0  -1024][ 0 -1]

    = [59049  58025]
      [    0   1024]

验证：A² = [9  5]，A³ = [27  16]...
          [0  4]        [0   8]
```

**例8：伪逆（Moore-Penrose逆）**
```
非方阵矩阵：
A = [ 1  0 ]
    [ 0  1 ]
    [ 1  1 ]  (3×2矩阵，列满秩)

伪逆公式（左逆）：A⁺ = (AᵀA)⁻¹Aᵀ

计算 AᵀA：
AᵀA = [1  0  1][1  0]   [2  1]
      [0  1  1][0  1] = [1  2]
                [1  1]

计算 (AᵀA)⁻¹：
det(AᵀA) = 2×2 - 1×1 = 3

(AᵀA)⁻¹ = (1/3)[ 2  -1]
                [-1   2]

计算伪逆：
A⁺ = (1/3)[ 2  -1][1  0  1]
          [-1   2][0  1  1]

   = (1/3)[2  -1   1]
          [-1  2   1]

验证：A⁺A = I
(1/3)[2  -1   1][1  0]   (1/3)[3  0]   [1  0]
    [-1  2   1][0  1] =      [0  3] = [0  1] ✓
                [1  1]
```

---

**PyTorch实现与应用**
```python
def matrix_decomposition_deep():
    """
    矩阵分解在深度学习中的应用
    """
    # 1. SVD用于降维和初始化
    A = torch.randn(100, 50)
    U, S, Vt = torch.linalg.svd(A, full_matrices=False)

    # 低秩近似（用于模型压缩）
    k = 10  # 保留前k个奇异值
    A_approx = U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]

    # 计算近似误差
    error = torch.norm(A - A_approx, p='fro')
    print(f"重构误差: {error.item()}")

    # 2. 特征值分解用于理解Hessian矩阵
    def compute_hessian_eigenvalues(loss_fn, params):
        """
        计算Hessian矩阵的特征值
        用于分析损失函数的曲率
        """
        # 这在优化理论中非常重要
        # 特征值全为正 → 凸函数
        # 有正有负 → 鞍点
        pass

    # 3. Cholesky分解用于协方差矩阵
    # L Lᵀ = Σ（协方差矩阵）
    cov_matrix = torch.tensor([[4., 2.], [2., 3.]])
    L = torch.linalg.cholesky(cov_matrix)
    print(f"Cholesky分解:\n{L}")

    # 应用：生成相关的随机变量
    z = torch.randn(2, 1000)
    x = L @ z  # x的协方差矩阵近似为cov_matrix

def comprehensive_matrix_decomposition_examples():
    """
    全面的矩阵分解数学运算示例
    """
    print("\n=== 矩阵分解综合示例 ===\n")

    # 1. 奇异值分解（SVD）的详细应用
    print("1. 奇异值分解（SVD）")
    A = torch.randn(100, 50)
    U, S, Vt = torch.linalg.svd(A, full_matrices=False)

    # 验证分解：A = U @ diag(S) @ Vt
    A_reconstructed = U @ torch.diag(S) @ Vt
    reconstruction_error = torch.norm(A - A_reconstructed, p='fro')
    print(f"SVD重构误差: {reconstruction_error.item():.2e}")

    # 计算有效秩（effective rank）
    s_sum = S.sum()
    cumulative_energy = torch.cumsum(S, dim=0) / s_sum
    rank_95 = (cumulative_energy < 0.95).sum().item() + 1
    print(f"保留95%能量需要的秩: {rank_95}/{len(S)}")

    # 低秩近似用于模型压缩
    for k in [5, 10, 20, 30]:
        A_k = U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]
        compression_ratio = (100 * k + k + 50 * k) / (100 * 50)
        error_k = torch.norm(A - A_k, p='fro') / torch.norm(A, p='fro')
        print(f"秩-{k} 近似: 压缩率={compression_ratio:.2%}, 相对误差={error_k:.4f}")

    # 2. 特征值分解（EVD）
    print("\n2. 特征值分解（EVD）")
    # 对称矩阵的特征值分解
    symmetric_matrix = torch.randn(5, 5)
    symmetric_matrix = symmetric_matrix + symmetric_matrix.t()  # 确保对称

    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_matrix)
    print(f"特征值: {eigenvalues}")

    # 验证 A v = λ v
    for i in range(5):
        lhs = symmetric_matrix @ eigenvectors[:, i]
        rhs = eigenvalues[i] * eigenvectors[:, i]
        error = torch.norm(lhs - rhs)
        print(f"特征对 {i}: λ={eigenvalues[i]:.4f}, 验证误差={error:.2e}")

    # 重构矩阵：A = Q Λ Q^T
    A_reconstructed = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.t()
    print(f"EVD重构误差: {torch.norm(symmetric_matrix - A_reconstructed):.2e}")

    # 3. QR分解
    print("\n3. QR分解")
    A = torch.randn(100, 50)
    Q, R = torch.linalg.qr(A)

    # 验证Q的正交性：Q^T Q = I
    orthogonality_error = torch.norm(Q.t() @ Q - torch.eye(50))
    print(f"Q正交性误差: {orthogonality_error:.2e}")

    # 验证R是上三角
    is_upper = torch.allclose(R, torch.triu(R))
    print(f"R是上三角矩阵: {is_upper}")

    # 验证分解：A = QR
    qr_error = torch.norm(A - Q @ R, p='fro')
    print(f"QR分解重构误差: {qr_error:.2e}")

    # 4. Cholesky分解
    print("\n4. Cholesky分解")
    # 构造正定矩阵
    A = torch.randn(5, 5)
    cov_matrix = A @ A.t() + torch.eye(5) * 0.1  # 确保正定

    L = torch.linalg.cholesky(cov_matrix)
    print(f"Cholesky因子 L:\n{L}")

    # 验证：A = L L^T
    chol_reconstructed = L @ L.t()
    chol_error = torch.norm(cov_matrix - chol_reconstructed)
    print(f"Cholesky重构误差: {chol_error:.2e}")

    # 应用：高效求解线性方程 Ax = b
    b = torch.randn(5)
    # 传统方法：x = A^{-1} b（需要矩阵求逆）
    x_inv = torch.linalg.solve(cov_matrix, b)

    # Cholesky方法：先解 Ly = b，再解 L^T x = y
    y = torch.linalg.solve_triangular(L, b, upper=False)
    x_chol = torch.linalg.solve_triangular(L.t(), y, upper=True)
    print(f"求解方法对比误差: {torch.norm(x_inv - x_chol):.2e}")

    # 5. LU分解
    print("\n5. LU分解（带部分主元）")
    A = torch.randn(5, 5)
    P, L, U = torch.lu_unpack(*torch.linalg.lu_factor(A))

    # 验证：PA = LU
    lu_reconstructed = L @ U
    lu_error = torch.norm(P @ A - lu_reconstructed)
    print(f"LU分解重构误差: {lu_error:.2e}")

    # 验证L是下三角，U是上三角
    is_lower = torch.allclose(L, torch.tril(L))
    is_upper = torch.allclose(U, torch.triu(U))
    print(f"L是下三角: {is_lower}, U是上三角: {is_upper}")

    # 6. 极分解（Polar Decomposition）
    print("\n6. 极分解")
    A = torch.randn(5, 5)
    U_svd, S, Vt = torch.linalg.svd(A)

    # A = U @ P，其中U是正交矩阵，P是半正定矩阵
    U_polar = U_svd @ Vt
    P = Vt.t() @ torch.diag(S) @ Vt

    polar_reconstructed = U_polar @ P
    polar_error = torch.norm(A - polar_reconstructed)
    print(f"极分解重构误差: {polar_error:.2e}")
    print(f"U是正交矩阵: {torch.allclose(U_polar @ U_polar.t(), torch.eye(5))}")

    # 7. 矩阵的秩、行列式、迹
    print("\n7. 矩阵基本性质")
    A = torch.randn(10, 10)

    rank = torch.linalg.matrix_rank(A)
    det = torch.linalg.det(A)
    trace = torch.trace(A)
    condition_number = torch.linalg.cond(A)

    print(f"秩: {rank}")
    print(f"行列式: {det:.6f}")
    print(f"迹: {trace:.6f}")
    print(f"条件数: {condition_number:.6f}")

    # 验证特征值之和 = 迹
    eigenvalues = torch.linalg.eigvals(A)
    eigenvalue_sum = eigenvalues.real.sum()
    print(f"特征值之和: {eigenvalue_sum:.6f}")
    print(f"迹与特征值和的差异: {abs(trace - eigenvalue_sum):.2e}")

    # 验证特征值之积 = 行列式
    eigenvalue_prod = torch.prod(eigenvalues.real)
    print(f"特征值之积: {eigenvalue_prod:.6f}")
    print(f"行列式与特征值积的差异: {abs(det - eigenvalue_prod):.2e}")
```

#### 1.1.3 线性变换与仿射变换

```python
def linear_transformations_in_dl():
    """
    深度学习中的线性变换数学本质
    """
    # 线性变换：f(x) = Wx
    # 仿射变换：f(x) = Wx + b

    # 全连接层的本质
    class LinearLayerMath(torch.nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.W = torch.nn.Parameter(torch.randn(out_features, in_features))
            self.b = torch.nn.Parameter(torch.zeros(out_features))

        def forward(self, x):
            # 数学形式：y = xWᵀ + b
            # 矩阵形式：(batch, in) @ (in, out)ᵀ + (out,)
            return x @ self.W.t() + self.b

    # 变换的几何意义
    def visualize_transformation():
        # 旋转矩阵
        theta = torch.tensor(np.pi / 4)  # 45度
        rotation = torch.tensor([
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)]
        ])

        # 缩放矩阵
        scaling = torch.tensor([[2., 0.], [0., 0.5]])

        # 组合变换
        combined = rotation @ scaling

        # 应用到数据
        points = torch.randn(100, 2)
        transformed = points @ combined.t()

        return transformed

def comprehensive_linear_transformation_examples():
    """
    线性变换和仿射变换的全面数学示例
    """
    print("\n=== 线性变换综合示例 ===\n")

    # 1. 基本几何变换矩阵
    print("1. 基本几何变换")

    # 旋转矩阵（2D）
    def rotation_matrix_2d(theta):
        """逆时针旋转theta弧度"""
        c, s = torch.cos(theta), torch.sin(theta)
        return torch.tensor([[c, -s], [s, c]])

    # 缩放矩阵
    def scaling_matrix(sx, sy):
        """x方向缩放sx倍，y方向缩放sy倍"""
        return torch.diag(torch.tensor([sx, sy]))

    # 剪切矩阵
    def shear_matrix(k):
        """沿x轴剪切"""
        return torch.tensor([[1., k], [0., 1.]])

    # 镜像矩阵
    def reflection_matrix():
        """关于y轴镜像"""
        return torch.tensor([[-1., 0.], [0., 1.]])

    # 应用变换
    points = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0.5, 0.5]])

    theta = torch.tensor(torch.pi / 4)  # 45度
    rotated = points @ rotation_matrix_2d(theta).t()
    scaled = points @ scaling_matrix(2., 0.5).t()
    sheared = points @ shear_matrix(0.5).t()
    reflected = points @ reflection_matrix().t()

    print(f"原始点:\n{points}")
    print(f"旋转45度后:\n{rotated}")
    print(f"缩放后:\n{scaled}")
    print(f"剪切后:\n{sheared}")

    # 2. 复合变换
    print("\n2. 复合变换")
    # 先缩放再旋转
    combined = rotation_matrix_2d(theta) @ scaling_matrix(2., 0.5)
    combined_result = points @ combined.t()
    print(f"先缩放再旋转:\n{combined_result}")

    # 3. 3D旋转矩阵
    print("\n3. 3D旋转��阵")

    def rotation_matrix_x(theta):
        """绕x轴旋转"""
        c, s = torch.cos(theta), torch.sin(theta)
        return torch.tensor([[1., 0., 0.],
                            [0., c, -s],
                            [0., s, c]])

    def rotation_matrix_y(theta):
        """绕y轴旋转"""
        c, s = torch.cos(theta), torch.sin(theta)
        return torch.tensor([[c, 0., s],
                            [0., 1., 0.],
                            [-s, 0., c]])

    def rotation_matrix_z(theta):
        """绕z轴旋转"""
        c, s = torch.cos(theta), torch.sin(theta)
        return torch.tensor([[c, -s, 0.],
                            [s, c, 0.],
                            [0., 0., 1.]])

    # Euler角旋转（ZYX顺序）
    alpha, beta, gamma = torch.pi/6, torch.pi/4, torch.pi/3
    R = rotation_matrix_z(alpha) @ rotation_matrix_y(beta) @ rotation_matrix_x(gamma)

    points_3d = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    rotated_3d = points_3d @ R.t()
    print(f"3D旋转后:\n{rotated_3d}")

    # 验证旋转矩阵的正交性：R^T R = I
    orthogonality = R.t() @ R
    print(f"正交性验证:\n{orthogonality}")
    print(f"行列式(应为1): {torch.linalg.det(R):.6f}")

    # 4. 仿射变换（线性变换 + 平移）
    print("\n4. 仿射变换")

    # 齐次坐标表示：[x, y, 1]^T
    # 仿射变换矩阵：[[A, b], [0, 1]]
    def affine_transform_matrix(A, b):
        """构造仿射变换的齐次坐标矩阵"""
        n = A.shape[0]
        T = torch.zeros(n + 1, n + 1)
        T[:n, :n] = A
        T[:n, n] = b
        T[n, n] = 1.
        return T

    A = rotation_matrix_2d(torch.tensor(torch.pi / 6))
    b = torch.tensor([1., 2.])
    T = affine_transform_matrix(A, b)

    # 应用到点
    points_homo = torch.cat([points, torch.ones(points.shape[0], 1)], dim=1)
    transformed_homo = points_homo @ T.t()
    transformed = transformed_homo[:, :2]
    print(f"仿射变换后:\n{transformed}")

    # 5. 投影变换
    print("\n5. 投影变换")

    # 正交投影到xy平面
    proj_xy = torch.tensor([[1., 0., 0.],
                           [0., 1., 0.],
                           [0., 0., 0.]])

    points_3d = torch.randn(10, 3)
    projected = points_3d @ proj_xy.t()
    print(f"投影后的点（z=0）:\n{projected[:3]}")

    # 投影到任意平面：n·x = d（法向量n，距离d）
    n = torch.tensor([1., 1., 1.]) / torch.sqrt(torch.tensor(3.))  # 单位法向量
    proj_plane = torch.eye(3) - torch.outer(n, n)
    projected_plane = points_3d @ proj_plane.t()
    print(f"投影到平面:\n{projected_plane[:3]}")

    # 6. 深度学习中的变换：全连接层
    print("\n6. 全连接层作为线性变换")

    batch_size, in_features, out_features = 32, 128, 64
    W = torch.randn(out_features, in_features) / torch.sqrt(torch.tensor(in_features))
    b = torch.zeros(out_features)

    x = torch.randn(batch_size, in_features)
    y = x @ W.t() + b  # 仿射变换

    print(f"输入形状: {x.shape}")
    print(f"权重形状: {W.shape}")
    print(f"输出形状: {y.shape}")

    # 计算变换的条件数（衡量数值稳定性）
    condition_number = torch.linalg.cond(W)
    print(f"权重矩阵条件数: {condition_number:.4f}")

    # 7. 批量变换（广播机制）
    print("\n7. 批量变换")

    # 对batch中的每个样本应用不同的变换
    batch_transforms = torch.randn(32, 2, 2)  # 32个2x2变换矩阵
    batch_points = torch.randn(32, 10, 2)      # 32个样本，每个10个点

    # 使用bmm进行批量矩阵乘法
    batch_transformed = torch.bmm(batch_points, batch_transforms.transpose(1, 2))
    print(f"批量变换后形状: {batch_transformed.shape}")

    # 使用einsum的更灵活方式
    batch_transformed_einsum = torch.einsum('bni,bij->bnj', batch_points, batch_transforms)
    print(f"einsum批量变换验证: {torch.allclose(batch_transformed, batch_transformed_einsum)}")
```

### 1.2 微积分与优化理论

#### 1.2.1 梯度、雅可比矩阵和Hessian矩阵

**数学定义**
```
梯度（Gradient）：
∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ

雅可比矩阵（Jacobian）：
J = [∂fᵢ/∂xⱼ], 对于 f: ℝⁿ → ℝᵐ

Hessian矩阵：
H = [∂²f/∂xᵢ∂xⱼ], 二阶导数矩阵
```

**纯数学例子：微分与梯度计算**

**例1：一元函数的导数**
```
函数：f(x) = x³ - 3x² + 2x + 5

一阶导数：
f'(x) = d/dx(x³ - 3x² + 2x + 5)
      = 3x² - 6x + 2

在 x = 2 处的导数：
f'(2) = 3(2)² - 6(2) + 2
      = 12 - 12 + 2
      = 2

几何意义：在点 (2, f(2)) 处的切线斜率为 2

二阶导数（曲率）：
f''(x) = d/dx(3x² - 6x + 2)
       = 6x - 6

f''(2) = 6(2) - 6 = 6 > 0（凹函数，局部最小值方向）

驻点（临界点）：f'(x) = 0
3x² - 6x + 2 = 0
x = (6 ± √(36-24))/6 = (6 ± √12)/6 = 1 ± √3/3
x₁ ≈ 0.423, x₂ ≈ 1.577

判断极值：
f''(0.423) = 6(0.423) - 6 ≈ -3.46 < 0 → 极大值
f''(1.577) = 6(1.577) - 6 ≈ 3.46 > 0 → 极小值
```

**例2：多元函数的偏导数**
```
函数：f(x, y) = x²y + 3xy² - 2x + y

偏导数：
∂f/∂x = 2xy + 3y² - 2
∂f/∂y = x² + 6xy + 1

在点 (1, 2) 处：
∂f/∂x|(1,2) = 2(1)(2) + 3(2)² - 2 = 4 + 12 - 2 = 14
∂f/∂y|(1,2) = (1)² + 6(1)(2) + 1 = 1 + 12 + 1 = 14

梯度向量：
∇f(1,2) = [14, 14]ᵀ

最速上升方向：梯度方向 [14, 14]
最速下降方向：负梯度方向 [-14, -14]

方向导数（沿单位向量 u = [1/√2, 1/√2]）：
D_u f = ∇f · u = 14(1/√2) + 14(1/√2) = 28/√2 ≈ 19.8
```

**例3：链式法则**
```
复合函数：z = f(g(x))

例：z = (3x² + 1)⁵

方法1：外层导数 × 内层导数
令 u = 3x² + 1
则 z = u⁵

dz/dx = dz/du · du/dx
      = 5u⁴ · 6x
      = 5(3x² + 1)⁴ · 6x
      = 30x(3x² + 1)⁴

在 x = 1:
dz/dx|_{x=1} = 30(1)(3 + 1)⁴ = 30(256) = 7680

多变量链式法则：
z = f(u, v), u = g(x, y), v = h(x, y)

∂z/∂x = ∂z/∂u · ∂u/∂x + ∂z/∂v · ∂v/∂x

例：z = u² + v², u = x + y, v = xy
∂z/∂x = 2u · 1 + 2v · y
       = 2(x+y) + 2xy · y
       = 2x + 2y + 2xy²
```

**例4：梯度向量的计算**
```
函数：f(x, y, z) = x²y + yz² + 3z

计算梯度：
∂f/∂x = 2xy
∂f/∂y = x² + z²
∂f/∂z = 2yz + 3

∇f = [2xy, x² + z², 2yz + 3]ᵀ

在点 P(1, 2, 3)：
∇f(1,2,3) = [2(1)(2), (1)² + (3)², 2(2)(3) + 3]
          = [4, 10, 15]ᵀ

梯度的模（变化率的最大值）：
||∇f|| = √(4² + 10² + 15²) = √(16 + 100 + 225) = √341 ≈ 18.47

等值面：f(x,y,z) = c 的法向量就是 ∇f
在点P处，等值面的法向量为 [4, 10, 15]
```

**例5：雅可比矩阵**
```
向量值函数：F: ℝ² → ℝ³
F(x, y) = [x² + y, xy, y²]ᵀ

即：f₁(x,y) = x² + y
    f₂(x,y) = xy
    f₃(x,y) = y²

雅可比矩阵：
J = [∂f₁/∂x  ∂f₁/∂y]   [2x   1]
    [∂f₂/∂x  ∂f₂/∂y] = [ y   x]
    [∂f₃/∂x  ∂f₃/∂y]   [ 0  2y]

在点 (2, 3)：
J(2,3) = [2(2)  1 ]   [4  1]
         [ 3    2 ] = [3  2]
         [ 0   2(3)]  [0  6]

应用：线性近似
F(x+Δx, y+Δy) ≈ F(x,y) + J·[Δx, Δy]ᵀ

在 (2,3) 附近，若 Δx=0.1, Δy=0.05：
F(2,3) = [7, 6, 9]ᵀ

ΔF ≈ [4  1][0.1 ]   [0.45]
     [3  2][0.05] = [0.40]
     [0  6]         [0.30]

F(2.1, 3.05) ≈ [7.45, 6.40, 9.30]ᵀ

验证（精确计算）：
F(2.1, 3.05) = [(2.1)² + 3.05, (2.1)(3.05), (3.05)²]
             = [7.46, 6.405, 9.3025]ᵀ
近似误差很小！
```

**例6：Hessian矩阵（二阶偏导数）**
```
函数：f(x, y) = x³ + 2x²y - y² + 3xy

一阶偏导数：
∂f/∂x = 3x² + 4xy + 3y
∂f/∂y = 2x² - 2y + 3x

二阶偏导数：
∂²f/∂x² = 6x + 4y
∂²f/∂y² = -2
∂²f/∂x∂y = 4x + 3
∂²f/∂y∂x = 4x + 3  （混合偏导数相等，Schwarz定理）

Hessian矩阵：
H = [∂²f/∂x²    ∂²f/∂x∂y]   [6x+4y   4x+3]
    [∂²f/∂y∂x   ∂²f/∂y²  ] = [4x+3     -2 ]

在点 (1, 1)：
H(1,1) = [6+4  4+3]   [10  7]
         [4+3   -2] = [ 7 -2]

判断凸凹性：
det(H) = 10(-2) - 7(7) = -20 - 49 = -69 < 0
特征值：λ₁ + λ₂ = trace(H) = 8
       λ₁λ₂ = det(H) = -69

因为行列式<0，特征值有正有负 → 鞍点（saddle point）

Hessian正定 → 局部最小值
Hessian负定 → 局部最大值
Hessian不定 → 鞍点
```

**例7：梯度下降的数学推导**
```
目标：最小化 f(x) = x² - 4x + 5

梯度：f'(x) = 2x - 4

梯度下降迭代：
x_{k+1} = x_k - α·f'(x_k)
        = x_k - α(2x_k - 4)

选择学习率 α = 0.1，初始值 x₀ = 0

迭代过程：
k=0: x₀ = 0
     f'(0) = -4
     x₁ = 0 - 0.1(-4) = 0.4

k=1: x₁ = 0.4
     f'(0.4) = 2(0.4) - 4 = -3.2
     x₂ = 0.4 - 0.1(-3.2) = 0.72

k=2: x₂ = 0.72
     f'(0.72) = 2(0.72) - 4 = -2.56
     x₃ = 0.72 - 0.1(-2.56) = 0.976

k=3: x₃ = 0.976
     f'(0.976) = 2(0.976) - 4 = -2.048
     x₄ = 0.976 - 0.1(-2.048) = 1.1808

继续迭代... → x* = 2（解析解：f'(x)=0 → x=2）

收敛速度分析：
误差 e_k = x_k - 2
e_{k+1} = x_k - α(2x_k - 4) - 2
        = x_k - 2αx_k + 4α - 2
        = (1 - 2α)e_k + (4α - 2)
        = (1 - 2α)e_k

当 0 < α < 1 时，|1-2α| < 1，线性收敛
```

**例8：牛顿法（使用Hessian）**
```
函数：f(x) = x³ - 2x + 2

牛顿法迭代：
x_{k+1} = x_k - f'(x_k)/f''(x_k)

计算导数：
f'(x) = 3x² - 2
f''(x) = 6x

初始值 x₀ = 1

迭代：
k=0: x₀ = 1
     f'(1) = 3 - 2 = 1
     f''(1) = 6
     x₁ = 1 - 1/6 = 0.8333

k=1: x₀ = 0.8333
     f'(0.8333) = 3(0.8333)² - 2 = 0.0833
     f''(0.8333) = 5
     x₂ = 0.8333 - 0.0833/5 = 0.8167

k=2: x₂ = 0.8167
     f'(0.8167) ≈ 0.0008
     f''(0.8167) ≈ 4.9
     x₃ = 0.8167 - 0.0008/4.9 ≈ 0.8165

快速收敛到根 x* ≈ 0.8165（二次收敛速度）

多元牛顿法：
x_{k+1} = x_k - H⁻¹(x_k)·∇f(x_k)
其中 H 是 Hessian 矩阵
```

**例9：方向导数**
```
函数：f(x, y) = x² - xy + y²

在点 P(2, 1) 沿方向 v = [3, 4]

步骤1：归一化方向向量
||v|| = √(3² + 4²) = 5
u = v/||v|| = [3/5, 4/5]

步骤2：计算梯度
∇f = [2x - y, -x + 2y]
∇f(2,1) = [2(2) - 1, -2 + 2(1)] = [3, 0]

步骤3：方向导数
D_u f = ∇f · u
      = [3, 0] · [3/5, 4/5]
      = 3(3/5) + 0(4/5)
      = 9/5 = 1.8

解释：沿方向 [3, 4] 移动单位距离，函数值增加 1.8

最大方向导数 = ||∇f|| = 3（沿梯度方向）
最小方向导数 = -3（沿负梯度方向）
```

**例10：拉格朗日乘数法**
```
优化问题：
最小化 f(x, y) = x² + y²
约束条件：g(x, y) = x + y - 2 = 0

拉格朗日函数：
L(x, y, λ) = f(x, y) - λg(x, y)
           = x² + y² - λ(x + y - 2)

必要条件（KKT条件）：
∂L/∂x = 2x - λ = 0  → x = λ/2
∂L/∂y = 2y - λ = 0  → y = λ/2
∂L/∂λ = -(x + y - 2) = 0  → x + y = 2

解方程组：
x = y = λ/2
x + y = 2
→ λ/2 + λ/2 = 2
→ λ = 2
→ x = y = 1

最优解：(x*, y*) = (1, 1)
最小值：f(1, 1) = 1² + 1² = 2

验证（几何解释）：
在约束 x+y=2 上，找到与原点最近的点
原点到直线 x+y=2 的距离 = 2/√2 = √2
最近点坐标：(1, 1) ✓
```

---

**PyTorch中的实现**
```python
def gradient_computation_internals():
    """
    梯度计算的内部机制
    """
    # 1. 标量函数的梯度
    x = torch.tensor([1., 2., 3.], requires_grad=True)
    y = torch.sum(x ** 2)  # y = x₁² + x₂² + x₃²

    # 计算梯度：∂y/∂x = [2x₁, 2x₂, 2x₃]
    y.backward()
    print(f"梯度: {x.grad}")  # [2., 4., 6.]

    # 2. 雅可比矩阵
    def compute_jacobian(func, x):
        """
        计算雅可比矩阵：J[i,j] = ∂fᵢ/∂xⱼ
        """
        x = x.detach().requires_grad_(True)
        y = func(x)

        jacobian = []
        for i in range(y.shape[0]):
            grad_outputs = torch.zeros_like(y)
            grad_outputs[i] = 1

            x.grad = None
            y.backward(grad_outputs, retain_graph=True)
            jacobian.append(x.grad.clone())

        return torch.stack(jacobian)

    # 示例：f(x) = [x₁², x₁x₂, x₂²]
    def f(x):
        return torch.stack([x[0]**2, x[0]*x[1], x[1]**2])

    x = torch.tensor([2., 3.])
    J = compute_jacobian(f, x)
    print(f"雅可比矩阵:\n{J}")

    # 3. Hessian矩阵（用于二阶优化）
    def compute_hessian(func, x):
        """
        计算Hessian矩阵：H[i,j] = ∂²f/∂xᵢ∂xⱼ
        """
        x = x.detach().requires_grad_(True)
        y = func(x)

        # 先计算一阶导数
        grad = torch.autograd.grad(y, x, create_graph=True)[0]

        # 再对每个梯度分量计算导数
        hessian = []
        for i in range(len(grad)):
            hessian.append(
                torch.autograd.grad(grad[i], x, retain_graph=True)[0]
            )

        return torch.stack(hessian)

    # 示例：f(x) = x₁² + 3x₁x₂ + x₂²
    def f_scalar(x):
        return x[0]**2 + 3*x[0]*x[1] + x[1]**2

    x = torch.tensor([1., 2.])
    H = compute_hessian(f_scalar, x)
    print(f"Hessian矩阵:\n{H}")
    # 理论值：[[2, 3], [3, 2]]

def comprehensive_gradient_examples():
    """
    梯度计算的全面数学示例
    """
    print("\n=== 梯度计算综合示例 ===\n")

    # 1. 标量函数的梯度
    print("1. 标量函数的梯度")

    # f(x, y, z) = x^2 + 2y^2 + 3z^2 + xy - yz
    # ∇f = [2x + y, 4y + x - z, 6z - y]
    x = torch.tensor([1., 2., 3.], requires_grad=True)

    f = x[0]**2 + 2*x[1]**2 + 3*x[2]**2 + x[0]*x[1] - x[1]*x[2]
    f.backward()

    print(f"f = {f.item():.4f}")
    print(f"∇f = {x.grad}")

    # 手动验证
    manual_grad = torch.tensor([
        2*1. + 2.,  # ∂f/∂x = 2x + y
        4*2. + 1. - 3.,  # ∂f/∂y = 4y + x - z
        6*3. - 2.   # ∂f/∂z = 6z - y
    ])
    print(f"手动计算: {manual_grad}")
    print(f"验证误差: {torch.norm(x.grad - manual_grad):.2e}")

    # 2. 向量函数的雅可比矩阵
    print("\n2. 向量函数的雅可比矩阵")

    # f: R^2 -> R^3
    # f(x, y) = [x^2 + y, xy, y^2]
    # J = [[2x, 1], [y, x], [0, 2y]]
    def vector_function(x):
        return torch.stack([
            x[0]**2 + x[1],
            x[0] * x[1],
            x[1]**2
        ])

    def compute_jacobian_functional(func, x):
        """使用torch.autograd.functional计算雅可比矩阵"""
        return torch.autograd.functional.jacobian(func, x)

    x = torch.tensor([2., 3.])
    J = compute_jacobian_functional(vector_function, x)
    print(f"雅可比矩阵:\n{J}")

    # 手动验证
    manual_J = torch.tensor([
        [2*2., 1.],
        [3., 2.],
        [0., 2*3.]
    ])
    print(f"手动计算:\n{manual_J}")
    print(f"验证误差: {torch.norm(J - manual_J):.2e}")

    # 3. Hessian矩阵的详细计算
    print("\n3. Hessian矩阵")

    # f(x, y) = x^3 + y^3 + 3xy
    # ∇f = [3x^2 + 3y, 3y^2 + 3x]
    # H = [[6x, 3], [3, 6y]]
    def f_hessian(x):
        return x[0]**3 + x[1]**3 + 3*x[0]*x[1]

    x = torch.tensor([2., 3.])
    H = torch.autograd.functional.hessian(f_hessian, x)
    print(f"Hessian矩阵:\n{H}")

    # 手动验证
    manual_H = torch.tensor([
        [6*2., 3.],
        [3., 6*3.]
    ])
    print(f"手动计算:\n{manual_H}")

    # Hessian的特征值决定了曲率
    eigenvalues = torch.linalg.eigvalsh(H)
    print(f"Hessian特征值: {eigenvalues}")
    if (eigenvalues > 0).all():
        print("所有特征值为正 → 局部极小值")
    elif (eigenvalues < 0).all():
        print("所有特征值为负 → 局部极大值")
    else:
        print("有正有负 → 鞍点")

    # 4. 方向导数
    print("\n4. 方向导数")

    # 方向导数：D_v f(x) = ∇f(x) · v
    x = torch.tensor([1., 2.], requires_grad=True)
    f = x[0]**2 + x[1]**2

    f.backward()
    grad = x.grad.clone()

    # 沿着方向 v = [1, 1]/√2
    v = torch.tensor([1., 1.]) / torch.sqrt(torch.tensor(2.))
    directional_derivative = torch.dot(grad, v)
    print(f"梯度: {grad}")
    print(f"方向: {v}")
    print(f"方向导数: {directional_derivative:.4f}")

    # 5. 链式法则
    print("\n5. 链式法则")

    # h(x) = g(f(x))，其中 f(x) = x^2, g(y) = sin(y)
    # h'(x) = g'(f(x)) * f'(x) = cos(x^2) * 2x
    x = torch.tensor([torch.pi/2], requires_grad=True)
    y = x**2
    z = torch.sin(y)

    z.backward()
    print(f"h({x.item():.4f}) = sin(x^2) = {z.item():.4f}")
    print(f"h'(x) = {x.grad.item():.4f}")

    # 手动验证：cos((π/2)^2) * 2(π/2)
    manual_deriv = torch.cos(y.detach()) * 2 * x.detach()
    print(f"手动计算: {manual_deriv.item():.4f}")

    # 6. 多元链式法则
    print("\n6. 多元链式法则")

    # z = f(u, v), u = g(x, y), v = h(x, y)
    # ∂z/∂x = ∂z/∂u * ∂u/∂x + ∂z/∂v * ∂v/∂x
    x = torch.tensor([1., 2.], requires_grad=True)

    u = x[0]**2 + x[1]  # u(x, y) = x^2 + y
    v = x[0] * x[1]     # v(x, y) = xy
    z = u**2 + v**2     # z(u, v) = u^2 + v^2

    z.backward()
    print(f"∇z = {x.grad}")

    # 手动验证
    # ∂z/∂x = 2u * 2x + 2v * y = 4x(x^2 + y) + 2xy^2
    # ∂z/∂y = 2u * 1 + 2v * x = 2(x^2 + y) + 2x^2y
    u_val = x[0].detach()**2 + x[1].detach()
    v_val = x[0].detach() * x[1].detach()
    manual_grad_x = 2 * u_val * 2 * x[0].detach() + 2 * v_val * x[1].detach()
    manual_grad_y = 2 * u_val * 1 + 2 * v_val * x[0].detach()
    manual_grad = torch.tensor([manual_grad_x, manual_grad_y])
    print(f"手动计算: {manual_grad}")

    # 7. 梯度的几何意义
    print("\n7. 梯度的几何意义")

    # 梯度指向函数增长最快的方向
    def rosenbrock(x):
        """Rosenbrock函数：(1-x)^2 + 100(y-x^2)^2"""
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    x = torch.tensor([0., 0.], requires_grad=True)
    f = rosenbrock(x)
    f.backward()

    grad = x.grad.clone()
    grad_norm = torch.norm(grad)
    grad_direction = grad / grad_norm

    print(f"函数值: {f.item():.4f}")
    print(f"梯度: {grad}")
    print(f"梯度范数: {grad_norm:.4f}")
    print(f"梯度方向: {grad_direction}")

    # 沿梯度方向的方向导数应该等于梯度范数
    directional_deriv = torch.dot(grad, grad_direction)
    print(f"沿梯度方向的方向导数: {directional_deriv:.4f}")

    # 8. 高阶导数
    print("\n8. 高阶导数")

    # f(x) = sin(x), f'(x) = cos(x), f''(x) = -sin(x)
    x = torch.tensor([torch.pi/4], requires_grad=True)
    y = torch.sin(x)

    # 一阶导数
    dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]

    # 二阶导数
    d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]

    print(f"f(π/4) = sin(π/4) = {y.item():.6f}")
    print(f"f'(π/4) = cos(π/4) = {dy_dx.item():.6f}")
    print(f"f''(π/4) = -sin(π/4) = {d2y_dx2.item():.6f}")

    # 验证
    print(f"理论值 sin(π/4) = {(torch.sqrt(torch.tensor(2.))/2).item():.6f}")
    print(f"理论值 cos(π/4) = {(torch.sqrt(torch.tensor(2.))/2).item():.6f}")

    # 9. 批量梯度计算
    print("\n9. 批量梯度计算")

    # 对一个batch的样本计算梯度
    batch_size = 32
    x = torch.randn(batch_size, 2, requires_grad=True)

    # 对每个样本计算损失
    loss_per_sample = (x ** 2).sum(dim=1)  # [batch_size]
    loss = loss_per_sample.mean()

    loss.backward()
    print(f"批量输入形状: {x.shape}")
    print(f"批量梯度形状: {x.grad.shape}")
    print(f"平均梯度范数: {torch.norm(x.grad, dim=1).mean():.4f}")
```

#### 1.2.2 凸优化理论

```python
class ConvexOptimizationTheory:
    """
    凸优化在深度学习中的应用
    """

    @staticmethod
    def check_convexity(func, x_range):
        """
        检查函数的凸性
        凸函数：f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
        """
        # 检查Hessian矩阵是否半正定
        pass

    @staticmethod
    def gradient_descent_convergence():
        """
        梯度下降的收敛性分析
        """
        # 对于L-Lipschitz连续梯度的凸函数
        # 学习率 α < 2/L 时，GD收敛

        # 强凸函数（条件数κ = L/μ）
        # 收敛率：O((1 - μ/L)^k)

        def smooth_convex_loss(x):
            """示例：二次损失（强凸且光滑）"""
            Q = torch.tensor([[2., 0.], [0., 1.]])
            return 0.5 * x @ Q @ x

        # 计算条件数
        Q = torch.tensor([[2., 0.], [0., 1.]])
        eigenvalues = torch.linalg.eigvalsh(Q)
        condition_number = eigenvalues.max() / eigenvalues.min()
        print(f"条件数: {condition_number}")

        # 理论最优学习率
        L = eigenvalues.max()
        optimal_lr = 1 / L
        print(f"理论最优学习率: {optimal_lr}")
```

### 1.3 概率论与信息论

#### 1.3.1 概率分布与采样

**纯数学例子：概率论基础计算**

**例1：离散概率分布**
```
抛硬币实验：
样本空间 Ω = {正面H, 反面T}
公平硬币：P(H) = P(T) = 0.5

掷骰子实验：
样本空间 Ω = {1, 2, 3, 4, 5, 6}
公平骰子：P(X=k) = 1/6, k=1,2,...,6

期望值：E[X] = Σ k·P(X=k)
E[X] = 1(1/6) + 2(1/6) + 3(1/6) + 4(1/6) + 5(1/6) + 6(1/6)
     = (1+2+3+4+5+6)/6 = 21/6 = 3.5

方差：Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²
E[X²] = 1²(1/6) + 2²(1/6) + ... + 6²(1/6)
      = (1+4+9+16+25+36)/6 = 91/6 ≈ 15.17

Var(X) = 91/6 - (21/6)² = 91/6 - 441/36
       = 546/36 - 441/36 = 105/36 ≈ 2.92

标准差：σ = √Var(X) = √(105/36) ≈ 1.71
```

**例2：条件概率与贝叶斯定理**
```
医学诊断问题：
- 疾病发病率：P(D) = 0.01（1%的人患病）
- 测试灵敏度：P(+|D) = 0.95（患病者测试阳性的概率）
- 测试特异度：P(-|D^c) = 0.90（健康者测试阴性的概率）

问：测试阳性时实际患病的概率 P(D|+) = ?

贝叶斯定理：
P(D|+) = P(+|D)·P(D) / P(+)

计算 P(+)（全概率公式）：
P(+) = P(+|D)·P(D) + P(+|D^c)·P(D^c)
     = 0.95 × 0.01 + 0.10 × 0.99
     = 0.0095 + 0.099
     = 0.1085

因此：
P(D|+) = (0.95 × 0.01) / 0.1085
       = 0.0095 / 0.1085
       ≈ 0.0876 ≈ 8.76%

解释：即使测试阳性，实际患病概率仅约9%！
（因为疾病本身很罕见）
```

**例3：连续概率分布 - 正态分布**
```
标准正态分布 N(0, 1)：
概率密度函数（PDF）：
φ(x) = (1/√(2π)) exp(-x²/2)

计算 P(-1 ≤ X ≤ 1)：
P(-1 ≤ X ≤ 1) = ∫_{-1}^{1} φ(x)dx
               = Φ(1) - Φ(-1)
               = 0.8413 - 0.1587
               = 0.6826 ≈ 68.26%

经验法则（68-95-99.7规则）：
- P(μ-σ ≤ X ≤ μ+σ) ≈ 68%
- P(μ-2σ ≤ X ≤ μ+2σ) ≈ 95%
- P(μ-3σ ≤ X ≤ μ+3σ) ≈ 99.7%

一般正态分布 N(μ, σ²)：
标准化变换：Z = (X - μ)/σ ~ N(0, 1)

例：X ~ N(100, 15²)（IQ分数）
求 P(X > 130)：
Z = (130 - 100)/15 = 2
P(X > 130) = P(Z > 2) = 1 - Φ(2) ≈ 1 - 0.9772 = 0.0228 ≈ 2.28%
```

**例4：联合概率与独立性**
```
掷两个骰子：
X = 第一个骰子的点数
Y = 第二个骰子的点数

联合概率（独立情况）：
P(X=i, Y=j) = P(X=i)·P(Y=j) = (1/6)·(1/6) = 1/36

事件A：两个骰子点数之和为7
A = {(1,6), (2,5), (3,4), (4,3), (5,2), (6,1)}
P(A) = 6/36 = 1/6

事件B：第一个骰子为偶数
B = {(2,y), (4,y), (6,y)}, y=1,2,...,6
P(B) = 18/36 = 1/2

条件概率：
P(A|B) = P(A∩B)/P(B)

A∩B = {(2,5), (4,3), (6,1)}
P(A∩B) = 3/36

P(A|B) = (3/36)/(18/36) = 3/18 = 1/6 = P(A)

结论：A和B独立（知道第一个骰子是偶数不影响和为7的概率）
```

**例5：期望值与方差的性质**
```
随机变量 X 和 Y

期望的线性性：
E[aX + bY + c] = aE[X] + bE[Y] + c

例：X是骰子点数，E[X] = 3.5
Y = 2X + 3（线性变换）

E[Y] = E[2X + 3] = 2E[X] + 3 = 2(3.5) + 3 = 10

方差的性质：
Var(aX + b) = a²Var(X)

Var(Y) = Var(2X + 3) = 2²Var(X) = 4(2.92) ≈ 11.67

独立情况下：
Var(X + Y) = Var(X) + Var(Y)
Var(X - Y) = Var(X) + Var(Y)

协方差：
Cov(X, Y) = E[(X - E[X])(Y - E[Y])]
          = E[XY] - E[X]E[Y]

相关系数：
ρ(X, Y) = Cov(X, Y) / (σ_X · σ_Y)
范围：-1 ≤ ρ ≤ 1
ρ = 1：完全正相关
ρ = 0：不相关
ρ = -1：完全负相关
```

**例6：大数定律与中心极限定理**
```
大数定律：
设 X₁, X₂, ..., Xₙ 独立同分布，E[Xᵢ] = μ

样本均值：X̄ₙ = (X₁ + X₂ + ... + Xₙ)/n

则：X̄ₙ → μ（当 n → ∞）

例：抛硬币1000次
设 Xᵢ = {1 (正面), 0 (反面)}
E[Xᵢ] = 0.5

X̄₁₀₀₀ = (正面次数)/1000 ≈ 0.5

中心极限定理：
√n(X̄ₙ - μ) → N(0, σ²)（分布收敛）

或等价地：
X̄ₙ ~ N(μ, σ²/n)（当n足够大）

例：掷骰子100次，求平均值大于4的概率
E[X] = 3.5, Var(X) = 2.92

X̄₁₀₀ ~ N(3.5, 2.92/100) = N(3.5, 0.0292)
σ_{X̄} = √0.0292 ≈ 0.171

P(X̄₁₀₀ > 4) = P(Z > (4-3.5)/0.171)
              = P(Z > 2.92)
              ≈ 0.0018 ≈ 0.18%
```

**例7：最大似然估计（MLE）**
```
观测数据：x₁, x₂, ..., xₙ ~ N(μ, σ²)

似然函数：
L(μ, σ²) = ∏ᵢ (1/√(2πσ²)) exp(-(xᵢ-μ)²/(2σ²))

对数似然：
ℓ(μ, σ²) = -n/2 log(2π) - n/2 log(σ²) - Σ(xᵢ-μ)²/(2σ²)

求导并令其为0：
∂ℓ/∂μ = Σ(xᵢ-μ)/σ² = 0
→ μ̂ = (1/n)Σxᵢ = x̄（样本均值）

∂ℓ/∂σ² = -n/(2σ²) + Σ(xᵢ-μ)²/(2σ⁴) = 0
→ σ̂² = (1/n)Σ(xᵢ-μ̂)²（样本方差）

数值例子：观测值 [2, 4, 3, 5, 6]
μ̂ = (2+4+3+5+6)/5 = 20/5 = 4

σ̂² = [(2-4)² + (4-4)² + (3-4)² + (5-4)² + (6-4)²]/5
    = [4 + 0 + 1 + 1 + 4]/5
    = 10/5 = 2

估计结果：N(4, 2)
```

**例8：KL散度（Kullback-Leibler Divergence）**
```
两个概率分布 P 和 Q

KL散度定义：
D_KL(P||Q) = Σ P(x) log(P(x)/Q(x))  （离散情况）
           = ∫ p(x) log(p(x)/q(x))dx  （连续情况）

性质：
1. D_KL(P||Q) ≥ 0
2. D_KL(P||Q) = 0 当且仅当 P = Q
3. 非对称：D_KL(P||Q) ≠ D_KL(Q||P)

例：离散分布
P = [0.5, 0.3, 0.2]
Q = [0.4, 0.4, 0.2]

D_KL(P||Q) = 0.5·log(0.5/0.4) + 0.3·log(0.3/0.4) + 0.2·log(0.2/0.2)
           = 0.5·log(1.25) + 0.3·log(0.75) + 0
           = 0.5·(0.223) + 0.3·(-0.288)
           = 0.112 - 0.086
           = 0.026

正态分布的KL散度：
P = N(μ₁, σ₁²), Q = N(μ₂, σ₂²)

D_KL(P||Q) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2

例：P = N(0, 1), Q = N(1, 2)
D_KL(P||Q) = log(√2) + (1 + (0-1)²)/(2·2) - 1/2
           = 0.347 + (1 + 1)/4 - 0.5
           = 0.347 + 0.5 - 0.5
           = 0.347
```

**例9：交叉熵（Cross-Entropy）**
```
定义：H(P, Q) = -Σ P(x) log Q(x)

与KL散度的关系：
D_KL(P||Q) = H(P, Q) - H(P)

其中 H(P) = -Σ P(x) log P(x) 是P的熵

分类问题中的交叉熵损失：
真实分布：y = [0, 1, 0]（one-hot编码，类别2）
预测分布：ŷ = [0.1, 0.7, 0.2]（softmax输出）

交叉熵损失：
L = -Σ yᵢ log(ŷᵢ)
  = -0·log(0.1) - 1·log(0.7) - 0·log(0.2)
  = -log(0.7)
  = 0.357

如果预测更准确：ŷ' = [0.05, 0.9, 0.05]
L' = -log(0.9) = 0.105 < 0.357 ✓

二元交叉熵（Binary Cross-Entropy）：
L = -[y log(ŷ) + (1-y) log(1-ŷ)]

例：y=1（正类），ŷ=0.8
L = -[1·log(0.8) + 0·log(0.2)]
  = -log(0.8)
  ≈ 0.223
```

**例10：贝塔分布与共轭先验**
```
贝叶斯统计中的共轭先验：

似然：伯努利分布 p(x|θ) = θ^x(1-θ)^(1-x)
先验：贝塔分布 p(θ) = Beta(α, β)

贝塔分布PDF：
p(θ) = [Γ(α+β)/(Γ(α)Γ(β))] θ^(α-1)(1-θ)^(β-1)

观测n次，k次成功：
后验分布：θ|data ~ Beta(α+k, β+n-k)

数值例子：
先验：α=2, β=2（温和信念，接近均匀）
E[θ] = α/(α+β) = 2/4 = 0.5

观测：抛硬币10次，7次正面
后验：θ|data ~ Beta(2+7, 2+3) = Beta(9, 5)
E[θ|data] = 9/14 ≈ 0.643

比较：
- 最大似然估计：θ̂ = 7/10 = 0.7
- 贝叶斯估计：0.643（融合了先验信息）

继续观测10次，又8次正面：
后验：θ ~ Beta(9+8, 5+2) = Beta(17, 7)
E[θ] = 17/24 ≈ 0.708

随着数据增多，后验逐渐接近MLE
```

---

```python
def probability_distributions_in_pytorch():
    """
    PyTorch中的概率分布实现
    """
    from torch.distributions import (
        Normal, Categorical, Bernoulli,
        MultivariateNormal, Exponential
    )

    # 1. 正态分布 N(μ, σ²)
    # PDF: p(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))
    normal = Normal(loc=0., scale=1.)
    samples = normal.sample((1000,))
    log_prob = normal.log_prob(samples)  # 对数概率密度

    # 2. 多元正态分布
    mean = torch.zeros(2)
    cov = torch.tensor([[1., 0.5], [0.5, 1.]])
    mvn = MultivariateNormal(mean, cov)
    samples = mvn.sample((1000,))

    # 3. 重参数化技巧（Reparameterization Trick）
    # 用于VAE等生成模型
    def reparameterize(mu, logvar):
        """
        z ~ N(μ, σ²)
        等价于 z = μ + σ * ε, 其中 ε ~ N(0, 1)
        这样梯度可以通过μ和σ反向传播
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # 4. 类别分布与Gumbel-Softmax
    # 用于离散采样的可微近似
    def gumbel_softmax_sample(logits, temperature=1.0):
        """
        Gumbel-Softmax技巧：离散分布的连续松弛
        """
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
        y = logits + gumbel_noise
        return torch.softmax(y / temperature, dim=-1)

def comprehensive_probability_examples():
    """
    概率论和信息论的全面数学示例
    """
    print("\n=== 概率论和信息论综合示例 ===\n")

    from torch.distributions import (
        Normal, MultivariateNormal, Categorical, Bernoulli,
        Beta, Gamma, Exponential, Poisson, Uniform
    )

    # 1. 基本概率分布
    print("1. 基本概率分布")

    # 正态分布 N(μ, σ²)
    normal = Normal(loc=0., scale=1.)
    samples = normal.sample((10000,))
    print(f"正态分布样本均值: {samples.mean():.4f} (理论值: 0.0)")
    print(f"正态分布样本标准差: {samples.std():.4f} (理论值: 1.0)")

    # 计算对数概率密度
    x = torch.tensor([0., 1., -1.])
    log_prob = normal.log_prob(x)
    prob = torch.exp(log_prob)
    print(f"x={x.tolist()} 的概率密度: {prob.tolist()}")

    # 2. 多元正态分布
    print("\n2. 多元正态分布")

    # 构造协方差矩阵
    mean = torch.zeros(3)
    # 相关系数矩阵
    corr = torch.tensor([[1.0, 0.5, 0.3],
                        [0.5, 1.0, 0.4],
                        [0.3, 0.4, 1.0]])
    # 标准差
    std = torch.tensor([1.0, 2.0, 3.0])
    # 协方差矩阵 Σ = D * R * D，其中D是标准差对角矩阵，R是相关系数矩阵
    cov = torch.diag(std) @ corr @ torch.diag(std)

    mvn = MultivariateNormal(mean, cov)
    samples = mvn.sample((10000,))

    print(f"样本均值: {samples.mean(dim=0)}")
    print(f"理论均值: {mean}")

    sample_cov = torch.cov(samples.t())
    print(f"样本协方差矩阵:\n{sample_cov}")
    print(f"理论协方差矩阵:\n{cov}")

    # 3. 重参数化技巧（Reparameterization Trick）
    print("\n3. 重参数化技巧")

    # VAE中的核心技巧：使采样操作可导
    mu = torch.tensor([1., 2., 3.], requires_grad=True)
    logvar = torch.tensor([0., -1., -2.], requires_grad=True)

    def reparameterize(mu, logvar):
        """z ~ N(μ, σ²) = μ + σ * ε, ε ~ N(0, 1)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    z = reparameterize(mu, logvar)
    loss = z.sum()
    loss.backward()

    print(f"μ的梯度: {mu.grad}")  # 梯度可以反向传播
    print(f"logvar的梯度: {logvar.grad}")

    # 4. KL散度
    print("\n4. KL散度")

    # KL(p||q) = ∫ p(x) log(p(x)/q(x)) dx
    # 对于两个正态分布 N(μ₁,σ₁²) 和 N(μ₂,σ₂²)：
    # KL = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2

    mu1, sigma1 = torch.tensor([0.]), torch.tensor([1.])
    mu2, sigma2 = torch.tensor([1.]), torch.tensor([2.])

    p = Normal(mu1, sigma1)
    q = Normal(mu2, sigma2)

    # 使用PyTorch内置函数
    kl_div = torch.distributions.kl_divergence(p, q)
    print(f"KL(p||q) = {kl_div.item():.6f}")

    # 手动计算验证
    kl_manual = (torch.log(sigma2 / sigma1) +
                (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5)
    print(f"手动计算: {kl_manual.item():.6f}")

    # 对于标准正态分布 N(0,1) 和 N(μ,σ²)：
    # KL = 0.5 * (μ² + σ² - log(σ²) - 1)
    mu, logvar = torch.tensor([1.]), torch.tensor([0.])
    kl_standard = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    print(f"KL(N(μ,σ²)||N(0,1)) = {kl_standard.item():.6f}")

    # 5. 交叉熵
    print("\n5. 交叉熵")

    # H(p, q) = -∑ p(x) log q(x)
    # 分类任务中的交叉熵损失
    logits = torch.randn(5, 10)  # 5个样本，10个类别
    targets = torch.randint(0, 10, (5,))

    # 方法1：使用F.cross_entropy
    import torch.nn.functional as F
    ce_loss = F.cross_entropy(logits, targets)
    print(f"交叉熵损失: {ce_loss.item():.6f}")

    # 方法2：手动计算
    log_probs = F.log_softmax(logits, dim=1)
    ce_manual = F.nll_loss(log_probs, targets)
    print(f"手动计算: {ce_manual.item():.6f}")

    # 方法3：完全手动
    probs = F.softmax(logits, dim=1)
    ce_full_manual = -torch.log(probs[range(5), targets]).mean()
    print(f"完全手动: {ce_full_manual.item():.6f}")

    # 6. 熵（Entropy）
    print("\n6. 熵")

    # H(p) = -∑ p(x) log p(x)
    # 衡量不确定性
    probs = torch.tensor([0.7, 0.2, 0.1])  # 低熵（较确定）
    entropy_low = -(probs * torch.log(probs + 1e-10)).sum()

    probs_uniform = torch.ones(10) / 10  # 均匀分布（高熵）
    entropy_high = -(probs_uniform * torch.log(probs_uniform)).sum()

    print(f"低熵（确定性高）: {entropy_low.item():.6f}")
    print(f"高熵（不确定性高）: {entropy_high.item():.6f}")
    print(f"最大熵（均匀分布）: {torch.log(torch.tensor(10.)).item():.6f}")

    # 使用分布对象计算熵
    cat = Categorical(probs=probs)
    entropy_pytorch = cat.entropy()
    print(f"PyTorch计算的熵: {entropy_pytorch.item():.6f}")

    # 7. 互信息（Mutual Information）
    print("\n7. 互信息近似")

    # I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
    # 在实践中通常通过变分方法估计
    # 这里演示概念

    # 生成相关的随机变量
    x = torch.randn(1000, 1)
    y = 2 * x + torch.randn(1000, 1) * 0.5  # y与x高度相关

    # 简化的互信息估计（基于相关系数）
    # 对于联合正态分布：I(X;Y) = -0.5 * log(1 - ρ²)
    corr_matrix = torch.corrcoef(torch.cat([x, y], dim=1).t())
    rho = corr_matrix[0, 1]
    mi_estimate = -0.5 * torch.log(1 - rho**2 + 1e-10)
    print(f"相关系数: {rho.item():.4f}")
    print(f"互信息估计: {mi_estimate.item():.4f}")

    # 8. 采样方法
    print("\n8. 高级采样方法")

    # 逆变换采样（Inverse Transform Sampling）
    # 示例：从指数分布采样
    lambda_param = 2.0
    u = torch.rand(1000)  # U(0,1)
    exp_samples = -torch.log(1 - u) / lambda_param

    exp_dist = Exponential(lambda_param)
    exp_samples_builtin = exp_dist.sample((1000,))

    print(f"逆变换采样均值: {exp_samples.mean():.4f}")
    print(f"内置采样均值: {exp_samples_builtin.mean():.4f}")
    print(f"理论均值: {1/lambda_param:.4f}")

    # 拒绝采样示例（Rejection Sampling）
    # 从 Beta(2, 2) 分布采样
    def rejection_sampling_beta():
        """使用拒绝采样从Beta(2,2)采样"""
        samples = []
        M = 1.5  # 包络常数

        while len(samples) < 1000:
            # 提议分布：均匀分布U(0,1)
            x = torch.rand(1).item()
            u = torch.rand(1).item()

            # 目标密度（未归一化）：x(1-x)
            target_density = x * (1 - x) * 6  # Beta(2,2)的密度
            proposal_density = 1.0  # U(0,1)的密度

            if u < target_density / (M * proposal_density):
                samples.append(x)

        return torch.tensor(samples)

    rejection_samples = rejection_sampling_beta()
    beta_dist = Beta(2., 2.)
    beta_samples = beta_dist.sample((1000,))

    print(f"拒绝采样均值: {rejection_samples.mean():.4f}")
    print(f"内置采样均值: {beta_samples.mean():.4f}")
    print(f"理论均值: 0.5000")

    # 9. 期望值和方差计算
    print("\n9. 期望值和方差")

    # 使用蒙特卡洛方法估计期望 E[f(X)]
    normal = Normal(0., 1.)
    samples = normal.sample((100000,))

    # E[X²] for X ~ N(0,1) = 1
    exp_x2 = (samples ** 2).mean()
    print(f"E[X²] = {exp_x2.item():.4f} (理论值: 1.0)")

    # E[exp(X)] for X ~ N(0,1) = exp(0.5)
    exp_exp_x = torch.exp(samples).mean()
    print(f"E[exp(X)] = {exp_exp_x.item():.4f} (理论值: {torch.exp(torch.tensor(0.5)).item():.4f})")

    # 方差：Var[X] = E[X²] - E[X]²
    var_x = (samples ** 2).mean() - samples.mean() ** 2
    print(f"Var[X] = {var_x.item():.4f} (理论值: 1.0)")

    # 10. 条件概率和贝叶斯推断
    print("\n10. 贝叶斯推断示例")

    # 贝叶斯定理：P(θ|D) ∝ P(D|θ) * P(θ)
    # 示例：估计硬币正面概率

    # 先验：Beta(2, 2)（温和先验）
    alpha_prior, beta_prior = 2., 2.
    prior = Beta(alpha_prior, beta_prior)

    # 观测数据：10次投掷，7次正面
    heads, tails = 7, 3

    # 后验：Beta(α + heads, β + tails)
    alpha_post = alpha_prior + heads
    beta_post = beta_prior + tails
    posterior = Beta(alpha_post, beta_post)

    # 比较先验和后验
    theta_grid = torch.linspace(0, 1, 100)
    prior_pdf = torch.exp(prior.log_prob(theta_grid))
    posterior_pdf = torch.exp(posterior.log_prob(theta_grid))

    print(f"先验均值: {prior.mean:.4f}")
    print(f"后验均值: {posterior.mean:.4f}")
    print(f"最大似然估计: {heads/(heads+tails):.4f}")

    # 计算95%置信区间
    credible_interval = [posterior.icdf(torch.tensor(0.025)),
                        posterior.icdf(torch.tensor(0.975))]
    print(f"95%可信区间: [{credible_interval[0]:.4f}, {credible_interval[1]:.4f}]")
```

#### 1.3.2 信息论基础

**核心概念**
```
熵（Entropy）：
H(X) = -Σ p(x) log p(x)

交叉熵（Cross-Entropy）：
H(p, q) = -Σ p(x) log q(x)

KL散度（Kullback-Leibler Divergence）：
KL(p||q) = Σ p(x) log(p(x)/q(x)) = H(p,q) - H(p)
```

```python
def information_theory_in_dl():
    """
    信息论在深度学习中的应用
    """

    # 1. 交叉熵损失的数学本质
    def cross_entropy_from_scratch(predictions, targets):
        """
        分类问题的交叉熵损失
        predictions: logits (未归一化)
        targets: 真实类别索引
        """
        # Softmax: qᵢ = exp(zᵢ) / Σexp(zⱼ)
        log_probs = torch.log_softmax(predictions, dim=-1)

        # 交叉熵: -Σ p(x) log q(x)
        # 对于one-hot编码，只有正确类别的项非零
        nll = -log_probs[range(len(targets)), targets]
        return nll.mean()

    # 2. KL散度的应用
    def kl_divergence(p_logits, q_logits):
        """
        计算两个分布之间的KL散度
        用于知识蒸馏、VAE等
        """
        p = torch.softmax(p_logits, dim=-1)
        log_p = torch.log_softmax(p_logits, dim=-1)
        log_q = torch.log_softmax(q_logits, dim=-1)

        # KL(p||q) = Σ p * (log p - log q)
        return (p * (log_p - log_q)).sum(dim=-1).mean()

    # 3. 互信息（Mutual Information）
    # I(X;Y) = KL(p(x,y) || p(x)p(y))
    # 用于特征选择和表示学习

    # 4. 熵正则化
    def entropy_regularization(logits):
        """
        熵作为正则化项，鼓励探索
        用于强化学习
        """
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy.mean()
```

---

## 第二部分：PyTorch核心架构

### 2.1 张量计算系统

#### 2.1.1 内存布局与存储机制

```python
class TensorStorageSystem:
    """
    深入理解PyTorch张量的存储系统
    """

    def storage_and_view(self):
        """
        存储（Storage）与视图（View）机制
        """
        # 1. 存储是一维连续内存块
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        print(f"存储内容: {x.storage()}")  # [1, 2, 3, 4, 5, 6]

        # 2. 张量是存储的视图
        # 通过offset, size, stride定义
        print(f"偏移量: {x.storage_offset()}")
        print(f"形状: {x.size()}")
        print(f"步幅: {x.stride()}")  # (3, 1) 表示行间隔3，列间隔1

        # 3. 视图操作零拷贝
        y = x.view(3, 2)  # 只改变形状元数据
        z = x.t()          # 转置只改变步幅

        # 4. 步幅的数学意义
        # 元素x[i,j]的内存地址：
        # addr = storage_offset + i*stride[0] + j*stride[1]

    def advanced_indexing(self):
        """
        高级索引与内存连续性
        """
        x = torch.randn(10, 20, 30)

        # 基础索引（不复制）
        y1 = x[0]         # shape: (20, 30)
        y2 = x[:, 0, :]   # shape: (10, 30)

        # 高级索引（复制数据）
        indices = torch.tensor([0, 2, 4])
        y3 = x[indices]   # 需要复制，因为索引不连续

        # 掩码索引
        mask = x > 0
        y4 = x[mask]      # 返回一维张量

    def memory_format(self):
        """
        内存格式：NCHW vs NHWC
        """
        # Channels Last格式（对某些硬件更优）
        x = torch.randn(8, 3, 224, 224)  # NCHW
        x_cl = x.to(memory_format=torch.channels_last)  # NHWC

        # 检查内存格式
        print(f"是否channels_last: {x_cl.is_contiguous(memory_format=torch.channels_last)}")
```

#### 2.1.2 数据类型系统

```python
def dtype_system_deep_dive():
    """
    PyTorch数据类型系统详解
    """
    # 1. 浮点类型
    dtypes_float = {
        torch.float64: "双精度（FP64）, 1 sign + 11 exp + 52 mantissa",
        torch.float32: "单精度（FP32）, 1 sign + 8 exp + 23 mantissa",
        torch.float16: "半精度（FP16）, 1 sign + 5 exp + 10 mantissa",
        torch.bfloat16: "Brain Float16, 1 sign + 8 exp + 7 mantissa"
    }

    # 2. BFloat16的优势
    # - 指数位与FP32相同（8位），动态范围大
    # - 尾数位少（7位），精度降低但够用
    # - 与FP32转换简单（截断尾数）

    x_fp32 = torch.randn(100, 100)
    x_bf16 = x_fp32.to(torch.bfloat16)

    # 3. 混合精度训练的数学原理
    class MixedPrecisionMath:
        """
        混合精度训练的数学考虑
        """
        def understand_precision_loss(self):
            # FP16的问题：
            # - 最小正数：2^-24 ≈ 6e-8
            # - 最大数：65504
            # - 梯度下溢/上溢风险

            # 解决方案：损失缩放
            scale = 2 ** 16

            # 前向：FP16
            # 损失：FP32
            # 梯度：scale × grad (FP16)
            # 更新：(grad / scale) (FP32)

    # 4. 整数类型与量化
    def quantization_basics():
        """
        量化：FP32 → INT8
        """
        x = torch.randn(100) * 10  # 范围 ~[-30, 30]

        # 对称量化
        scale = x.abs().max() / 127
        x_int8 = torch.round(x / scale).to(torch.int8)
        x_dequant = x_int8.float() * scale

        # 非对称量化
        x_min, x_max = x.min(), x.max()
        scale = (x_max - x_min) / 255
        zero_point = -torch.round(x_min / scale).to(torch.int32)

        x_uint8 = torch.round(x / scale + zero_point).to(torch.uint8)
```

### 2.2 设备管理与CUDA编程

#### 2.2.1 CUDA内存层级

```python
class CUDAMemoryHierarchy:
    """
    CUDA内存层级与优化
    """

    def memory_types(self):
        """
        CUDA内存类型：
        1. 全局内存（Global Memory）：大但慢
        2. 共享内存（Shared Memory）：快但小
        3. 寄存器（Registers）：最快但最小
        4. 常量内存（Constant Memory）：只读，有缓存
        5. 纹理内存（Texture Memory）：优化的2D访问
        """
        # PyTorch自动管理，但理解有助于优化

        # 检查设备属性
        if torch.cuda.is_available():
            device = torch.device('cuda')
            props = torch.cuda.get_device_properties(0)

            print(f"全局内存: {props.total_memory / 1e9:.2f} GB")
            print(f"共享内存/块: {props.shared_memory_per_block / 1024:.2f} KB")
            print(f"寄存器/块: {props.regs_per_block}")
            print(f"计算能力: {props.major}.{props.minor}")

    def memory_coalescing(self):
        """
        内存合并访问（Memory Coalescing）
        """
        # 好的访问模式：连续访问
        x = torch.randn(1024, 1024, device='cuda')
        y = x[:, 0]  # 跨度1024，可能不合并

        # 优化：转置后访问
        x_t = x.t().contiguous()
        y_opt = x_t[:, 0]  # 现在是连续的

    def shared_memory_usage(self):
        """
        利用共享内存的矩阵乘法示例（概念）
        """
        # 标准矩阵乘法：C = A @ B
        # 优化策略：
        # 1. 分块加载到共享内存
        # 2. 重用数据减少全局内存访问
        # 3. 同步线程块内的线程

        # PyTorch的torch.mm已高度优化
        # 但理解原理有助于自定义CUDA内核
        pass
```

#### 2.2.2 异步执行与流

```python
def cuda_streams_and_async():
    """
    CUDA流与异步执行
    """
    if not torch.cuda.is_available():
        return

    device = torch.device('cuda')

    # 1. 默认流
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = x @ y  # 在默认流中执行

    # 2. 自定义流实现并行
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    with torch.cuda.stream(stream1):
        a1 = torch.randn(1000, 1000, device=device)
        b1 = a1 @ a1

    with torch.cuda.stream(stream2):
        a2 = torch.randn(1000, 1000, device=device)
        b2 = a2 @ a2

    # stream1和stream2可以并行执行

    # 3. 同步
    torch.cuda.synchronize()  # 等待所有流完成

    # 4. 事件用于测量和同步
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    result = x @ y
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    print(f"运行时间: {elapsed_time:.2f} ms")

def memory_management_advanced():
    """
    高级内存管理技巧
    """
    if not torch.cuda.is_available():
        return

    # 1. 预分配与缓存分配器
    # PyTorch使用缓存分配器避免频繁cudaMalloc
    torch.cuda.empty_cache()  # 释放缓存但不删除张量

    # 2. 查看内存使用
    print(f"已分配: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"缓存: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # 3. 内存碎片化
    # 长期运行的程序可能遇到碎片化
    # 解决：定期清空缓存或使用统一大小的张量

    # 4. 内存池
    # 自定义内存池用于细粒度控制
    # 参考：torch.cuda.memory.CUDAPluggableAllocator
```

### 2.3 计算图与自动微分

#### 2.3.1 动态计算图原理

```python
class ComputationGraphInternals:
    """
    PyTorch计算图的内部实现
    """

    def graph_construction(self):
        """
        计算图的构建过程
        """
        # 1. 每个Tensor有grad_fn属性
        x = torch.tensor([1., 2., 3.], requires_grad=True)
        print(f"叶节点的grad_fn: {x.grad_fn}")  # None（叶节点）

        y = x ** 2
        print(f"y的grad_fn: {y.grad_fn}")  # PowBackward

        z = y.sum()
        print(f"z的grad_fn: {z.grad_fn}")  # SumBackward

        # 2. grad_fn形成链式结构
        # z → SumBackward → PowBackward → AccumulateGrad

        # 3. 保存用于反向传播的中间值
        # saved_tensors属性

    def autograd_function_custom(self):
        """
        自定义autograd函数
        """
        class MyExp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # ctx用于保存反向传播需要的值
                result = torch.exp(x)
                ctx.save_for_backward(result)
                return result

            @staticmethod
            def backward(ctx, grad_output):
                # grad_output是L对输出的梯度
                # 返回L对输入的梯度
                result, = ctx.saved_tensors
                return grad_output * result  # d(exp(x))/dx = exp(x)

        # 使用
        x = torch.tensor([1., 2., 3.], requires_grad=True)
        y = MyExp.apply(x)
        loss = y.sum()
        loss.backward()
        print(f"梯度: {x.grad}")

    def higher_order_gradients(self):
        """
        高阶梯度计算
        """
        x = torch.tensor([1., 2., 3.], requires_grad=True)
        y = x ** 3  # y = x³

        # 一阶导数: dy/dx = 3x²
        grad_y = torch.autograd.grad(y, x, torch.ones_like(y),
                                      create_graph=True)[0]
        print(f"一阶导数: {grad_y}")

        # 二阶导数: d²y/dx² = 6x
        grad2_y = torch.autograd.grad(grad_y, x, torch.ones_like(grad_y))[0]
        print(f"二阶导数: {grad2_y}")
```

#### 2.3.2 反向传播算法详解

**数学原理**
```
链式法则（Chain Rule）：
∂L/∂x = (∂L/∂y) · (∂y/∂x)

向量链式法则：
∂L/∂x = Σᵢ (∂L/∂yᵢ) · (∂yᵢ/∂x)

矩阵情况：
∂L/∂X = (∂L/∂Y) · (∂Y/∂X)
```

```python
def backpropagation_from_scratch():
    """
    从零实现反向传播，理解原理
    """
    class MicroGrad:
        """
        简化版自动微分引擎
        """
        def __init__(self, data, _children=(), _op=''):
            self.data = data
            self.grad = 0.0
            self._backward = lambda: None
            self._prev = set(_children)
            self._op = _op

        def __add__(self, other):
            other = other if isinstance(other, MicroGrad) else MicroGrad(other)
            out = MicroGrad(self.data + other.data, (self, other), '+')

            def _backward():
                # 加法的导数分配规则
                self.grad += out.grad
                other.grad += out.grad
            out._backward = _backward

            return out

        def __mul__(self, other):
            other = other if isinstance(other, MicroGrad) else MicroGrad(other)
            out = MicroGrad(self.data * other.data, (self, other), '*')

            def _backward():
                # 乘法的导数规则：d(xy)/dx = y, d(xy)/dy = x
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
            out._backward = _backward

            return out

        def relu(self):
            out = MicroGrad(max(0, self.data), (self,), 'ReLU')

            def _backward():
                self.grad += (out.data > 0) * out.grad
            out._backward = _backward

            return out

        def backward(self):
            # 拓扑排序
            topo = []
            visited = set()

            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        build_topo(child)
                    topo.append(v)

            build_topo(self)

            # 反向传播
            self.grad = 1.0
            for node in reversed(topo):
                node._backward()

    # 测试
    x = MicroGrad(2.0)
    y = MicroGrad(3.0)
    z = x * y + x
    z.backward()
    print(f"∂z/∂x = {x.grad}")  # 应该是 y + 1 = 4.0
    print(f"∂z/∂y = {y.grad}")  # 应该是 x = 2.0

def gradient_checkpointing():
    """
    梯度检查点：以时间换空间
    """
    # 原理：不保存所有中间激活值
    # 反向传播时重新计算

    class CheckpointedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(100, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, 10)
            )

        def forward(self, x):
            # 使用checkpoint包装
            from torch.utils.checkpoint import checkpoint

            # 将layers分段，每段使用checkpoint
            x = checkpoint(lambda x: self.layers[:2](x), x)
            x = checkpoint(lambda x: self.layers[2:4](x), x)
            x = self.layers[4](x)
            return x

    # 内存使用：O(√n) 而非 O(n)
    # 时间开销：增加~30%
```

---

## 第三部分：数值稳定性与数学技巧

### 3.1 数值稳定性问题

#### 3.1.1 指数函数的数值稳定实现

```python
def numerical_stability_techniques():
    """
    数值稳定性技巧
    """

    # 1. Softmax的稳定实现
    def softmax_naive(x):
        """不稳定的实现"""
        exp_x = torch.exp(x)
        return exp_x / exp_x.sum(dim=-1, keepdim=True)

    def softmax_stable(x):
        """
        稳定实现：减去最大值
        softmax(x) = softmax(x - c)对任意常数c成立
        """
        x_max = x.max(dim=-1, keepdim=True)[0]
        exp_x = torch.exp(x - x_max)
        return exp_x / exp_x.sum(dim=-1, keepdim=True)

    # 测试
    x = torch.tensor([1000., 1001., 1002.])
    print(f"不稳定: {softmax_naive(x)}")      # 可能inf/nan
    print(f"稳定: {softmax_stable(x)}")        # 正常
    print(f"PyTorch: {torch.softmax(x, 0)}")  # 使用稳定实现

    # 2. Log-Softmax的稳定实现
    def log_softmax_stable(x):
        """
        log(softmax(x)) = x - log(Σexp(x))
        稳定版本：x - max(x) - log(Σexp(x - max(x)))
        """
        x_max = x.max(dim=-1, keepdim=True)[0]
        return x - x_max - torch.log(torch.exp(x - x_max).sum(dim=-1, keepdim=True))

    # 3. LogSumExp技巧
    def logsumexp_stable(x):
        """
        log(Σexp(xᵢ)) = max(x) + log(Σexp(xᵢ - max(x)))
        """
        x_max = x.max()
        return x_max + torch.log(torch.exp(x - x_max).sum())

    # 4. 数值范围问题
    print(f"FP32最大值: {torch.finfo(torch.float32).max}")  # ~3.4e38
    print(f"exp(100) = {torch.exp(torch.tensor(100.))}")    # 溢出风险
```

#### 3.1.2 梯度消失与爆炸

```python
def gradient_flow_analysis():
    """
    梯度流动分析
    """

    # 1. 梯度消失示例
    def demonstrate_vanishing_gradients():
        x = torch.randn(1, 10, requires_grad=True)

        # 深层Sigmoid网络
        y = x
        for _ in range(100):
            y = torch.sigmoid(y @ torch.randn(10, 10))

        loss = y.sum()
        loss.backward()

        print(f"输入梯度范数: {x.grad.norm().item()}")
        # 梯度接近0：消失

    # 2. 梯度爆炸示例
    def demonstrate_exploding_gradients():
        x = torch.randn(1, 10, requires_grad=True)

        # 大权重矩阵
        W = torch.randn(10, 10) * 2  # 特征值>1
        y = x
        for _ in range(20):
            y = y @ W

        loss = y.sum()
        loss.backward()

        print(f"输入梯度范数: {x.grad.norm().item()}")
        # 梯度爆炸

    # 3. 梯度裁剪（Gradient Clipping）
    def gradient_clipping(parameters, max_norm):
        """
        梯度裁剪：防止梯度爆炸
        """
        # 计算总梯度范数
        total_norm = torch.sqrt(sum(
            p.grad.data.norm() ** 2 for p in parameters
        ))

        # 裁剪系数
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.data.mul_(clip_coef)

        return total_norm

    # PyTorch内置
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 3.2 初始化策略

```python
class InitializationStrategies:
    """
    权重初始化的数学原理
    """

    @staticmethod
    def xavier_initialization():
        """
        Xavier初始化（Glorot初始化）

        目标：保持方差在层间传播
        假设：激活函数线性（如tanh在0附近）

        W ~ U(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))
        或 W ~ N(0, 2/(n_in + n_out))
        """
        n_in, n_out = 100, 50

        # 均匀分布版本
        limit = torch.sqrt(torch.tensor(6.0 / (n_in + n_out)))
        W = torch.empty(n_out, n_in).uniform_(-limit, limit)

        # 正态分布版本
        std = torch.sqrt(torch.tensor(2.0 / (n_in + n_out)))
        W = torch.randn(n_out, n_in) * std

        # PyTorch内置
        layer = torch.nn.Linear(n_in, n_out)
        torch.nn.init.xavier_uniform_(layer.weight)

    @staticmethod
    def kaiming_initialization():
        """
        Kaiming初始化（He初始化）

        专为ReLU设计
        考虑ReLU会将一半输出置零

        W ~ N(0, 2/n_in)  # fan_in模式
        或 W ~ N(0, 2/n_out)  # fan_out模式
        """
        n_in, n_out = 100, 50

        # fan_in模式（前向传播）
        std = torch.sqrt(torch.tensor(2.0 / n_in))
        W = torch.randn(n_out, n_in) * std

        # PyTorch内置
        layer = torch.nn.Linear(n_in, n_out)
        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in',
                                       nonlinearity='relu')

    @staticmethod
    def orthogonal_initialization():
        """
        正交初始化

        好处：
        - 保持向量长度（等距变换）
        - 梯度范数保持为1
        - 适用于RNN
        """
        n = 100
        W = torch.randn(n, n)

        # QR分解得到正交矩阵
        Q, R = torch.linalg.qr(W)
        # 调整符号使对角线为正
        W_ortho = Q * torch.sign(torch.diag(R))

        # PyTorch内置
        torch.nn.init.orthogonal_(W)
```

---

## 第四部分：性能分析与优化

### 4.1 性能分析工具

```python
def performance_profiling():
    """
    性能分析与Profiling
    """

    # 1. 基础计时
    import time

    start = time.time()
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = x @ y
    torch.cuda.synchronize()  # 重要！CUDA是异步的
    end = time.time()
    print(f"运行时间: {(end - start) * 1000:.2f} ms")

    # 2. PyTorch Profiler
    from torch.profiler import profile, ProfilerActivity

    def model_forward(x):
        return torch.nn.functional.relu(x @ torch.randn(1000, 1000, device='cuda'))

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        x = torch.randn(1000, 1000, device='cuda')
        for _ in range(10):
            y = model_forward(x)

    # 打印统计信息
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # 3. 导出Chrome trace
    prof.export_chrome_trace("trace.json")

    # 4. 基准测试
    from torch.utils.benchmark import Timer

    timer = Timer(
        stmt='x @ y',
        setup='x = torch.randn(1000, 1000, device="cuda"); y = torch.randn(1000, 1000, device="cuda")',
        globals={}
    )

    result = timer.blocked_autorange()
    print(result)
```

### 4.2 算子融合与JIT编译

```python
def operator_fusion_and_jit():
    """
    算子融合与JIT编译优化
    """

    # 1. TorchScript JIT编译
    @torch.jit.script
    def fused_gelu(x):
        """
        融合的GELU激活函数
        GELU(x) = x * Φ(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
        """
        return 0.5 * x * (1.0 + torch.tanh(
            0.7978845608 * (x + 0.044715 * x ** 3)
        ))

    # 编译后的版本更快
    x = torch.randn(1000, 1000, device='cuda')

    # 未编译
    def gelu_python(x):
        return 0.5 * x * (1.0 + torch.tanh(
            0.7978845608 * (x + 0.044715 * x ** 3)
        ))

    # 2. 自定义CUDA融合内核
    # 使用torch.cuda.jit或自定义C++扩展

    # 3. torch.compile (PyTorch 2.0+)
    @torch.compile
    def optimized_function(x, y):
        a = torch.sin(x)
        b = torch.cos(y)
        return a + b

    # torch.compile会：
    # - 分析计算图
    # - 融合算子
    # - 生成优化的内核
```

---

## 总结与路线图

### 学习检查清单

**数学基础**
- [ ] 理解张量的数学定义
- [ ] 掌握矩阵分解（SVD, EVD, Cholesky）
- [ ] 熟悉梯度、Jacobian、Hessian的计算
- [ ] 理解凸优化基础理论
- [ ] 掌握概率分布与采样方法
- [ ] 理解信息论基本概念（熵、KL散度）

**PyTorch核心**
- [ ] 理解张量存储机制（Storage, View, Stride）
- [ ] 掌握数据类型与混合精度训练
- [ ] 熟悉CUDA内存层级与优化
- [ ] 理解计算图与自动微分原理
- [ ] 能实现自定义autograd函数
- [ ] 掌握梯度检查点技术

**数值计算**
- [ ] 掌握数值稳定性技巧
- [ ] 理解梯度消失/爆炸问题
- [ ] 熟悉各种初始化策略
- [ ] 能进行性能分析与优化

### 下一步学习方向

1. **PyTorch深度教程（二）**：张量运算与自动微分深度解析
2. **PyTorch深度教程（三）**：神经网络架构设计
3. **PyTorch深度教程（四）**：优化器与训练技巧
4. **PyTorch深度教程（五）**：分布式训练与性能优化

---

## 参考资源

### 数学基础
- *Deep Learning* by Goodfellow, Bengio, Courville
- *Matrix Computations* by Golub and Van Loan
- *Convex Optimization* by Boyd and Vandenberghe

### PyTorch源码
- GitHub: pytorch/pytorch
- 重点阅读：
  - `torch/csrc/autograd/` - 自动微分
  - `aten/src/ATen/` - 张量库
  - `torch/nn/` - 神经网络模块

### 论文
- *Automatic Differentiation in Machine Learning: a Survey*
- *Mixed Precision Training*
- *Large Batch Training of Convolutional Networks*
