# Python 工程化实践指南 - Conda & UV 环境管理

## 目录
- [核心概念](#核心概念)
- [Conda 环境管理](#conda-环境管理)
- [UV 环境管理](#uv-环境管理)
- [Conda vs UV 对比](#conda-vs-uv-对比)
- [最佳实践与工作流](#最佳实践与工作流)

---

## 核心概念

### 什么是环境管理

Python 环境管理是指创建隔离的 Python 运行环境，每个环境拥有独立的：
- Python 解释器版本
- 安装的包及其版本
- 环境变量配置

**为什么需要环境管理？**
- **隔离依赖**：不同项目依赖不同版本的包
- **避免冲突**：防止全局包污染
- **可复现性**：确保团队成员环境一致
- **安全性**：隔离测试新包，不影响其他项目

### Conda vs UV 简介

**Conda：**
- 跨语言包管理系统（不仅限于 Python）
- 科学计算领域的事实标准
- 管理 Python 解释器和系统级依赖
- 生态成熟，包仓库丰富（conda-forge）

**UV：**
- 极速的 Python 包管理工具（Rust 编写）
- 替代 pip、pip-tools、virtualenv、poetry
- 性能强大：10-100x 快于传统工具
- 现代化设计，符合 Python 打包标准

---

## Conda 环境管理

### 1. Conda 安装

**Miniconda（推荐，轻量级）：**
```bash
# Linux/macOS
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Windows - 下载安装器
# https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe

# 验证安装
conda --version
```

**Anaconda（完整版，包含大量科学计算包）：**
```bash
# 下载地址
# https://www.anaconda.com/download

# 安装后验证
conda --version
```

**Mamba（Conda 的高性能替代品）：**
```bash
# 安装 mamba（推荐）
conda install -n base -c conda-forge mamba

# 使用 mamba 替代 conda（更快）
mamba install package_name
```

### 2. Conda 环境创建与管理

#### 基础操作

```bash
# 创建环境（默认 Python 版本）
conda create -n myenv

# 创建环境并指定 Python 版本
conda create -n myenv python=3.11

# 创建环境并安装包
conda create -n myenv python=3.11 numpy pandas matplotlib

# 激活环境
conda activate myenv

# 退出当前环境
conda deactivate

# 列出所有环境
conda env list
conda info --envs

# 删除环境
conda remove -n myenv --all

# 克隆环境
conda create -n myenv_backup --clone myenv
```

#### 包管理

```bash
# 安装包
conda install numpy

# 安装指定版本
conda install numpy=1.24.0

# 安装多个包
conda install numpy pandas scikit-learn

# 从 conda-forge 安装
conda install -c conda-forge package_name

# 更新包
conda update numpy

# 更新所有包
conda update --all

# 卸载包
conda remove numpy

# 列出已安装的包
conda list

# 搜索包
conda search package_name

# 查看包信息
conda info package_name
```

### 3. 环境配置文件

#### environment.yml（推荐）

```yaml
# environment.yml - Conda 环境配置文件
name: my_project
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy=1.24.*
  - pandas=2.0.*
  - scikit-learn>=1.3.0
  - matplotlib
  - jupyter
  - pytest
  # pip 安装的包
  - pip
  - pip:
      - requests==2.31.0
      - python-dotenv
      - custom-package @ git+https://github.com/user/repo.git

# 可选：指定构建字符串
# - numpy=1.24.0=py311h1234567_0
```

**使用配置文件：**

```bash
# 从配置文件创建环境
conda env create -f environment.yml

# 更新现有环境
conda env update -f environment.yml --prune

# 导出当前环境
conda env export > environment.yml

# 导出跨平台环境（不包含构建信息）
conda env export --no-builds > environment.yml

# 只导出手动安装的包（推荐）
conda env export --from-history > environment.yml
```

#### requirements.txt（兼容 pip）

```bash
# 导出为 requirements.txt
conda list --export > requirements.txt

# 从 requirements.txt 安装（使用 pip）
pip install -r requirements.txt
```

### 4. Conda 高级配置

#### .condarc 配置文件

```yaml
# ~/.condarc - Conda 全局配置
channels:
  - conda-forge
  - defaults

channel_priority: strict  # 严格优先级

show_channel_urls: true  # 显示包的来源

auto_activate_base: false  # 不自动激活 base 环境

# 添加镜像源（国内用户加速）
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free

custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge

# 环境目录
envs_dirs:
  - ~/conda/envs
  - ~/.conda/envs

# 包缓存目录
pkgs_dirs:
  - ~/conda/pkgs
  - ~/.conda/pkgs

# SSL 验证
ssl_verify: true

# 安装时显示进度条
show_progress: true
```

**配置命令：**

```bash
# 查看配置
conda config --show

# 添加频道
conda config --add channels conda-forge

# 设置频道优先级
conda config --set channel_priority strict

# 禁用自动激活 base
conda config --set auto_activate_base false

# 设置镜像源
conda config --add default_channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
```

### 5. Conda 项目实战工作流

#### 科学计算项目示例

```bash
# 1. 创建项目目录
mkdir ml_project && cd ml_project

# 2. 创建环境配置文件
cat > environment.yml << EOF
name: ml_project
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy=1.24.*
  - pandas=2.0.*
  - scikit-learn=1.3.*
  - matplotlib=3.7.*
  - seaborn=0.12.*
  - jupyter
  - ipykernel
  - pytest=7.*
  - black
  - flake8
  - pip
  - pip:
      - mlflow
      - wandb
EOF

# 3. 创建环境
conda env create -f environment.yml

# 4. 激活环境
conda activate ml_project

# 5. 注册 Jupyter 内核
python -m ipykernel install --user --name ml_project --display-name "Python (ML Project)"

# 6. 启动 Jupyter
jupyter lab

# 7. 项目结束后导出环境
conda env export --from-history > environment.yml
```

#### 深度学习项目（GPU 支持）

```yaml
# environment.yml - 深度学习环境
name: dl_project
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pytorch=2.0.*
  - torchvision
  - torchaudio
  - pytorch-cuda=11.8  # CUDA 版本
  - cudatoolkit=11.8
  - cudnn=8.9.*
  - numpy
  - pandas
  - matplotlib
  - jupyter
  - tensorboard
  - pip
  - pip:
      - transformers
      - datasets
      - accelerate
```

```bash
# 创建深度学习环境
conda env create -f environment.yml

# 验证 GPU 可用性
conda activate dl_project
python -c "import torch; print(torch.cuda.is_available())"
```

### 6. Conda 包构建与分发

#### 创建 Conda 包

```yaml
# meta.yaml - Conda 包配置
package:
  name: my_package
  version: "0.1.0"

source:
  path: ..

build:
  number: 0
  script: python -m pip install . -vv
  noarch: python  # 纯 Python 包

requirements:
  host:
    - python >=3.9
    - pip
    - setuptools
  run:
    - python >=3.9
    - numpy >=1.20
    - pandas >=1.3

test:
  imports:
    - my_package
  commands:
    - pytest tests/

about:
  home: https://github.com/user/my_package
  license: MIT
  summary: 'My awesome package'
```

```bash
# 构建 Conda 包
conda install conda-build
conda build .

# 安装本地构建的包
conda install --use-local my_package

# 上传到 Anaconda Cloud
conda install anaconda-client
anaconda upload /path/to/my_package-0.1.0-py_0.tar.bz2
```

### 7. Conda 性能优化

```bash
# 使用 mamba 加速（强烈推荐）
conda install -n base -c conda-forge mamba
mamba install numpy pandas  # 比 conda 快 10-100 倍

# 使用 libmamba solver（Conda 22.11+）
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

# 清理缓存
conda clean --all

# 减少包缓存大小
conda config --set pkgs_dirs ~/conda/pkgs

# 并行下载
conda config --set remote_max_retries 3
conda config --set remote_backoff_factor 2
```

### 8. Conda 常见问题与解决方案

#### 环境激活问题

```bash
# 问题：conda activate 不工作
# 解决：初始化 shell
conda init bash  # 或 zsh, fish, powershell

# 重新加载 shell 配置
source ~/.bashrc  # Linux/macOS
```

#### 包冲突解决

```bash
# 问题：依赖冲突
# 解决方案1：使用 mamba（更好的依赖求解器）
mamba install conflicting_package

# 解决方案2：放宽版本约束
conda install "package>=1.0" --no-pin

# 解决方案3：创建新环境
conda create -n new_env python=3.11 package1 package2
```

#### 清理与修复

```bash
# 清理未使用的包和缓存
conda clean --all

# 修复损坏的环境
conda install --force-reinstall python

# 重建环境索引
conda index ~/conda/pkgs

# 验证环境完整性
conda list --explicit > spec-file.txt
conda create -n test_env --file spec-file.txt
```

---

## UV 环境管理

### 1. UV 安装

```bash
# Linux/macOS（推荐）
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows（PowerShell）
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 使用 pip 安装
pip install uv

# 使用 cargo 安装（Rust 用户）
cargo install --git https://github.com/astral-sh/uv uv

# 验证安装
uv --version
```

### 2. UV 环境创建与管理

#### 基础操作

```bash
# 创建虚拟环境（使用系统 Python）
uv venv

# 指定 Python 版本（UV 会自动下载）
uv venv --python 3.11

# 指定环境目录
uv venv myenv

# 激活环境
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# 退出环境
deactivate

# 删除环境
rm -rf .venv
```

#### 包管理

```bash
# 安装包（极速）
uv pip install numpy

# 安装多个包
uv pip install numpy pandas matplotlib

# 安装指定版本
uv pip install "numpy==1.24.0"

# 安装版本范围
uv pip install "numpy>=1.20,<2.0"

# 从 requirements.txt 安装
uv pip install -r requirements.txt

# 从 pyproject.toml 安装
uv pip install -e .

# 更新包
uv pip install --upgrade numpy

# 卸载包
uv pip uninstall numpy

# 列出已安装的包
uv pip list

# 显示包详情
uv pip show numpy

# 冻结依赖
uv pip freeze > requirements.txt
```

### 3. UV 项目配置

#### pyproject.toml（现代化方式）

```toml
# pyproject.toml - UV 项目配置
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-project"
version = "0.1.0"
description = "My awesome project"
authors = [
    {name = "Your Name", email = "you@example.com"}
]
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "requests>=2.31.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "black>=23.0",
    "ruff>=0.1.0",
]
docs = [
    "mkdocs>=1.5",
    "mkdocs-material>=9.0",
]

[project.scripts]
my-cli = "my_project.cli:main"

[tool.uv]
# UV 特定配置
index-url = "https://pypi.org/simple"
extra-index-url = [
    "https://download.pytorch.org/whl/cu118",
]

[tool.uv.sources]
# 从 Git 安装
my-custom-package = { git = "https://github.com/user/repo.git", tag = "v1.0.0" }
```

**使用 UV 管理项目：**

```bash
# 创建新项目
uv init my-project
cd my-project

# 安装项目依赖
uv pip install -e .

# 安装开发依赖
uv pip install -e ".[dev]"

# 安装所有可选依赖
uv pip install -e ".[dev,docs]"

# 同步依赖（确保环境与配置一致）
uv pip sync requirements.txt
```

#### requirements.txt（传统方式）

```txt
# requirements.txt
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2

# 从 Git 安装
git+https://github.com/user/repo.git@v1.0.0

# 可编辑模式（本地开发）
-e ./packages/my_local_package

# 包含其他文件
-r requirements-dev.txt
```

```txt
# requirements-dev.txt
pytest==7.4.0
black==23.7.0
ruff==0.0.285
mypy==1.5.0
```

**锁定依赖（类似 Poetry）：**

```bash
# 生成锁定文件
uv pip compile pyproject.toml -o requirements.lock

# 从锁定文件安装
uv pip sync requirements.lock

# 升级锁定文件
uv pip compile --upgrade pyproject.toml -o requirements.lock

# 只升级特定包
uv pip compile --upgrade-package numpy pyproject.toml -o requirements.lock
```

### 4. UV 高级特性

#### Python 版本管理

```bash
# UV 可以自动下载和管理 Python 版本
uv venv --python 3.11  # 自动下载 Python 3.11
uv venv --python 3.12  # 自动下载 Python 3.12

# 列出可用的 Python 版本
uv python list

# 安装特定 Python 版本
uv python install 3.11.5

# 使用本地 Python
uv venv --python /usr/bin/python3.11
```

#### 缓存管理

```bash
# UV 使用全局缓存，大幅提升性能

# 查看缓存信息
uv cache info

# 清理缓存
uv cache clean

# 清理特定包的缓存
uv cache clean numpy
```

#### 工作空间（Monorepo）

```toml
# workspace 根目录的 pyproject.toml
[tool.uv.workspace]
members = [
    "packages/*",
]

# packages/package_a/pyproject.toml
[project]
name = "package-a"
dependencies = [
    "package-b",  # 引用同一 workspace 的包
]
```

```bash
# 在 workspace 根目录安装所有包
uv pip install -e packages/package_a -e packages/package_b

# UV 会自动解析 workspace 依赖
```

### 5. UV 项目实战工作流

#### Web 应用项目

```toml
# pyproject.toml - FastAPI 项目
[project]
name = "my-api"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "pydantic>=2.4.0",
    "sqlalchemy>=2.0.0",
    "alembic>=1.12.0",
    "python-jose[cryptography]>=3.3.0",
    "passlib[bcrypt]>=1.7.4",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "httpx>=0.25.0",  # 测试 FastAPI
    "ruff>=0.1.0",
    "black>=23.0.0",
]
```

```bash
# 初始化项目
mkdir my-api && cd my-api
uv venv --python 3.11
source .venv/bin/activate

# 创建 pyproject.toml（如上）
# 安装依赖
uv pip install -e ".[dev]"

# 运行应用
uvicorn main:app --reload

# 运行测试
pytest
```

#### 数据科学项目

```bash
# 创建项目
mkdir data_analysis && cd data_analysis
uv venv --python 3.11
source .venv/bin/activate

# 快速安装数据科学栈
uv pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# 启动 Jupyter
jupyter lab

# 生成依赖文件
uv pip freeze > requirements.txt
```

#### 多环境管理

```bash
# 项目结构
my_project/
├── .venv/              # 开发环境
├── .venv-test/         # 测试环境
├── .venv-prod/         # 生产环境
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml

# 开发环境
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# 测试环境
uv venv .venv-test --python 3.11
source .venv-test/bin/activate
uv pip install -r requirements.txt -r requirements-dev.txt

# 生产环境（最小依赖）
uv venv .venv-prod --python 3.11
source .venv-prod/bin/activate
uv pip install -r requirements.txt
```

### 6. UV 与 Docker 集成

```dockerfile
# Dockerfile - 使用 UV 加速构建
FROM python:3.11-slim

# 安装 UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 复制依赖文件
COPY pyproject.toml ./

# 创建虚拟环境并安装依赖（极速）
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install -e .

# 复制应用代码
COPY . .

# 设置环境变量
ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "-m", "my_package"]
```

**多阶段构建优化：**

```dockerfile
# Dockerfile - 多阶段构建
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml ./
RUN uv venv && . .venv/bin/activate && uv pip install -e .

# 生产镜像
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY . .

ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "-m", "my_package"]
```

### 7. UV 性能对比

```bash
# 性能测试示例（安装 100 个包）

# 传统 pip
time pip install -r requirements.txt
# 真实: 2m 30s

# UV
time uv pip install -r requirements.txt
# 真实: 8s

# 性能提升：~18x 🚀
```

**UV 性能优势：**
- **并行下载**：同时下载多个包
- **全局缓存**：跨项目共享包
- **零拷贝安装**：硬链接或符号链接
- **Rust 实现**：极低开销
- **智能依赖解析**：快速求解依赖树

---

## Conda vs UV 对比

### 功能对比表

| 特性 | Conda | UV |
|------|-------|-----|
| **安装速度** | 慢（Python 实现） | 极快（Rust 实现，10-100x） |
| **Python 版本管理** | ✅ 内置 | ✅ 自动下载 |
| **系统依赖** | ✅ 可管理（C/C++ 库） | ❌ 仅 Python 包 |
| **科学计算支持** | ✅ 优秀（conda-forge） | ⚠️ 依赖 PyPI |
| **跨平台** | ✅ 优秀 | ✅ 优秀 |
| **包生态** | conda-forge（>20k 包） | PyPI（>450k 包） |
| **配置文件** | environment.yml | pyproject.toml / requirements.txt |
| **磁盘占用** | 大（每个环境独立） | 小（全局缓存） |
| **依赖解析** | 慢但准确 | 快且现代 |
| **适用场景** | 科学计算、深度学习 | Web 开发、通用 Python |

### 使用场景推荐

**优先选择 Conda：**
- 科学计算项目（NumPy、SciPy、Pandas）
- 深度学习项目（PyTorch、TensorFlow + CUDA）
- 需要非 Python 依赖（libhdf5、GDAL、OpenCV）
- 跨语言项目（Python + R + Julia）
- 团队已使用 Conda 生态

**优先选择 UV：**
- Web 应用开发（FastAPI、Django、Flask）
- 通用 Python 项目
- CI/CD 流程（追求速度）
- 容器化部署（Docker）
- 需要现代化工具链

### 混合使用策略

```bash
# 场景1：Conda 管理 Python + UV 管理包
conda create -n myenv python=3.11
conda activate myenv
pip install uv
uv pip install numpy pandas

# 场景2：Conda 管理系统依赖 + UV 管理 Python 依赖
conda create -n myenv python=3.11 cudatoolkit=11.8
conda activate myenv
uv pip install torch torchvision

# 场景3：在 Conda 环境中使用 UV 加速
conda activate myenv
uv pip install -r requirements.txt  # 比 conda install 快得多
```

---

## 最佳实践与工作流

### 1. 项目目录结构

```bash
my_project/
├── .venv/                    # UV 虚拟环境（或 Conda 环境）
├── src/
│   └── my_package/
│       ├── __init__.py
│       └── core.py
├── tests/
│   └── test_core.py
├── pyproject.toml           # 项目配置（UV 推荐）
├── environment.yml          # Conda 配置（科学计算推荐）
├── requirements.txt         # 锁定的依赖
├── .gitignore
├── README.md
└── .python-version          # 指定 Python 版本
```

### 2. .gitignore 配置

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# 虚拟环境
.venv/
venv/
ENV/
env/

# Conda
.conda/

# UV
.uv/

# 依赖锁定文件（可选）
# requirements.lock

# IDE
.vscode/
.idea/
*.swp

# 测试
.pytest_cache/
.coverage
htmlcov/

# 构建
build/
dist/
*.egg-info/
```

### 3. Conda 工作流示例

```bash
# 1. 初始化项目
mkdir my_ml_project && cd my_ml_project

# 2. 创建环境配置
cat > environment.yml << EOF
name: my_ml_project
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy
  - pandas
  - scikit-learn
  - jupyter
  - pytest
EOF

# 3. 创建环境
conda env create -f environment.yml

# 4. 激活环境
conda activate my_ml_project

# 5. 开发...

# 6. 添加新依赖
conda install matplotlib

# 7. 导出环境（团队共享）
conda env export --from-history > environment.yml

# 8. 提交到版本控制
git add environment.yml
git commit -m "Update dependencies"
```

### 4. UV 工作流示例

```bash
# 1. 初始化项目
mkdir my_web_app && cd my_web_app
uv venv --python 3.11
source .venv/bin/activate

# 2. 创建项目配置
cat > pyproject.toml << EOF
[project]
name = "my-web-app"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "ruff>=0.1.0",
]
EOF

# 3. 安装依赖
uv pip install -e ".[dev]"

# 4. 锁定依赖
uv pip freeze > requirements.txt

# 5. 开发...

# 6. 添加新依赖（修改 pyproject.toml 后）
uv pip install -e ".[dev]"

# 7. 更新锁定文件
uv pip freeze > requirements.txt

# 8. 提交到版本控制
git add pyproject.toml requirements.txt
git commit -m "Update dependencies"
```

### 5. 团队协作规范

#### Conda 项目

```bash
# 新成员加入项目
git clone https://github.com/team/project.git
cd project

# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate project_name

# 开始工作
```

#### UV 项目

```bash
# 新成员加入项目
git clone https://github.com/team/project.git
cd project

# 安装 UV（如果还没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建环境并安装依赖
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# 或使用 pyproject.toml
uv pip install -e ".[dev]"
```

### 6. CI/CD 集成

#### GitHub Actions - Conda

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
      - uses: actions/checkout@v3

      - name: Setup Miniconda
        uses: conda-incubator/setup-miniconda@v2
        with:
          python-version: ${{ matrix.python-version }}
          environment-file: environment.yml
          activate-environment: myenv
          auto-activate-base: false

      - name: Run tests
        shell: bash -l {0}
        run: |
          pytest tests/
```

#### GitHub Actions - UV

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
      - uses: actions/checkout@v3

      - name: Install UV
        run: curl -LsSf https://astral.sh/uv/install.sh | sh

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Create venv and install dependencies
        run: |
          uv venv --python ${{ matrix.python-version }}
          source .venv/bin/activate
          uv pip install -r requirements.txt

      - name: Run tests
        run: |
          source .venv/bin/activate
          pytest tests/
```

### 7. 常见问题与解决方案

#### Conda 问题

**问题1：依赖解析缓慢**
```bash
# 解决：使用 mamba
conda install -n base -c conda-forge mamba
mamba install package_name

# 或使用 libmamba solver
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

**问题2：环境体积过大**
```bash
# 只安装必需的包
conda install --no-deps package_name

# 使用 --from-history 导出
conda env export --from-history > environment.yml

# 清理缓存
conda clean --all
```

#### UV 问题

**问题1：需要系统级依赖（如 CUDA）**
```bash
# 解决：先用 Conda 安装系统依赖
conda create -n myenv python=3.11 cudatoolkit=11.8
conda activate myenv
pip install uv
uv pip install torch torchvision
```

**问题2：与现有 pip 冲突**
```bash
# UV 完全兼容 pip，可直接替换
alias pip="uv pip"

# 或在虚拟环境中使用 UV
uv venv
source .venv/bin/activate
uv pip install package_name
```

### 8. 性能优化技巧

#### Conda 优化

```bash
# 1. 使用 mamba（最重要）
mamba install package_name

# 2. 禁用不必要的频道
conda config --show channels
conda config --remove channels unnecessary_channel

# 3. 使用本地缓存
conda install --offline package_name

# 4. 预先下载包
conda install --download-only package_name
```

#### UV 优化

```bash
# 1. 使用全局缓存（默认启用）
# UV 自动跨项目共享包

# 2. 并行安装（默认启用）
# UV 自动并行下载和安装

# 3. 预编译 wheel
uv pip install --only-binary :all: package_name

# 4. 使用镜像源
uv pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple package_name
```

### 9. 迁移指南

#### 从 pip 迁移到 UV

```bash
# 1. 安装 UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 创建虚拟环境
uv venv

# 3. 激活环境
source .venv/bin/activate

# 4. 安装现有依赖
uv pip install -r requirements.txt

# 5. 替换 pip 命令
# 将所有 pip 命令替换为 uv pip
# pip install -> uv pip install
```

#### 从 Conda 迁移到 UV

```bash
# 1. 导出 Conda 环境
conda list --export > conda_packages.txt

# 2. 转换为 requirements.txt
# 手动创建 requirements.txt，只包含纯 Python 包

# 3. 使用 UV 创建环境
uv venv --python 3.11
source .venv/bin/activate

# 4. 安装依赖
uv pip install -r requirements.txt

# 注意：如果有系统级依赖（CUDA、MKL），仍需使用 Conda
```

---

## 总结

### 选择建议

**使用 Conda 的场景：**
- ✅ 科学计算和数据分析
- ✅ 深度学习（需要 CUDA）
- ✅ 需要非 Python 依赖（C/C++ 库）
- ✅ 团队已标准化使用 Conda

**使用 UV 的场景：**
- ✅ Web 应用开发
- ✅ 通用 Python 项目
- ✅ 追求极致性能和速度
- ✅ 容器化部署（Docker）
- ✅ 现代化 Python 开发工作流

### 核心要点

1. **Conda**：成熟稳定，科学计算首选，管理系统依赖
2. **UV**：极速现代，通用开发首选，完美兼容 pip
3. **混合使用**：Conda 管理底层（Python + CUDA），UV 管理上层包
4. **版本控制**：始终提交 `environment.yml` 或 `requirements.txt`
5. **性能优化**：Conda 用 mamba，UV 默认已是最快

### 推荐工具链

**科学计算项目：**
```
Conda (mamba) + Jupyter + conda-forge
```

**Web 开发项目：**
```
UV + pyproject.toml + Docker
```

**混合项目：**
```
Conda (Python + CUDA) + UV (Python 包)
```

---

**最后更新：2025-12-24**
