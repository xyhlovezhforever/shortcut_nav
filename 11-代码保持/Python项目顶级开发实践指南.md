# Python项目顶级开发实践指南

> 十年Python开发经验总结 - 真正让代码质量提升一个档次的实战方案

---

## 目录

1. [从零开始搭建项目](#1-从零开始搭建项目)
2. [架构设计的真相](#2-架构设计的真相)
3. [写出让人惊叹的代码](#3-写出让人惊叹的代码)
4. [测试的正确姿势](#4-测试的正确姿势)
5. [性能优化实战](#5-性能优化实战)
6. [踩过的坑与解决方案](#6-踩过的坑与解决方案)
7. [团队协作最佳实践](#7-团队协作最佳实践)
8. [生产环境部署经验](#8-生产环境部署经验)

---

## 1. 从零开始搭建项目

### 1.1 不要一上来就写代码

**问题**: 很多人拿到需求就开始写 `app.py`，结果三个月后代码变成屎山。

**正解**: 先花2小时做这些事：

```bash
# 1. 创建项目结构（5分钟）
cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage

# 或者手动创建一个清晰的结构
mkdir -p myproject/{src/myproject,tests,docs,scripts}
```

```
myproject/
├── .github/workflows/      # CI/CD配置
├── src/myproject/          # 源代码（注意用src布局！）
│   ├── __init__.py
│   ├── api/               # API层
│   ├── domain/            # 业务逻辑
│   ├── infra/             # 基础设施
│   └── utils/             # 工具函数
├── tests/                 # 测试代码
├── scripts/               # 脚本工具
├── pyproject.toml         # 项目配置
├── .env.example           # 环境变量示例
└── README.md
```

**为什么用 src 布局？**

```python
# ❌ 没有src，导入会有问题
# 项目根目录会被加入PYTHONPATH，可能导入到未安装的代码
from myproject import something  # 可能导入到本地文件

# ✅ 有src，强制使用安装后的版本
from myproject import something  # 只能导入已安装的包
```

### 1.2 pyproject.toml 的正确配置

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "myproject"
version = "0.1.0"
description = "A real-world Python project"
authors = [{name = "Your Name", email = "you@example.com"}]
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.109.0",
    "sqlalchemy[asyncio]>=2.0",
    "pydantic>=2.5",
    "pydantic-settings>=2.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "pytest-asyncio>=0.23",
    "ruff>=0.1",
    "mypy>=1.8",
    "pre-commit>=3.6",
]

[project.scripts]
myproject = "myproject.cli:main"

# Ruff配置 - 比Black+Flake8+isort更快
[tool.ruff]
target-version = "py311"
line-length = 100
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
]

# MyPy配置
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

# Pytest配置
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --cov=src/myproject --cov-report=term-missing"
```

### 1.3 第一个commit就要配置的东西

```bash
# 1. 初始化git
git init
echo "__pycache__/" > .gitignore
echo ".env" >> .gitignore
echo ".venv/" >> .gitignore
echo "*.pyc" >> .gitignore

# 2. 配置pre-commit（强制代码质量）
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic, sqlalchemy]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=500']
EOF

pre-commit install
```

---

## 2. 架构设计的真相

### 2.1 不要过度设计

**错误示范**：一个简单的TODO应用搞出10个层

```python
# ❌ 过度设计
class TodoRepositoryInterface(ABC):
    @abstractmethod
    async def get_todo(self, id: int) -> Todo: ...

class TodoRepositoryImplementation(TodoRepositoryInterface):
    async def get_todo(self, id: int) -> Todo: ...

class TodoServiceInterface(ABC):
    @abstractmethod
    async def get_todo(self, id: int) -> TodoDTO: ...

class TodoService(TodoServiceInterface):
    def __init__(self, repo: TodoRepositoryInterface):
        self.repo = repo
    async def get_todo(self, id: int) -> TodoDTO: ...

# 只是为了查询一条数据...写了100行代码
```

**正确做法**：根据项目规模选择架构

```python
# ✅ 小项目（<5个表）：直接写就行
from fastapi import FastAPI
from sqlalchemy import select
from .models import Todo

app = FastAPI()

@app.get("/todos/{todo_id}")
async def get_todo(todo_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Todo).where(Todo.id == todo_id))
    return result.scalar_one_or_none()


# ✅ 中型项目（5-20个表）：分层但不抽象
# api/ 处理HTTP
# services/ 处理业务逻辑
# repositories/ 处理数据访问

# api/todos.py
@router.get("/todos/{todo_id}")
async def get_todo(todo_id: int, db: AsyncSession = Depends(get_db)):
    todo = await TodoService(db).get_todo(todo_id)
    return TodoResponse.from_orm(todo)

# services/todos.py
class TodoService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_todo(self, todo_id: int) -> Todo:
        result = await self.db.execute(
            select(Todo).where(Todo.id == todo_id)
        )
        return result.scalar_one_or_none()


# ✅ 大型项目（>20个表）：DDD + 清晰边界
# domain/ 领域模型（纯业务逻辑）
# application/ 应用服务（用例编排）
# infrastructure/ 基础设施（数据库、缓存等）
# api/ API接口
```

### 2.2 依赖注入的现实做法

**问题**：很多人照搬Java那套，结果Python反而更复杂了。

```python
# ❌ 过度使用DI框架
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    database = providers.Singleton(Database, config.db.url)
    user_repository = providers.Factory(UserRepository, database)
    user_service = providers.Factory(UserService, user_repository)
    # ...配置地狱

# ✅ FastAPI的Depends就够用了
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

async def get_db() -> AsyncSession:
    async with async_session_maker() as session:
        yield session

@app.post("/users")
async def create_user(
    data: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    user = User(**data.model_dump())
    db.add(user)
    await db.commit()
    return user


# ✅ 如果真的需要复杂DI，用Protocol + 工厂函数
from typing import Protocol

class UserRepository(Protocol):
    async def get(self, id: int) -> User: ...
    async def save(self, user: User) -> None: ...

# 实现
class PostgresUserRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get(self, id: int) -> User:
        result = await self.db.execute(select(User).where(User.id == id))
        return result.scalar_one()

# 工厂
def get_user_repository(db: AsyncSession = Depends(get_db)) -> UserRepository:
    return PostgresUserRepository(db)

# 使用
@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
    repo: UserRepository = Depends(get_user_repository),
):
    return await repo.get(user_id)
```

### 2.3 数据库模型的实战经验

```python
# ❌ 常见错误：把ORM模型当万能对象
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True)
    hashed_password = Column(String)

    # ❌ 在模型里写业务逻辑
    def send_welcome_email(self):
        send_email(self.email, "Welcome!")

    # ❌ 在模型里验证
    def validate_email(self):
        if "@" not in self.email:
            raise ValueError("Invalid email")


# ✅ 正确：分离关注点
# models/user.py - 只管数据库映射
class UserModel(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(default=func.now())

    # 只在这里定义关系
    posts: Mapped[List["PostModel"]] = relationship(back_populates="author")


# domain/user.py - 业务逻辑
from dataclasses import dataclass
from datetime import datetime

@dataclass
class User:
    id: int
    email: str
    created_at: datetime

    def is_email_verified(self) -> bool:
        # 业务逻辑
        return True

    @staticmethod
    def validate_email(email: str) -> None:
        if "@" not in email:
            raise ValueError("Invalid email")


# schemas/user.py - API输入输出
from pydantic import BaseModel, EmailStr, Field

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)

class UserResponse(BaseModel):
    id: int
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}
```

---

## 3. 写出让人惊叹的代码

### 3.1 函数设计的黄金法则

```python
# ❌ 烂代码的典型特征
def process_order(order_id, user_id, items, coupon_code=None,
                  use_points=False, gift_wrap=False, message=None,
                  shipping_method="standard", insurance=False):
    # 100行代码...
    pass


# ✅ 改进：参数对象化
from dataclasses import dataclass
from typing import Optional

@dataclass
class OrderRequest:
    order_id: int
    user_id: int
    items: list[OrderItem]
    coupon_code: Optional[str] = None
    use_points: bool = False
    gift_options: Optional[GiftOptions] = None
    shipping: ShippingOptions = ShippingOptions.STANDARD

def process_order(request: OrderRequest) -> Order:
    # 清晰多了
    pass


# ❌ 函数做太多事
def create_user_and_send_email_and_log(email: str, password: str):
    # 创建用户
    user = User(email=email, password=hash_password(password))
    db.add(user)
    db.commit()

    # 发邮件
    send_email(email, "Welcome!")

    # 记录日志
    logger.info(f"User created: {email}")

    # 还要通知Slack
    notify_slack(f"New user: {email}")

    return user


# ✅ 单一职责 + 组合
async def create_user(email: str, password: str, db: AsyncSession) -> User:
    """只负责创建用户"""
    user = User(email=email, hashed_password=hash_password(password))
    db.add(user)
    await db.commit()
    return user

async def handle_user_registration(
    email: str,
    password: str,
    db: AsyncSession,
    background_tasks: BackgroundTasks,
) -> User:
    """编排注册流程"""
    user = await create_user(email, password, db)

    # 异步执行后续任务
    background_tasks.add_task(send_welcome_email, user.email)
    background_tasks.add_task(log_user_creation, user.id)
    background_tasks.add_task(notify_slack_new_user, user.email)

    return user
```

### 3.2 异常处理的实战技巧

```python
# ❌ 吞掉异常
try:
    result = dangerous_operation()
except Exception:
    pass  # 🔥 地狱之门


# ❌ 捕获太宽泛
try:
    user = get_user(user_id)
    send_email(user.email)
except Exception as e:
    # 数据库错误？邮件服务错误？网络错误？
    logger.error(f"Something went wrong: {e}")


# ✅ 精确捕获 + 转换为领域异常
from sqlalchemy.exc import NoResultFound
from myproject.exceptions import UserNotFound, EmailServiceError

try:
    user = await get_user(user_id)
except NoResultFound:
    raise UserNotFound(f"User {user_id} not found") from None

try:
    await send_email(user.email, "Hello")
except SMTPException as e:
    raise EmailServiceError(f"Failed to send email to {user.email}") from e


# ✅ 使用上下文管理器自动清理
from contextlib import asynccontextmanager

@asynccontextmanager
async def database_transaction(db: AsyncSession):
    """自动处理事务"""
    try:
        yield db
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()

async def create_order(data: OrderCreate):
    async with database_transaction(db) as session:
        order = Order(**data.dict())
        session.add(order)
        # commit自动执行
    return order


# ✅ 重试机制
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
)
async def call_external_api(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=5.0)
        response.raise_for_status()
        return response.json()
```

### 3.3 类型注解的正确姿势

```python
# ❌ 不写类型
def process_data(data):
    return data.get("items")


# ❌ 类型太模糊
def process_data(data: dict) -> list:
    return data.get("items")


# ✅ 精确的类型
from typing import TypedDict, NotRequired

class OrderData(TypedDict):
    order_id: int
    items: list[dict[str, int]]
    total: float
    discount: NotRequired[float]  # Python 3.11+

def process_data(data: OrderData) -> list[dict[str, int]]:
    return data["items"]


# ✅ 使用NewType增加类型安全
from typing import NewType

UserId = NewType('UserId', int)
OrderId = NewType('OrderId', int)

def get_user(user_id: UserId) -> User: ...
def get_order(order_id: OrderId) -> Order: ...

# 类型检查会报错
user_id = UserId(123)
get_order(user_id)  # ❌ mypy error: Expected OrderId, got UserId


# ✅ 泛型的实际应用
from typing import Generic, TypeVar

T = TypeVar('T')

class Repository(Generic[T]):
    def __init__(self, model_class: type[T]):
        self.model_class = model_class

    async def get(self, id: int) -> T | None:
        result = await db.execute(
            select(self.model_class).where(self.model_class.id == id)
        )
        return result.scalar_one_or_none()

# 使用
user_repo = Repository[User](User)
user = await user_repo.get(123)  # mypy知道user是User类型
```

### 3.4 避免常见陷阱

```python
# ❌ 可变默认参数
def add_item(item: str, items: list = []):
    items.append(item)
    return items

# 坑：默认参数只创建一次！
add_item("a")  # ["a"]
add_item("b")  # ["a", "b"] ❌


# ✅ 使用None + 内部创建
def add_item(item: str, items: list | None = None) -> list:
    if items is None:
        items = []
    items.append(item)
    return items


# ❌ 循环中的闭包陷阱
functions = []
for i in range(5):
    functions.append(lambda: i)

print([f() for f in functions])  # [4, 4, 4, 4, 4] ❌


# ✅ 立即绑定
functions = []
for i in range(5):
    functions.append(lambda i=i: i)

print([f() for f in functions])  # [0, 1, 2, 3, 4] ✅


# ❌ 字符串拼接SQL（SQL注入）
user_id = request.query_params["user_id"]
query = f"SELECT * FROM users WHERE id = {user_id}"  # 💀


# ✅ 使用参数化查询
stmt = select(User).where(User.id == user_id)


# ❌ 裸露的secrets
DATABASE_URL = "postgresql://admin:password123@localhost/mydb"


# ✅ 使用环境变量 + pydantic-settings
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str
    secret_key: str
    redis_url: str

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
```

---

## 4. 测试的正确姿势

### 4.1 不要写无用的测试

```python
# ❌ 测试框架本身
def test_list_append():
    items = []
    items.append(1)
    assert items == [1]  # 你在测试Python的list实现？


# ❌ 测试getter/setter
def test_user_email():
    user = User()
    user.email = "test@example.com"
    assert user.email == "test@example.com"  # 无意义


# ✅ 测试业务逻辑
def test_user_cannot_delete_others_post():
    user1 = User(id=1)
    user2 = User(id=2)
    post = Post(id=1, author_id=1)

    with pytest.raises(PermissionDenied):
        post.delete_by(user2)


# ✅ 测试边界条件
def test_order_total_calculation():
    order = Order()
    order.add_item(Item(price=10.00, quantity=2))
    order.add_item(Item(price=5.50, quantity=1))
    order.apply_discount(Decimal("0.1"))  # 10% off

    assert order.total == Decimal("22.95")  # (20 + 5.5) * 0.9
```

### 4.2 Fixtures的高级用法

```python
# conftest.py
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def engine():
    """创建测试数据库引擎"""
    engine = create_async_engine(
        "postgresql+asyncpg://test:test@localhost/test_db",
        echo=True,
    )

    # 创建所有表
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # 清理
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

@pytest.fixture
async def db(engine) -> AsyncSession:
    """每个测试一个独立的session"""
    async with AsyncSession(engine) as session:
        yield session
        await session.rollback()  # 测试后回滚

@pytest.fixture
async def user(db: AsyncSession) -> User:
    """创建测试用户"""
    user = User(email="test@example.com", hashed_password="fake_hash")
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


# test_orders.py
async def test_create_order(db: AsyncSession, user: User):
    order = Order(user_id=user.id, total=100.0)
    db.add(order)
    await db.commit()

    result = await db.execute(select(Order).where(Order.user_id == user.id))
    saved_order = result.scalar_one()

    assert saved_order.total == 100.0
```

### 4.3 Mock的实战技巧

```python
# ❌ 过度mock
@pytest.mark.asyncio
async def test_create_user():
    mock_db = Mock()
    mock_db.add = Mock()
    mock_db.commit = AsyncMock()

    user_service = UserService(mock_db)
    await user_service.create_user("test@example.com")

    mock_db.add.assert_called_once()
    # 你只是在测试mock框架


# ✅ 只mock外部依赖
import httpx
from unittest.mock import patch

@pytest.mark.asyncio
async def test_fetch_user_data_from_external_api():
    mock_response = {
        "id": 123,
        "name": "John",
        "email": "john@example.com"
    }

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.status_code = 200

        service = ExternalAPIService()
        user_data = await service.fetch_user(123)

        assert user_data["email"] == "john@example.com"


# ✅ 使用pytest-httpx (更好的HTTP mock)
from pytest_httpx import HTTPXMock

async def test_external_api(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.example.com/users/123",
        json={"id": 123, "name": "John"},
    )

    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/users/123")
        assert response.json()["name"] == "John"
```

---

## 5. 性能优化实战

### 5.1 数据库查询优化

```python
# ❌ N+1查询问题
users = await db.execute(select(User).limit(100))
for user in users.scalars():
    # 每个用户都查询一次！
    orders = await db.execute(
        select(Order).where(Order.user_id == user.id)
    )
    print(f"{user.name}: {len(orders.scalars().all())} orders")


# ✅ 使用joinedload
from sqlalchemy.orm import selectinload

stmt = (
    select(User)
    .options(selectinload(User.orders))
    .limit(100)
)
users = await db.execute(stmt)
for user in users.scalars():
    print(f"{user.name}: {len(user.orders)} orders")


# ❌ 查询太多列
users = await db.execute(select(User))  # 返回所有列


# ✅ 只查询需要的列
stmt = select(User.id, User.email)
users = await db.execute(stmt)


# ✅ 使用索引
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(index=True)  # 经常查询，加索引

    __table_args__ = (
        Index('idx_user_email_created', 'email', 'created_at'),  # 复合索引
    )
```

### 5.2 缓存策略

```python
# ✅ 使用Redis缓存
import redis.asyncio as redis
from functools import wraps
import json

redis_client = redis.from_url("redis://localhost")

def cached(ttl: int = 3600):
    """缓存装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{func.__name__}:{args}:{kwargs}"

            # 尝试从缓存获取
            cached_value = await redis_client.get(cache_key)
            if cached_value:
                return json.loads(cached_value)

            # 执行函数
            result = await func(*args, **kwargs)

            # 存入缓存
            await redis_client.setex(
                cache_key,
                ttl,
                json.dumps(result, default=str),
            )

            return result
        return wrapper
    return decorator

@cached(ttl=1800)
async def get_user_stats(user_id: int) -> dict:
    # 复杂的统计查询
    pass


# ✅ 缓存失效策略
async def update_user(user_id: int, data: UserUpdate):
    user = await get_user(user_id)
    for key, value in data.dict(exclude_unset=True).items():
        setattr(user, key, value)

    await db.commit()

    # 删除相关缓存
    await redis_client.delete(f"get_user:{user_id}")
    await redis_client.delete(f"get_user_stats:{user_id}")
```

### 5.3 异步编程最佳实践

```python
# ❌ 串行执行
async def get_dashboard_data(user_id: int):
    profile = await get_user_profile(user_id)  # 100ms
    orders = await get_user_orders(user_id)    # 150ms
    stats = await get_user_stats(user_id)      # 200ms
    # 总耗时: 450ms


# ✅ 并行执行
import asyncio

async def get_dashboard_data(user_id: int):
    profile, orders, stats = await asyncio.gather(
        get_user_profile(user_id),
        get_user_orders(user_id),
        get_user_stats(user_id),
    )
    # 总耗时: ~200ms (最慢的那个)

    return {
        "profile": profile,
        "orders": orders,
        "stats": stats,
    }


# ❌ 在async函数中使用阻塞调用
async def process_data():
    data = requests.get("https://api.example.com")  # 阻塞！
    return data.json()


# ✅ 使用async库
async def process_data():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com")
        return response.json()


# ✅ 如果必须用同步库，用线程池
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=10)

async def run_sync_task():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        sync_blocking_function,  # 同步函数
    )
    return result
```

---

## 6. 踩过的坑与解决方案

### 6.1 时区问题

```python
# ❌ 使用本地时间
from datetime import datetime

user.created_at = datetime.now()  # 本地时间，部署到其他时区就炸


# ✅ 始终使用UTC
from datetime import datetime, timezone

user.created_at = datetime.now(timezone.utc)

# 数据库存储
class User(Base):
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),  # 强制带时区
        default=lambda: datetime.now(timezone.utc),
    )

# 转换给前端
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    user = await get_user_from_db(user_id)
    return {
        "id": user.id,
        "created_at": user.created_at.isoformat(),  # ISO 8601格式
    }
```

### 6.2 内存泄漏

```python
# ❌ 全局缓存无限增长
cache = {}

def get_user(user_id: int):
    if user_id not in cache:
        cache[user_id] = fetch_user_from_db(user_id)
    return cache[user_id]


# ✅ 使用LRU缓存
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_user_config(user_id: int):
    return fetch_config_from_db(user_id)


# ✅ 或使用第三方库
from cachetools import TTLCache

cache = TTLCache(maxsize=1000, ttl=3600)

def get_user(user_id: int):
    if user_id not in cache:
        cache[user_id] = fetch_user_from_db(user_id)
    return cache[user_id]
```

### 6.3 并发安全问题

```python
# ❌ 全局变量不安全
current_user_id = None

async def process_request(user_id: int):
    global current_user_id
    current_user_id = user_id  # 并发请求会互相覆盖！
    await do_something()


# ✅ 使用上下文变量
from contextvars import ContextVar

current_user: ContextVar[int | None] = ContextVar('current_user', default=None)

async def process_request(user_id: int):
    current_user.set(user_id)  # 每个请求独立
    await do_something()

def get_current_user() -> int:
    user_id = current_user.get()
    if user_id is None:
        raise ValueError("No user context")
    return user_id
```

---

## 7. 团队协作最佳实践

### 7.1 Code Review清单

在提交PR前自查：

```markdown
## 功能性
- [ ] 代码实现了需求中的所有功能
- [ ] 边界条件都考虑到了
- [ ] 错误处理完善

## 质量
- [ ] 所有函数都有类型注解
- [ ] 复杂逻辑有注释说明
- [ ] 没有被注释掉的代码
- [ ] 没有print()调试语句
- [ ] 所有新代码有单元测试

## 性能
- [ ] 没有N+1查询
- [ ] 大数据集使用分页
- [ ] 耗时操作使用异步

## 安全
- [ ] 没有硬编码的密钥
- [ ] SQL使用参数化查询
- [ ] 用户输入都经过验证

## 其他
- [ ] 运行了ruff和mypy
- [ ] 测试全部通过
- [ ] 更新了相关文档
```

### 7.2 Git工作流

```bash
# 功能分支命名规范
git checkout -b feature/user-authentication
git checkout -b bugfix/fix-order-calculation
git checkout -b refactor/improve-database-queries

# Commit信息规范
git commit -m "feat: add user authentication with JWT"
git commit -m "fix: correct order total calculation"
git commit -m "refactor: optimize database queries"
git commit -m "docs: update API documentation"
git commit -m "test: add tests for order service"

# 类型前缀
# feat: 新功能
# fix: 修复bug
# refactor: 重构
# docs: 文档
# test: 测试
# chore: 构建/工具
```

---

## 8. 生产环境部署经验

### 8.1 配置管理

```python
# settings.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # 数据库
    database_url: str
    database_pool_size: int = 20

    # Redis
    redis_url: str

    # 安全
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # 外部服务
    email_api_key: str
    sms_api_key: str

    # 日志
    log_level: str = "INFO"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
```

### 8.2 结构化日志

```python
import structlog
from datetime import datetime

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()

# 使用
logger.info(
    "user_login",
    user_id=user.id,
    ip_address=request.client.host,
    user_agent=request.headers.get("user-agent"),
)

# 输出:
# {"event": "user_login", "user_id": 123, "ip_address": "1.2.3.4", ...}
```

### 8.3 健康检查

```python
from fastapi import FastAPI, status
from sqlalchemy import text

app = FastAPI()

@app.get("/health")
async def health_check():
    """Kubernetes liveness probe"""
    return {"status": "ok"}

@app.get("/health/ready")
async def readiness_check(db: AsyncSession = Depends(get_db)):
    """Kubernetes readiness probe"""
    try:
        # 检查数据库
        await db.execute(text("SELECT 1"))

        # 检查Redis
        await redis_client.ping()

        return {"status": "ready"}
    except Exception as e:
        return Response(
            content=f"Not ready: {str(e)}",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
```

### 8.4 监控指标

```python
from prometheus_client import Counter, Histogram
import time

# 定义指标
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

# 中间件
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time

    http_requests_total.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()

    http_request_duration_seconds.labels(
        method=request.method,
        endpoint=request.url.path,
    ).observe(duration)

    return response
```

---

## 总结：真正的专业主义

1. **不要追求完美架构** - 根据项目规模选择合适的复杂度
2. **测试有价值的代码** - 业务逻辑 > 边界条件 > 集成点
3. **性能优化要有数据支撑** - 先测量，再优化
4. **安全问题零容忍** - 参数化查询、环境变量、输入验证
5. **代码是写给人看的** - 清晰 > 聪明
6. **自动化一切可自动化的** - CI/CD、代码检查、测试
7. **保持学习** - 技术栈一直在进化

记住：**写出让同事感激、让未来的自己感激的代码。**
