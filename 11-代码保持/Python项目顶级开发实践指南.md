# Python项目顶级开发实践指南

> 十年Python开发经验总结 - 从编码技巧到架构思维的完整实战方案

---

## 目录

1. [项目搭建与工程化](#1-项目搭建与工程化)
2. [函数与类设计模式](#2-函数与类设计模式)
3. [异常处理与错误设计](#3-异常处理与错误设计)
4. [类型注解实战](#4-类型注解实战)
5. [异步编程最佳实践](#5-异步编程最佳实践)
6. [数据库操作精华](#6-数据库操作精华)
7. [API 设计模式](#7-api-设计模式)
8. [性能优化技巧](#8-性能优化技巧)
9. [测试策略与技巧](#9-测试策略与技巧)
10. [生产环境实战](#10-生产环境实战)

---

## 1. 项目搭建与工程化

### 1.1 现代 Python 项目结构

**推荐结构（src布局）**
```
myproject/
├── src/myproject/          # 源代码
│   ├── __init__.py
│   ├── api/                # API 层
│   │   ├── dependencies.py
│   │   └── routes/
│   ├── domain/             # 领域模型
│   │   └── models.py
│   ├── services/           # 业务逻辑
│   │   └── user_service.py
│   └── utils/              # 工具函数
├── tests/                  # 测试
├── scripts/                # 脚本
├── pyproject.toml          # 项目配置
├── .env.example            # 环境变量模板
└── README.md
```

### 1.2 pyproject.toml 完整配置

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "myproject"
version = "0.1.0"
description = "Production-ready Python project"
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
]

[tool.ruff]
target-version = "py311"
line-length = 100
select = ["E", "W", "F", "I", "N", "UP", "B", "C4", "SIM"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=src/myproject --cov-report=term-missing"
```

### 1.3 依赖管理

```bash
# 使用 uv (推荐 - 比 pip 快 10-100倍)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 或使用 pip
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

---

## 2. 函数与类设计模式

### 2.1 函数参数最佳实践

**参数对象化**
```python
from dataclasses import dataclass
from typing import Optional

# ❌ 参数地狱
def create_order(user_id: int, items: list, coupon: str | None = None,
                 use_points: bool = False, gift_wrap: bool = False):
    pass

# ✅ 使用 dataclass
@dataclass
class OrderRequest:
    user_id: int
    items: list[OrderItem]
    coupon: Optional[str] = None
    use_points: bool = False
    gift_wrap: bool = False

def create_order(request: OrderRequest) -> Order:
    pass
```

**避免可变默认参数**
```python
# ❌ 经典陷阱
def add_item(item: str, items: list = []):
    items.append(item)
    return items

# ✅ 正确做法
def add_item(item: str, items: list | None = None) -> list:
    if items is None:
        items = []
    items.append(item)
    return items
```

**关键字参数强制**
```python
# ✅ 使用 * 强制关键字参数
def create_user(*, email: str, password: str, is_admin: bool = False):
    pass

# 调用必须指定参数名
create_user(email="a@b.com", password="123")  # ✅
create_user("a@b.com", "123")  # ❌ TypeError
```

### 2.2 上下文管理器

**自定义上下文管理器**
```python
from contextlib import contextmanager
from sqlalchemy.ext.asyncio import AsyncSession

# ✅ 自动事务管理
@contextmanager
async def database_transaction(db: AsyncSession):
    try:
        yield db
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()

# 使用
async with database_transaction(db) as session:
    user = User(email="test@example.com")
    session.add(user)
    # 自动 commit
```

**文件操作**
```python
# ✅ 使用 with 自动关闭
with open('file.txt', 'r') as f:
    content = f.read()
# 文件自动关闭

# ✅ 多个上下文
with open('input.txt') as fin, open('output.txt', 'w') as fout:
    fout.write(fin.read())
```

### 2.3 装饰器模式

**缓存装饰器**
```python
from functools import wraps, lru_cache
import time

# ✅ 简单缓存
@lru_cache(maxsize=128)
def expensive_function(n: int) -> int:
    time.sleep(1)
    return n * 2

# ✅ 自定义缓存装饰器
def cache_with_ttl(ttl: int):
    def decorator(func):
        cache = {}
        cache_times = {}

        @wraps(func)
        def wrapper(*args):
            now = time.time()
            if args in cache and now - cache_times[args] < ttl:
                return cache[args]

            result = func(*args)
            cache[args] = result
            cache_times[args] = now
            return result

        return wrapper
    return decorator

@cache_with_ttl(ttl=60)
def get_user_stats(user_id: int):
    # 复杂的统计查询
    pass
```

**重试装饰器**
```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

# ✅ 自动重试
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
async def call_external_api(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=5.0)
        response.raise_for_status()
        return response.json()
```

---

## 3. 异常处理与错误设计

### 3.1 自定义异常体系

```python
# ✅ 构建异常层次结构
class AppException(Exception):
    """应用基础异常"""
    def __init__(self, message: str, code: str | None = None):
        self.message = message
        self.code = code
        super().__init__(message)

class ValidationError(AppException):
    """验证错误"""
    pass

class NotFoundError(AppException):
    """资源未找到"""
    pass

class UnauthorizedError(AppException):
    """未授权"""
    pass

class DatabaseError(AppException):
    """数据库错误"""
    pass
```

### 3.2 异常转换模式

```python
from sqlalchemy.exc import NoResultFound, IntegrityError

# ✅ 在边界转换异常
async def get_user(user_id: int) -> User:
    try:
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one()
    except NoResultFound:
        raise NotFoundError(
            f"User {user_id} not found",
            code="USER_NOT_FOUND"
        ) from None

async def create_user(email: str) -> User:
    try:
        user = User(email=email)
        db.add(user)
        await db.commit()
        return user
    except IntegrityError:
        raise ValidationError(
            f"User with email {email} already exists",
            code="EMAIL_ALREADY_EXISTS"
        ) from None
```

### 3.3 错误处理最佳实践

```python
# ❌ 吞掉异常
try:
    result = dangerous_operation()
except Exception:
    pass  # 灾难

# ❌ 捕获太宽泛
try:
    result = operation()
except Exception as e:
    logger.error(f"Error: {e}")  # 记录了但没处理

# ✅ 精确捕获 + 处理
try:
    result = await api_call()
except httpx.TimeoutException:
    # 超时使用缓存
    result = get_from_cache()
except httpx.HTTPStatusError as e:
    if e.response.status_code == 404:
        raise NotFoundError("Resource not found")
    elif e.response.status_code >= 500:
        # 服务器错误，记录并重试
        logger.error(f"API error: {e}")
        raise
```

---

## 4. 类型注解实战

### 4.1 基础类型注解

```python
from typing import Optional, Union, List, Dict, Tuple, Any

# ✅ 基础类型
def process_user(
    user_id: int,
    name: str,
    email: str | None = None  # Python 3.10+
) -> User:
    pass

# ✅ 集合类型
def get_users(
    ids: list[int],  # Python 3.9+
    filters: dict[str, str]
) -> list[User]:
    pass

# ✅ 元组
def get_stats() -> tuple[int, int, float]:
    return (100, 50, 0.5)

# ✅ 可调用对象
from collections.abc import Callable

def execute(
    func: Callable[[int, str], bool],
    timeout: int
) -> bool:
    return func(10, "test")
```

### 4.2 高级类型注解

**TypedDict**
```python
from typing import TypedDict, NotRequired

# ✅ 精确的字典类型
class UserData(TypedDict):
    id: int
    name: str
    email: str
    age: NotRequired[int]  # Python 3.11+

def process_user(data: UserData) -> None:
    print(data["name"])  # 类型检查通过
    print(data["invalid"])  # mypy 报错
```

**泛型**
```python
from typing import TypeVar, Generic

T = TypeVar('T')

class Repository(Generic[T]):
    def __init__(self, model_class: type[T]):
        self.model_class = model_class

    async def get(self, id: int) -> T | None:
        result = await db.execute(
            select(self.model_class).where(self.model_class.id == id)
        )
        return result.scalar_one_or_none()

    async def save(self, entity: T) -> T:
        db.add(entity)
        await db.commit()
        return entity

# 使用
user_repo = Repository[User](User)
user = await user_repo.get(123)  # mypy 知道 user 是 User 类型
```

**Protocol（结构化类型）**
```python
from typing import Protocol

# ✅ 定义接口协议
class Serializable(Protocol):
    def to_dict(self) -> dict: ...
    def from_dict(self, data: dict) -> None: ...

# 任何实现了这些方法的类都符合协议
class User:
    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name}

    def from_dict(self, data: dict) -> None:
        self.id = data["id"]
        self.name = data["name"]

def serialize(obj: Serializable) -> dict:
    return obj.to_dict()

# User 自动符合 Serializable 协议
user = User()
serialize(user)  # ✅ 类型检查通过
```

### 4.3 类型守卫

```python
from typing import TypeGuard

# ✅ 自定义类型守卫
def is_user(obj: object) -> TypeGuard[User]:
    return (
        isinstance(obj, dict) and
        "id" in obj and
        "name" in obj and
        isinstance(obj["id"], int)
    )

def process_data(data: dict | User):
    if is_user(data):
        # mypy 知道这里 data 是 User
        print(data.name)
    else:
        # mypy 知道这里 data 是 dict
        print(data.keys())
```

---

## 5. 异步编程最佳实践

### 5.1 异步函数基础

**并发执行**
```python
import asyncio

# ❌ 串行执行
async def fetch_all_serial():
    user = await fetch_user()      # 100ms
    posts = await fetch_posts()    # 150ms
    stats = await fetch_stats()    # 200ms
    # 总耗时: 450ms

# ✅ 并行执行
async def fetch_all_parallel():
    user, posts, stats = await asyncio.gather(
        fetch_user(),
        fetch_posts(),
        fetch_stats()
    )
    # 总耗时: ~200ms (最慢的那个)

# ✅ 带错误处理的并行
async def fetch_all_safe():
    results = await asyncio.gather(
        fetch_user(),
        fetch_posts(),
        fetch_stats(),
        return_exceptions=True  # 不会因为一个失败而全部失败
    )

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Failed: {result}")
```

**超时控制**
```python
import asyncio

# ✅ 单个操作超时
async def fetch_with_timeout():
    try:
        result = await asyncio.wait_for(
            fetch_slow_api(),
            timeout=5.0
        )
    except asyncio.TimeoutError:
        logger.error("Request timed out")
        raise
```

### 5.2 异步上下文管理器

```python
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession

# ✅ 异步上下文管理器
@asynccontextmanager
async def get_db_session():
    session = AsyncSession(engine)
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()

# 使用
async with get_db_session() as db:
    user = User(email="test@example.com")
    db.add(user)
    # 自动 commit
```

### 5.3 异步迭代器

```python
from typing import AsyncIterator

# ✅ 异步生成器
async def fetch_users_paginated(
    page_size: int = 100
) -> AsyncIterator[User]:
    offset = 0
    while True:
        result = await db.execute(
            select(User).limit(page_size).offset(offset)
        )
        users = result.scalars().all()

        if not users:
            break

        for user in users:
            yield user

        offset += page_size

# 使用
async for user in fetch_users_paginated():
    await process_user(user)
```

### 5.4 同步代码调用异步

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=10)

# ✅ 在异步函数中运行同步代码
async def run_sync_in_async():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        sync_blocking_function,  # 同步函数
        arg1, arg2
    )
    return result

# ✅ 在同步代码中运行异步函数
def sync_function():
    result = asyncio.run(async_function())
    return result
```

---

## 6. 数据库操作精华

### 6.1 SQLAlchemy 2.0 最佳实践

**模型定义**
```python
from sqlalchemy import String, Integer, DateTime, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from datetime import datetime

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    username: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )

    # 关系
    posts: Mapped[list["Post"]] = relationship(back_populates="author")

class Post(Base):
    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    author_id: Mapped[int] = mapped_column(ForeignKey("users.id"))

    author: Mapped[User] = relationship(back_populates="posts")
```

**查询模式**
```python
from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import selectinload, joinedload

# ✅ 基础查询
stmt = select(User).where(User.id == 123)
result = await session.execute(stmt)
user = result.scalar_one_or_none()

# ✅ 复杂条件
stmt = select(User).where(
    and_(
        User.created_at >= start_date,
        or_(User.role == 'admin', User.is_verified == True)
    )
)

# ✅ JOIN 查询
stmt = (
    select(User, Post)
    .join(Post, User.id == Post.author_id)
    .where(Post.published == True)
)

# ✅ 聚合查询
stmt = select(
    User.role,
    func.count(User.id).label('count')
).group_by(User.role)
```

**N+1 问题解决**
```python
# ❌ N+1 问题
users = await session.execute(select(User).limit(100))
for user in users.scalars():
    # 每个用户都查询一次！
    posts = await session.execute(
        select(Post).where(Post.author_id == user.id)
    )

# ✅ 使用 selectinload（分两次查询）
stmt = select(User).options(selectinload(User.posts)).limit(100)
users = await session.execute(stmt)
for user in users.scalars():
    # posts 已经加载
    print(len(user.posts))

# ✅ 使用 joinedload（一次查询）
stmt = select(User).options(joinedload(User.posts))
```

### 6.2 事务管理

```python
from sqlalchemy.ext.asyncio import AsyncSession

# ✅ 手动事务控制
async def transfer_money(from_id: int, to_id: int, amount: float):
    async with AsyncSession(engine) as session:
        async with session.begin():  # 自动 commit/rollback
            # 扣款
            result = await session.execute(
                select(Account).where(Account.id == from_id).with_for_update()
            )
            from_account = result.scalar_one()
            from_account.balance -= amount

            # 加款
            result = await session.execute(
                select(Account).where(Account.id == to_id).with_for_update()
            )
            to_account = result.scalar_one()
            to_account.balance += amount

            # 事务结束自动 commit
```

---

## 7. API 设计模式

### 7.1 FastAPI 最佳实践

**依赖注入**
```python
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

app = FastAPI()

# ✅ 数据库依赖
async def get_db() -> AsyncSession:
    async with async_session_maker() as session:
        yield session

# ✅ 当前用户依赖
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    user = await verify_token(token, db)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user

# ✅ 使用依赖
@app.get("/users/me")
async def get_me(
    current_user: User = Depends(get_current_user)
):
    return current_user
```

**Pydantic Schema**
```python
from pydantic import BaseModel, EmailStr, Field, field_validator

# ✅ 请求模型
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=100)
    username: str = Field(min_length=3, max_length=50)

    @field_validator('username')
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v

# ✅ 响应模型
class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    created_at: datetime

    model_config = {"from_attributes": True}

# ✅ API 端点
@app.post("/users", response_model=UserResponse)
async def create_user(
    data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    user = User(**data.model_dump(exclude={'password'}))
    user.hashed_password = hash_password(data.password)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user
```

**后台任务**
```python
from fastapi import BackgroundTasks

# ✅ 后台任务
async def send_welcome_email(email: str):
    await email_service.send(email, "Welcome!")

@app.post("/users")
async def create_user(
    data: UserCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    user = await create_user_in_db(data, db)

    # 异步执行后台任务
    background_tasks.add_task(send_welcome_email, user.email)
    background_tasks.add_task(log_user_creation, user.id)

    return user
```

### 7.2 错误处理

```python
from fastapi import Request, status
from fastapi.responses import JSONResponse

# ✅ 全局异常处理
@app.exception_handler(NotFoundError)
async def not_found_handler(request: Request, exc: NotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": exc.message, "code": exc.code}
    )

@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": exc.message, "code": exc.code}
    )

# ✅ 请求验证错误
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"errors": exc.errors()}
    )
```

---

## 8. 性能优化技巧

### 8.1 缓存策略

**函数级缓存**
```python
from functools import lru_cache

# ✅ 简单缓存
@lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

**Redis 缓存**
```python
import redis.asyncio as redis
import json

redis_client = redis.from_url("redis://localhost")

# ✅ 缓存装饰器
def cached(key_prefix: str, ttl: int = 3600):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache_key = f"{key_prefix}:{args}:{kwargs}"

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
                json.dumps(result, default=str)
            )

            return result
        return wrapper
    return decorator

@cached("user_stats", ttl=1800)
async def get_user_stats(user_id: int) -> dict:
    # 复杂的统计查询
    pass
```

### 8.2 批量操作

**批量插入**
```python
# ❌ 逐条插入
for user_data in users_data:
    user = User(**user_data)
    session.add(user)
    await session.commit()  # 每次都提交！

# ✅ 批量插入
session.add_all([User(**data) for data in users_data])
await session.commit()  # 一次提交

# ✅ bulk_insert_mappings（更快）
await session.execute(
    insert(User),
    users_data
)
await session.commit()
```

### 8.3 生成器节省内存

```python
# ❌ 一次性加载所有数据
def process_large_file(filename: str):
    with open(filename) as f:
        lines = f.readlines()  # 全部加载到内存
        for line in lines:
            process_line(line)

# ✅ 使用生成器
def process_large_file(filename: str):
    with open(filename) as f:
        for line in f:  # 逐行读取
            process_line(line)

# ✅ 数据库分页查询
async def process_all_users():
    page_size = 1000
    offset = 0

    while True:
        result = await session.execute(
            select(User).limit(page_size).offset(offset)
        )
        users = result.scalars().all()

        if not users:
            break

        for user in users:
            await process_user(user)

        offset += page_size
```

---

## 9. 测试策略与技巧

### 9.1 Pytest 最佳实践

**Fixtures**
```python
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

@pytest.fixture(scope="session")
async def engine():
    engine = create_async_engine("postgresql+asyncpg://test:test@localhost/test")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

@pytest.fixture
async def db(engine) -> AsyncSession:
    async with AsyncSession(engine) as session:
        yield session
        await session.rollback()  # 测试后回滚

@pytest.fixture
async def user(db: AsyncSession) -> User:
    user = User(email="test@example.com")
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user
```

**参数化测试**
```python
import pytest

@pytest.mark.parametrize("email,expected", [
    ("valid@example.com", True),
    ("invalid.com", False),
    ("@example.com", False),
    ("user@", False),
])
def test_email_validation(email: str, expected: bool):
    assert is_valid_email(email) == expected
```

**Mock 外部依赖**
```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_fetch_user_from_api():
    mock_response = {"id": 123, "name": "John"}

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value.json = AsyncMock(return_value=mock_response)
        mock_get.return_value.status_code = 200

        service = ExternalAPIService()
        user_data = await service.fetch_user(123)

        assert user_data["name"] == "John"
```

---

## 10. 生产环境实战

### 10.1 配置管理

```python
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

    # 外部服务
    email_api_key: str

    # 日志
    log_level: str = "INFO"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
```

### 10.2 结构化日志

```python
import structlog

# ✅ 配置结构化日志
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()

# ✅ 使用
logger.info(
    "user_login",
    user_id=user.id,
    ip_address=request.client.host,
    user_agent=request.headers.get("user-agent")
)
# 输出: {"event":"user_login","user_id":123,"ip_address":"1.2.3.4",...}
```

### 10.3 健康检查

```python
from fastapi import FastAPI, status, Response

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

        # 检查 Redis
        await redis_client.ping()

        return {"status": "ready"}
    except Exception as e:
        return Response(
            content=f"Not ready: {str(e)}",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )
```

### 10.4 监控指标

```python
from prometheus_client import Counter, Histogram
import time

# ✅ 定义指标
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

# ✅ 中间件
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time

    http_requests_total.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    http_request_duration_seconds.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)

    return response
```

---

## 总结：10年Python的编码智慧

### 核心原则

1. **代码可读性**
   - 命名清晰，见名知意
   - 函数单一职责
   - 类型注解完整

2. **异常处理**
   - 精确捕获异常
   - 边界转换异常
   - 永不吞掉异常

3. **异步编程**
   - 区分 IO密集 vs CPU密集
   - 并发执行提升性能
   - 清理资源避免泄漏

4. **数据库优化**
   - 避免 N+1 查询
   - 使用索引
   - 批量操作

5. **测试覆盖**
   - 核心业务逻辑必测
   - 边界条件必测
   - Mock 外部依赖

### 记住

> 代码是写给6个月后的自己看的。
>
> 简单的设计胜过复杂的设计。
>
> 显式优于隐式，可读性胜过简洁性。
>
> 测试不是负担，是信心的来源。
