# Python 注解系统高阶详解

## 目录
- [核心概念](#核心概念)
- [基础注解语法](#基础注解语法)
- [类型注解进阶](#类型注解进阶)
- [泛型注解](#泛型注解)
- [运行时注解处理](#运行时注解处理)
- [装饰器与注解结合](#装饰器与注解结合)
- [协议与结构化子类型](#协议与结构化子类型)
- [字面量类型与类型窄化](#字面量类型与类型窄化)
- [类型守卫与类型保护](#类型守卫与类型保护)
- [注解的高级应用场景](#注解的高级应用场景)
- [性能与最佳实践](#性能与最佳实践)

---

## 核心概念

### 什么是注解（Annotations）

Python 注解是一种在 **不改变运行时行为** 的前提下，为函数参数、返回值、变量和类属性添加元数据的机制。注解在 Python 3.0 引入，并在后续版本中不断完善。

**关键特性：**
- 注解本质上是附加到对象上的元数据
- 默认情况下不影响代码执行（运行时忽略）
- 存储在 `__annotations__` 字典中
- 可以是任意 Python 表达式

### 注解 vs 类型提示

```python
# 注解可以是任意表达式
def func(x: "任意字符串", y: 42, z: print) -> list:
    pass

# 类型提示是注解的一种语义化应用
from typing import List

def func(x: str, y: int, z: callable) -> List[str]:
    pass
```

**区别：**
- **注解**：语法层面的元数据机制
- **类型提示**：使用注解表达类型信息的约定

---

## 基础注解语法

### 1. 函数注解

```python
def greet(name: str, age: int = 18) -> str:
    return f"Hello {name}, you are {age} years old"

# 查看注解
print(greet.__annotations__)
# {'name': <class 'str'>, 'age': <class 'int'>, 'return': <class 'str'>}
```

### 2. 变量注解（Python 3.6+）

```python
# 简单变量注解
name: str = "Alice"
age: int = 30

# 类属性注解
class Person:
    name: str
    age: int
    address: str = "Unknown"  # 带默认值

    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

# 查看类注解
print(Person.__annotations__)
# {'name': <class 'str'>, 'age': <class 'int'>, 'address': <class 'str'>}
```

### 3. 延迟注解（Deferred Annotations）

```python
# 使用字符串避免前向引用问题
class Node:
    def __init__(self, value: int, next: "Node" = None):
        self.value = value
        self.next = next

# Python 3.7+ 可使用 from __future__ import annotations
from __future__ import annotations

class Node:
    def __init__(self, value: int, next: Node = None):  # 不需要引号
        self.value = value
        self.next = next
```

---

## 类型注解进阶

### 1. typing 模块核心类型

```python
from typing import (
    List, Dict, Set, Tuple, Optional, Union,
    Any, Callable, Iterable, Iterator, Sequence
)

# 容器类型
numbers: List[int] = [1, 2, 3]
mapping: Dict[str, int] = {"a": 1, "b": 2}
unique: Set[str] = {"apple", "banana"}

# 元组类型
coordinates: Tuple[float, float] = (10.5, 20.3)
variable_tuple: Tuple[int, ...] = (1, 2, 3, 4, 5)  # 可变长度

# 可选类型
def find_user(user_id: int) -> Optional[str]:
    # 返回 str 或 None
    return None

# 联合类型
def process(value: Union[int, str]) -> str:
    if isinstance(value, int):
        return str(value)
    return value

# Python 3.10+ 简化语法
def process(value: int | str) -> str:
    pass
```

### 2. Callable 类型

```python
from typing import Callable

# 函数类型注解
def apply(func: Callable[[int, int], int], x: int, y: int) -> int:
    return func(x, y)

def add(a: int, b: int) -> int:
    return a + b

result = apply(add, 3, 5)  # 8

# 复杂回调
Callback = Callable[[str, int], bool]

def register_callback(cb: Callback) -> None:
    result = cb("test", 42)
```

### 3. Any 与 NoReturn

```python
from typing import Any, NoReturn

# Any - 跳过类型检查
def legacy_function(data: Any) -> Any:
    return data

# NoReturn - 永不返回
def raise_error() -> NoReturn:
    raise RuntimeError("This function never returns")

def infinite_loop() -> NoReturn:
    while True:
        pass
```

---

## 泛型注解

### 1. TypeVar - 类型变量

```python
from typing import TypeVar, List, Sequence

T = TypeVar('T')

def first(items: Sequence[T]) -> T:
    return items[0]

# 类型推断
num = first([1, 2, 3])  # int
name = first(["Alice", "Bob"])  # str

# 带约束的 TypeVar
NumberType = TypeVar('NumberType', int, float)

def add(x: NumberType, y: NumberType) -> NumberType:
    return x + y

add(1, 2)      # OK
add(1.5, 2.5)  # OK
add(1, 2.5)    # 类型检查器会警告
```

### 2. 泛型类

```python
from typing import Generic, TypeVar

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self):
        self._items: List[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()

# 使用泛型类
int_stack: Stack[int] = Stack()
int_stack.push(1)
int_stack.push(2)

str_stack: Stack[str] = Stack()
str_stack.push("hello")
```

### 3. 多类型变量

```python
from typing import TypeVar, Generic, Tuple

K = TypeVar('K')
V = TypeVar('V')

class Pair(Generic[K, V]):
    def __init__(self, key: K, value: V):
        self.key = key
        self.value = value

    def get_tuple(self) -> Tuple[K, V]:
        return (self.key, self.value)

# 使用
pair: Pair[str, int] = Pair("age", 30)
```

### 4. 协变与逆变

```python
from typing import TypeVar, Generic, List

# 不变（Invariant）- 默认行为
T = TypeVar('T')

# 协变（Covariant）- 可以用子类型替代
T_co = TypeVar('T_co', covariant=True)

# 逆变（Contravariant）- 可以用父类型替代
T_contra = TypeVar('T_contra', contravariant=True)

# 实际应用
class Animal: pass
class Dog(Animal): pass

class Producer(Generic[T_co]):
    def produce(self) -> T_co:
        ...

# 协变允许
dog_producer: Producer[Dog] = Producer()
animal_producer: Producer[Animal] = dog_producer  # OK

class Consumer(Generic[T_contra]):
    def consume(self, item: T_contra) -> None:
        ...

# 逆变允许
animal_consumer: Consumer[Animal] = Consumer()
dog_consumer: Consumer[Dog] = animal_consumer  # OK
```

---

## 运行时注解处理

### 1. 访问注解信息

```python
from typing import get_type_hints
import inspect

def example(name: str, age: int = 18) -> str:
    return f"{name} is {age}"

# 方法1：直接访问 __annotations__
print(example.__annotations__)
# {'name': <class 'str'>, 'age': <class 'int'>, 'return': <class 'str'>}

# 方法2：使用 get_type_hints（推荐，会解析字符串注解）
print(get_type_hints(example))

# 方法3：inspect 模块
sig = inspect.signature(example)
for param_name, param in sig.parameters.items():
    print(f"{param_name}: {param.annotation}")
```

### 2. 运行时类型检查

```python
from typing import get_type_hints, get_origin, get_args
import inspect

def validate_types(func):
    """装饰器：运行时类型验证"""
    hints = get_type_hints(func)
    sig = inspect.signature(func)

    def wrapper(*args, **kwargs):
        # 绑定参数
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # 验证参数类型
        for param_name, param_value in bound.arguments.items():
            if param_name in hints:
                expected_type = hints[param_name]
                origin = get_origin(expected_type)

                # 处理泛型类型
                if origin is not None:
                    if not isinstance(param_value, origin):
                        raise TypeError(
                            f"{param_name} expected {expected_type}, "
                            f"got {type(param_value)}"
                        )
                else:
                    if not isinstance(param_value, expected_type):
                        raise TypeError(
                            f"{param_name} expected {expected_type}, "
                            f"got {type(param_value)}"
                        )

        # 执行函数
        result = func(*args, **kwargs)

        # 验证返回值类型
        if 'return' in hints:
            expected_return = hints['return']
            origin = get_origin(expected_return)

            if origin is not None:
                if not isinstance(result, origin):
                    raise TypeError(
                        f"Return value expected {expected_return}, "
                        f"got {type(result)}"
                    )
            else:
                if not isinstance(result, expected_return):
                    raise TypeError(
                        f"Return value expected {expected_return}, "
                        f"got {type(result)}"
                    )

        return result

    return wrapper

# 使用
@validate_types
def add(x: int, y: int) -> int:
    return x + y

add(1, 2)      # OK
add("1", "2")  # TypeError
```

### 3. 深度类型验证

```python
from typing import get_origin, get_args, Union
import collections.abc

def deep_type_check(value, expected_type) -> bool:
    """递归验证复杂类型"""
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    # 处理 None 和 Any
    if expected_type is type(None):
        return value is None

    # 处理 Union 和 Optional
    if origin is Union:
        return any(deep_type_check(value, t) for t in args)

    # 处理泛型容器
    if origin is list or origin is collections.abc.Sequence:
        if not isinstance(value, list):
            return False
        if args:
            return all(deep_type_check(item, args[0]) for item in value)
        return True

    if origin is dict or origin is collections.abc.Mapping:
        if not isinstance(value, dict):
            return False
        if args:
            key_type, value_type = args
            return all(
                deep_type_check(k, key_type) and deep_type_check(v, value_type)
                for k, v in value.items()
            )
        return True

    # 处理普通类型
    if origin is None:
        return isinstance(value, expected_type)

    return isinstance(value, origin)

# 测试
from typing import List, Dict

print(deep_type_check([1, 2, 3], List[int]))  # True
print(deep_type_check([1, "2", 3], List[int]))  # False
print(deep_type_check({"a": 1}, Dict[str, int]))  # True
```

---

## 装饰器与注解结合

### 1. 保留注解的装饰器

```python
from functools import wraps
from typing import TypeVar, Callable, Any

F = TypeVar('F', bound=Callable[..., Any])

def preserve_annotations(decorator: Callable[[F], F]) -> Callable[[F], F]:
    """确保装饰器保留原函数的注解"""
    @wraps(decorator)
    def wrapper(func: F) -> F:
        decorated = decorator(func)
        decorated.__annotations__ = func.__annotations__
        return decorated
    return wrapper

# 使用示例
def timer(func: F) -> F:
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Executed in {time.time() - start:.4f}s")
        return result
    return wrapper

@timer
def calculate(x: int, y: int) -> int:
    return x + y

print(calculate.__annotations__)  # 注解被保留
```

### 2. 基于注解的依赖注入

```python
from typing import get_type_hints, Type, Dict, Any
import inspect

class Container:
    """简单的依赖注入容器"""
    def __init__(self):
        self._services: Dict[Type, Any] = {}

    def register(self, service_type: Type, instance: Any) -> None:
        self._services[service_type] = instance

    def resolve(self, func: Callable) -> Callable:
        """解析函数依赖并注入"""
        hints = get_type_hints(func)
        sig = inspect.signature(func)

        def wrapper(*args, **kwargs):
            # 自动注入未提供的参数
            for param_name, param in sig.parameters.items():
                if param_name not in kwargs and param_name in hints:
                    param_type = hints[param_name]
                    if param_type in self._services:
                        kwargs[param_name] = self._services[param_type]

            return func(*args, **kwargs)

        return wrapper

# 使用示例
class Database:
    def query(self, sql: str) -> list:
        return [{"id": 1, "name": "test"}]

class Logger:
    def log(self, message: str) -> None:
        print(f"[LOG] {message}")

# 配置容器
container = Container()
container.register(Database, Database())
container.register(Logger, Logger())

# 自动注入依赖
@container.resolve
def process_data(db: Database, logger: Logger) -> None:
    logger.log("Starting query")
    result = db.query("SELECT * FROM users")
    logger.log(f"Found {len(result)} records")

process_data()  # 自动注入 db 和 logger
```

---

## 协议与结构化子类型

### 1. Protocol 基础

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Drawable(Protocol):
    """任何实现了 draw 方法的对象都满足此协议"""
    def draw(self) -> str:
        ...

class Circle:
    def draw(self) -> str:
        return "Drawing circle"

class Square:
    def draw(self) -> str:
        return "Drawing square"

def render(shape: Drawable) -> None:
    print(shape.draw())

# Duck Typing - 无需显式继承
render(Circle())  # OK
render(Square())  # OK

# 运行时检查
print(isinstance(Circle(), Drawable))  # True
```

### 2. 复杂协议

```python
from typing import Protocol, Iterator

class SupportsIter(Protocol):
    def __iter__(self) -> Iterator[int]:
        ...

    def __len__(self) -> int:
        ...

class MyRange:
    def __init__(self, n: int):
        self.n = n

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.n))

    def __len__(self) -> int:
        return self.n

def process(items: SupportsIter) -> None:
    print(f"Processing {len(items)} items")
    for item in items:
        print(item)

process(MyRange(5))  # OK
```

### 3. 协议组合

```python
from typing import Protocol

class Comparable(Protocol):
    def __lt__(self, other: Any) -> bool:
        ...

class Hashable(Protocol):
    def __hash__(self) -> int:
        ...

class SortableHashable(Comparable, Hashable, Protocol):
    """组合多个协议"""
    pass

class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def __lt__(self, other: "Person") -> bool:
        return self.age < other.age

    def __hash__(self) -> int:
        return hash((self.name, self.age))

def process_item(item: SortableHashable) -> None:
    print(f"Hash: {hash(item)}")

process_item(Person("Alice", 30))  # OK
```

---

## 字面量类型与类型窄化

### 1. Literal 类型

```python
from typing import Literal

Mode = Literal["read", "write", "append"]

def open_file(filename: str, mode: Mode) -> None:
    print(f"Opening {filename} in {mode} mode")

open_file("test.txt", "read")    # OK
open_file("test.txt", "delete")  # 类型检查器报错

# 组合字面量
Status = Literal["success", "error", "pending"]
Priority = Literal[1, 2, 3]

def set_priority(level: Priority) -> None:
    pass

set_priority(1)  # OK
set_priority(5)  # 类型检查器报错
```

### 2. 类型窄化（Type Narrowing）

```python
from typing import Union

def process(value: Union[int, str]) -> str:
    # isinstance 窄化类型
    if isinstance(value, int):
        # 此处 value 被窄化为 int
        return str(value * 2)
    else:
        # 此处 value 被窄化为 str
        return value.upper()

# None 检查窄化
from typing import Optional

def greet(name: Optional[str]) -> str:
    if name is None:
        return "Hello, stranger"
    # 此处 name 被窄化为 str
    return f"Hello, {name}"

# 自定义类型守卫
def is_list_of_ints(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(x, int) for x in value)

def process_data(data: Union[List[int], List[str]]) -> None:
    if is_list_of_ints(data):
        # 理想情况下 data 应被窄化为 List[int]
        # 但 Python 类型检查器可能不支持
        total = sum(data)
```

### 3. TypeGuard（Python 3.10+）

```python
from typing import TypeGuard, List

def is_list_of_strings(value: List[object]) -> TypeGuard[List[str]]:
    """类型守卫函数"""
    return all(isinstance(item, str) for item in value)

def process(items: List[object]) -> None:
    if is_list_of_strings(items):
        # items 被窄化为 List[str]
        for item in items:
            print(item.upper())  # OK，item 是 str
    else:
        print("Not all strings")

# 使用
process(["a", "b", "c"])  # OK
process([1, 2, 3])        # 打印 "Not all strings"
```

---

## 类型守卫与类型保护

### 1. 自定义类型守卫

```python
from typing import TypeGuard, Union

class Cat:
    def meow(self) -> str:
        return "Meow!"

class Dog:
    def bark(self) -> str:
        return "Woof!"

Animal = Union[Cat, Dog]

def is_cat(animal: Animal) -> TypeGuard[Cat]:
    return isinstance(animal, Cat)

def make_sound(animal: Animal) -> str:
    if is_cat(animal):
        return animal.meow()  # animal 被窄化为 Cat
    else:
        return animal.bark()  # animal 被窄化为 Dog
```

### 2. 复杂数据验证

```python
from typing import TypedDict, TypeGuard

class UserDict(TypedDict):
    name: str
    age: int
    email: str

def is_valid_user(data: dict) -> TypeGuard[UserDict]:
    """验证字典是否符合 UserDict 结构"""
    return (
        isinstance(data.get("name"), str) and
        isinstance(data.get("age"), int) and
        isinstance(data.get("email"), str)
    )

def process_user(data: dict) -> None:
    if is_valid_user(data):
        # data 被窄化为 UserDict
        print(f"User: {data['name']}, Age: {data['age']}")
    else:
        print("Invalid user data")

# 使用
process_user({"name": "Alice", "age": 30, "email": "alice@example.com"})
process_user({"name": "Bob"})  # Invalid
```

---

## 注解的高级应用场景

### 1. ORM 模型定义

```python
from typing import Optional, ClassVar
from dataclasses import dataclass, field

@dataclass
class User:
    id: int
    name: str
    email: str
    age: Optional[int] = None
    is_active: bool = True

    # 类变量
    table_name: ClassVar[str] = "users"

    # 带验证的字段
    def __post_init__(self):
        if self.age is not None and self.age < 0:
            raise ValueError("Age cannot be negative")

# 使用
user = User(id=1, name="Alice", email="alice@example.com", age=30)
print(User.table_name)  # "users"
```

### 2. API 请求验证

```python
from typing import TypedDict, List, Optional
from typing import get_type_hints

class CreateUserRequest(TypedDict):
    username: str
    email: str
    age: int
    roles: List[str]
    metadata: Optional[dict]

def validate_request(data: dict, expected_type: type) -> bool:
    """基于 TypedDict 验证请求数据"""
    hints = get_type_hints(expected_type)

    for field, field_type in hints.items():
        if field not in data:
            # 检查是否为 Optional
            origin = get_origin(field_type)
            if origin is Union:
                args = get_args(field_type)
                if type(None) not in args:
                    return False
            else:
                return False
        else:
            # 简化的类型检查
            value = data[field]
            if not isinstance(value, field_type):
                return False

    return True

# 使用
request_data = {
    "username": "alice",
    "email": "alice@example.com",
    "age": 30,
    "roles": ["admin", "user"]
}

if validate_request(request_data, CreateUserRequest):
    print("Valid request")
```

### 3. 配置管理

```python
from typing import Annotated, get_type_hints, get_origin, get_args
from dataclasses import dataclass

# 使用 Annotated 添加元数据
@dataclass
class Config:
    # 数据库配置
    db_host: Annotated[str, "Database host", "localhost"]
    db_port: Annotated[int, "Database port", 5432]
    db_name: Annotated[str, "Database name", "mydb"]

    # 应用配置
    debug: Annotated[bool, "Debug mode", False]
    max_connections: Annotated[int, "Max connections", 100]

def load_config_from_env() -> Config:
    """从环境变量加载配置，使用注解中的默认值"""
    import os
    hints = get_type_hints(Config, include_extras=True)

    kwargs = {}
    for field_name, annotated_type in hints.items():
        # 获取 Annotated 的参数
        if get_origin(annotated_type) is Annotated:
            args = get_args(annotated_type)
            actual_type = args[0]
            default_value = args[2] if len(args) > 2 else None

            # 从环境变量读取，或使用默认值
            env_value = os.getenv(field_name.upper(), default_value)

            # 类型转换
            if actual_type is bool:
                kwargs[field_name] = env_value in ("True", "true", "1")
            elif actual_type is int:
                kwargs[field_name] = int(env_value)
            else:
                kwargs[field_name] = env_value

    return Config(**kwargs)

# 使用
config = load_config_from_env()
print(config)
```

### 4. 序列化与反序列化

```python
from typing import get_type_hints, get_origin, get_args, List, Dict, Any
from dataclasses import dataclass, asdict
import json

@dataclass
class Address:
    street: str
    city: str
    country: str

@dataclass
class Person:
    name: str
    age: int
    addresses: List[Address]
    metadata: Dict[str, Any]

def deserialize(data: dict, cls: type):
    """基于类型注解的反序列化"""
    hints = get_type_hints(cls)
    kwargs = {}

    for field, field_type in hints.items():
        if field not in data:
            continue

        value = data[field]
        origin = get_origin(field_type)

        # 处理 List
        if origin is list:
            args = get_args(field_type)
            if args and hasattr(args[0], '__annotations__'):
                # List[DataClass]
                kwargs[field] = [deserialize(item, args[0]) for item in value]
            else:
                kwargs[field] = value

        # 处理嵌套 dataclass
        elif hasattr(field_type, '__annotations__'):
            kwargs[field] = deserialize(value, field_type)

        else:
            kwargs[field] = value

    return cls(**kwargs)

# 使用
data = {
    "name": "Alice",
    "age": 30,
    "addresses": [
        {"street": "123 Main St", "city": "NYC", "country": "USA"},
        {"street": "456 Elm St", "city": "LA", "country": "USA"}
    ],
    "metadata": {"role": "admin"}
}

person = deserialize(data, Person)
print(person)

# 序列化（使用 dataclasses.asdict）
print(json.dumps(asdict(person), indent=2))
```

---

## 性能与最佳实践

### 1. 性能影响

```python
import timeit

# 注解对运行时性能的影响微乎其微
def without_annotations(x, y):
    return x + y

def with_annotations(x: int, y: int) -> int:
    return x + y

# 性能测试
print("Without annotations:", timeit.timeit(lambda: without_annotations(1, 2)))
print("With annotations:", timeit.timeit(lambda: with_annotations(1, 2)))
# 几乎没有性能差异
```

### 2. 延迟注解求值（PEP 563）

```python
from __future__ import annotations  # Python 3.7+

# 注解不会在定义时求值，而是保存为字符串
class Node:
    def __init__(self, value: int, left: Node = None, right: Node = None):
        self.value = value
        self.left = left
        self.right = right

# 查看注解
print(Node.__init__.__annotations__)
# {'value': 'int', 'left': 'Node', 'right': 'Node', 'return': 'None'}

# 需要时使用 get_type_hints 解析
from typing import get_type_hints
print(get_type_hints(Node.__init__))
# {'value': <class 'int'>, 'left': <class '__main__.Node'>, ...}
```

### 3. 最佳实践总结

```python
# ✅ 推荐做法

# 1. 为公共 API 添加类型注解
def public_function(data: dict) -> list:
    """公共接口应该有清晰的类型签名"""
    return list(data.values())

# 2. 使用 TypedDict 替代普通字典
from typing import TypedDict

class UserData(TypedDict):
    name: str
    age: int

def create_user(data: UserData) -> None:
    pass

# 3. 使用 Protocol 实现鸭子类型
from typing import Protocol

class SupportsClose(Protocol):
    def close(self) -> None:
        ...

def cleanup(resource: SupportsClose) -> None:
    resource.close()

# 4. 使用 Annotated 添加元数据
from typing import Annotated

UserId = Annotated[int, "Unique user identifier"]

def get_user(user_id: UserId) -> UserData:
    pass

# 5. 对复杂类型使用类型别名
from typing import Union, List, Dict

JSON = Union[dict, list, str, int, float, bool, None]
JSONObject = Dict[str, JSON]
JSONArray = List[JSON]

def parse_json(data: str) -> JSON:
    import json
    return json.loads(data)

# ❌ 避免的做法

# 1. 不要过度使用 Any
def bad_function(data: Any) -> Any:  # 失去类型检查的价值
    pass

# 2. 不要在内部实现细节上过度注解
def _internal_helper(x, y):  # 私有函数可以省略注解
    return x + y

# 3. 不要使用过于复杂的嵌套类型
# 考虑定义类型别名
ComplexType = Dict[str, List[Tuple[int, str]]]  # 好
def func(data: ComplexType) -> None:
    pass

# 4. 不要忽略运行时类型检查
# 如果需要运行时验证，使用 pydantic 或自定义验证
```

### 4. 工具链集成

```python
# mypy - 静态类型检查器
# 安装: pip install mypy
# 运行: mypy your_script.py

# 配置文件 mypy.ini
"""
[mypy]
python_version = 3.10
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
"""

# pyright - 微软开发的类型检查器
# 安装: npm install -g pyright
# 运行: pyright your_script.py

# pydantic - 运行时数据验证
from pydantic import BaseModel, validator

class User(BaseModel):
    name: str
    age: int
    email: str

    @validator('age')
    def age_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Age must be positive')
        return v

# 自动验证和类型转换
user = User(name="Alice", age="30", email="alice@example.com")
print(user.age)  # 30 (int)
```

### 5. 渐进式类型化策略

```python
# 阶段1：为新代码添加注解
def new_feature(data: dict) -> list:
    return process_data(data)  # process_data 可能没有注解

# 阶段2：为核心 API 添加注解
from typing import Any

def legacy_function(data):  # 旧代码
    return data

def new_wrapper(data: dict) -> Any:
    """逐步迁移，先包装后重构"""
    return legacy_function(data)

# 阶段3：使用类型存根文件 (.pyi)
# my_module.pyi
"""
def legacy_function(data: dict) -> list: ...
"""

# 阶段4：全面类型化
# 所有公共 API 都有完整的类型注解
```

---

## 总结

### 核心要点

1. **注解是元数据**：默认不影响运行时，但可被工具利用
2. **类型提示是约定**：使用注解表达类型信息的标准化方式
3. **渐进式采用**：可以逐步为代码库添加类型注解
4. **工具辅助**：mypy、pyright 等工具提供静态分析
5. **运行时验证**：需要时使用 pydantic 或自定义验证器

### 学习路径

1. 掌握基础注解语法（函数、变量、类）
2. 熟悉 typing 模块核心类型
3. 理解泛型和类型变量
4. 学习 Protocol 和结构化子类型
5. 探索高级特性（TypeGuard、Literal、Annotated）
6. 实践运行时类型处理

### 参考资源

- [PEP 484](https://peps.python.org/pep-0484/) - Type Hints
- [PEP 526](https://peps.python.org/pep-0526/) - Variable Annotations
- [PEP 544](https://peps.python.org/pep-0544/) - Protocols
- [PEP 586](https://peps.python.org/pep-0586/) - Literal Types
- [PEP 593](https://peps.python.org/pep-0593/) - Annotated
- [PEP 647](https://peps.python.org/pep-0647/) - TypeGuard
- [typing 官方文档](https://docs.python.org/3/library/typing.html)
- [mypy 文档](https://mypy.readthedocs.io/)

---

**最后更新：2025-12-24**
