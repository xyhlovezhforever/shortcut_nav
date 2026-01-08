# Python 装饰器深度解析

## 目录
- [核心概念](#核心概念)
- [函数装饰器基础](#函数装饰器基础)
- [带参数的装饰器](#带参数的装饰器)
- [类装饰器](#类装饰器)
- [装饰器与functools](#装饰器与functools)
- [装饰器链与执行顺序](#装饰器链与执行顺序)
- [类方法装饰器](#类方法装饰器)
- [高级装饰器模式](#高级装饰器模式)
- [装饰器实战应用](#装饰器实战应用)
- [性能优化与陷阱](#性能优化与陷阱)

---

## 核心概念

### 什么是装饰器

装饰器是 Python 中的一种设计模式，允许在不修改原函数代码的情况下，动态地增强或修改函数的行为。装饰器本质上是一个**接受函数作为参数并返回新函数的高阶函数**。

**核心特性：**
- 装饰器是可调用对象（函数或类）
- 遵循开闭原则（对扩展开放，对修改封闭）
- 使用 `@decorator` 语法糖
- 可以嵌套使用

### 装饰器的本质

```python
# 语法糖形式
@decorator
def func():
    pass

# 等价于
def func():
    pass
func = decorator(func)
```

**关键点：**
1. 装饰器在函数定义时执行（导入时）
2. 被装饰函数被替换为装饰器返回的新函数
3. 装饰器可以修改、增强或完全替换原函数

---

## 函数装饰器基础

### 1. 最简单的装饰器

```python
def simple_decorator(func):
    """最基础的装饰器"""
    def wrapper():
        print("Before function call")
        result = func()
        print("After function call")
        return result
    return wrapper

@simple_decorator
def say_hello():
    print("Hello!")
    return "Done"

# 调用
say_hello()
# 输出:
# Before function call
# Hello!
# After function call
```

### 2. 处理任意参数的装饰器

```python
def universal_decorator(func):
    """接受任意参数的装饰器"""
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"Result: {result}")
        return result
    return wrapper

@universal_decorator
def add(a, b):
    return a + b

@universal_decorator
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# 使用
add(3, 5)           # Calling add with args=(3, 5), kwargs={}
greet("Alice")      # Calling greet with args=('Alice',), kwargs={}
greet("Bob", greeting="Hi")
```

### 3. 保留元数据的装饰器

```python
from functools import wraps

def better_decorator(func):
    """使用 @wraps 保留原函数元数据"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper function"""
        return func(*args, **kwargs)
    return wrapper

@better_decorator
def documented_function():
    """This is the original docstring"""
    pass

# 查看元数据
print(documented_function.__name__)  # documented_function (而非 wrapper)
print(documented_function.__doc__)   # This is the original docstring
```

---

## 带参数的装饰器

### 1. 装饰器工厂

```python
def repeat(times):
    """接受参数的装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(times):
                result = func(*args, **kwargs)
                results.append(result)
            return results
        return wrapper
    return decorator

@repeat(times=3)
def greet(name):
    return f"Hello, {name}!"

# 调用
result = greet("Alice")
# ['Hello, Alice!', 'Hello, Alice!', 'Hello, Alice!']
```

### 2. 可选参数的装饰器

```python
from functools import wraps, partial

def smart_decorator(func=None, *, prefix=">>>"):
    """可以带参数也可以不带参数的装饰器"""
    if func is None:
        # 带参数调用: @smart_decorator(prefix="***")
        return partial(smart_decorator, prefix=prefix)

    # 不带参数调用: @smart_decorator
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"{prefix} {result}")
        return result
    return wrapper

# 不带参数使用
@smart_decorator
def add1(a, b):
    return a + b

# 带参数使用
@smart_decorator(prefix="RESULT:")
def add2(a, b):
    return a + b

add1(1, 2)  # >>> 3
add2(3, 4)  # RESULT: 7
```

### 3. 复杂参数处理

```python
def validate(*, min_value=None, max_value=None, allowed_types=None):
    """参数验证装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取函数签名
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # 验证每个参数
            for param_name, param_value in bound.arguments.items():
                # 类型验证
                if allowed_types and not isinstance(param_value, allowed_types):
                    raise TypeError(
                        f"{param_name} must be one of {allowed_types}, "
                        f"got {type(param_value)}"
                    )

                # 范围验证
                if min_value is not None and param_value < min_value:
                    raise ValueError(f"{param_name} must be >= {min_value}")

                if max_value is not None and param_value > max_value:
                    raise ValueError(f"{param_name} must be <= {max_value}")

            return func(*args, **kwargs)
        return wrapper
    return decorator

@validate(min_value=0, max_value=100, allowed_types=(int, float))
def set_volume(level):
    print(f"Volume set to {level}")

set_volume(50)    # OK
set_volume(-10)   # ValueError: level must be >= 0
set_volume("50")  # TypeError
```

---

## 类装饰器

### 1. 使用类实现装饰器

```python
class CallCounter:
    """计数装饰器 - 使用类实现"""
    def __init__(self, func):
        self.func = func
        self.count = 0
        # 保留原函数的元数据
        wraps(func)(self)

    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"Call #{self.count} to {self.func.__name__}")
        return self.func(*args, **kwargs)

    def reset(self):
        """重置计数器"""
        self.count = 0

@CallCounter
def say_hello():
    print("Hello!")

say_hello()  # Call #1 to say_hello
say_hello()  # Call #2 to say_hello
print(say_hello.count)  # 2
say_hello.reset()
```

### 2. 带参数的类装饰器

```python
class Retry:
    """重试装饰器"""
    def __init__(self, max_attempts=3, delay=1):
        self.max_attempts = max_attempts
        self.delay = delay

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time

            for attempt in range(1, self.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == self.max_attempts:
                        raise
                    print(f"Attempt {attempt} failed: {e}")
                    print(f"Retrying in {self.delay} seconds...")
                    time.sleep(self.delay)

        return wrapper

@Retry(max_attempts=3, delay=2)
def unstable_function():
    import random
    if random.random() < 0.7:
        raise ConnectionError("Network error")
    return "Success!"

# 使用
result = unstable_function()
```

### 3. 装饰类的装饰器

```python
def singleton(cls):
    """单例模式装饰器"""
    instances = {}

    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class Database:
    def __init__(self):
        print("Initializing database connection")
        self.connection = "DB Connection"

# 使用
db1 = Database()  # Initializing database connection
db2 = Database()  # 不会再次初始化
print(db1 is db2)  # True
```

### 4. 添加属性和方法到类

```python
def add_repr(cls):
    """自动添加 __repr__ 方法"""
    def __repr__(self):
        attrs = ', '.join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{cls.__name__}({attrs})"

    cls.__repr__ = __repr__
    return cls

@add_repr
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

p = Person("Alice", 30)
print(p)  # Person(name='Alice', age=30)
```

---

## 装饰器与functools

### 1. @wraps - 保留元数据

```python
from functools import wraps

def without_wraps(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def with_wraps(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def original():
    """Original docstring"""
    pass

# 比较
f1 = without_wraps(original)
print(f1.__name__)  # wrapper
print(f1.__doc__)   # None

f2 = with_wraps(original)
print(f2.__name__)  # original
print(f2.__doc__)   # Original docstring
```

### 2. @lru_cache - 缓存装饰器

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    """带缓存的斐波那契数列"""
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 性能测试
import time

start = time.time()
result = fibonacci(35)
print(f"Result: {result}, Time: {time.time() - start:.4f}s")
# 第二次调用会非常快（使用缓存）

# 查看缓存信息
print(fibonacci.cache_info())
# CacheInfo(hits=..., misses=..., maxsize=128, currsize=...)

# 清除缓存
fibonacci.cache_clear()
```

### 3. @singledispatch - 单分派泛型函数

```python
from functools import singledispatch

@singledispatch
def process(value):
    """默认处理"""
    return f"Processing {value}"

@process.register(int)
def _(value):
    return f"Integer: {value * 2}"

@process.register(str)
def _(value):
    return f"String: {value.upper()}"

@process.register(list)
def _(value):
    return f"List with {len(value)} items"

# 使用
print(process(42))           # Integer: 84
print(process("hello"))      # String: HELLO
print(process([1, 2, 3]))    # List with 3 items
print(process(3.14))         # Processing 3.14 (默认)
```

### 4. @cached_property - 缓存属性

```python
from functools import cached_property

class DataProcessor:
    def __init__(self, data):
        self.data = data

    @cached_property
    def processed_data(self):
        """计算密集型属性，只计算一次"""
        print("Processing data... (expensive operation)")
        return [x * 2 for x in self.data]

    @property
    def size(self):
        """普通属性，每次都重新计算"""
        return len(self.data)

# 使用
processor = DataProcessor([1, 2, 3, 4, 5])
print(processor.processed_data)  # Processing data...
print(processor.processed_data)  # 使用缓存，不再打印提示
```

---

## 装饰器链与执行顺序

### 1. 多个装饰器的执行顺序

```python
def decorator_a(func):
    print(f"Decorator A: Wrapping {func.__name__}")
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Decorator A: Before")
        result = func(*args, **kwargs)
        print("Decorator A: After")
        return result
    return wrapper

def decorator_b(func):
    print(f"Decorator B: Wrapping {func.__name__}")
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Decorator B: Before")
        result = func(*args, **kwargs)
        print("Decorator B: After")
        return result
    return wrapper

@decorator_a
@decorator_b
def my_function():
    print("Original function")

# 装饰时的输出（从下到上）:
# Decorator B: Wrapping my_function
# Decorator A: Wrapping wrapper

# 调用时的输出（从上到下）:
my_function()
# Decorator A: Before
# Decorator B: Before
# Original function
# Decorator B: After
# Decorator A: After
```

### 2. 装饰器顺序的影响

```python
from functools import wraps

def convert_to_int(func):
    """将返回值转换为整数"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return int(result)
    return wrapper

def double_result(func):
    """将返回值翻倍"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

# 顺序1: 先翻倍，后转整数
@convert_to_int
@double_result
def get_value1():
    return 3.7

print(get_value1())  # int(3.7 * 2) = int(7.4) = 7

# 顺序2: 先转整数，后翻倍
@double_result
@convert_to_int
def get_value2():
    return 3.7

print(get_value2())  # int(3.7) * 2 = 3 * 2 = 6
```

---

## 类方法装饰器

### 1. 内置装饰器

```python
class MyClass:
    class_var = "I'm a class variable"

    def __init__(self, value):
        self.value = value

    # 实例方法
    def instance_method(self):
        return f"Instance method: {self.value}"

    # 类方法
    @classmethod
    def class_method(cls):
        return f"Class method: {cls.class_var}"

    # 静态方法
    @staticmethod
    def static_method(x, y):
        return f"Static method: {x + y}"

    # 属性装饰器
    @property
    def formatted_value(self):
        return f"Value: {self.value}"

    @formatted_value.setter
    def formatted_value(self, new_value):
        self.value = new_value

    @formatted_value.deleter
    def formatted_value(self):
        del self.value

# 使用
obj = MyClass(42)
print(obj.instance_method())      # Instance method: 42
print(MyClass.class_method())     # Class method: I'm a class variable
print(MyClass.static_method(1, 2)) # Static method: 3
print(obj.formatted_value)        # Value: 42
obj.formatted_value = 100
```

### 2. 装饰类方法的装饰器

```python
def method_logger(func):
    """记录方法调用的装饰器"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        class_name = self.__class__.__name__
        print(f"[LOG] {class_name}.{func.__name__} called")
        result = func(self, *args, **kwargs)
        print(f"[LOG] {class_name}.{func.__name__} returned: {result}")
        return result
    return wrapper

class Calculator:
    @method_logger
    def add(self, a, b):
        return a + b

    @method_logger
    def multiply(self, a, b):
        return a * b

calc = Calculator()
calc.add(3, 5)
# [LOG] Calculator.add called
# [LOG] Calculator.add returned: 8
```

### 3. 装饰类的所有方法

```python
def log_all_methods(cls):
    """装饰类的所有方法"""
    for name, method in cls.__dict__.items():
        if callable(method) and not name.startswith('_'):
            setattr(cls, name, method_logger(method))
    return cls

@log_all_methods
class Service:
    def start(self):
        return "Service started"

    def stop(self):
        return "Service stopped"

    def _internal(self):
        return "Internal method (not logged)"

# 使用
service = Service()
service.start()  # 自动记录日志
service.stop()   # 自动记录日志
```

---

## 高级装饰器模式

### 1. 上下文管理装饰器

```python
from contextlib import contextmanager

def with_context(context_manager):
    """使用上下文管理器的装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with context_manager as resource:
                return func(resource, *args, **kwargs)
        return wrapper
    return decorator

# 使用示例
@contextmanager
def database_connection():
    print("Opening database connection")
    conn = "DB Connection"
    try:
        yield conn
    finally:
        print("Closing database connection")

@with_context(database_connection())
def query_database(conn, query):
    print(f"Executing query: {query}")
    return f"Results from {conn}"

# 调用
result = query_database("SELECT * FROM users")
```

### 2. 异步装饰器

```python
import asyncio
from functools import wraps

def async_timer(func):
    """异步函数计时装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.4f} seconds")
        return result
    return wrapper

@async_timer
async def fetch_data(url):
    await asyncio.sleep(1)  # 模拟网络请求
    return f"Data from {url}"

# 使用
asyncio.run(fetch_data("https://example.com"))
```

### 3. 条件装饰器

```python
def conditional_decorator(condition):
    """根据条件决定是否应用装饰器"""
    def decorator(func):
        if condition:
            @wraps(func)
            def wrapper(*args, **kwargs):
                print(f"Decorated: {func.__name__}")
                return func(*args, **kwargs)
            return wrapper
        else:
            return func  # 不装饰
    return decorator

DEBUG = True

@conditional_decorator(DEBUG)
def debug_function():
    print("Function body")

debug_function()
# 输出: Decorated: debug_function
#      Function body

# 如果 DEBUG = False，则只输出: Function body
```

### 4. 元类与装饰器结合

```python
def auto_register(registry):
    """自动注册类的装饰器"""
    def decorator(cls):
        registry[cls.__name__] = cls
        return cls
    return decorator

# 插件注册系统
PLUGINS = {}

@auto_register(PLUGINS)
class PluginA:
    pass

@auto_register(PLUGINS)
class PluginB:
    pass

print(PLUGINS)
# {'PluginA': <class '__main__.PluginA'>, 'PluginB': <class '__main__.PluginB'>}
```

---

## 装饰器实战应用

### 1. 权限检查装饰器

```python
from functools import wraps

class User:
    def __init__(self, username, roles):
        self.username = username
        self.roles = roles

# 全局当前用户（实际应用中应使用线程本地存储）
current_user = None

def require_role(*required_roles):
    """权限检查装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if current_user is None:
                raise PermissionError("Not authenticated")

            if not any(role in current_user.roles for role in required_roles):
                raise PermissionError(
                    f"Requires one of {required_roles}, "
                    f"but user has {current_user.roles}"
                )

            return func(*args, **kwargs)
        return wrapper
    return decorator

@require_role("admin", "moderator")
def delete_user(user_id):
    return f"User {user_id} deleted"

@require_role("user")
def view_profile():
    return "Viewing profile"

# 使用
current_user = User("alice", ["user"])
print(view_profile())  # OK

try:
    delete_user(123)  # PermissionError
except PermissionError as e:
    print(f"Error: {e}")
```

### 2. API 限流装饰器

```python
import time
from collections import defaultdict
from functools import wraps

class RateLimiter:
    """简单的限流器"""
    def __init__(self, max_calls, time_window):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = defaultdict(list)

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # 获取调用者标识（简化示例）
            caller_id = "default"

            # 清理过期的调用记录
            self.calls[caller_id] = [
                call_time for call_time in self.calls[caller_id]
                if now - call_time < self.time_window
            ]

            # 检查是否超过限制
            if len(self.calls[caller_id]) >= self.max_calls:
                raise Exception(
                    f"Rate limit exceeded: {self.max_calls} calls "
                    f"per {self.time_window} seconds"
                )

            # 记录本次调用
            self.calls[caller_id].append(now)

            return func(*args, **kwargs)
        return wrapper

@RateLimiter(max_calls=3, time_window=10)
def api_call():
    return "API response"

# 测试
for i in range(5):
    try:
        print(f"Call {i+1}: {api_call()}")
    except Exception as e:
        print(f"Call {i+1}: {e}")
    time.sleep(1)
```

### 3. 缓存装饰器（带过期时间）

```python
import time
from functools import wraps

def timed_cache(seconds=60):
    """带过期时间的缓存装饰器"""
    def decorator(func):
        cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            # 创建缓存键
            key = (args, tuple(sorted(kwargs.items())))

            # 检查缓存
            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < seconds:
                    print(f"Cache hit for {func.__name__}")
                    return result
                else:
                    print(f"Cache expired for {func.__name__}")

            # 调用函数并缓存结果
            print(f"Cache miss for {func.__name__}")
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            return result

        # 添加清除缓存的方法
        wrapper.clear_cache = lambda: cache.clear()

        return wrapper
    return decorator

@timed_cache(seconds=5)
def expensive_operation(x, y):
    time.sleep(1)  # 模拟耗时操作
    return x + y

# 测试
print(expensive_operation(1, 2))  # Cache miss
print(expensive_operation(1, 2))  # Cache hit
time.sleep(6)
print(expensive_operation(1, 2))  # Cache expired
```

### 4. 重试装饰器（指数退避）

```python
import time
from functools import wraps

def retry_with_backoff(
    max_retries=3,
    initial_delay=1,
    backoff_factor=2,
    exceptions=(Exception,)
):
    """带指数退避的重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        raise

                    print(f"Attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= backoff_factor

        return wrapper
    return decorator

@retry_with_backoff(max_retries=4, initial_delay=1, backoff_factor=2)
def unstable_api_call():
    import random
    if random.random() < 0.8:
        raise ConnectionError("API temporarily unavailable")
    return "Success!"

# 使用
result = unstable_api_call()
```

### 5. 性能分析装饰器

```python
import time
import functools
from collections import defaultdict

class PerformanceProfiler:
    """性能分析装饰器"""
    stats = defaultdict(lambda: {"calls": 0, "total_time": 0, "min_time": float('inf'), "max_time": 0})

    @classmethod
    def profile(cls, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start

            # 更新统计信息
            stats = cls.stats[func.__name__]
            stats["calls"] += 1
            stats["total_time"] += elapsed
            stats["min_time"] = min(stats["min_time"], elapsed)
            stats["max_time"] = max(stats["max_time"], elapsed)

            return result
        return wrapper

    @classmethod
    def report(cls):
        """打印性能报告"""
        print("\n=== Performance Report ===")
        for func_name, stats in cls.stats.items():
            avg_time = stats["total_time"] / stats["calls"]
            print(f"\n{func_name}:")
            print(f"  Calls: {stats['calls']}")
            print(f"  Total time: {stats['total_time']:.4f}s")
            print(f"  Avg time: {avg_time:.4f}s")
            print(f"  Min time: {stats['min_time']:.4f}s")
            print(f"  Max time: {stats['max_time']:.4f}s")

@PerformanceProfiler.profile
def process_data(n):
    time.sleep(0.01 * n)
    return n * 2

# 测试
for i in range(1, 6):
    process_data(i)

PerformanceProfiler.report()
```

### 6. 参数验证装饰器

```python
from functools import wraps
import inspect

def validate_params(**validators):
    """参数验证装饰器"""
    def decorator(func):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # 绑定参数
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # 验证每个参数
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]

                    # 执行验证函数
                    if callable(validator):
                        if not validator(value):
                            raise ValueError(
                                f"Validation failed for {param_name}={value}"
                            )
                    # 或检查类型
                    elif isinstance(validator, type):
                        if not isinstance(value, validator):
                            raise TypeError(
                                f"{param_name} must be {validator}, "
                                f"got {type(value)}"
                            )

            return func(*args, **kwargs)
        return wrapper
    return decorator

# 使用示例
@validate_params(
    age=lambda x: 0 <= x <= 150,
    name=str,
    email=lambda x: '@' in x
)
def create_user(name, age, email):
    return f"User {name} created"

# 测试
create_user("Alice", 30, "alice@example.com")  # OK
try:
    create_user("Bob", -5, "bob@example.com")  # ValueError
except ValueError as e:
    print(e)
```

---

## 性能优化与陷阱

### 1. 装饰器的性能开销

```python
import time
from functools import wraps

def simple_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def no_decorator(x):
    return x + 1

@simple_decorator
def with_decorator(x):
    return x + 1

# 性能测试
iterations = 1_000_000

start = time.perf_counter()
for i in range(iterations):
    no_decorator(i)
time_no_dec = time.perf_counter() - start

start = time.perf_counter()
for i in range(iterations):
    with_decorator(i)
time_with_dec = time.perf_counter() - start

print(f"Without decorator: {time_no_dec:.4f}s")
print(f"With decorator: {time_with_dec:.4f}s")
print(f"Overhead: {(time_with_dec/time_no_dec - 1) * 100:.2f}%")
```

### 2. 避免闭包陷阱

```python
# ❌ 错误：循环中的闭包陷阱
def create_multipliers_wrong():
    multipliers = []
    for i in range(5):
        def multiplier(x):
            return x * i  # i 是闭包变量
        multipliers.append(multiplier)
    return multipliers

funcs = create_multipliers_wrong()
print([f(2) for f in funcs])  # [8, 8, 8, 8, 8] - 全部使用最后的 i=4

# ✅ 正确：使用默认参数捕获值
def create_multipliers_correct():
    multipliers = []
    for i in range(5):
        def multiplier(x, i=i):  # 默认参数在定义时求值
            return x * i
        multipliers.append(multiplier)
    return multipliers

funcs = create_multipliers_correct()
print([f(2) for f in funcs])  # [0, 2, 4, 6, 8] - 正确
```

### 3. 装饰器与描述符的交互

```python
class Method:
    """方法描述符"""
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return lambda *args, **kwargs: self.func(instance, *args, **kwargs)

def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Decorated!")
        return func(*args, **kwargs)
    return wrapper

class MyClass:
    # ❌ 错误顺序：装饰器破坏了描述符协议
    @my_decorator
    @Method
    def method1(self):
        return "method1"

    # ✅ 正确顺序：描述符在最外层
    @Method
    @my_decorator
    def method2(self):
        return "method2"

obj = MyClass()
# obj.method1()  # 可能出错
obj.method2()    # OK
```

### 4. 内存泄漏风险

```python
from functools import wraps
import weakref

# ❌ 可能导致内存泄漏
def caching_decorator_bad(func):
    cache = {}  # 强引用，可能导致对象无法释放

    @wraps(func)
    def wrapper(obj):
        if obj not in cache:
            cache[obj] = func(obj)
        return cache[obj]
    return wrapper

# ✅ 使用弱引用避免内存泄漏
def caching_decorator_good(func):
    cache = weakref.WeakValueDictionary()

    @wraps(func)
    def wrapper(obj):
        if obj not in cache:
            cache[obj] = func(obj)
        return cache[obj]
    return wrapper
```

### 5. 装饰器的最佳实践

```python
from functools import wraps

# ✅ 好的装饰器实践
def good_decorator(func):
    """
    1. 有清晰的文档说明
    2. 使用 @wraps 保留元数据
    3. 接受任意参数
    4. 保持简单明了
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 装饰逻辑
        return func(*args, **kwargs)

    # 5. 提供实用的辅助方法
    wrapper.original = func

    return wrapper

# ❌ 避免的做法
def bad_decorator(func):
    # 1. 没有文档
    # 2. 不使用 @wraps
    def wrapper(x):  # 3. 参数固定，不够通用
        result = func(x)
        # 4. 复杂的逻辑应该提取到单独的函数
        # ... 100 行代码 ...
        return result
    return wrapper

# ✅ 装饰器应该可组合
@good_decorator
@another_decorator
def my_function():
    pass

# ✅ 装饰器应该可测试
def test_decorator():
    @good_decorator
    def sample():
        return 42

    assert sample() == 42
    assert sample.original() == 42  # 可以访问原函数
```

---

## 总结

### 核心要点

1. **装饰器本质**：接受函数并返回新函数的高阶函数
2. **语法糖**：`@decorator` 等价于 `func = decorator(func)`
3. **保留元数据**：始终使用 `@wraps(func)`
4. **通用性**：使用 `*args, **kwargs` 接受任意参数
5. **可组合性**：装饰器应该能够链式使用

### 应用场景

- **日志记录**：记录函数调用和返回值
- **性能监控**：计时、性能分析
- **缓存**：避免重复计算
- **权限控制**：验证用户权限
- **重试机制**：处理临时性失败
- **参数验证**：检查输入有效性
- **限流**：控制调用频率

### 学习路径

1. 理解闭包和高阶函数
2. 掌握基础函数装饰器
3. 学习带参数的装饰器
4. 探索类装饰器
5. 熟悉 functools 模块
6. 实践装饰器模式

### 参考资源

- [PEP 318](https://peps.python.org/pep-0318/) - Decorators for Functions and Methods
- [PEP 3129](https://peps.python.org/pep-3129/) - Class Decorators
- [functools 文档](https://docs.python.org/3/library/functools.html)
- [Python Decorator Library](https://wiki.python.org/moin/PythonDecoratorLibrary)

---

**最后更新：2025-12-24**
