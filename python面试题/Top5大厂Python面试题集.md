# Top5 大厂 Python 面试题集（2小时完整版）

## 目录
- [第一部分：Python基础与进阶（30分钟）](#第一部分python基础与进阶30分钟)
- [第二部分：数据结构与算法（25分钟）](#第二部分数据结构与算法25分钟)
- [第三部分：并发编程与异步（20分钟）](#第三部分并发编程与异步20分钟)
- [第四部分：Web开发与框架（15分钟）](#第四部分web开发与框架15分钟)
- [第五部分：系统设计与架构（20分钟）](#第五部分系统设计与架构20分钟)
- [第六部分：性能优化与最佳实践（10分钟）](#第六部分性能优化与最佳实践10分钟)

---

## 第一部分：Python基础与进阶（30分钟）

### 1. 请解释Python的GIL（全局解释器锁），它对多线程的影响是什么？如何绕过GIL的限制？

**标准答案：**

**GIL的定义：**
- GIL是CPython解释器中的一个互斥锁，确保同一时刻只有一个线程执行Python字节码
- 它是CPython内存管理的实现细节，用于保护对Python对象的访问，防止多线程并发时的竞态条件

**对多线程的影响：**
1. **CPU密集型任务**：由于GIL的存在，多线程无法利用多核CPU，性能可能还不如单线程
2. **I/O密集型任务**：多线程仍然有效，因为I/O操作会释放GIL
3. **线程切换开销**：频繁的线程切换会带来额外的性能损耗

**绕过GIL的方法：**

```python
# 方法1：使用多进程
from multiprocessing import Pool
import time

def cpu_intensive_task(n):
    """CPU密集型任务"""
    result = 0
    for i in range(n):
        result += i ** 2
    return result

if __name__ == '__main__':
    # 多进程方式，每个进程有独立的GIL
    with Pool(processes=4) as pool:
        results = pool.map(cpu_intensive_task, [10000000] * 4)
    print(sum(results))

# 方法2：使用C扩展或Cython
# example.pyx
def fast_computation(int n):
    cdef int i
    cdef long long result = 0
    with nogil:  # 释放GIL
        for i in range(n):
            result += i * i
    return result

# 方法3：使用其他Python实现
# - Jython（基于JVM，无GIL）
# - IronPython（基于.NET，无GIL）
# - PyPy（JIT编译，仍有GIL但性能更好）

# 方法4：使用ctypes调用C库
import ctypes
import threading

lib = ctypes.CDLL('./mylib.so')
lib.compute.argtypes = [ctypes.c_int]
lib.compute.restype = ctypes.c_int

# C函数执行时会释放GIL
result = lib.compute(1000000)
```

**关键点：**
- GIL只影响CPython，不是Python语言特性
- I/O密集型任务用多线程，CPU密集型任务用多进程
- 使用asyncio可以在单线程中实现高并发I/O

---

### 2. 深入解释Python的元类（Metaclass），并给出实际应用场景

**标准答案：**

**元类的概念：**
- 元类是类的类，type是所有类的默认元类
- 类是元类的实例，对象是类的实例
- 关系链：对象 -> 类 -> 元类 -> type

**基础示例：**

```python
# 类的创建过程
class MyClass:
    pass

# 等价于
MyClass = type('MyClass', (), {})

# 自定义元类
class SingletonMeta(type):
    """单例模式元类"""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.connection = "Connected"

# 测试单例
db1 = Database()
db2 = Database()
print(db1 is db2)  # True
```

**实际应用场景1：ORM框架（类似Django ORM）**

```python
class ModelMeta(type):
    """ORM模型元类"""
    def __new__(mcs, name, bases, attrs):
        # 收集字段定义
        if name == 'Model':
            return super().__new__(mcs, name, bases, attrs)

        fields = {}
        for key, value in list(attrs.items()):
            if isinstance(value, Field):
                fields[key] = value
                attrs.pop(key)

        attrs['_fields'] = fields
        attrs['_table_name'] = name.lower()
        return super().__new__(mcs, name, bases, attrs)

class Field:
    def __init__(self, field_type, **kwargs):
        self.field_type = field_type
        self.kwargs = kwargs

class Model(metaclass=ModelMeta):
    def __init__(self, **kwargs):
        for field_name in self._fields:
            setattr(self, field_name, kwargs.get(field_name))

    def save(self):
        """生成SQL插入语句"""
        fields = ', '.join(self._fields.keys())
        placeholders = ', '.join(['?' for _ in self._fields])
        values = [getattr(self, f) for f in self._fields]
        sql = f"INSERT INTO {self._table_name} ({fields}) VALUES ({placeholders})"
        return sql, values

class User(Model):
    name = Field('VARCHAR(100)')
    email = Field('VARCHAR(100)')
    age = Field('INTEGER')

# 使用
user = User(name='Alice', email='alice@example.com', age=25)
sql, values = user.save()
print(sql)  # INSERT INTO user (name, email, age) VALUES (?, ?, ?)
print(values)  # ['Alice', 'alice@example.com', 25]
```

**实际应用场景2：API参数验证**

```python
class ValidatorMeta(type):
    """自动参数验证元类"""
    def __new__(mcs, name, bases, attrs):
        # 收集验证规则
        validators = {}
        for key, value in attrs.items():
            if hasattr(value, '__validator__'):
                validators[key] = value

        # 包装__init__方法
        original_init = attrs.get('__init__', lambda self: None)

        def new_init(self, **kwargs):
            for field, validator in validators.items():
                if field in kwargs:
                    validator.validate(kwargs[field])
            original_init(self, **kwargs)

        attrs['__init__'] = new_init
        return super().__new__(mcs, name, bases, attrs)

class Validator:
    def __init__(self, validator_func):
        self.validator_func = validator_func
        self.__validator__ = True

    def validate(self, value):
        if not self.validator_func(value):
            raise ValueError(f"Validation failed for {value}")

class APIRequest(metaclass=ValidatorMeta):
    user_id = Validator(lambda x: isinstance(x, int) and x > 0)
    email = Validator(lambda x: '@' in str(x))
    age = Validator(lambda x: 0 < x < 150)

# 使用
try:
    req = APIRequest(user_id=123, email='test@test.com', age=25)  # OK
    req = APIRequest(user_id=-1, email='invalid', age=200)  # 抛出异常
except ValueError as e:
    print(e)
```

**实际应用场景3：插件系统**

```python
class PluginMeta(type):
    """插件注册元类"""
    plugins = {}

    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        if name != 'Plugin':
            plugin_name = attrs.get('name', name)
            mcs.plugins[plugin_name] = cls
        return cls

class Plugin(metaclass=PluginMeta):
    pass

class ImagePlugin(Plugin):
    name = 'image_processor'

    def process(self, data):
        return f"Processing image: {data}"

class VideoPlugin(Plugin):
    name = 'video_processor'

    def process(self, data):
        return f"Processing video: {data}"

# 动态获取插件
def get_plugin(name):
    return PluginMeta.plugins.get(name)

processor = get_plugin('image_processor')()
print(processor.process('photo.jpg'))  # Processing image: photo.jpg
```

**关键点：**
- 元类通过`__new__`方法控制类的创建过程
- 元类通过`__init__`方法初始化类
- 元类通过`__call__`方法控制实例的创建
- 元类适用于框架开发，不适合日常业务代码

---

### 3. 解释Python的描述符协议，并实现一个类型检查和缓存的描述符

**标准答案：**

**描述符协议：**
- `__get__(self, instance, owner)` - 属性访问
- `__set__(self, instance, value)` - 属性设置
- `__delete__(self, instance)` - 属性删除
- `__set_name__(self, owner, name)` - 描述符被赋值给类属性时调用

**实现示例：**

```python
from functools import wraps
from typing import Any, Callable
import time

class TypedProperty:
    """类型检查描述符"""
    def __init__(self, name: str, expected_type: type):
        self.name = name
        self.expected_type = expected_type

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)

    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(
                f"{self.name} must be {self.expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        del instance.__dict__[self.name]

class CachedProperty:
    """带缓存的计算属性描述符"""
    def __init__(self, func: Callable):
        self.func = func
        self.name = func.__name__
        self.__doc__ = func.__doc__

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self

        # 检查缓存
        cache_name = f'_cache_{self.name}'
        if not hasattr(instance, cache_name):
            # 计算并缓存结果
            value = self.func(instance)
            setattr(instance, cache_name, value)

        return getattr(instance, cache_name)

    def __set__(self, instance, value):
        raise AttributeError(f"Can't set attribute {self.name}")

class LazyProperty:
    """延迟加载描述符"""
    def __init__(self, func: Callable):
        self.func = func
        self.name = func.__name__

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self

        # 首次访问时计算，并替换为普通属性
        value = self.func(instance)
        setattr(instance, self.name, value)
        return value

class ValidatedProperty:
    """带验证的描述符"""
    def __init__(self, validator: Callable[[Any], bool], error_msg: str = "Invalid value"):
        self.validator = validator
        self.error_msg = error_msg

    def __set_name__(self, owner, name):
        self.name = f'_{name}'

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self.name, None)

    def __set__(self, instance, value):
        if not self.validator(value):
            raise ValueError(self.error_msg)
        setattr(instance, self.name, value)

# 综合使用示例
class Person:
    # 类型检查
    name = TypedProperty('name', str)
    age = TypedProperty('age', int)

    # 验证
    email = ValidatedProperty(
        lambda x: '@' in x and '.' in x,
        "Invalid email format"
    )

    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email

    @CachedProperty
    def expensive_calculation(self):
        """模拟耗时计算"""
        print("Computing expensive result...")
        time.sleep(1)
        return self.age * 1000

    @LazyProperty
    def lazy_resource(self):
        """延迟加载资源"""
        print("Loading resource...")
        return f"Resource for {self.name}"

# 测试
person = Person("Alice", 25, "alice@example.com")
print(person.name)  # Alice

# 缓存属性
print(person.expensive_calculation)  # Computing expensive result... 25000
print(person.expensive_calculation)  # 25000 (从缓存读取)

# 延迟加载
print(person.lazy_resource)  # Loading resource... Resource for Alice
print(person.lazy_resource)  # Resource for Alice (已变为普通属性)

# 类型检查
try:
    person.age = "invalid"  # TypeError
except TypeError as e:
    print(e)

# 验证
try:
    person.email = "invalid-email"  # ValueError
except ValueError as e:
    print(e)
```

**进阶：实现ORM风格的字段描述符**

```python
class Field:
    """数据库字段基类"""
    def __init__(self, db_type: str, required: bool = False, default=None):
        self.db_type = db_type
        self.required = required
        self.default = default

    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = f'_{name}'

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self.private_name, self.default)

    def __set__(self, instance, value):
        if value is None and self.required:
            raise ValueError(f"{self.name} is required")
        self.validate(value)
        setattr(instance, self.private_name, value)

    def validate(self, value):
        """子类实现具体验证逻辑"""
        pass

class IntegerField(Field):
    def __init__(self, min_value=None, max_value=None, **kwargs):
        super().__init__('INTEGER', **kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value):
        if value is None:
            return
        if not isinstance(value, int):
            raise TypeError(f"{self.name} must be an integer")
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"{self.name} must be >= {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"{self.name} must be <= {self.max_value}")

class StringField(Field):
    def __init__(self, max_length: int = 255, **kwargs):
        super().__init__(f'VARCHAR({max_length})', **kwargs)
        self.max_length = max_length

    def validate(self, value):
        if value is None:
            return
        if not isinstance(value, str):
            raise TypeError(f"{self.name} must be a string")
        if len(value) > self.max_length:
            raise ValueError(f"{self.name} exceeds max length {self.max_length}")

class User:
    id = IntegerField(min_value=1, required=True)
    username = StringField(max_length=50, required=True)
    age = IntegerField(min_value=0, max_value=150)
    email = StringField(max_length=100)

    def __init__(self, id, username, age=None, email=None):
        self.id = id
        self.username = username
        self.age = age
        self.email = email

# 测试
user = User(id=1, username="alice", age=25)
print(user.username)  # alice

try:
    user.age = 200  # ValueError: age must be <= 150
except ValueError as e:
    print(e)
```

**关键点：**
- 数据描述符（定义了`__set__`）优先级高于实例字典
- 非数据描述符（只定义`__get__`）优先级低于实例字典
- `__set_name__`在Python 3.6+可用，简化描述符初始化
- property是描述符的语法糖实现

---

### 4. 深入解释Python的内存管理机制，包括引用计数、垃圾回收和内存池

**标准答案：**

**1. 引用计数（Reference Counting）**

```python
import sys

# 查看引用计数
a = []
print(sys.getrefcount(a))  # 2（a本身 + getrefcount的参数）

b = a
print(sys.getrefcount(a))  # 3

del b
print(sys.getrefcount(a))  # 2

# 引用计数的问题：循环引用
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

node1 = Node(1)
node2 = Node(2)
node1.next = node2
node2.next = node1  # 循环引用

# 即使删除node1和node2，它们仍相互引用，引用计数不为0
del node1, node2  # 需要垃圾回收器处理
```

**2. 垃圾回收（Garbage Collection）**

```python
import gc

# 查看垃圾回收器配置
print(gc.get_threshold())  # (700, 10, 10)
# 700: 当分配对象数 - 释放对象数 > 700时触发0代回收
# 10: 0代回收10次后触发1代回收
# 10: 1代回收10次后触发2代回收

# 手动触发垃圾回收
gc.collect()

# 查看垃圾回收统计
print(gc.get_count())  # (当前0代对象数, 当前1代对象数, 当前2代对象数)

# 查看被垃圾回收的对象
gc.set_debug(gc.DEBUG_SAVEALL)

class Cyclic:
    def __init__(self, name):
        self.name = name
        self.ref = None

    def __del__(self):
        print(f"Deleting {self.name}")

a = Cyclic("A")
b = Cyclic("B")
a.ref = b
b.ref = a

del a, b
print("After del")

# 手动触发垃圾回收
collected = gc.collect()
print(f"Collected {collected} objects")
print(f"Garbage: {gc.garbage}")

# 禁用/启用垃圾回收
gc.disable()
# ... 执行代码 ...
gc.enable()
```

**3. 内存池机制（Memory Pool）**

```python
# Python的小对象内存池（PyMalloc）
# 管理小于512字节的对象

# Arena（256KB）
# ├── Pool（4KB）
# │   ├── Block（固定大小）
# │   ├── Block
# │   └── ...
# └── Pool
#     └── ...

# 演示内存复用
import sys

# 小整数对象池（-5 到 256）
a = 256
b = 256
print(a is b)  # True，同一个对象

a = 257
b = 257
print(a is b)  # False，不在小整数池范围

# 字符串驻留（String Interning）
s1 = "hello"
s2 = "hello"
print(s1 is s2)  # True

s1 = "hello world"
s2 = "hello world"
print(s1 is s2)  # False（包含空格，不自动驻留）

# 手动驻留
import sys
s1 = sys.intern("hello world")
s2 = sys.intern("hello world")
print(s1 is s2)  # True
```

**4. 内存优化实践**

```python
# 使用__slots__减少内存占用
class Point:
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = x
        self.y = y

class PointWithDict:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 比较内存占用
import sys
p1 = Point(1, 2)
p2 = PointWithDict(1, 2)

print(sys.getsizeof(p1))  # 更小
print(sys.getsizeof(p2))  # 更大（包含__dict__）

# 使用生成器节省内存
def read_large_file(file_path):
    """使用生成器读取大文件"""
    with open(file_path, 'r') as f:
        for line in f:
            yield line.strip()

# 而不是
def read_large_file_bad(file_path):
    """一次性加载所有内容到内存"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

# 弱引用避免循环引用
import weakref

class CacheNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        # 使用弱引用避免缓存对象无法被回收
        self.cache = weakref.WeakValueDictionary()

    def get(self, key):
        return self.cache.get(key)

    def put(self, key, value):
        self.cache[key] = value

# 内存分析工具
from memory_profiler import profile

@profile
def memory_intensive_function():
    """使用memory_profiler分析内存"""
    large_list = [i for i in range(1000000)]
    large_dict = {i: i**2 for i in range(1000000)}
    return len(large_list) + len(large_dict)

# 使用tracemalloc跟踪内存
import tracemalloc

tracemalloc.start()

# 执行代码
data = [i for i in range(1000000)]

# 获取内存快照
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("[ Top 10 memory allocations ]")
for stat in top_stats[:10]:
    print(stat)

tracemalloc.stop()
```

**5. 内存泄漏检测和预防**

```python
import weakref
import gc

# 常见内存泄漏场景1：全局变量
global_cache = {}  # 永远不会被回收

def add_to_cache(key, value):
    global_cache[key] = value  # 内存泄漏

# 解决方案：使用弱引用或定期清理
weak_cache = weakref.WeakValueDictionary()

# 常见内存泄漏场景2：闭包
def create_leak():
    large_data = [i for i in range(1000000)]

    def inner():
        return len(large_data)  # large_data被闭包引用

    return inner

leak = create_leak()  # large_data永远不会被回收

# 解决方案：显式删除或使用弱引用
def create_no_leak():
    large_data = [i for i in range(1000000)]
    result = len(large_data)

    def inner():
        return result  # 只保存结果，不保存large_data

    return inner

# 常见内存泄漏场景3：循环引用
class Parent:
    def __init__(self):
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        child.parent = self  # 循环引用

class Child:
    def __init__(self):
        self.parent = None

# 解决方案：使用弱引用
class ChildFixed:
    def __init__(self):
        self._parent = None

    @property
    def parent(self):
        return self._parent() if self._parent else None

    @parent.setter
    def parent(self, value):
        self._parent = weakref.ref(value) if value else None
```

**关键点：**
- 引用计数是主要机制，垃圾回收处理循环引用
- 小对象使用内存池，大对象直接使用系统malloc
- 使用`__slots__`、生成器、弱引用等优化内存
- 使用tracemalloc和memory_profiler进行内存分析

---

### 5. 解释Python的装饰器原理，实现一个支持参数、类方法和异步函数的通用装饰器

**标准答案：**

**装饰器本质：**
- 装饰器是一个接收函数作为参数并返回函数的可调用对象
- `@decorator`是语法糖，等价于`func = decorator(func)`

**完整实现：**

```python
import functools
import time
import asyncio
from typing import Callable, Any
import inspect

class UniversalDecorator:
    """支持同步/异步函数、实例方法、类方法的通用装饰器"""

    def __init__(self, prefix: str = ""):
        """装饰器可以接收参数"""
        self.prefix = prefix
        self.call_count = 0

    def __call__(self, func: Callable) -> Callable:
        """装饰器的核心逻辑"""

        # 处理异步函数
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                self.call_count += 1
                print(f"{self.prefix}[Async Call #{self.call_count}] {func.__name__}")
                start = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    elapsed = time.time() - start
                    print(f"{self.prefix}[Async] {func.__name__} took {elapsed:.4f}s")

            return async_wrapper

        # 处理同步函数
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            self.call_count += 1
            print(f"{self.prefix}[Call #{self.call_count}] {func.__name__}")
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start
                print(f"{self.prefix}{func.__name__} took {elapsed:.4f}s")

        return sync_wrapper

    def __get__(self, instance, owner):
        """支持作为方法装饰器（描述符协议）"""
        if instance is None:
            return self
        return functools.partial(self.__call__, instance)

# 常用装饰器实现集合

def retry(max_attempts: int = 3, delay: float = 1.0,
          exceptions: tuple = (Exception,)):
    """重试装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed: {e}, retrying...")
                    await asyncio.sleep(delay)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(delay)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def cache_result(ttl: int = 60):
    """缓存装饰器（支持TTL）"""
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_times = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            key = (args, tuple(sorted(kwargs.items())))

            # 检查缓存是否过期
            if key in cache:
                if time.time() - cache_times[key] < ttl:
                    print(f"Cache hit for {func.__name__}")
                    return cache[key]
                else:
                    del cache[key]
                    del cache_times[key]

            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache[key] = result
            cache_times[key] = time.time()
            return result

        # 添加清除缓存的方法
        wrapper.clear_cache = lambda: (cache.clear(), cache_times.clear())
        return wrapper

    return decorator

def validate_types(**type_hints):
    """类型验证装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取函数签名
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # 验证类型
            for param_name, expected_type in type_hints.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"Parameter '{param_name}' must be {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )

            return func(*args, **kwargs)

        return wrapper

    return decorator

def rate_limit(calls: int, period: float):
    """速率限制装饰器"""
    def decorator(func: Callable) -> Callable:
        timestamps = []

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal timestamps
            now = time.time()

            # 清理过期的时间戳
            timestamps = [ts for ts in timestamps if now - ts < period]

            # 检查是否超过速率限制
            if len(timestamps) >= calls:
                wait_time = period - (now - timestamps[0])
                raise RuntimeError(
                    f"Rate limit exceeded. Wait {wait_time:.2f}s"
                )

            timestamps.append(now)
            return func(*args, **kwargs)

        return wrapper

    return decorator

def log_calls(logger=None, level='INFO'):
    """日志装饰器"""
    import logging

    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)

        log_method = getattr(logger, level.lower())

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)

            log_method(f"Calling {func.__name__}({signature})")
            try:
                result = func(*args, **kwargs)
                log_method(f"{func.__name__} returned {result!r}")
                return result
            except Exception as e:
                logger.exception(f"{func.__name__} raised {e.__class__.__name__}")
                raise

        return wrapper

    return decorator

# 装饰器链式使用示例
class APIClient:
    @UniversalDecorator(prefix="[API] ")
    @retry(max_attempts=3, delay=1.0)
    @cache_result(ttl=30)
    @validate_types(user_id=int, page=int)
    @rate_limit(calls=10, period=60)
    def get_user_data(self, user_id: int, page: int = 1):
        """获取用户数据"""
        print(f"Fetching user {user_id}, page {page}")
        return {"user_id": user_id, "page": page, "data": "..."}

    @UniversalDecorator(prefix="[Async API] ")
    @retry(max_attempts=3, delay=1.0)
    async def fetch_async(self, url: str):
        """异步获取数据"""
        await asyncio.sleep(0.5)
        return f"Data from {url}"

# 测试
async def test_decorators():
    client = APIClient()

    # 测试同步方法
    result1 = client.get_user_data(123, page=1)
    result2 = client.get_user_data(123, page=1)  # 从缓存获取

    # 测试异步方法
    result3 = await client.fetch_async("https://api.example.com")

    print(result1, result2, result3)

# 运行测试
# asyncio.run(test_decorators())

# 类装饰器
def singleton(cls):
    """单例装饰器"""
    instances = {}

    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper

@singleton
class Database:
    def __init__(self):
        self.connection = "Connected"

# 参数化类装饰器
def add_methods(**methods):
    """动态添加方法的类装饰器"""
    def decorator(cls):
        for method_name, method_func in methods.items():
            setattr(cls, method_name, method_func)
        return cls
    return decorator

@add_methods(
    greet=lambda self: f"Hello, {self.name}",
    shout=lambda self: f"HEY {self.name.upper()}!"
)
class Person:
    def __init__(self, name):
        self.name = name

person = Person("Alice")
print(person.greet())  # Hello, Alice
print(person.shout())  # HEY ALICE!
```

**关键点：**
- 使用`functools.wraps`保留原函数元信息
- 区分装饰器、装饰器工厂和参数化装饰器
- 处理异步函数需要检查`asyncio.iscoroutinefunction`
- 装饰器可以实现为函数、类或描述符
- 装饰器顺序很重要：从下到上应用

---

## 第二部分：数据结构与算法（25分钟）

### 6. 实现一个LRU缓存，要求O(1)时间复杂度的get和put操作

**标准答案：**

```python
from collections import OrderedDict
from typing import Any, Optional

class LRUCache:
    """
    LRU (Least Recently Used) 缓存实现
    使用有序字典实现O(1)的get和put操作
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: Any) -> Optional[Any]:
        """
        获取键对应的值，如果存在则移到末尾（最近使用）
        时间复杂度：O(1)
        """
        if key not in self.cache:
            return None

        # 移动到末尾表示最近使用
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: Any, value: Any) -> None:
        """
        设置键值对，如果超过容量则删除最久未使用的项
        时间复杂度：O(1)
        """
        if key in self.cache:
            # 更新现有键并移到末尾
            self.cache.move_to_end(key)
        else:
            # 检查容量
            if len(self.cache) >= self.capacity:
                # 删除最旧的项（第一个）
                self.cache.popitem(last=False)

        self.cache[key] = value

    def __repr__(self):
        return f"LRUCache({dict(self.cache)})"

# 方法2：使用双向链表 + 哈希表手动实现
class DLinkedNode:
    """双向链表节点"""
    def __init__(self, key: Any = None, value: Any = None):
        self.key = key
        self.value = value
        self.prev: Optional[DLinkedNode] = None
        self.next: Optional[DLinkedNode] = None

class LRUCacheManual:
    """
    手动实现LRU缓存（不使用OrderedDict）
    更好地展示底层原理
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> node

        # 虚拟头尾节点
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_node(self, node: DLinkedNode) -> None:
        """在头部后面添加节点"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: DLinkedNode) -> None:
        """移除节点"""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _move_to_head(self, node: DLinkedNode) -> None:
        """移动节点到头部"""
        self._remove_node(node)
        self._add_node(node)

    def _pop_tail(self) -> DLinkedNode:
        """弹出尾部节点"""
        node = self.tail.prev
        self._remove_node(node)
        return node

    def get(self, key: Any) -> Optional[Any]:
        """O(1) 获取操作"""
        if key not in self.cache:
            return None

        node = self.cache[key]
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any) -> None:
        """O(1) 设置操作"""
        if key in self.cache:
            # 更新现有节点
            node = self.cache[key]
            node.value = value
            self._move_to_head(node)
        else:
            # 创建新节点
            new_node = DLinkedNode(key, value)
            self.cache[key] = new_node
            self._add_node(new_node)

            # 检查容量
            if len(self.cache) > self.capacity:
                # 移除最久未使用的节点
                tail_node = self._pop_tail()
                del self.cache[tail_node.key]

    def __repr__(self):
        items = []
        current = self.head.next
        while current != self.tail:
            items.append(f"{current.key}: {current.value}")
            current = current.next
        return f"LRUCache([{', '.join(items)}])"

# 方法3：支持过期时间的LRU缓存
import time
from typing import Tuple

class LRUCacheWithTTL:
    """带过期时间的LRU缓存"""

    def __init__(self, capacity: int, default_ttl: float = 60):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache = OrderedDict()  # key -> (value, expire_time)

    def _is_expired(self, expire_time: float) -> bool:
        """检查是否过期"""
        return time.time() > expire_time

    def _clean_expired(self) -> None:
        """清理所有过期项"""
        expired_keys = [
            key for key, (_, expire_time) in self.cache.items()
            if self._is_expired(expire_time)
        ]
        for key in expired_keys:
            del self.cache[key]

    def get(self, key: Any) -> Optional[Any]:
        """获取值，如果过期则返回None"""
        if key not in self.cache:
            return None

        value, expire_time = self.cache[key]

        # 检查是否过期
        if self._is_expired(expire_time):
            del self.cache[key]
            return None

        # 移到末尾
        self.cache.move_to_end(key)
        return value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """设置值和过期时间"""
        ttl = ttl if ttl is not None else self.default_ttl
        expire_time = time.time() + ttl

        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            # 清理过期项
            self._clean_expired()

            # 检查容量
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)

        self.cache[key] = (value, expire_time)

# 测试所有实现
def test_lru_cache():
    print("=== 测试基础LRU缓存 ===")
    cache = LRUCache(3)

    cache.put(1, "one")
    cache.put(2, "two")
    cache.put(3, "three")
    print(cache)  # LRUCache({1: 'one', 2: 'two', 3: 'three'})

    print(cache.get(1))  # "one"
    print(cache)  # 1移到末尾

    cache.put(4, "four")  # 删除2
    print(cache)  # LRUCache({3: 'three', 1: 'one', 4: 'four'})

    print(cache.get(2))  # None

    print("\n=== 测试手动实现LRU缓存 ===")
    manual_cache = LRUCacheManual(3)
    manual_cache.put(1, "one")
    manual_cache.put(2, "two")
    manual_cache.put(3, "three")
    print(manual_cache)

    print(manual_cache.get(1))  # "one"
    manual_cache.put(4, "four")
    print(manual_cache)

    print("\n=== 测试带TTL的LRU缓存 ===")
    ttl_cache = LRUCacheWithTTL(3, default_ttl=2)
    ttl_cache.put("a", 1, ttl=1)  # 1秒后过期
    ttl_cache.put("b", 2, ttl=3)  # 3秒后过期

    print(ttl_cache.get("a"))  # 1
    time.sleep(1.5)
    print(ttl_cache.get("a"))  # None (已过期)
    print(ttl_cache.get("b"))  # 2 (未过期)

# test_lru_cache()

# 性能测试
import random

def benchmark_lru():
    """性能对比测试"""
    import timeit

    def test_ordered_dict():
        cache = LRUCache(1000)
        for i in range(10000):
            cache.put(i, i * 2)
            if i % 3 == 0:
                cache.get(i // 2)

    def test_manual():
        cache = LRUCacheManual(1000)
        for i in range(10000):
            cache.put(i, i * 2)
            if i % 3 == 0:
                cache.get(i // 2)

    time1 = timeit.timeit(test_ordered_dict, number=10)
    time2 = timeit.timeit(test_manual, number=10)

    print(f"OrderedDict实现: {time1:.4f}s")
    print(f"手动实现: {time2:.4f}s")

# benchmark_lru()
```

**关键点：**
- OrderedDict实现最简洁，性能也很好
- 手动实现可以更深入理解底层原理
- 双向链表保证O(1)的插入和删除
- 哈希表保证O(1)的查找
- 可以扩展支持TTL、容量动态调整等特性

---

### 7. 实现一个支持通配符的字典树（Trie），用于高效的字符串匹配

**标准答案：**

```python
from typing import List, Dict, Optional, Set
from collections import defaultdict

class TrieNode:
    """字典树节点"""
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_end_of_word = False
        self.word: Optional[str] = None  # 存储完整单词

class Trie:
    """
    字典树（Trie）实现
    支持插入、搜索、前缀匹配和通配符搜索
    """

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """
        插入单词
        时间复杂度：O(m)，m是单词长度
        """
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        node.is_end_of_word = True
        node.word = word

    def search(self, word: str) -> bool:
        """
        搜索完整单词
        时间复杂度：O(m)
        """
        node = self._search_prefix(word)
        return node is not None and node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        """
        检查是否有以prefix开头的单词
        时间复杂度：O(m)
        """
        return self._search_prefix(prefix) is not None

    def _search_prefix(self, prefix: str) -> Optional[TrieNode]:
        """辅助方法：搜索前缀"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def search_with_wildcard(self, pattern: str) -> List[str]:
        """
        支持通配符的搜索
        '.' 匹配任意单个字符
        '*' 匹配任意数量的任意字符

        时间复杂度：最坏O(n)，n是树中的节点总数
        """
        results = []
        self._search_wildcard_helper(self.root, pattern, 0, results)
        return results

    def _search_wildcard_helper(self, node: TrieNode, pattern: str,
                                 index: int, results: List[str]) -> None:
        """通配符搜索辅助方法（DFS）"""
        if index == len(pattern):
            if node.is_end_of_word:
                results.append(node.word)
            return

        char = pattern[index]

        if char == '.':
            # '.' 匹配任意单个字符
            for child in node.children.values():
                self._search_wildcard_helper(child, pattern, index + 1, results)

        elif char == '*':
            # '*' 匹配任意数量的任意字符
            # 情况1：* 匹配0个字符
            self._search_wildcard_helper(node, pattern, index + 1, results)

            # 情况2：* 匹配1个或多个字符
            for child in node.children.values():
                self._search_wildcard_helper(child, pattern, index, results)

        else:
            # 普通字符
            if char in node.children:
                self._search_wildcard_helper(
                    node.children[char], pattern, index + 1, results
                )

    def find_all_with_prefix(self, prefix: str) -> List[str]:
        """
        找到所有以prefix开头的单词
        时间复杂度：O(p + n)，p是前缀长度，n是匹配节点数
        """
        results = []
        node = self._search_prefix(prefix)

        if node is None:
            return results

        # DFS收集所有单词
        self._collect_words(node, results)
        return results

    def _collect_words(self, node: TrieNode, results: List[str]) -> None:
        """DFS收集所有单词"""
        if node.is_end_of_word:
            results.append(node.word)

        for child in node.children.values():
            self._collect_words(child, results)

    def delete(self, word: str) -> bool:
        """
        删除单词
        时间复杂度：O(m)
        """
        return self._delete_helper(self.root, word, 0)

    def _delete_helper(self, node: TrieNode, word: str, index: int) -> bool:
        """删除辅助方法（递归）"""
        if index == len(word):
            if not node.is_end_of_word:
                return False

            node.is_end_of_word = False
            node.word = None

            # 如果没有子节点，可以删除此节点
            return len(node.children) == 0

        char = word[index]
        if char not in node.children:
            return False

        child = node.children[char]
        should_delete_child = self._delete_helper(child, word, index + 1)

        if should_delete_child:
            del node.children[char]
            # 当前节点是否应该被删除
            return not node.is_end_of_word and len(node.children) == 0

        return False

    def longest_common_prefix(self) -> str:
        """
        找到所有单词的最长公共前缀
        时间复杂度：O(m)，m是最短单词的长度
        """
        prefix = []
        node = self.root

        while len(node.children) == 1 and not node.is_end_of_word:
            char = next(iter(node.children))
            prefix.append(char)
            node = node.children[char]

        return ''.join(prefix)

# 进阶：支持权重的Trie（用于自动补全）
class WeightedTrieNode:
    """带权重的Trie节点"""
    def __init__(self):
        self.children: Dict[str, WeightedTrieNode] = {}
        self.is_end = False
        self.word: Optional[str] = None
        self.weight = 0  # 权重（如搜索频率）

class AutocompleteTrie:
    """
    自动补全Trie
    支持按权重排序的搜索建议
    """

    def __init__(self):
        self.root = WeightedTrieNode()

    def insert(self, word: str, weight: int = 1) -> None:
        """插入单词及其权重"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = WeightedTrieNode()
            node = node.children[char]

        node.is_end = True
        node.word = word
        node.weight += weight  # 累加权重

    def get_suggestions(self, prefix: str, top_k: int = 10) -> List[tuple]:
        """
        获取自动补全建议
        返回：[(word, weight), ...]，按权重降序
        """
        node = self.root

        # 找到前缀节点
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        # 收集所有候选词
        candidates = []
        self._collect_weighted_words(node, candidates)

        # 按权重排序并返回top_k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def _collect_weighted_words(self, node: WeightedTrieNode,
                                 results: List[tuple]) -> None:
        """收集带权重的单词"""
        if node.is_end:
            results.append((node.word, node.weight))

        for child in node.children.values():
            self._collect_weighted_words(child, results)

# 实际应用：文件路径匹配
class FileSystemTrie:
    """
    文件系统路径Trie
    支持路径通配符匹配
    """

    def __init__(self):
        self.root = TrieNode()

    def add_path(self, path: str) -> None:
        """添加文件路径"""
        parts = path.strip('/').split('/')
        node = self.root

        for part in parts:
            if part not in node.children:
                node.children[part] = TrieNode()
            node = node.children[part]

        node.is_end_of_word = True
        node.word = path

    def find_paths(self, pattern: str) -> List[str]:
        """
        查找匹配模式的所有路径
        支持 * 和 ** 通配符
        * 匹配单个目录名
        ** 匹配任意深度的目录
        """
        parts = pattern.strip('/').split('/')
        results = []
        self._find_paths_helper(self.root, parts, 0, [], results)
        return results

    def _find_paths_helper(self, node: TrieNode, parts: List[str],
                           index: int, current_path: List[str],
                           results: List[str]) -> None:
        """路径匹配辅助方法"""
        if index == len(parts):
            if node.is_end_of_word:
                results.append('/'.join(current_path))
            return

        part = parts[index]

        if part == '**':
            # ** 匹配任意深度
            # 情况1：** 匹配0层
            self._find_paths_helper(node, parts, index + 1, current_path, results)

            # 情况2：** 匹配1层或多层
            for child_name, child_node in node.children.items():
                self._find_paths_helper(
                    child_node, parts, index,
                    current_path + [child_name], results
                )

        elif part == '*':
            # * 匹配单层任意目录名
            for child_name, child_node in node.children.items():
                self._find_paths_helper(
                    child_node, parts, index + 1,
                    current_path + [child_name], results
                )

        else:
            # 精确匹配
            if part in node.children:
                self._find_paths_helper(
                    node.children[part], parts, index + 1,
                    current_path + [part], results
                )

# 测试所有实现
def test_trie():
    print("=== 测试基础Trie ===")
    trie = Trie()

    words = ["apple", "app", "apricot", "banana", "band", "bandana"]
    for word in words:
        trie.insert(word)

    print(f"搜索 'app': {trie.search('app')}")  # True
    print(f"搜索 'appl': {trie.search('appl')}")  # False
    print(f"前缀 'app': {trie.starts_with('app')}")  # True
    print(f"前缀 'ap' 的所有单词: {trie.find_all_with_prefix('ap')}")

    print(f"\n通配符搜索 'a.p': {trie.search_with_wildcard('a.p')}")
    print(f"通配符搜索 'a*e': {trie.search_with_wildcard('a*e')}")
    print(f"通配符搜索 'ban*': {trie.search_with_wildcard('ban*')}")

    print(f"\n删除 'app': {trie.delete('app')}")
    print(f"删除后搜索 'app': {trie.search('app')}")  # False
    print(f"删除后搜索 'apple': {trie.search('apple')}")  # True (仍存在)

    print("\n=== 测试自动补全Trie ===")
    autocomplete = AutocompleteTrie()

    # 模拟搜索日志
    searches = [
        ("python", 100),
        ("python tutorial", 80),
        ("python pandas", 60),
        ("java", 90),
        ("javascript", 85),
        ("java spring", 70),
    ]

    for word, weight in searches:
        autocomplete.insert(word, weight)

    print(f"输入 'py' 的建议: {autocomplete.get_suggestions('py', 3)}")
    print(f"输入 'java' 的建议: {autocomplete.get_suggestions('java', 3)}")

    print("\n=== 测试文件系统Trie ===")
    fs_trie = FileSystemTrie()

    paths = [
        "/home/user/documents/file1.txt",
        "/home/user/documents/file2.txt",
        "/home/user/photos/vacation.jpg",
        "/home/admin/config.yml",
        "/var/log/system.log",
    ]

    for path in paths:
        fs_trie.add_path(path)

    print(f"匹配 '/home/*/documents/*': ")
    print(fs_trie.find_paths('/home/*/documents/*'))

    print(f"\n匹配 '/home/**/file*': ")
    print(fs_trie.find_paths('/home/**/file*'))

# test_trie()
```

**关键点：**
- Trie的核心优势是前缀查询性能优秀
- 通配符搜索使用DFS递归实现
- 可以扩展支持权重、删除、路径匹配等功能
- 空间换时间的典型数据结构

---

### 8. 实现一个线程安全的生产者-消费者队列，支持优先级和超时

**标准答案：**

```python
import threading
import heapq
import time
from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class QueueEmpty(Exception):
    """队列为空异常"""
    pass

class QueueFull(Exception):
    """队列已满异常"""
    pass

@dataclass(order=True)
class PriorityItem:
    """优先级队列项"""
    priority: int
    timestamp: float = field(compare=True)
    item: Any = field(compare=False)

    def __init__(self, priority: int, item: Any):
        self.priority = priority
        self.timestamp = time.time()
        self.item = item

class ProducerConsumerQueue:
    """
    线程安全的生产者-消费者队列
    支持优先级、超时、容量限制
    """

    def __init__(self, maxsize: int = 0, enable_priority: bool = False):
        """
        初始化队列
        :param maxsize: 最大容量，0表示无限制
        :param enable_priority: 是否启用优先级队列
        """
        self.maxsize = maxsize
        self.enable_priority = enable_priority

        # 队列存储
        if enable_priority:
            self._queue = []  # 使用堆实现优先级队列
        else:
            from collections import deque
            self._queue = deque()

        # 线程同步原语
        self._mutex = threading.Lock()  # 保护队列的互斥锁
        self._not_empty = threading.Condition(self._mutex)  # 队列非空条件变量
        self._not_full = threading.Condition(self._mutex)  # 队列非满条件变量

        # 统计信息
        self._unfinished_tasks = 0
        self._all_tasks_done = threading.Condition(self._mutex)

    def qsize(self) -> int:
        """返回队列大小"""
        with self._mutex:
            return len(self._queue)

    def empty(self) -> bool:
        """检查队列是否为空"""
        with self._mutex:
            return len(self._queue) == 0

    def full(self) -> bool:
        """检查队列是否已满"""
        with self._mutex:
            return self.maxsize > 0 and len(self._queue) >= self.maxsize

    def put(self, item: Any, priority: int = 0, block: bool = True,
            timeout: Optional[float] = None) -> None:
        """
        放入元素
        :param item: 要放入的元素
        :param priority: 优先级（数字越小优先级越高）
        :param block: 是否阻塞
        :param timeout: 超时时间（秒）
        """
        with self._not_full:
            # 检查队列是否已满
            if self.maxsize > 0:
                if not block:
                    if len(self._queue) >= self.maxsize:
                        raise QueueFull("Queue is full")
                elif timeout is None:
                    while len(self._queue) >= self.maxsize:
                        self._not_full.wait()
                elif timeout < 0:
                    raise ValueError("'timeout' must be a non-negative number")
                else:
                    endtime = time.time() + timeout
                    while len(self._queue) >= self.maxsize:
                        remaining = endtime - time.time()
                        if remaining <= 0.0:
                            raise QueueFull("Queue is full (timeout)")
                        self._not_full.wait(remaining)

            # 放入元素
            if self.enable_priority:
                heapq.heappush(self._queue, PriorityItem(priority, item))
            else:
                self._queue.append(item)

            self._unfinished_tasks += 1
            self._not_empty.notify()

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        """
        获取元素
        :param block: 是否阻塞
        :param timeout: 超时时间（秒）
        :return: 队列元素
        """
        with self._not_empty:
            # 等待队列非空
            if not block:
                if len(self._queue) == 0:
                    raise QueueEmpty("Queue is empty")
            elif timeout is None:
                while len(self._queue) == 0:
                    self._not_empty.wait()
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                endtime = time.time() + timeout
                while len(self._queue) == 0:
                    remaining = endtime - time.time()
                    if remaining <= 0.0:
                        raise QueueEmpty("Queue is empty (timeout)")
                    self._not_empty.wait(remaining)

            # 获取元素
            if self.enable_priority:
                priority_item = heapq.heappop(self._queue)
                item = priority_item.item
            else:
                item = self._queue.popleft()

            self._not_full.notify()
            return item

    def put_nowait(self, item: Any, priority: int = 0) -> None:
        """非阻塞放入"""
        return self.put(item, priority, block=False)

    def get_nowait(self) -> Any:
        """非阻塞获取"""
        return self.get(block=False)

    def task_done(self) -> None:
        """
        标记任务完成
        消费者在处理完从队列获取的任务后调用
        """
        with self._all_tasks_done:
            unfinished = self._unfinished_tasks - 1
            if unfinished <= 0:
                if unfinished < 0:
                    raise ValueError('task_done() called too many times')
                self._all_tasks_done.notify_all()
            self._unfinished_tasks = unfinished

    def join(self) -> None:
        """
        阻塞直到所有任务完成
        """
        with self._all_tasks_done:
            while self._unfinished_tasks:
                self._all_tasks_done.wait()

# 测试示例
def test_producer_consumer():
    """测试生产者-消费者模式"""

    # 创建优先级队列
    queue = ProducerConsumerQueue(maxsize=10, enable_priority=True)

    # 生产者线程
    def producer(name: str, items: list):
        for item, priority in items:
            time.sleep(0.1)  # 模拟生产时间
            queue.put(item, priority)
            print(f"{name} produced: {item} (priority: {priority})")

    # 消费者线程
    def consumer(name: str):
        while True:
            try:
                item = queue.get(timeout=2)
                print(f"{name} consumed: {item}")
                time.sleep(0.2)  # 模拟处理时间
                queue.task_done()
            except QueueEmpty:
                print(f"{name} timeout, exiting")
                break

    # 创建生产者和消费者
    producers = [
        threading.Thread(target=producer, args=("Producer-1", [("A", 2), ("B", 1), ("C", 3)])),
        threading.Thread(target=producer, args=("Producer-2", [("D", 1), ("E", 2), ("F", 1)])),
    ]

    consumers = [
        threading.Thread(target=consumer, args=("Consumer-1",)),
        threading.Thread(target=consumer, args=("Consumer-2",)),
    ]

    # 启动所有线程
    for p in producers:
        p.start()
    for c in consumers:
        c.start()

    # 等待所有生产者完成
    for p in producers:
        p.join()

    # 等待所有任务完成
    queue.join()
    print("All tasks completed")

    # 等待所有消费者退出
    for c in consumers:
        c.join()

# test_producer_consumer()

# 进阶：批量处理队列
class BatchProcessingQueue(ProducerConsumerQueue):
    """
    批量处理队列
    当积累了足够的元素或超时时，批量返回
    """

    def get_batch(self, batch_size: int = 10, timeout: float = 1.0) -> list:
        """
        批量获取元素
        :param batch_size: 批次大小
        :param timeout: 超时时间
        :return: 元素列表
        """
        batch = []
        endtime = time.time() + timeout

        while len(batch) < batch_size:
            remaining = endtime - time.time()
            if remaining <= 0:
                break

            try:
                item = self.get(block=True, timeout=remaining)
                batch.append(item)
            except QueueEmpty:
                break

        return batch

# 进阶：带回调的队列
class CallbackQueue(ProducerConsumerQueue):
    """
    支持回调的队列
    当队列大小达到阈值时触发回调
    """

    def __init__(self, maxsize: int = 0, on_high_watermark: callable = None,
                 high_watermark: float = 0.8):
        super().__init__(maxsize)
        self.on_high_watermark = on_high_watermark
        self.high_watermark = high_watermark
        self._high_watermark_notified = False

    def put(self, item: Any, priority: int = 0, block: bool = True,
            timeout: Optional[float] = None) -> None:
        super().put(item, priority, block, timeout)

        # 检查高水位
        if self.maxsize > 0:
            current_usage = len(self._queue) / self.maxsize
            if current_usage >= self.high_watermark and not self._high_watermark_notified:
                self._high_watermark_notified = True
                if self.on_high_watermark:
                    threading.Thread(
                        target=self.on_high_watermark,
                        args=(current_usage,)
                    ).start()

        # 检查是否回落到安全水位
        if self._high_watermark_notified:
            current_usage = len(self._queue) / self.maxsize
            if current_usage < self.high_watermark * 0.7:  # 回落到70%
                self._high_watermark_notified = False
```

**关键点：**
- 使用`threading.Lock`保护共享数据
- 使用`threading.Condition`实现等待/通知机制
- 优先级队列使用heapq实现
- 支持阻塞/非阻塞、超时等多种模式
- `task_done()`和`join()`实现任务完成同步

---

## 第三部分：并发编程与异步（20分钟）

### 9. 解释Python的协程（Coroutine）和异步编程，实现一个异步爬虫框架

**标准答案：**

**协程的概念：**
- 协程是可以暂停和恢复的函数
- 使用`async def`定义，用`await`暂停
- 在单线程中实现并发，避免GIL问题
- 基于事件循环（Event Loop）调度

**完整实现：**

```python
import asyncio
import aiohttp
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import re
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CrawlResult:
    """爬取结果"""
    url: str
    status: int
    content: str
    headers: dict
    elapsed: float
    error: Optional[str] = None

class RateLimiter:
    """速率限制器"""

    def __init__(self, rate: int, per: float = 1.0):
        """
        :param rate: 每个时间段允许的请求数
        :param per: 时间段长度（秒）
        """
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """获取许可"""
        async with self._lock:
            current = time.time()
            time_passed = current - self.last_check
            self.last_check = current

            # 补充许可
            self.allowance += time_passed * (self.rate / self.per)
            if self.allowance > self.rate:
                self.allowance = self.rate

            # 检查是否有足够的许可
            if self.allowance < 1.0:
                sleep_time = (1.0 - self.allowance) * (self.per / self.rate)
                await asyncio.sleep(sleep_time)
                self.allowance = 0.0
            else:
                self.allowance -= 1.0

class AsyncCrawler:
    """
    异步网络爬虫框架
    支持并发控制、速率限制、重试机制
    """

    def __init__(self,
                 max_concurrent: int = 10,
                 rate_limit: int = 20,
                 timeout: int = 30,
                 max_retries: int = 3,
                 user_agent: Optional[str] = None):
        """
        初始化爬虫
        :param max_concurrent: 最大并发请求数
        :param rate_limit: 每秒最大请求数
        :param timeout: 请求超时时间
        :param max_retries: 最大重试次数
        :param user_agent: User-Agent头
        """
        self.max_concurrent = max_concurrent
        self.rate_limiter = RateLimiter(rate_limit)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries

        self.headers = {
            'User-Agent': user_agent or 'Mozilla/5.0 (AsyncCrawler/1.0)'
        }

        # 并发控制
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # 统计信息
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'retried': 0
        }

    async def fetch(self, session: aiohttp.ClientSession,
                   url: str) -> CrawlResult:
        """
        获取单个URL
        """
        async with self.semaphore:
            await self.rate_limiter.acquire()

            self.stats['total'] += 1
            retries = 0

            while retries <= self.max_retries:
                start_time = time.time()

                try:
                    async with session.get(url, headers=self.headers) as response:
                        content = await response.text()
                        elapsed = time.time() - start_time

                        self.stats['success'] += 1
                        logger.info(f"✓ {url} ({response.status}) - {elapsed:.2f}s")

                        return CrawlResult(
                            url=url,
                            status=response.status,
                            content=content,
                            headers=dict(response.headers),
                            elapsed=elapsed
                        )

                except asyncio.TimeoutError:
                    retries += 1
                    self.stats['retried'] += 1
                    if retries <= self.max_retries:
                        wait_time = 2 ** retries  # 指数退避
                        logger.warning(f"⟳ {url} timeout, retry {retries}/{self.max_retries} after {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        self.stats['failed'] += 1
                        logger.error(f"✗ {url} failed after {retries} retries")
                        return CrawlResult(
                            url=url, status=0, content='', headers={},
                            elapsed=time.time() - start_time,
                            error="Timeout after retries"
                        )

                except Exception as e:
                    self.stats['failed'] += 1
                    logger.error(f"✗ {url} error: {e}")
                    return CrawlResult(
                        url=url, status=0, content='', headers={},
                        elapsed=time.time() - start_time,
                        error=str(e)
                    )

    async def crawl(self, urls: List[str],
                   callback: Optional[Callable] = None) -> List[CrawlResult]:
        """
        爬取多个URL
        :param urls: URL列表
        :param callback: 回调函数，处理每个结果
        :return: 爬取结果列表
        """
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            tasks = [self.fetch(session, url) for url in urls]

            results = []
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if callback:
                    await callback(result) if asyncio.iscoroutinefunction(callback) else callback(result)
                results.append(result)

            return results

    def run(self, urls: List[str],
            callback: Optional[Callable] = None) -> List[CrawlResult]:
        """
        同步接口运行爬虫
        """
        return asyncio.run(self.crawl(urls, callback))

    def print_stats(self):
        """打印统计信息"""
        print("\n" + "="*50)
        print("Crawling Statistics:")
        print(f"Total Requests: {self.stats['total']}")
        print(f"Successful: {self.stats['success']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Retried: {self.stats['retried']}")
        success_rate = (self.stats['success'] / self.stats['total'] * 100) if self.stats['total'] > 0 else 0
        print(f"Success Rate: {success_rate:.2f}%")
        print("="*50)

# 进阶：递归爬虫（爬取链接）
class RecursiveCrawler(AsyncCrawler):
    """
    递归爬虫
    从起始URL开始，递归爬取页面中的链接
    """

    def __init__(self, *args, max_depth: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_depth = max_depth
        self.visited = set()
        self.to_visit = deque()

    def extract_links(self, html: str, base_url: str) -> List[str]:
        """从HTML中提取链接"""
        # 简单的正则提取链接（实际应用应使用BeautifulSoup或lxml）
        pattern = r'href=["\']([^"\']+)["\']'
        links = re.findall(pattern, html)

        # 转换为绝对URL
        absolute_links = []
        for link in links:
            absolute_url = urljoin(base_url, link)
            # 只保留同域名的链接
            if urlparse(absolute_url).netloc == urlparse(base_url).netloc:
                absolute_links.append(absolute_url)

        return list(set(absolute_links))  # 去重

    async def crawl_recursive(self, start_url: str,
                              callback: Optional[Callable] = None) -> Dict[str, CrawlResult]:
        """
        递归爬取
        :param start_url: 起始URL
        :param callback: 回调函数
        :return: URL -> CrawlResult的映射
        """
        results = {}
        self.to_visit.append((start_url, 0))  # (url, depth)

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            while self.to_visit:
                # 批量处理当前层级的URL
                current_batch = []
                while self.to_visit and len(current_batch) < self.max_concurrent:
                    url, depth = self.to_visit.popleft()

                    if url in self.visited:
                        continue

                    self.visited.add(url)
                    current_batch.append((url, depth))

                # 并发爬取当前批次
                tasks = [
                    self.fetch(session, url)
                    for url, depth in current_batch
                ]

                batch_results = await asyncio.gather(*tasks)

                # 处理结果并提取新链接
                for (url, depth), result in zip(current_batch, batch_results):
                    results[url] = result

                    if callback:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(result)
                        else:
                            callback(result)

                    # 提取并添加新链接
                    if depth < self.max_depth and result.status == 200:
                        links = self.extract_links(result.content, url)
                        for link in links:
                            if link not in self.visited:
                                self.to_visit.append((link, depth + 1))

        return results

# 测试示例
async def test_async_crawler():
    """测试异步爬虫"""

    # 测试URL列表
    urls = [
        'https://httpbin.org/delay/1',
        'https://httpbin.org/delay/2',
        'https://httpbin.org/status/200',
        'https://httpbin.org/status/404',
        'https://httpbin.org/html',
    ]

    # 创建爬虫
    crawler = AsyncCrawler(
        max_concurrent=3,
        rate_limit=5,
        timeout=10,
        max_retries=2
    )

    # 回调函数
    def process_result(result: CrawlResult):
        if result.error:
            print(f"Error crawling {result.url}: {result.error}")
        else:
            print(f"Successfully crawled {result.url}: {len(result.content)} bytes")

    # 开始爬取
    print("Starting crawler...")
    start_time = time.time()

    results = await crawler.crawl(urls, callback=process_result)

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f}s")
    crawler.print_stats()

    return results

# 异步上下文管理器示例
class AsyncDatabaseConnection:
    """异步数据库连接示例"""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.connection = None

    async def __aenter__(self):
        """异步进入上下文"""
        print(f"Connecting to {self.host}:{self.port}...")
        await asyncio.sleep(0.1)  # 模拟连接延迟
        self.connection = f"Connection to {self.host}:{self.port}"
        print("Connected!")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步退出上下文"""
        print("Closing connection...")
        await asyncio.sleep(0.1)  # 模拟关闭延迟
        self.connection = None
        print("Connection closed!")
        return False

    async def query(self, sql: str):
        """执行查询"""
        if not self.connection:
            raise RuntimeError("Not connected")
        print(f"Executing: {sql}")
        await asyncio.sleep(0.2)  # 模拟查询延迟
        return f"Results for: {sql}"

# 异步迭代器示例
class AsyncRange:
    """异步范围迭代器"""

    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
        self.current = start

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.current >= self.end:
            raise StopAsyncIteration

        await asyncio.sleep(0.1)  # 模拟异步操作
        self.current += 1
        return self.current - 1

async def demo_async_features():
    """演示各种异步特性"""

    print("=== 异步上下文管理器 ===")
    async with AsyncDatabaseConnection("localhost", 5432) as db:
        result = await db.query("SELECT * FROM users")
        print(result)

    print("\n=== 异步迭代器 ===")
    async for num in AsyncRange(0, 5):
        print(f"Number: {num}")

    print("\n=== 异步任务并发 ===")
    async def task(name: str, delay: float):
        print(f"Task {name} starting...")
        await asyncio.sleep(delay)
        print(f"Task {name} completed!")
        return f"Result from {name}"

    # 创建并发任务
    tasks = [
        asyncio.create_task(task("A", 1)),
        asyncio.create_task(task("B", 2)),
        asyncio.create_task(task("C", 1.5)),
    ]

    # 等待所有任务完成
    results = await asyncio.gather(*tasks)
    print(f"All tasks completed: {results}")

# 运行测试
# asyncio.run(test_async_crawler())
# asyncio.run(demo_async_features())
```

**关键点：**
- `async`/`await`是协程的核心语法
- 使用`asyncio.gather()`并发执行多个协程
- `async with`支持异步上下文管理器
- `async for`支持异步迭代器
- 使用`asyncio.Semaphore`控制并发数
- 事件循环负责调度所有协程

---

### 10. 实现一个支持超时和取消的异步任务池

**标准答案：**

```python
import asyncio
import time
from typing import Callable, Any, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

@dataclass
class TaskResult:
    """任务结果"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def elapsed(self) -> float:
        """执行时间"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

class AsyncTaskPool:
    """
    异步任务池
    支持任务提交、取消、超时、优先级等功能
    """

    def __init__(self, max_workers: int = 10):
        """
        初始化任务池
        :param max_workers: 最大并发工作数
        """
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)

        # 任务管理
        self.tasks: Dict[str, asyncio.Task] = {}
        self.results: Dict[str, TaskResult] = {}
        self.task_id_counter = 0

        # 统计信息
        self.stats = {
            'submitted': 0,
            'completed': 0,
            'failed': 0,
            'cancelled': 0,
            'timeout': 0
        }

        self._lock = asyncio.Lock()

    def _generate_task_id(self) -> str:
        """生成唯一任务ID"""
        self.task_id_counter += 1
        return f"task-{self.task_id_counter}"

    async def submit(self,
                    coro: Callable,
                    *args,
                    timeout: Optional[float] = None,
                    task_id: Optional[str] = None,
                    **kwargs) -> str:
        """
        提交异步任务
        :param coro: 协程函数
        :param args: 位置参数
        :param timeout: 超时时间（秒）
        :param task_id: 任务ID（可选）
        :param kwargs: 关键字参数
        :return: 任务ID
        """
        if task_id is None:
            task_id = self._generate_task_id()

        # 创建任务
        task = asyncio.create_task(
            self._execute_task(task_id, coro, args, kwargs, timeout)
        )

        async with self._lock:
            self.tasks[task_id] = task
            self.results[task_id] = TaskResult(
                task_id=task_id,
                status=TaskStatus.PENDING
            )
            self.stats['submitted'] += 1

        logger.info(f"Task {task_id} submitted")
        return task_id

    async def _execute_task(self,
                           task_id: str,
                           coro: Callable,
                           args: tuple,
                           kwargs: dict,
                           timeout: Optional[float]) -> TaskResult:
        """
        执行任务的内部方法
        """
        result = self.results[task_id]
        result.start_time = time.time()
        result.status = TaskStatus.RUNNING

        async with self.semaphore:  # 控制并发数
            try:
                # 执行任务（带超时）
                if timeout:
                    task_result = await asyncio.wait_for(
                        coro(*args, **kwargs),
                        timeout=timeout
                    )
                else:
                    task_result = await coro(*args, **kwargs)

                # 任务成功完成
                result.status = TaskStatus.COMPLETED
                result.result = task_result
                self.stats['completed'] += 1
                logger.info(f"Task {task_id} completed successfully")

            except asyncio.TimeoutError:
                # 超时
                result.status = TaskStatus.TIMEOUT
                result.error = TimeoutError(f"Task timeout after {timeout}s")
                self.stats['timeout'] += 1
                logger.warning(f"Task {task_id} timeout")

            except asyncio.CancelledError:
                # 被取消
                result.status = TaskStatus.CANCELLED
                result.error = asyncio.CancelledError("Task cancelled")
                self.stats['cancelled'] += 1
                logger.warning(f"Task {task_id} cancelled")
                raise  # 重新抛出以正确处理取消

            except Exception as e:
                # 执行失败
                result.status = TaskStatus.FAILED
                result.error = e
                self.stats['failed'] += 1
                logger.error(f"Task {task_id} failed: {e}")

            finally:
                result.end_time = time.time()

        return result

    async def cancel(self, task_id: str) -> bool:
        """
        取消任务
        :param task_id: 任务ID
        :return: 是否成功取消
        """
        async with self._lock:
            if task_id not in self.tasks:
                logger.warning(f"Task {task_id} not found")
                return False

            task = self.tasks[task_id]
            if task.done():
                logger.warning(f"Task {task_id} already completed")
                return False

            task.cancel()
            logger.info(f"Task {task_id} cancellation requested")
            return True

    async def cancel_all(self) -> int:
        """
        取消所有未完成的任务
        :return: 取消的任务数
        """
        async with self._lock:
            cancelled_count = 0
            for task_id, task in self.tasks.items():
                if not task.done():
                    task.cancel()
                    cancelled_count += 1

            logger.info(f"Cancelled {cancelled_count} tasks")
            return cancelled_count

    async def wait(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """
        等待任务完成
        :param task_id: 任务ID
        :param timeout: 超时时间
        :return: 任务结果
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self.tasks[task_id]

        try:
            if timeout:
                await asyncio.wait_for(task, timeout=timeout)
            else:
                await task
        except asyncio.TimeoutError:
            logger.warning(f"Waiting for task {task_id} timeout")

        return self.results[task_id]

    async def wait_all(self, timeout: Optional[float] = None) -> List[TaskResult]:
        """
        等待所有任务完成
        :param timeout: 超时时间
        :return: 所有任务结果
        """
        if not self.tasks:
            return []

        try:
            if timeout:
                await asyncio.wait_for(
                    asyncio.gather(*self.tasks.values(), return_exceptions=True),
                    timeout=timeout
                )
            else:
                await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        except asyncio.TimeoutError:
            logger.warning("Waiting for all tasks timeout")

        return list(self.results.values())

    def get_result(self, task_id: str) -> Optional[TaskResult]:
        """
        获取任务结果（非阻塞）
        :param task_id: 任务ID
        :return: 任务结果或None
        """
        return self.results.get(task_id)

    def get_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        获取任务状态
        :param task_id: 任务ID
        :return: 任务状态或None
        """
        result = self.results.get(task_id)
        return result.status if result else None

    def print_stats(self):
        """打印统计信息"""
        print("\n" + "="*50)
        print("Task Pool Statistics:")
        print(f"Submitted: {self.stats['submitted']}")
        print(f"Completed: {self.stats['completed']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Cancelled: {self.stats['cancelled']}")
        print(f"Timeout: {self.stats['timeout']}")

        active_tasks = sum(1 for task in self.tasks.values() if not task.done())
        print(f"Active Tasks: {active_tasks}")
        print("="*50)

# 测试示例
async def example_task(task_name: str, duration: float, should_fail: bool = False):
    """示例任务"""
    print(f"[{task_name}] Starting (duration: {duration}s)")
    await asyncio.sleep(duration)

    if should_fail:
        raise ValueError(f"[{task_name}] Intentional failure")

    print(f"[{task_name}] Completed")
    return f"Result from {task_name}"

async def test_task_pool():
    """测试任务池"""
    pool = AsyncTaskPool(max_workers=3)

    # 提交任务
    task_ids = []

    # 正常任务
    task_ids.append(await pool.submit(example_task, "Task-1", 2.0))
    task_ids.append(await pool.submit(example_task, "Task-2", 1.0))

    # 会失败的任务
    task_ids.append(await pool.submit(example_task, "Task-3", 1.5, should_fail=True))

    # 会超时的任务
    task_ids.append(await pool.submit(example_task, "Task-4", 5.0, timeout=2.0))

    # 会被取消的任务
    cancel_task_id = await pool.submit(example_task, "Task-5", 3.0)
    task_ids.append(cancel_task_id)

    # 等待一会儿后取消任务
    await asyncio.sleep(0.5)
    await pool.cancel(cancel_task_id)

    # 等待所有任务完成
    results = await pool.wait_all(timeout=10.0)

    # 打印结果
    print("\n" + "="*50)
    print("Task Results:")
    for result in results:
        print(f"\n{result.task_id}:")
        print(f"  Status: {result.status.value}")
        print(f"  Elapsed: {result.elapsed:.2f}s")
        if result.result:
            print(f"  Result: {result.result}")
        if result.error:
            print(f"  Error: {result.error}")

    # 打印统计
    pool.print_stats()

# asyncio.run(test_task_pool())

# 进阶：带回调的任务池
class CallbackTaskPool(AsyncTaskPool):
    """
    支持回调的任务池
    """

    async def submit_with_callback(self,
                                   coro: Callable,
                                   *args,
                                   on_complete: Optional[Callable] = None,
                                   on_error: Optional[Callable] = None,
                                   **kwargs) -> str:
        """
        提交带回调的任务
        :param coro: 协程函数
        :param on_complete: 完成回调
        :param on_error: 错误回调
        :return: 任务ID
        """
        task_id = await self.submit(coro, *args, **kwargs)

        # 创建回调任务
        async def callback_wrapper():
            result = await self.wait(task_id)

            if result.status == TaskStatus.COMPLETED and on_complete:
                try:
                    if asyncio.iscoroutinefunction(on_complete):
                        await on_complete(result.result)
                    else:
                        on_complete(result.result)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

            elif result.status in [TaskStatus.FAILED, TaskStatus.TIMEOUT] and on_error:
                try:
                    if asyncio.iscoroutinefunction(on_error):
                        await on_error(result.error)
                    else:
                        on_error(result.error)
                except Exception as e:
                    logger.error(f"Error callback error: {e}")

        asyncio.create_task(callback_wrapper())
        return task_id
```

**关键点：**
- 使用`asyncio.Semaphore`控制并发数
- `asyncio.wait_for()`实现超时控制
- `task.cancel()`取消任务
- `asyncio.gather()`批量等待任务
- 使用枚举管理任务状态
- 异常处理和资源清理很重要

---

*（由于篇幅限制，剩余部分包括：第四部分Web开发、第五部分系统设计、第六部分性能优化将在后续补充）*

## 总结

这份面试题集涵盖了：

1. **Python基础与进阶**
   - GIL原理与绕过
   - 元类深入应用
   - 描述符协议
   - 内存管理机制
   - 装饰器高级用法

2. **数据结构与算法**
   - LRU缓存实现
   - 字典树（Trie）
   - 线程安全队列

3. **并发编程与异步**
   - 协程与异步编程
   - 异步爬虫框架
   - 异步任务池

每个问题都包含：
- 深入的原理解释
- 完整的代码实现
- 实际应用场景
- 性能优化技巧

**面试建议：**
- 不要死记硬背，理解原理最重要
- 准备好讲解自己的代码实现
- 关注边界条件和异常处理
- 了解性能优化的trade-off
- 准备实际项目中的应用案例
