# Python 魔法方法(Magic Methods)详解

## 目录
- [什么是魔法方法](#什么是魔法方法)
- [对象的生命周期](#对象的生命周期)
- [属性访问](#属性访问)
- [字符串表示](#字符串表示)
- [比较运算符](#比较运算符)
- [数值运算符](#数值运算符)
- [容器类型](#容器类型)
- [可调用对象](#可调用对象)
- [上下文管理器](#上下文管理器)
- [描述符](#描述符)
- [其他魔法方法](#其他魔法方法)

---

## 什么是魔法方法

魔法方法(Magic Methods)也称为双下划线方法(Dunder Methods),是Python中以双下划线开头和结尾的特殊方法,如 `__init__`、`__str__` 等。这些方法允许我们自定义类的行为,使其支持Python的内置操作。

**特点:**
- 以双下划线 `__` 开头和结尾
- 不需要直接调用,会被Python解释器自动调用
- 让自定义对象支持内置函数和运算符

---

## 对象的生命周期

### `__new__(cls, *args, **kwargs)`
创建实例的第一步,在 `__init__` 之前调用。

```python
class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, value):
        self.value = value

# 测试单例模式
s1 = Singleton(1)
s2 = Singleton(2)
print(s1 is s2)  # True
print(s1.value)  # 2
```

### `__init__(self, *args, **kwargs)`
初始化实例,最常用的魔法方法。

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

p = Person("Alice", 25)
```

### `__del__(self)`
对象销毁时调用的析构方法。

```python
class FileHandler:
    def __init__(self, filename):
        self.file = open(filename, 'w')

    def __del__(self):
        if hasattr(self, 'file'):
            self.file.close()
            print(f"文件已关闭")
```

**注意:** 不建议依赖 `__del__`,因为调用时机不确定。使用上下文管理器更可靠。

---

## 属性访问

### `__getattr__(self, name)`
访问不存在的属性时调用。

```python
class DynamicAttrs:
    def __init__(self):
        self.existing = "I exist"

    def __getattr__(self, name):
        return f"属性 '{name}' 不存在,返回默认值"

obj = DynamicAttrs()
print(obj.existing)      # I exist
print(obj.nonexistent)   # 属性 'nonexistent' 不存在,返回默认值
```

### `__getattribute__(self, name)`
访问任何属性时都会调用(包括存在的属性)。

```python
class LoggedAccess:
    def __init__(self):
        self.value = 42

    def __getattribute__(self, name):
        print(f"访问属性: {name}")
        return super().__getattribute__(name)

obj = LoggedAccess()
print(obj.value)
# 输出:
# 访问属性: value
# 42
```

**注意:** 使用 `__getattribute__` 要小心无限递归,必须使用 `super().__getattribute__()`.

### `__setattr__(self, name, value)`
设置属性时调用。

```python
class ValidatedAttrs:
    def __setattr__(self, name, value):
        if name == 'age' and (value < 0 or value > 150):
            raise ValueError("年龄必须在0-150之间")
        super().__setattr__(name, value)

person = ValidatedAttrs()
person.age = 25   # OK
# person.age = 200  # ValueError
```

### `__delattr__(self, name)`
删除属性时调用。

```python
class ProtectedAttrs:
    def __init__(self):
        self.public = "可删除"
        self._protected = "受保护"

    def __delattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"不能删除受保护的属性: {name}")
        super().__delattr__(name)

obj = ProtectedAttrs()
del obj.public      # OK
# del obj._protected  # AttributeError
```

### `__dir__(self)`
自定义 `dir()` 函数的输出。

```python
class CustomDir:
    def __init__(self):
        self.x = 1
        self.y = 2

    def __dir__(self):
        return ['x', 'y', 'custom_method']

obj = CustomDir()
print(dir(obj))  # ['custom_method', 'x', 'y']
```

---

## 字符串表示

### `__str__(self)`
定义 `str()` 和 `print()` 的输出,面向用户。

```python
class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author

    def __str__(self):
        return f"《{self.title}》 by {self.author}"

book = Book("Python编程", "张三")
print(book)  # 《Python编程》 by 张三
```

### `__repr__(self)`
定义对象的"官方"字符串表示,面向开发者。

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def __str__(self):
        return f"({self.x}, {self.y})"

p = Point(3, 4)
print(str(p))   # (3, 4)
print(repr(p))  # Point(3, 4)
print([p])      # [Point(3, 4)]  # 列表使用__repr__
```

### `__format__(self, format_spec)`
自定义字符串格式化。

```python
class Currency:
    def __init__(self, amount):
        self.amount = amount

    def __format__(self, format_spec):
        if format_spec == 'usd':
            return f"${self.amount:.2f}"
        elif format_spec == 'cny':
            return f"¥{self.amount:.2f}"
        return str(self.amount)

money = Currency(1234.5)
print(f"{money:usd}")  # $1234.50
print(f"{money:cny}")  # ¥1234.50
```

### `__bytes__(self)`
定义 `bytes()` 的行为。

```python
class ByteString:
    def __init__(self, text):
        self.text = text

    def __bytes__(self):
        return self.text.encode('utf-8')

obj = ByteString("Hello")
print(bytes(obj))  # b'Hello'
```

---

## 比较运算符

### `__eq__(self, other)` - 等于 (==)
```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __eq__(self, other):
        if not isinstance(other, Person):
            return False
        return self.name == other.name and self.age == other.age

p1 = Person("Alice", 25)
p2 = Person("Alice", 25)
print(p1 == p2)  # True
```

### `__ne__(self, other)` - 不等于 (!=)
```python
class Number:
    def __init__(self, value):
        self.value = value

    def __ne__(self, other):
        return self.value != other.value
```

### `__lt__(self, other)` - 小于 (<)
### `__le__(self, other)` - 小于等于 (<=)
### `__gt__(self, other)` - 大于 (>)
### `__ge__(self, other)` - 大于等于 (>=)

```python
from functools import total_ordering

@total_ordering  # 只需定义 __eq__ 和一个比较方法,自动生成其他
class Version:
    def __init__(self, major, minor, patch):
        self.major = major
        self.minor = minor
        self.patch = patch

    def __eq__(self, other):
        return (self.major, self.minor, self.patch) == \
               (other.major, other.minor, other.patch)

    def __lt__(self, other):
        return (self.major, self.minor, self.patch) < \
               (other.major, other.minor, other.patch)

v1 = Version(1, 2, 3)
v2 = Version(1, 3, 0)
print(v1 < v2)   # True
print(v1 <= v2)  # True (自动生成)
print(v1 > v2)   # False (自动生成)
```

---

## 数值运算符

### 算术运算符

#### `__add__(self, other)` - 加法 (+)
```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(1, 2)
v2 = Vector(3, 4)
print(v1 + v2)  # Vector(4, 6)
```

#### `__sub__(self, other)` - 减法 (-)
#### `__mul__(self, other)` - 乘法 (*)
#### `__truediv__(self, other)` - 真除法 (/)
#### `__floordiv__(self, other)` - 整除 (//)
#### `__mod__(self, other)` - 取模 (%)
#### `__pow__(self, other)` - 幂运算 (**)

```python
class ComplexNumber:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return ComplexNumber(
            self.real + other.real,
            self.imag + other.imag
        )

    def __mul__(self, other):
        # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        return ComplexNumber(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real
        )

    def __repr__(self):
        return f"{self.real}+{self.imag}i"

c1 = ComplexNumber(1, 2)
c2 = ComplexNumber(3, 4)
print(c1 + c2)  # 4+6i
print(c1 * c2)  # -5+10i
```

### 反射运算符 (右侧运算)

#### `__radd__(self, other)` - 右加法
当左操作数不支持运算时调用右操作数的反射方法。

```python
class MyInt:
    def __init__(self, value):
        self.value = value

    def __radd__(self, other):
        print(f"调用 __radd__: {other} + MyInt({self.value})")
        return other + self.value

obj = MyInt(10)
result = 5 + obj  # int 不支持 + MyInt,调用 MyInt.__radd__
print(result)     # 15
```

同理还有: `__rsub__`, `__rmul__`, `__rtruediv__`, `__rfloordiv__`, `__rmod__`, `__rpow__`

### 增强赋值运算符

#### `__iadd__(self, other)` - 加法赋值 (+=)
```python
class Counter:
    def __init__(self, value=0):
        self.value = value

    def __iadd__(self, other):
        self.value += other
        return self  # 必须返回self

    def __repr__(self):
        return f"Counter({self.value})"

c = Counter(10)
c += 5
print(c)  # Counter(15)
```

同理还有: `__isub__`, `__imul__`, `__itruediv__`, `__ifloordiv__`, `__imod__`, `__ipow__`

### 一元运算符

#### `__neg__(self)` - 负号 (-)
#### `__pos__(self)` - 正号 (+)
#### `__abs__(self)` - 绝对值 abs()
#### `__invert__(self)` - 按位取反 (~)

```python
class SignedNumber:
    def __init__(self, value):
        self.value = value

    def __neg__(self):
        return SignedNumber(-self.value)

    def __pos__(self):
        return SignedNumber(+self.value)

    def __abs__(self):
        return SignedNumber(abs(self.value))

    def __repr__(self):
        return f"SignedNumber({self.value})"

n = SignedNumber(-5)
print(-n)     # SignedNumber(5)
print(+n)     # SignedNumber(-5)
print(abs(n)) # SignedNumber(5)
```

### 位运算符

#### `__and__(self, other)` - 按位与 (&)
#### `__or__(self, other)` - 按位或 (|)
#### `__xor__(self, other)` - 按位异或 (^)
#### `__lshift__(self, other)` - 左移 (<<)
#### `__rshift__(self, other)` - 右移 (>>)

```python
class BitSet:
    def __init__(self, value):
        self.value = value

    def __and__(self, other):
        return BitSet(self.value & other.value)

    def __or__(self, other):
        return BitSet(self.value | other.value)

    def __repr__(self):
        return f"BitSet({bin(self.value)})"

b1 = BitSet(0b1010)
b2 = BitSet(0b1100)
print(b1 & b2)  # BitSet(0b1000)
print(b1 | b2)  # BitSet(0b1110)
```

### 类型转换

#### `__int__(self)` - int()
#### `__float__(self)` - float()
#### `__complex__(self)` - complex()
#### `__bool__(self)` - bool()

```python
class Temperature:
    def __init__(self, celsius):
        self.celsius = celsius

    def __int__(self):
        return int(self.celsius)

    def __float__(self):
        return float(self.celsius)

    def __bool__(self):
        return self.celsius != 0

temp = Temperature(36.5)
print(int(temp))    # 36
print(float(temp))  # 36.5
print(bool(temp))   # True
```

### `__round__(self, n=0)` - round()
### `__floor__(self)` - math.floor()
### `__ceil__(self)` - math.ceil()
### `__trunc__(self)` - math.trunc()

```python
import math

class Decimal:
    def __init__(self, value):
        self.value = value

    def __round__(self, n=0):
        return round(self.value, n)

    def __floor__(self):
        return math.floor(self.value)

    def __ceil__(self):
        return math.ceil(self.value)

d = Decimal(3.7)
print(round(d))        # 4
print(math.floor(d))   # 3
print(math.ceil(d))    # 4
```

---

## 容器类型

### `__len__(self)` - len()
返回容器的长度。

```python
class Playlist:
    def __init__(self):
        self.songs = []

    def add(self, song):
        self.songs.append(song)

    def __len__(self):
        return len(self.songs)

playlist = Playlist()
playlist.add("Song 1")
playlist.add("Song 2")
print(len(playlist))  # 2
```

### `__getitem__(self, key)` - 索引访问 []
```python
class MyList:
    def __init__(self, items):
        self.items = items

    def __getitem__(self, index):
        # 支持负索引
        if isinstance(index, int):
            return self.items[index]
        # 支持切片
        elif isinstance(index, slice):
            return MyList(self.items[index])

lst = MyList([1, 2, 3, 4, 5])
print(lst[0])      # 1
print(lst[-1])     # 5
print(lst[1:3])    # MyList([2, 3])
```

### `__setitem__(self, key, value)` - 索引赋值
```python
class Grid:
    def __init__(self, rows, cols):
        self.data = [[0] * cols for _ in range(rows)]

    def __setitem__(self, pos, value):
        row, col = pos
        self.data[row][col] = value

    def __getitem__(self, pos):
        row, col = pos
        return self.data[row][col]

grid = Grid(3, 3)
grid[1, 2] = 5
print(grid[1, 2])  # 5
```

### `__delitem__(self, key)` - 删除元素
```python
class RemovableList:
    def __init__(self, items):
        self.items = list(items)

    def __delitem__(self, index):
        del self.items[index]

    def __repr__(self):
        return f"RemovableList({self.items})"

lst = RemovableList([1, 2, 3, 4])
del lst[1]
print(lst)  # RemovableList([1, 3, 4])
```

### `__contains__(self, item)` - in 运算符
```python
class Team:
    def __init__(self):
        self.members = set()

    def add_member(self, name):
        self.members.add(name)

    def __contains__(self, name):
        return name in self.members

team = Team()
team.add_member("Alice")
print("Alice" in team)  # True
print("Bob" in team)    # False
```

### `__iter__(self)` - 迭代器
返回一个迭代器对象。

```python
class Countdown:
    def __init__(self, start):
        self.start = start

    def __iter__(self):
        self.current = self.start
        return self

    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

for num in Countdown(5):
    print(num)  # 5, 4, 3, 2, 1
```

### `__reversed__(self)` - reversed()
```python
class ReversibleList:
    def __init__(self, items):
        self.items = items

    def __reversed__(self):
        return iter(reversed(self.items))

lst = ReversibleList([1, 2, 3, 4])
for item in reversed(lst):
    print(item)  # 4, 3, 2, 1
```

---

## 可调用对象

### `__call__(self, *args, **kwargs)`
使对象可以像函数一样被调用。

```python
class Multiplier:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return x * self.factor

double = Multiplier(2)
triple = Multiplier(3)

print(double(5))  # 10
print(triple(5))  # 15
```

**实际应用:**
```python
class Counter:
    def __init__(self):
        self.count = 0

    def __call__(self):
        self.count += 1
        return self.count

counter = Counter()
print(counter())  # 1
print(counter())  # 2
print(counter())  # 3
```

---

## 上下文管理器

### `__enter__(self)` 和 `__exit__(self, exc_type, exc_val, exc_tb)`
实现 `with` 语句支持。

```python
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        print(f"打开文件: {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"关闭文件: {self.filename}")
        if self.file:
            self.file.close()
        # 返回False表示不抑制异常,返回True表示抑制异常
        return False

with FileManager('test.txt', 'w') as f:
    f.write('Hello World')
```

**异常处理:**
```python
class ExceptionHandler:
    def __enter__(self):
        print("进入上下文")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"捕获异常: {exc_type.__name__}: {exc_val}")
            return True  # 抑制异常
        print("正常退出")
        return False

with ExceptionHandler():
    print("执行代码")
    raise ValueError("出错了")
print("继续执行")  # 会执行,因为异常被抑制
```

---

## 描述符

### `__get__(self, instance, owner)`
### `__set__(self, instance, value)`
### `__delete__(self, instance)`

描述符是实现了描述符协议的对象,用于自定义属性访问。

```python
class ValidatedAttribute:
    def __init__(self, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        self.data = {}

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self.data.get(id(instance))

    def __set__(self, instance, value):
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"{self.name} 不能小于 {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"{self.name} 不能大于 {self.max_value}")
        self.data[id(instance)] = value

    def __delete__(self, instance):
        del self.data[id(instance)]

class Person:
    age = ValidatedAttribute(min_value=0, max_value=150)

    def __init__(self, age):
        self.age = age

person = Person(25)
print(person.age)  # 25
person.age = 30    # OK
# person.age = 200   # ValueError
```

**property 装饰器的实现:**
```python
class MyProperty:
    def __init__(self, fget=None, fset=None, fdel=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if self.fget is None:
            raise AttributeError("无法读取")
        return self.fget(instance)

    def __set__(self, instance, value):
        if self.fset is None:
            raise AttributeError("无法设置")
        self.fset(instance, value)

    def __delete__(self, instance):
        if self.fdel is None:
            raise AttributeError("无法删除")
        self.fdel(instance)

class Circle:
    def __init__(self, radius):
        self._radius = radius

    def get_radius(self):
        return self._radius

    def set_radius(self, value):
        if value < 0:
            raise ValueError("半径不能为负")
        self._radius = value

    radius = MyProperty(get_radius, set_radius)

c = Circle(5)
print(c.radius)  # 5
c.radius = 10    # OK
```

### `__set_name__(self, owner, name)`
在描述符被赋值给类属性时自动调用。

```python
class NamedDescriptor:
    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = f'_{name}'

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self.private_name)

    def __set__(self, instance, value):
        print(f"设置 {self.name} = {value}")
        setattr(instance, self.private_name, value)

class MyClass:
    attr = NamedDescriptor()  # 自动获取名称 'attr'

    def __init__(self, value):
        self.attr = value

obj = MyClass(42)
print(obj.attr)  # 42
```

---

## 其他魔法方法

### `__hash__(self)`
返回对象的哈希值,用于字典键和集合。

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

p1 = Point(1, 2)
p2 = Point(1, 2)
print(hash(p1) == hash(p2))  # True

# 可以用作字典键
points = {p1: "第一个点"}
print(points[p2])  # 第一个点
```

**注意:** 如果定义了 `__eq__`,也应该定义 `__hash__`,否则对象不可哈希。

### `__index__(self)`
将对象转换为整数索引。

```python
class IntLike:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value

idx = IntLike(2)
lst = [10, 20, 30, 40]
print(lst[idx])  # 30
print(bin(idx))  # '0b10'
```

### `__sizeof__(self)`
返回对象占用的内存大小(字节)。

```python
import sys

class MyClass:
    def __init__(self, data):
        self.data = data

    def __sizeof__(self):
        return super().__sizeof__() + sys.getsizeof(self.data)

obj = MyClass([1, 2, 3, 4, 5])
print(sys.getsizeof(obj))  # 返回自定义的大小
```

### `__copy__(self)` 和 `__deepcopy__(self, memo)`
自定义拷贝行为。

```python
import copy

class MyList:
    def __init__(self, items):
        self.items = items

    def __copy__(self):
        print("执行浅拷贝")
        return MyList(self.items)

    def __deepcopy__(self, memo):
        print("执行深拷贝")
        return MyList(copy.deepcopy(self.items, memo))

original = MyList([[1, 2], [3, 4]])
shallow = copy.copy(original)
deep = copy.deepcopy(original)
```

### `__reduce__(self)` 和 `__reduce_ex__(self, protocol)`
用于 pickle 序列化。

```python
import pickle

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __reduce__(self):
        return (
            self.__class__,
            (self.name, self.age)
        )

person = Person("Alice", 25)
serialized = pickle.dumps(person)
restored = pickle.loads(serialized)
print(restored.name, restored.age)  # Alice 25
```

### `__missing__(self, key)`
在字典子类中,当键不存在时调用。

```python
class DefaultDict(dict):
    def __init__(self, default_factory):
        super().__init__()
        self.default_factory = default_factory

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        value = self.default_factory()
        self[key] = value
        return value

dd = DefaultDict(list)
dd['items'].append(1)
dd['items'].append(2)
print(dd['items'])  # [1, 2]
```

### `__instancecheck__(self, instance)` 和 `__subclasscheck__(self, subclass)`
自定义 isinstance() 和 issubclass() 行为。

```python
class ABCMeta(type):
    def __instancecheck__(cls, instance):
        print(f"检查 {instance} 是否是 {cls} 的实例")
        return type.__instancecheck__(cls, instance)

class MyABC(metaclass=ABCMeta):
    pass

class MyClass(MyABC):
    pass

obj = MyClass()
isinstance(obj, MyABC)  # 会调用 __instancecheck__
```

### `__class_getitem__(cls, item)`
用于泛型类型提示(Python 3.7+)。

```python
class GenericList:
    @classmethod
    def __class_getitem__(cls, item):
        return f"{cls.__name__}[{item}]"

print(GenericList[int])       # GenericList[int]
print(GenericList[str])       # GenericList[str]

# 用于类型提示
from typing import Generic, TypeVar

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self):
        self.items: list[T] = []

    def push(self, item: T):
        self.items.append(item)

stack: Stack[int] = Stack()
```

### `__init_subclass__(cls, **kwargs)`
当类被继承时自动调用(Python 3.6+)。

```python
class PluginBase:
    plugins = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        print(f"注册插件: {cls.__name__}")
        cls.plugins.append(cls)

class Plugin1(PluginBase):
    pass

class Plugin2(PluginBase):
    pass

print(PluginBase.plugins)  # [Plugin1, Plugin2]
```

### `__prepare__(metacls, name, bases, **kwargs)`
在创建类之前准备命名空间(元类方法)。

```python
from collections import OrderedDict

class OrderedMeta(type):
    @classmethod
    def __prepare__(metacls, name, bases, **kwargs):
        return OrderedDict()

    def __new__(cls, name, bases, namespace, **kwargs):
        namespace['_order'] = list(namespace.keys())
        return super().__new__(cls, name, bases, dict(namespace))

class MyClass(metaclass=OrderedMeta):
    z = 1
    y = 2
    x = 3

print(MyClass._order)  # ['__module__', '__qualname__', 'z', 'y', 'x']
```

### `__mro_entries__(self, bases)`
自定义方法解析顺序(MRO)。

```python
class MyBaseMixin:
    def __mro_entries__(self, bases):
        print(f"处理 MRO: {bases}")
        return (MyMixin,)

class MyMixin:
    def greet(self):
        return "Hello from MyMixin"

# 使用 MyBaseMixin() 会被替换为 MyMixin
class MyClass(MyBaseMixin()):
    pass

obj = MyClass()
print(obj.greet())  # Hello from MyMixin
```

### `__await__(self)`
使对象可等待(用于异步编程)。

```python
class Awaitable:
    def __await__(self):
        yield from some_coroutine().__await__()

# 实际例子
import asyncio

class MyFuture:
    def __init__(self, value):
        self.value = value

    def __await__(self):
        # 模拟异步操作
        future = asyncio.Future()
        future.set_result(self.value)
        return future.__await__()

async def main():
    result = await MyFuture(42)
    print(result)  # 42

# asyncio.run(main())
```

### `__aiter__(self)` 和 `__anext__(self)`
异步迭代器。

```python
class AsyncRange:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __aiter__(self):
        self.current = self.start
        return self

    async def __anext__(self):
        if self.current >= self.end:
            raise StopAsyncIteration
        await asyncio.sleep(0.1)  # 模拟异步操作
        self.current += 1
        return self.current - 1

async def main():
    async for num in AsyncRange(0, 5):
        print(num)  # 0, 1, 2, 3, 4

# asyncio.run(main())
```

### `__aenter__(self)` 和 `__aexit__(self, exc_type, exc_val, exc_tb)`
异步上下文管理器。

```python
class AsyncResource:
    async def __aenter__(self):
        print("获取资源")
        await asyncio.sleep(0.1)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("释放资源")
        await asyncio.sleep(0.1)
        return False

async def main():
    async with AsyncResource() as resource:
        print("使用资源")

# asyncio.run(main())
```

---

## 魔法方法速查表

### 构造与析构
- `__new__(cls, ...)` - 创建实例
- `__init__(self, ...)` - 初始化实例
- `__del__(self)` - 销毁实例

### 属性访问
- `__getattr__(self, name)` - 访问不存在的属性
- `__getattribute__(self, name)` - 访问任何属性
- `__setattr__(self, name, value)` - 设置属性
- `__delattr__(self, name)` - 删除属性
- `__dir__(self)` - dir() 函数

### 字符串表示
- `__str__(self)` - str() 和 print()
- `__repr__(self)` - repr() 和交互式环境
- `__format__(self, format_spec)` - format() 和 f-string
- `__bytes__(self)` - bytes()

### 比较运算
- `__eq__(self, other)` - ==
- `__ne__(self, other)` - !=
- `__lt__(self, other)` - <
- `__le__(self, other)` - <=
- `__gt__(self, other)` - >
- `__ge__(self, other)` - >=

### 数值运算
- `__add__(self, other)` - +
- `__sub__(self, other)` - -
- `__mul__(self, other)` - *
- `__truediv__(self, other)` - /
- `__floordiv__(self, other)` - //
- `__mod__(self, other)` - %
- `__pow__(self, other)` - **
- `__matmul__(self, other)` - @ (矩阵乘法)

### 反射运算
- `__radd__`, `__rsub__`, `__rmul__` 等

### 增强赋值
- `__iadd__`, `__isub__`, `__imul__` 等

### 一元运算
- `__neg__(self)` - -obj
- `__pos__(self)` - +obj
- `__abs__(self)` - abs()
- `__invert__(self)` - ~obj

### 位运算
- `__and__(self, other)` - &
- `__or__(self, other)` - |
- `__xor__(self, other)` - ^
- `__lshift__(self, other)` - <<
- `__rshift__(self, other)` - >>

### 类型转换
- `__int__(self)` - int()
- `__float__(self)` - float()
- `__complex__(self)` - complex()
- `__bool__(self)` - bool()
- `__index__(self)` - 整数索引
- `__round__(self, n)` - round()
- `__floor__(self)` - math.floor()
- `__ceil__(self)` - math.ceil()
- `__trunc__(self)` - math.trunc()

### 容器类型
- `__len__(self)` - len()
- `__getitem__(self, key)` - obj[key]
- `__setitem__(self, key, value)` - obj[key] = value
- `__delitem__(self, key)` - del obj[key]
- `__contains__(self, item)` - in
- `__iter__(self)` - iter()
- `__reversed__(self)` - reversed()

### 可调用对象
- `__call__(self, ...)` - obj()

### 上下文管理
- `__enter__(self)` - with 进入
- `__exit__(self, ...)` - with 退出

### 描述符
- `__get__(self, instance, owner)` - 获取属性
- `__set__(self, instance, value)` - 设置属性
- `__delete__(self, instance)` - 删除属性
- `__set_name__(self, owner, name)` - 设置名称

### 异步编程
- `__await__(self)` - await
- `__aiter__(self)` - async for
- `__anext__(self)` - async for 下一个
- `__aenter__(self)` - async with 进入
- `__aexit__(self, ...)` - async with 退出

### 其他
- `__hash__(self)` - hash()
- `__sizeof__(self)` - sys.getsizeof()
- `__copy__(self)` - copy.copy()
- `__deepcopy__(self, memo)` - copy.deepcopy()
- `__reduce__(self)` - pickle
- `__missing__(self, key)` - 字典键缺失
- `__instancecheck__(self, instance)` - isinstance()
- `__subclasscheck__(self, subclass)` - issubclass()
- `__class_getitem__(cls, item)` - 泛型类型
- `__init_subclass__(cls, **kwargs)` - 继承时调用
- `__prepare__(metacls, name, bases, **kwargs)` - 元类命名空间

---

## 最佳实践

1. **实现 `__repr__` 优于 `__str__`**
   - `__repr__` 应该返回可重新创建对象的字符串
   - 如果只定义 `__repr__`,它也会作为 `__str__` 的后备

2. **实现比较运算符时使用 `@total_ordering`**
   - 只需定义 `__eq__` 和一个顺序比较方法
   - 装饰器会自动生成其他比较方法

3. **实现 `__eq__` 时也要实现 `__hash__`**
   - 如果两个对象相等,它们的哈希值也应该相等
   - 可变对象通常不应该实现 `__hash__`

4. **增强赋值运算符应该返回 `self`**
   - `__iadd__` 等方法必须返回 `self`

5. **使用 `super()` 调用父类方法**
   - 在 `__new__`, `__init__`, `__getattribute__` 等方法中

6. **避免在 `__getattribute__` 中使用 `self.attr`**
   - 会导致无限递归
   - 使用 `super().__getattribute__(name)`

7. **上下文管理器的 `__exit__` 返回值很重要**
   - 返回 `True` 抑制异常
   - 返回 `False` 或 `None` 让异常继续传播

8. **描述符应该使用弱引用或 `id(instance)` 存储数据**
   - 避免内存泄漏

---

## 总结

Python 的魔法方法提供了强大的机制来自定义类的行为,使得自定义对象能够像内置类型一样使用。掌握这些魔法方法可以:

- 让代码更加 Pythonic
- 使自定义类与 Python 内置函数和运算符无缝集成
- 实现更加优雅和直观的 API
- 提供更好的用户体验

关键是要理解每个魔法方法的调用时机和用途,并在适当的场景下使用它们。不要过度使用 - 只在能够提升代码可读性和易用性时才实现这些方法。
