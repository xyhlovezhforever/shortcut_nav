# Redis 技术详解与应用实战

## 一、Redis的历史背景与发展历程

### 1.1 Redis诞生的历史背景和原因

2009年,意大利开发者Salvatore Sanfilippo(网名antirez)在开发一个实时日志分析系统时,发现MySQL无法满足实时性要求。他需要一个能够快速写入、快速读取、支持复杂数据结构的存储系统。现有的解决方案要么太慢(关系型数据库),要么功能太简单(Memcached只支持字符串)。

于是antirez用C语言开发了Redis(Remote Dictionary Server,远程字典服务器)。Redis的诞生填补了"需要持久化的高性能缓存"这一市场空白。

**Redis解决的核心问题**:

- **性能问题**: 提供10万级QPS的读写性能,远超关系型数据库
- **数据结构问题**: 支持String、List、Hash、Set、ZSet等丰富数据结构
- **持久化问题**: 兼顾性能和数据安全,提供RDB和AOF两种持久化方案
- **原子性问题**: 单线程模型天然保证命令的原子性

### 1.2 发展历程中的重要里程碑

**2009年 - Redis 1.0发布**
- 支持基本数据类型(String、List、Set)
- 实现RDB快照持久化
- 单线程事件驱动模型

**2010年 - Redis 2.0**
- 引入Hash和ZSet数据类型
- 支持主从复制(Master-Slave)
- 发布订阅(Pub/Sub)功能

**2012年 - Redis 2.6**
- 引入Lua脚本支持
- 实现服务器端脚本执行
- 毫秒级过期时间

**2013年 - Redis 2.8**
- 引入Sentinel(哨兵)高可用方案
- 部分复制(PSYNC)优化主从同步
- 发布订阅模式改进

**2015年 - Redis 3.0**
- **重大突破**: 引入Redis Cluster(集群模式)
- 支持分布式横向扩展
- 16384个哈希槽分片机制

**2018年 - Redis 5.0**
- 引入Stream数据类型(消息队列场景)
- 新增ZPOPMIN、ZPOPMAX等命令
- 改进RDB、AOF持久化性能

**2020年 - Redis 6.0**
- **多线程网络IO**: 充分利用多核CPU处理网络请求
- ACL访问控制列表(多用户权限管理)
- 客户端缓存(Client-side caching)

**2022年 - Redis 7.0**
- Redis Functions(函数库)替代部分Lua脚本场景
- Sharded Pub/Sub(分片发布订阅)
- ACL v2增强权限控制

### 1.3 目前在业界的地位和影响力

**市场占有率**:
- DB-Engines排名: 键值存储数据库第1名,整体数据库排名前10
- GitHub Stars: 超过65,000(最受欢迎的数据库项目之一)
- 被超过90%的互联网公司使用

**典型应用企业**:
- **Twitter**: 存储时间线数据,每天数十亿次操作
- **GitHub**: Session存储、排行榜、计数器
- **StackOverflow**: 实时访问统计、页面缓存
- **微博**: 粉丝关系、Feed流、消息队列
- **淘宝/京东**: 商品缓存、购物车、库存扣减

**技术影响**:
- 推动了内存数据库的普及
- 证明了单线程+IO多路复用的高性能模型
- 影响了Memcached、LevelDB等后续项目设计

### 1.4 未来发展趋势和方向

**1. 持久化内存(PMem)支持**
- 利用Intel傲腾等持久化内存技术
- 兼顾内存性能和磁盘持久化
- 降低硬件成本

**2. 原生JSON支持增强**
- RedisJSON模块逐渐成为标配
- 支持JSONPath查询
- 替代部分文档数据库场景

**3. AI场景优化**
- 向量搜索(Vector Search)
- 支持机器学习模型推理缓存
- 实时特征存储

**4. 云原生演进**
- Serverless Redis(按需付费)
- 自动扩缩容
- 多租户隔离优化

**5. 安全性增强**
- 端到端加密
- 数据脱敏
- 审计日志完善

---

## 二、核心概念与设计理念

### 2.1 Redis的核心设计理念

**1. 简单性(Simplicity)**

Redis的设计哲学是"简单即美"。它没有复杂的查询语言(如SQL),没有表结构定义,只提供最基础的键值操作和少量数据结构。这种简单性带来了:
- 极低的学习曲线
- 极高的运行效率
- 极少的Bug和维护成本

**2. 高性能(Performance)**

Redis将性能作为第一优先级:
- 纯内存操作(避免磁盘IO)
- 单线程模型(避免锁竞争)
- 高效的数据结构(针对每种场景优化)
- IO多路复用(epoll/kqueue)

**3. 持久化(Persistence)**

区别于Memcached,Redis提供可靠的数据持久化:
- RDB快照: 定期保存内存数据到磁盘
- AOF日志: 记录每条写命令
- 混合持久化: RDB+AOF结合

**4. 数据结构丰富性(Rich Data Structures)**

Redis不仅仅是"缓存",更是"数据结构服务器":
- String: 计数器、分布式锁
- List: 消息队列、时间线
- Hash: 对象存储
- Set: 标签系统、共同好友
- ZSet: 排行榜、延迟队列
- Bitmap: 用户签到、布隆过滤器
- HyperLogLog: UV统计
- Geo: LBS位置服务
- Stream: 消息队列

### 2.2 架构设计的独特之处

**1. 单线程事件驱动模型**

**为什么选择单线程?**

传统多线程数据库的痛点:
- 线程上下文切换开销(每次切换数��微秒级)
- 锁竞争导致性能下降(互斥锁、自旋锁)
- 缓存一致性问题(多核CPU的L1/L2缓存同步)
- 并发Bug难以排查(死锁、竞态条件)

Redis单线程的优势:
- **无锁设计**: 所有操作串行执行,天然线程安全
- **高缓存命中率**: 单线程绑定CPU核心,L1/L2缓存充分利用
- **代码简洁**: 无需考虑并发控制,代码更易维护
- **原子性保证**: 每个命令都是原子执行

**单线程为何还能达到10万QPS?**

关键在于:
- **内存操作极快**: 内存访问时间约100纳秒,磁盘IO约10毫秒(差距10万倍)
- **IO多路复用**: 使用epoll监听数千个客户端连接,无需为每个连接创建线程
- **高效数据结构**: C语言实现,内存布局优化

**2. IO多路复用机制**

Redis使用epoll(Linux)、kqueue(BSD)、evport(Solaris)等系统调用实现IO多路复用:

```
while (true) {
    // 1. 等待事件发生(可读、可写、错误)
    events = epoll_wait(epfd, events, maxevents, timeout);

    // 2. 处理文件事件(网络IO)
    for (event in events) {
        if (event.readable) {
            // 读取客户端命令
            readQueryFromClient();
        }
        if (event.writable) {
            // 发送响应给客户端
            sendReplyToClient();
        }
    }

    // 3. 处理时间事件(定时任务)
    processTimeEvents();
}
```

**3. Reactor模式**

Redis采用经典的Reactor事件处理模式:
- **事件分离器**: epoll监听所有连接的读写事件
- **事件处理器**: 对应不同命令的处理函数
- **事件队列**: 将就绪事件加入队列依次处理

### 2.3 与同类工具的核心差异

**Redis vs Memcached**

| 对比维度 | Redis | Memcached |
|---------|-------|-----------|
| 数据结构 | String、List、Hash、Set、ZSet等 | 仅支持String |
| 持久化 | RDB、AOF | 不支持 |
| 分布式 | 原生Cluster | 客户端分片 |
| 单线程/多线程 | 单线程(6.0后网络IO多线程) | 多线程 |
| 内存淘汰 | 8种策略 | LRU |
| 过期策略 | 定期删除+惰性删除 | 惰性删除 |
| 适用场景 | 缓存+数据库 | 纯缓存 |

**Redis vs MongoDB**

| 对比维度 | Redis | MongoDB |
|---------|-------|---------|
| 数据模型 | 键值+数据结构 | 文档(JSON) |
| 查询能力 | 简单键值查询 | 丰富的查询语言 |
| 性能 | 极高(纯内存) | 较高(内存+磁盘) |
| 事务 | 简单事务 | ACID事务 |
| 适用场景 | 高速缓存、实时计算 | 复杂查询、文档存储 |

### 2.4 底层工作原理

**1. 内存分配器(jemalloc)**

Redis默认使用jemalloc作为内存分配器(也支持tcmalloc、libc malloc):

**jemalloc的优势**:
- **减少内存碎片**: 按固定大小类别分配(8B、16B、32B...)
- **多线程优化**: 每个线程独立内存池,减少锁竞争
- **大小对象分离**: 小对象(<14KB)和大对象分别管理

**内存碎片问题**:
```
// 示例:内存碎片产生
SET key1 "1234567890"  // 分配16字节
SET key2 "ABC"         // 分配8字节
DEL key1               // 释放16字节
SET key3 "XYZ"         // 只需8字节,但有16字节空闲

// 结果:浪费了8字节,形成碎片
```

**查看内存碎片率**:
```
INFO memory
# mem_fragmentation_ratio:1.25  // >1.5需要整理
```

**2. 对象编码(Object Encoding)**

Redis为每种数据类型设计了多种编码方式,根据数据量动态切换:

**String类型编码**:
- **int**: 存储整数,节省内存
- **embstr**: 短字符串(≤44字节),一次内存分配
- **raw**: 长字符串,SDS(Simple Dynamic String)

**List类型编码**:
- **ziplist**(压缩列表): 元素少且小时使用,连续内存,节省空间
- **linkedlist**(双向链表): 元素多时使用,插入删除O(1)
- **quicklist**(快速列表): Redis 3.2后,结合ziplist和linkedlist优点

**Hash类型编码**:
- **ziplist**: 字段少时使用
- **hashtable**: 字段多时使用

**Set类型编码**:
- **intset**(整数集合): 元素都是整数且少于512个
- **hashtable**: 其他情况

**ZSet类型编码**:
- **ziplist**: 元素少时
- **skiplist**(跳表)+hashtable: 元素多时

**为什么用跳表而不是红黑树?**
- 跳表实现简单,代码可读性高
- 跳表支持范围查询(ZRANGEBYSCORE),红黑树需要中序遍历
- 跳表插入删除不需要复杂的旋转操作
- 性能相当:查找、插入、删除都是O(logN)

**3. 事件驱动机制**

Redis的事件分为两类:

**文件事件(File Events)**: 网络IO事件
- AE_READABLE: 客户端发送命令
- AE_WRITABLE: 向客户端发送响应

**时间事件(Time Events)**: 定时任务
- serverCron: 每100ms执行一次,处理后台任务
  - 清理过期键
  - 更新统计信息
  - 触发BGSAVE、AOF重写
  - 关闭超时客户端

**事件处理优先级**:
文件事件 > 时间事件(保证响应优先)

---

## 三、基础知识

### 3.1 安装部署的完整流程

**Linux环境安装(源码编译)**

```bash
# 1. 下载源码
wget https://download.redis.io/releases/redis-7.0.0.tar.gz
tar xzf redis-7.0.0.tar.gz
cd redis-7.0.0

# 2. 编译(需要gcc)
make

# 3. 安装到/usr/local/bin
make install

# 4. 创建配置文件目录
mkdir /etc/redis
cp redis.conf /etc/redis/redis.conf

# 5. 修改配置文件
vim /etc/redis/redis.conf
```

**Docker安装(推荐)**

```bash
# 拉取镜像
docker pull redis:7.0

# 运行容器
docker run -d \
  --name redis-server \
  -p 6379:6379 \
  -v /data/redis:/data \
  redis:7.0 redis-server --appendonly yes
```

**Windows安装(开发环境)**

下载Redis for Windows: https://github.com/tporadowski/redis/releases

### 3.2 基本配置和环境设置

**核心配置项详解**

```conf
# ===== 网络配置 =====
bind 0.0.0.0                    # 允许所有IP访问(生产环境建议限制)
port 6379                       # 监听端口
protected-mode yes              # 保护模式(需要密码才能远程访问)
timeout 300                     # 客户端空闲300秒后断开

# ===== 通用配置 =====
daemonize yes                   # 后台运行
pidfile /var/run/redis.pid      # PID文件路径
loglevel notice                 # 日志级别(debug|verbose|notice|warning)
logfile /var/log/redis/redis.log  # 日志文件

# ===== 安全配置 =====
requirepass yourpassword        # 设置密码

# ===== 内存配置 =====
maxmemory 2gb                   # 最大内存限制
maxmemory-policy allkeys-lru    # 内存淘汰策略

# ===== 持久化配置 =====
# RDB配置
save 900 1                      # 900秒内至少1次修改则保存
save 300 10                     # 300秒内至少10次修改
save 60 10000                   # 60秒内至少10000次修改
dbfilename dump.rdb             # RDB文件名
dir /var/lib/redis              # 数据文件目录

# AOF配置
appendonly yes                  # 启用AOF
appendfilename "appendonly.aof" # AOF文件名
appendfsync everysec            # 每秒刷盘一次

# ===== 主从复制配置 =====
# replicaof 192.168.1.100 6379  # 配置主节点
# masterauth yourpassword        # 主节点密码

# ===== 集群配置 =====
# cluster-enabled yes             # 启用集群模式
# cluster-config-file nodes.conf  # 集群配置文件
# cluster-node-timeout 5000       # 节点超时时间
```

**性能优化配置**

```conf
# 慢查询日志
slowlog-log-slower-than 10000   # 记录超过10ms的命令
slowlog-max-len 128             # 最多保存128条慢查询

# TCP配置
tcp-backlog 511                 # TCP连接队列长度
tcp-keepalive 300               # TCP心跳间隔

# 客户端连接
maxclients 10000                # 最大客户端连接数
```

### 3.3 核心数据类型基础使用

**1. String(字符串)**

```bash
# 基本操作
SET name "Redis"              # 设置值
GET name                      # 获取值
DEL name                      # 删除键

# 数值操作
SET counter 100
INCR counter                  # 自增1 → 101
INCRBY counter 10             # 增加10 → 111
DECR counter                  # 自减1 → 110

# 批量操作
MSET key1 "value1" key2 "value2"
MGET key1 key2

# 带过期时间
SET session:1001 "user_data" EX 3600  # 1小时过期
SETEX session:1002 3600 "user_data"   # 等价写法

# 原子操作
SETNX lock:order:1001 "locked"   # 仅当不存在时设置(分布式锁)
```

**应用场景**:
- 缓存: 存储用户信息、商品详情
- 计数器: 文章阅读量、点赞数
- 分布式锁: SETNX实现互斥

**2. List(列表)**

```bash
# 左侧插入
LPUSH tasks "task1" "task2"   # tasks: [task2, task1]

# 右侧插入
RPUSH tasks "task3"           # tasks: [task2, task1, task3]

# 弹出元素
LPOP tasks                    # 返回task2, tasks: [task1, task3]
RPOP tasks                    # 返回task3, tasks: [task1]

# 阻塞弹出(队列场景)
BRPOP tasks 30                # 30秒内等待元素,否则返回nil

# 范围查询
LRANGE tasks 0 -1             # 获取所有元素
LRANGE tasks 0 9              # 获取前10个元素

# 修改元素
LSET tasks 0 "new_task"       # 修改索引0的元素
LTRIM tasks 0 99              # 只保留前100个元素
```

**应用场景**:
- 消息队列: LPUSH生产,BRPOP消费
- 时间线: 朋友圈、微博Feed流
- 最新列表: 最新文章、最新评论

**3. Hash(哈希表)**

```bash
# 设置字段
HSET user:1001 name "Alice" age 25

# 批量设置
HMSET user:1001 name "Alice" age 25 city "Beijing"

# 获取字段
HGET user:1001 name           # "Alice"
HMGET user:1001 name age      # ["Alice", "25"]
HGETALL user:1001             # {name: "Alice", age: "25", city: "Beijing"}

# 数值操作
HINCRBY user:1001 age 1       # age增加1

# 判断字段是否存在
HEXISTS user:1001 email       # 0(不存在)

# 删除字段
HDEL user:1001 city
```

**应用场景**:
- 对象存储: 用户信息、商品属性
- 购物车: key=user_id, field=product_id, value=数量

**4. Set(集合)**

```bash
# 添加元素
SADD tags:article:1001 "Redis" "Database" "NoSQL"

# 查看所有元素
SMEMBERS tags:article:1001

# 判断元素是否存在
SISMEMBER tags:article:1001 "Redis"  # 1(存在)

# 随机弹出元素
SPOP tags:article:1001 1             # 随机返回1个元素并删除

# 集合运算
SADD user:1001:following "Alice" "Bob"
SADD user:1002:following "Bob" "Charlie"
SINTER user:1001:following user:1002:following  # 交集:Bob(共同关注)
SUNION user:1001:following user:1002:following  # 并集
SDIFF user:1001:following user:1002:following   # 差集
```

**应用场景**:
- 标签系统: 文章标签、商品分类
- 共同好友: SINTER计算交集
- 抽奖系统: SPOP随机抽取

**5. ZSet(有序集合)**

```bash
# 添加元素(带分数)
ZADD rank:game 1000 "Alice" 1200 "Bob" 800 "Charlie"

# 查询排名(从小到大)
ZRANGE rank:game 0 -1              # [Charlie, Alice, Bob]
ZRANGE rank:game 0 -1 WITHSCORES   # 带分数

# 查询排名(从大到小)
ZREVRANGE rank:game 0 2            # Top 3: [Bob, Alice, Charlie]

# 根据分数范围查询
ZRANGEBYSCORE rank:game 900 1100   # [Alice]

# 获取排名(索引从0开始)
ZRANK rank:game "Alice"            # 1
ZREVRANK rank:game "Alice"         # 1(倒序排名)

# 获取分数
ZSCORE rank:game "Bob"             # 1200

# 增加分数
ZINCRBY rank:game 50 "Charlie"     # Charlie分数变为850
```

**应用场景**:
- 排行榜: 游戏积分榜、热搜榜
- 延迟队列: 分数=时间戳,ZRANGEBYSCORE获取到期任务
- 时间线: 朋友圈按时间排序

### 3.4 常用命令详解

**键管理命令**

```bash
# 查看键是否存在
EXISTS key                    # 存在返回1,不存在返回0

# 查看键类型
TYPE key                      # string|list|hash|set|zset

# 设置过期时间
EXPIRE key 3600               # 3600秒后过期
EXPIREAT key 1672531200       # Unix时间戳过期
PEXPIRE key 60000             # 60000毫秒后过期

# 查看剩余过期时间
TTL key                       # 返回秒数,-1表示永不过期,-2表示已过期
PTTL key                      # 返回毫秒数

# 移除过期时间
PERSIST key

# 重命名键
RENAME old_key new_key

# 查找键(生产环境禁用)
KEYS pattern                  # 如: KEYS user:*
SCAN cursor MATCH pattern COUNT 100  # 渐进式遍历(推荐)
```

**数据库命令**

```bash
# 切换数据库(0-15,默认16个)
SELECT 1

# 查看当前数据库键数量
DBSIZE

# 清空当前数据库
FLUSHDB

# 清空所有数据库(危险操作)
FLUSHALL
```

**事务命令**

```bash
# 开启事务
MULTI

# 添加命令到队列
SET key1 "value1"
INCR counter

# 执行事务
EXEC

# 取消事务
DISCARD

# 监视键(乐观锁)
WATCH key1
MULTI
SET key1 "new_value"
EXEC                          # 如果key1被其他客户端修改,事务失败
```

### 3.5 新手必须掌握的基础概念

**1. 键的命名规范**

推荐使用冒号分隔的层级结构:
```
user:1001:profile           # 用户1001的资料
order:2024:1001             # 2024年订单1001
session:abc123              # 会话abc123
cache:product:1001          # 商品1001的缓存
```

**2. 过期策略理解**

Redis采用"定期删除+惰性删除":
- **定期删除**: 每秒10次随机抽取设置了过期时间的键,删除已过期的
- **惰性删除**: 访问键时检查是否过期,过期则删除

**为什么不全量扫描?**
1亿个键全量扫描会阻塞CPU数秒,影响正常服务。

**3. 内存淘汰策略**

当内存达到maxmemory时,触发淘汰策略:

- **noeviction**: 拒绝写入,只允许读取和删除
- **allkeys-lru**: 在所有键中,淘汰最少使用的(LRU)
- **allkeys-lfu**: 在所有键中,淘汰访问频率最低的(LFU)
- **allkeys-random**: 在所有键中,随机淘汰
- **volatile-lru**: 在设置了过期时间的键中,淘汰最少使用的
- **volatile-lfu**: 在设置了过期时间的键中,淘汰访问频率最低的
- **volatile-random**: 在设置了过期时间的键中,随机淘汰
- **volatile-ttl**: 在设置了过期时间的键中,优先淘汰快过期的

**如何选择?**
- 纯缓存场景: allkeys-lru
- 缓存+持久化混合: volatile-lru
- 时效性数据: volatile-ttl

**4. 持久化方式选择**

- **只用RDB**: 允许丢失少量数据,追求性能
- **只用AOF**: 数据安全性要求高
- **RDB+AOF**: 混合持久化,兼顾性能和安全(推荐)

---

## 四、核心功能详解

### 4.1 持久化机制深度解析

**RDB(Redis Database)快照持久化**

**工作原理**:
1. Redis调用fork()创建子进程
2. 子进程将内存数据写入临时RDB文件
3. 写入完成后,替换旧的RDB文件
4. 整个过程主进程继续处理请求

**触发方式**:
```bash
# 手动触发
SAVE                          # 阻塞主线程(生产禁用)
BGSAVE                        # 后台保存(推荐)

# 自动触发(配置文件)
save 900 1                    # 900秒内至少1次修改
save 300 10                   # 300秒内至少10次修改
save 60 10000                 # 60秒内至少10000次修改
```

**RDB文件格式**:
```
REDIS<version><db_number><data><EOF><checksum>
```

**优点**:
- 文件紧凑,适合备份和灾难恢复
- 加载速度快(二进制格式直接加载到内存)
- 对性能影响小(fork后主进程继续服务)

**缺点**:
- 数据丢失风险(最多丢失最后一次快照后的数据)
- fork()可能阻塞主进程数十毫秒
- 数据量大时,fork()占用内存翻倍(copy-on-write)

**AOF(Append Only File)日志持久化**

**工作原理**:
1. 每条写命令追加到AOF缓冲区
2. 根据刷盘策略写入磁盘
3. AOF文件膨胀后,后台重写压缩

**刷盘策略**:
```conf
appendfsync always            # 每条命令立即刷盘(最安全,最慢)
appendfsync everysec          # 每秒刷盘一次(平衡,推荐)
appendfsync no                # 由操作系统决定(最快,最不安全)
```

**AOF重写机制**:
```bash
# 手动触发
BGREWRITEAOF

# 自动触发(配置文件)
auto-aof-rewrite-percentage 100  # AOF文件比上次重写后增长100%
auto-aof-rewrite-min-size 64mb   # AOF文件至少64MB
```

**AOF重写原理**:
遍历当前内存数据,用当前状态重新生成AOF文件。

示例:
```
# 重写前AOF(100条SET命令)
SET counter 1
SET counter 2
...
SET counter 100

# 重写后AOF(1条SET命令)
SET counter 100
```

**混合持久化(Redis 4.0+)**

```conf
aof-use-rdb-preamble yes      # 启用混合持久化
```

**原理**:
AOF重写时,将当前内存数据以RDB格式写入AOF文件头部,增量命令以AOF格式追加。

**优势**:
- 恢复速度快(RDB部分快速加载)
- 数据安全性高(AOF增量命令补充)

### 4.2 主从复制机制

**配置主从复制**

```bash
# 从节点配置
replicaof 192.168.1.100 6379
masterauth yourpassword
```

**复制流程**:

1. **全量同步(首次连接)**:
   - 从节点发送PSYNC命令
   - 主节点执行BGSAVE生成RDB
   - 主节点将RDB发送给从节点
   - 从节点清空旧数据,加载RDB
   - 主节点将缓冲区的增量命令发送给从节点

2. **增量同步(网络中断恢复)**:
   - 从节点重新连接后,发送offset
   - 主节点从replication buffer发送缺失的命令

**复制积压缓冲区**:
```conf
repl-backlog-size 1mb         # 缓冲区大小,越大越能避免全量同步
```

**主从延迟问题**:

**原因**:
- 网络延迟(跨机房可能数十ms)
- 从节点负载高,执行命令慢
- 主节点写入量大,从节点追不上

**解决方案**:
- 监控主从延迟(info replication中的offset)
- 关键业务读主库
- 使用消息队列异步通知

### 4.3 哨兵(Sentinel)高可用方案

**哨兵的职责**:
1. **监控**: 定期检查主从节点是否正常
2. **通知**: 节点故障时通知管理员
3. **自动故障转移**: 主节点宕机时,选举新主节点

**部署架构**:
```
Sentinel 1 ----\
Sentinel 2 ------> Master
Sentinel 3 ----/    /  \
                 Slave1 Slave2
```

**配置文件(sentinel.conf)**:
```conf
# 监控主节点
sentinel monitor mymaster 192.168.1.100 6379 2

# 主节点密码
sentinel auth-pass mymaster yourpassword

# 主观下线时间(30秒无响应)
sentinel down-after-milliseconds mymaster 30000

# 故障转移超时时间
sentinel failover-timeout mymaster 180000

# 同时同步的从节点数量
sentinel parallel-syncs mymaster 1
```

**故障转移流程**:

1. **主观下线(SDOWN)**: 单个哨兵认为主节点下线
2. **客观下线(ODOWN)**: 超过quorum个哨兵认为主节点下线
3. **选举Leader哨兵**: Raft算法选举
4. **选择新主节点**:
   - 优先级(replica-priority)最高
   - 复制offset最大(数据最新)
   - run_id最小(字典序)
5. **通知其他从节点**: 复制新主节点
6. **通知客户端**: 更新主节点地址

**脑裂问题**:

**场景**:
主节点因网络分区与哨兵失联,哨兵选举新主节点,但旧主节点仍接受写入。

**后果**:
旧主节点的写入数据在网络恢复后会丢失。

**解决方案**:
```conf
min-replicas-to-write 1       # 至少1个从节点在线,主节点才接受写入
min-replicas-max-lag 10       # 从节点延迟不超过10秒
```

### 4.4 集群(Cluster)分片方案

**集群特性**:
- 数据分片存储(16384个槽位)
- 自动故障转移
- 横向扩展

**槽位分配**:
```
节点A: 槽位 0-5460
节点B: 槽位 5461-10922
节点C: 槽位 10923-16383
```

**键的槽位计算**:
```
slot = CRC16(key) % 16384
```

**创建集群**:
```bash
redis-cli --cluster create \
  192.168.1.101:6379 \
  192.168.1.102:6379 \
  192.168.1.103:6379 \
  192.168.1.104:6379 \
  192.168.1.105:6379 \
  192.168.1.106:6379 \
  --cluster-replicas 1    # 每个主节点1个从节点
```

**槽位迁移**:
```bash
# 重新分配槽位
redis-cli --cluster reshard 192.168.1.101:6379

# 添加节点
redis-cli --cluster add-node 192.168.1.107:6379 192.168.1.101:6379

# 删除节点
redis-cli --cluster del-node 192.168.1.107:6379 <node-id>
```

**重定向机制**:

客户端访问键时:
1. 计算槽位
2. 如果槽位在当前节点,直接返回
3. 如果槽位在其他节点,返回MOVED重定向
4. 如果槽位正在迁移,返回ASK重定向

```bash
# MOVED重定向
GET mykey
-MOVED 12345 192.168.1.102:6379

# ASK重定向(槽位迁移中)
GET mykey
-ASK 12345 192.168.1.102:6379
```

**哈希标签(Hash Tag)**:

强制多个键分配到同一槽位:
```bash
# {user123}是哈希标签,只计算标签部分的CRC16
SET {user123}:name "Alice"
SET {user123}:age 25
# 两个键在同一槽位,支持MGET等多键操作
MGET {user123}:name {user123}:age
```

### 4.5 发布订阅(Pub/Sub)

**基本使用**:
```bash
# 订阅频道
SUBSCRIBE news sports

# 订阅模式(通配符)
PSUBSCRIBE news:*

# 发布消息
PUBLISH news "Breaking news!"

# 取消订阅
UNSUBSCRIBE news
```

**应用场景**:
- 实时消息推送
- 事件通知
- 聊天室

**缺点**:
- 消息不持久化(订阅者离线期间的消息丢失)
- 无ACK机制(消息可能丢失)

**替代方案**:
使用Stream数据类型(Redis 5.0+),支持消息持久化和消费组。

---

## 五、应用场景深度解析

### 5.1 缓存系统

**场景1: 电商商品详情缓存**

**业务需求**:
- 商品信息查询QPS高达10万+
- MySQL无法支撑如此高的并发
- 商品信息变更频率低

**解决方案**:
```bash
# 1. 查询流程
GET cache:product:1001

# 2. 缓存未命中,查询数据库
# 3. 将数据写入Redis
SETEX cache:product:1001 3600 '{"id":1001,"name":"iPhone 15","price":5999}'

# 4. 缓存更新策略
# 方案A: 定时过期(简单,但可能读到旧数据)
# 方案B: 主动更新(商品信息变更时,删除缓存)
DEL cache:product:1001
```

**缓存击穿防护**:
```bash
# 使用互斥锁(Lua脚本保证原子性)
local lock = redis.call('setnx', 'lock:product:1001', '1')
if lock == 1 then
    redis.call('expire', 'lock:product:1001', 10)
    return 1  -- 获取锁成功,查询数据库
else
    return 0  -- 其他线程等待
end
```

**场景2: Session共享**

**传统问题**:
- 负载均衡后,用户请求可能路由到不同服务器
- Session存储在单个服务器内存中,无法共享

**Redis方案**:
```bash
# 登录后存储Session
SETEX session:abc123 1800 '{"user_id":1001,"username":"Alice"}'

# 每次请求验证Session
GET session:abc123

# 活跃时延长过期时间
EXPIRE session:abc123 1800
```

### 5.2 分布式锁

**场景: 秒杀库存扣减**

**问题**:
多个用户同时抢购,需要保证库存不超卖。

**错误示例(存在竞态条件)**:
```python
# 线程A读取库存:10
stock = redis.get('stock:product:1001')  # 10
# 线程B读取库存:10
# 线程A扣减库存
redis.set('stock:product:1001', stock - 1)  # 9
# 线程B扣减库存
redis.set('stock:product:1001', stock - 1)  # 9(错误,应该是8)
```

**正确方案1: 分布式锁**

```python
import redis
import uuid
import time

client = redis.Redis()

# 加锁
lock_key = 'lock:stock:product:1001'
lock_value = str(uuid.uuid4())  # 唯一标识,防止误删其他线程的锁

# SET NX EX原子操作
locked = client.set(lock_key, lock_value, nx=True, ex=10)

if locked:
    try:
        # 业务逻辑:扣减库存
        stock = int(client.get('stock:product:1001'))
        if stock > 0:
            client.set('stock:product:1001', stock - 1)
    finally:
        # 释放锁(Lua脚本保证原子性)
        lua_script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('del', KEYS[1])
        else
            return 0
        end
        """
        client.eval(lua_script, 1, lock_key, lock_value)
else:
    # 获取锁失败,返回错误
    print("System busy, please retry")
```

**正确方案2: Lua脚本原子扣减**

```lua
-- 库存扣减脚本
local stock = redis.call('get', KEYS[1])
if tonumber(stock) > 0 then
    redis.call('decr', KEYS[1])
    return 1
else
    return 0
end
```

```python
lua_script = """..."""
result = client.eval(lua_script, 1, 'stock:product:1001')
if result == 1:
    print("Purchase success")
else:
    print("Out of stock")
```

### 5.3 排行榜系统

**场景: 游戏积分排行榜**

**需求**:
- 实时更新玩家积分
- 查询Top N排名
- 查询某玩家排名

**实现**:
```bash
# 更新积分
ZADD rank:game:2024 1250 "player:Alice"
ZADD rank:game:2024 1180 "player:Bob"
ZADD rank:game:2024 1320 "player:Charlie"

# 增加积分
ZINCRBY rank:game:2024 50 "player:Alice"  # Alice积分变为1300

# 查询Top 10
ZREVRANGE rank:game:2024 0 9 WITHSCORES

# 查询玩家排名(从0开始)
ZREVRANK rank:game:2024 "player:Alice"    # 返回1(第2名)

# 查询玩家分数
ZSCORE rank:game:2024 "player:Alice"      # 1300

# 查询某分数区间的玩家
ZRANGEBYSCORE rank:game:2024 1200 1400    # 1200-1400分的玩家
```

**优化方案: 分段排行榜**

全服排行榜数据量过大时:
```bash
# 按区服分片
ZADD rank:server1:2024 1250 "player:Alice"
ZADD rank:server2:2024 1180 "player:Bob"

# 定时合并Top 100到全服榜
ZUNIONSTORE rank:global:2024 2 rank:server1:2024 rank:server2:2024
ZREMRANGEBYRANK rank:global:2024 0 -101  # 只保留Top 100
```

### 5.4 限流器

**场景: API接口限流**

**需求**:
某个用户每分钟最多请求100次API。

**方案1: 固定窗口(简单但不精确)**

```python
import redis
import time

client = redis.Redis()
user_id = 1001
key = f'rate_limit:{user_id}:{int(time.time() / 60)}'  # 按分钟分桶

current = client.incr(key)
if current == 1:
    client.expire(key, 60)  # 第一次设置过期时间

if current > 100:
    print("Rate limit exceeded")
else:
    print("Request allowed")
```

**问题**:
在00:59和01:00分别请求100次,1秒内处理200次请求。

**方案2: 滑动窗口(精确)**

```python
import redis
import time

client = redis.Redis()
user_id = 1001
key = f'rate_limit:{user_id}'

# Lua脚本保证原子性
lua_script = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

-- 删除窗口外的记录
redis.call('zremrangebyscore', key, 0, now - window)

-- 统计窗口内请求数
local count = redis.call('zcard', key)

if count < limit then
    redis.call('zadd', key, now, now)
    redis.call('expire', key, window)
    return 1
else
    return 0
end
"""

allowed = client.eval(
    lua_script,
    1,
    key,
    100,  # limit
    60,   # window(60秒)
    int(time.time() * 1000)  # now(毫秒)
)

if allowed:
    print("Request allowed")
else:
    print("Rate limit exceeded")
```

### 5.5 延迟队列

**场景: 订单超时自动取消**

**需求**:
用户下单后30分钟未支付,自动取消订单。

**实现**:
```python
import redis
import time
import json

client = redis.Redis()

# 创建订单时,加入延迟队列
order_id = 'order:1001'
expire_time = int(time.time()) + 1800  # 30分钟后
client.zadd('delay_queue:order', {order_id: expire_time})

# 定时任务扫描到期订单
while True:
    now = int(time.time())
    # 查询到期订单
    expired_orders = client.zrangebyscore('delay_queue:order', 0, now)

    for order_id in expired_orders:
        # 处理订单取消逻辑
        print(f"Cancel order: {order_id}")
        # 从队列删除
        client.zrem('delay_queue:order', order_id)

    time.sleep(1)  # 每秒扫描一次
```

### 5.6 用户签到统计

**场景: 统计用户每月签到情况**

**需求**:
- 记录用户每天是否签到
- 查询用户某月签到天数
- 查询用户连续签到天数

**实现(Bitmap)**:
```bash
# 用户1001在2024年1月1日签到
SETBIT sign:1001:202401 0 1

# 用户1001在2024年1月5日签到
SETBIT sign:1001:202401 4 1

# 查询用户1月5日是否签到
GETBIT sign:1001:202401 4  # 返回1

# 统计用户1月签到天数
BITCOUNT sign:1001:202401  # 返回2

# 查询首次签到日期
BITPOS sign:1001:202401 1  # 返回0(第1天)
```

**内存占用**:
1个月31天,每个用户占用31 bit ≈ 4字节,100万用户仅需4MB内存。

### 5.7 UV统计(HyperLogLog)

**场景: 统计网站每日独立访客数**

**传统方案问题**:
用Set存储1亿个用户ID,占用内存约800MB。

**HyperLogLog方案**:
```bash
# 记录用户访问
PFADD uv:2024-01-01 "user:1001" "user:1002" "user:1003"

# 统计UV
PFCOUNT uv:2024-01-01  # 返回3

# 合并多天UV
PFMERGE uv:2024-01 uv:2024-01-01 uv:2024-01-02 uv:2024-01-03
PFCOUNT uv:2024-01     # 返回1月总UV
```

**内存占用**:
每个HyperLogLog仅占用12KB,统计1亿UV也只需12KB,误差率0.81%。

### 5.8 地理位置服务(Geo)

**场景: 查询附近的餐厅**

**实现**:
```bash
# 添加餐厅位置(经度、纬度、名称)
GEOADD restaurants 116.404 39.915 "restaurant:A"
GEOADD restaurants 116.408 39.920 "restaurant:B"

# 查询半径3公里内的餐厅
GEORADIUS restaurants 116.405 39.916 3 km WITHDIST

# 查询两个餐厅的距离
GEODIST restaurants "restaurant:A" "restaurant:B" km  # 返回0.63km

# 获取餐厅的经纬度
GEOPOS restaurants "restaurant:A"  # [116.404, 39.915]

# 获取GeoHash编码
GEOHASH restaurants "restaurant:A"  # wx4g0b7xrt0
```

**底层实现**:
Geo基于ZSet实现,使用GeoHash将二维坐标编码为一维字符串。

---

## 六、进阶技巧

### 6.1 性能优化方法

**1. 慢查询优化**

```bash
# 查看慢查询配置
CONFIG GET slowlog-log-slower-than  # 默认10000微秒(10ms)
CONFIG GET slowlog-max-len          # 默认128条

# 查看慢查询日志
SLOWLOG GET 10

# 分析慢查询
# 常见慢查询命令:KEYS、FLUSHALL、FLUSHDB、SMEMBERS(大集合)、HGETALL(大哈希)
```

**优化建议**:
- 禁用KEYS命令,改用SCAN
- 避免一次性获取大集合(SMEMBERS),改用SSCAN
- 控制集合大小,单个Hash/Set/ZSet不超过1万元素

**2. 管道(Pipeline)批量操作**

```python
import redis

client = redis.Redis()

# 错误示例:多次网络往返
for i in range(10000):
    client.set(f'key:{i}', f'value:{i}')  # 10000次网络IO

# 优化:使用Pipeline
pipe = client.pipeline()
for i in range(10000):
    pipe.set(f'key:{i}', f'value:{i}')
pipe.execute()  # 1次网络IO
```

**性能提升**:
Pipeline可将性能提升10-100倍(取决于网络延迟)。

**3. 数据结构优化**

**案例:存储100万用户信息**

**方案A: String存储(内存浪费)**
```bash
SET user:1001:name "Alice"
SET user:1001:age 25
SET user:1001:city "Beijing"
# 每个键占用约60字节(键名+元数据),100万用户×3字段=180MB
```

**方案B: Hash存储(推荐)**
```bash
HSET user:1001 name "Alice" age 25 city "Beijing"
# 每个Hash占用约40字节,100万用户=40MB,节省75%内存
```

**方案C: Hash分片(超大数据集)**
```bash
# 按用户ID分片,每100个用户一个Hash
HSET user:10 user:1001 '{"name":"Alice","age":25}'
HSET user:10 user:1002 '{"name":"Bob","age":30}'
# 减少Hash大小,提升性能
```

**4. 网络优化**

- **长连接**: 避免频繁创建连接
- **连接池**: 复用连接,减少握手开销
- **就近部署**: Redis与应用部署在同一机房,减少网络延迟

### 6.2 集成方案

**Spring Boot集成Redis**

```xml
<!-- pom.xml -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-redis</artifactId>
</dependency>
```

```yaml
# application.yml
spring:
  redis:
    host: localhost
    port: 6379
    password: yourpassword
    database: 0
    lettuce:
      pool:
        max-active: 8
        max-idle: 8
        min-idle: 0
        max-wait: -1ms
```

```java
@Service
public class UserService {
    @Autowired
    private StringRedisTemplate redisTemplate;

    public void cacheUser(String userId, String userData) {
        redisTemplate.opsForValue().set(
            "user:" + userId,
            userData,
            Duration.ofHours(1)
        );
    }

    public String getUser(String userId) {
        return redisTemplate.opsForValue().get("user:" + userId);
    }
}
```

**Node.js集成Redis**

```javascript
const redis = require('redis');
const client = redis.createClient({
    host: 'localhost',
    port: 6379,
    password: 'yourpassword'
});

// 存储数据
await client.set('user:1001', JSON.stringify({name: 'Alice', age: 25}));

// 获取数据
const userData = await client.get('user:1001');
console.log(JSON.parse(userData));

// 使用Hash
await client.hSet('user:1001', 'name', 'Alice');
await client.hSet('user:1001', 'age', '25');
const user = await client.hGetAll('user:1001');
```

### 6.3 Lua脚本高级应用

**原子性保证**:
Lua脚本在Redis中以原子方式执行,等同于单条命令。

**案例1: 限流器(令牌桶)**

```lua
-- 令牌桶算法
local key = KEYS[1]
local limit = tonumber(ARGV[1])      -- 桶容量
local rate = tonumber(ARGV[2])       -- 每秒生成令牌数
local now = tonumber(ARGV[3])        -- 当前时间戳

local info = redis.call('hmget', key, 'last_time', 'tokens')
local last_time = tonumber(info[1]) or now
local tokens = tonumber(info[2]) or limit

-- 计算新生成的令牌
local delta = math.max(0, now - last_time)
local new_tokens = math.min(limit, tokens + delta * rate)

if new_tokens >= 1 then
    redis.call('hmset', key, 'last_time', now, 'tokens', new_tokens - 1)
    redis.call('expire', key, 60)
    return 1  -- 允许请求
else
    return 0  -- 拒绝请求
end
```

**案例2: 分布式锁续期**

```lua
-- 检查锁是否属于自己,如果是则续期
local lock_key = KEYS[1]
local lock_value = ARGV[1]
local expire_time = tonumber(ARGV[2])

if redis.call('get', lock_key) == lock_value then
    redis.call('expire', lock_key, expire_time)
    return 1
else
    return 0
end
```

### 6.4 监控和调试

**1. INFO命令查看状态**

```bash
INFO                      # 所有信息
INFO server               # 服务器信息
INFO clients              # 客户端连接信息
INFO memory               # 内存使用情况
INFO persistence          # 持久化信息
INFO stats                # 统计信息
INFO replication          # 主从复制信息
INFO cpu                  # CPU使用情况
INFO keyspace             # 数据库键统计
```

**关键指标**:
```bash
# 内存
used_memory_human: 2.50G            # 已使用内存
used_memory_rss_human: 3.10G        # 系统分配内存
mem_fragmentation_ratio: 1.24       # 内存碎片率

# 持久化
rdb_last_save_time: 1672531200      # 最后RDB保存时间
aof_enabled: 1                      # AOF是否启用
aof_current_size: 1048576           # AOF文件大小

# 复制
role: master                        # 角色(master/slave)
connected_slaves: 2                 # 从节点数量
master_repl_offset: 123456          # 复制偏移量

# 统计
total_connections_received: 10000   # 总连接数
total_commands_processed: 1000000   # 总命令数
instantaneous_ops_per_sec: 5000     # 当前QPS
```

**2. MONITOR命令实时监控**

```bash
MONITOR  # 实时显示所有命令(生产环境慎用,影响性能)
```

**3. 使用Redis客户端工具**

- **RedisInsight**: 官方GUI工具
- **redis-cli**: 命令行工具
- **prometheus + grafana**: 监控大盘

---

## 七、高阶知识

### 7.1 源码层面的实现原理

**1. SDS(Simple Dynamic String)字符串实现**

```c
struct sdshdr {
    int len;        // 已使用长度
    int free;       // 未使用长度
    char buf[];     // 字符数组
};
```

**优势**:
- O(1)获取字符串长度(len字段)
- 避免缓冲区溢出(free字段记录剩余空间)
- 减少内存重分配(预分配空间)
- 二进制安全(可存储任意二进制数据)

**2. 跳表(Skip List)实现**

```c
typedef struct zskiplistNode {
    sds ele;                      // 元素值
    double score;                 // 分数
    struct zskiplistNode *backward;  // 后退指针
    struct zskiplistLevel {
        struct zskiplistNode *forward;  // 前进指针
        unsigned long span;              // 跨度
    } level[];                    // 层级数组
} zskiplistNode;
```

**查找过程**:
1. 从最高层开始
2. 如果下一个节点分数小于目标,继续前进
3. 否则降到下一层
4. 重复直到找到目标或到达底层

**时间复杂度**:
- 查找: O(logN)
- 插入: O(logN)
- 删除: O(logN)

**3. ZipList(压缩列表)编码**

**目的**: 节省内存,用于小数据集

**结构**:
```
<zlbytes><zltail><zllen><entry1><entry2>...<zlend>
```

- zlbytes: 总字节数
- zltail: 尾节点偏移量
- zllen: 节点数量
- entry: 节点数据

**entry编码**:
```
<prevlen><encoding><data>
```

- prevlen: 前一个节点长度(用于反向遍历)
- encoding: 数据类型和长度
- data: 实际数据

**4. 事件循环(Event Loop)实现**

```c
void aeMain(aeEventLoop *eventLoop) {
    eventLoop->stop = 0;
    while (!eventLoop->stop) {
        // 执行beforeSleep(如AOF写入)
        if (eventLoop->beforesleep != NULL)
            eventLoop->beforesleep(eventLoop);

        // 处理文件事件和时间事件
        aeProcessEvents(eventLoop, AE_ALL_EVENTS|AE_CALL_AFTER_SLEEP);
    }
}
```

### 7.2 分布式场景的高级应用

**1. RedLock分布式锁算法**

**单Redis实例的问题**:
主节点宕机后,锁丢失。

**RedLock方案**:
在N个独立Redis实例上加锁,超过半数成功才算成功。

```python
import redis
import time
import uuid

redis_clients = [
    redis.Redis(host='192.168.1.101'),
    redis.Redis(host='192.168.1.102'),
    redis.Redis(host='192.168.1.103'),
    redis.Redis(host='192.168.1.104'),
    redis.Redis(host='192.168.1.105'),
]

def acquire_lock(lock_name, expire_time=10):
    lock_value = str(uuid.uuid4())
    start_time = time.time()
    success_count = 0

    # 在所有实例上尝试加锁
    for client in redis_clients:
        try:
            if client.set(lock_name, lock_value, nx=True, ex=expire_time):
                success_count += 1
        except:
            pass

    # 超过半数成功
    if success_count >= 3:
        elapsed = time.time() - start_time
        if elapsed < expire_time:
            return lock_value  # 加锁成功

    # 加锁失败,释放已获取的锁
    release_lock(lock_name, lock_value)
    return None

def release_lock(lock_name, lock_value):
    lua_script = """
    if redis.call('get', KEYS[1]) == ARGV[1] then
        return redis.call('del', KEYS[1])
    else
        return 0
    end
    """
    for client in redis_clients:
        try:
            client.eval(lua_script, 1, lock_name, lock_value)
        except:
            pass
```

**2. 分布式Session一致性**

**问题**:
主从复制异步,写主库后立即从从库读,可能读不到。

**解决方案**:

**方案A: 强制读主库(牺牲性能)**
```python
# 写入Session后,短时间内读主库
redis_master.set('session:abc123', session_data)
# 接下来30秒读主库
redis_master.get('session:abc123')
```

**方案B: Session Sticky(会话粘滞)**
```nginx
# Nginx配置,同一用户路由到同一服务器
upstream backend {
    ip_hash;  # 根据客户端IP哈希
    server 192.168.1.101;
    server 192.168.1.102;
}
```

**方案C: 本地缓存+Redis**
```python
# 写入时同时更新本地缓存和Redis
local_cache.set('session:abc123', session_data)
redis.set('session:abc123', session_data)

# 读取时优先读本地缓存
session = local_cache.get('session:abc123')
if not session:
    session = redis.get('session:abc123')
    local_cache.set('session:abc123', session)
```

### 7.3 高可用架构设计

**架构演进路径**:

**1. 单实例(适合低并发场景)**
```
Application --> Redis
```

**2. 主从复制(读写分离)**
```
Application --写--> Master
           --读--> Slave1
           --读--> Slave2
```

**3. 哨兵模式(自动故障转移)**
```
Sentinel1 --监控--> Master
Sentinel2 --监控--> Slave1
Sentinel3 --监控--> Slave2
```

**4. 集群模式(水平扩展)**
```
Application --> Cluster
                  ├─ Master1 (Slot 0-5460)
                  │    └─ Slave1
                  ├─ Master2 (Slot 5461-10922)
                  │    └─ Slave2
                  └─ Master3 (Slot 10923-16383)
                       └─ Slave3
```

**5. 混合架构(哨兵+集群)**
```
┌──────────────────────────────────┐
│  Cluster Shard 1                 │
│  ┌─────────┐      ┌─────────┐   │
│  │ Master1 │ <--> │ Slave1  │   │
│  └─────────┘      └─────────┘   │
│       ↑ Sentinel监控             │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│  Cluster Shard 2                 │
│  ┌─────────┐      ┌─────────┐   │
│  │ Master2 │ <--> │ Slave2  │   │
│  └─────────┘      └─────────┘   │
│       ↑ Sentinel监控             │
└──────────────────────────────────┘
```

### 7.4 安全性加固

**1. 访问控制**

```bash
# Redis 6.0+ ACL
# 创建用户
ACL SETUSER alice on >password123 ~cache:* +get +set

# 查看用户
ACL LIST

# 删除用户
ACL DELUSER alice
```

**2. 网络隔离**

```conf
# 绑定内网IP
bind 192.168.1.100

# 禁用危险命令
rename-command FLUSHALL ""
rename-command FLUSHDB ""
rename-command CONFIG "CONFIG_ADMIN"
rename-command KEYS ""
```

**3. TLS加密**

```conf
# 启用TLS
tls-port 6380
tls-cert-file /path/to/redis.crt
tls-key-file /path/to/redis.key
tls-ca-cert-file /path/to/ca.crt
```

**4. 定期备份**

```bash
# 每天凌晨2点备份
0 2 * * * redis-cli BGSAVE && cp /var/lib/redis/dump.rdb /backup/dump-$(date +\%Y\%m\%d).rdb
```

---

## 八、生产环境实践

### 8.1 生产环境部署方案

**硬件选型**:

| 配置项 | 推荐配置 | 说明 |
|--------|----------|------|
| CPU | 4核+ | 单线程,主频比核心数重要 |
| 内存 | 根据数据量,建议预留50% | 避免swap |
| 磁盘 | SSD | 持久化性能更好 |
| 网络 | 万兆网卡 | 减少网络延迟 |

**内存规划**:

```
总内存 = 数据内存 + 碎片内存 + 系统预留

数据内存: 根据业务预估
碎片内存: 数据内存 × 20%
系统预留: 数据内存 × 30%(fork等操作)

示例: 10GB数据
总内存 = 10GB + 2GB + 3GB = 15GB
配置maxmemory = 12GB(留3GB给系统)
```

**系统参数优化**:

```bash
# /etc/sysctl.conf

# 关闭透明大页(THP)
echo never > /sys/kernel/mm/transparent_hugepage/enabled

# 增加TCP连接队列
net.core.somaxconn = 65535

# 禁用swap
vm.swappiness = 0

# OOM时不杀Redis进程
vm.overcommit_memory = 1
```

**集群部署拓扑**:

```
机房A:
  Master1 + Slave2
  Master2 + Slave3

机房B:
  Master3 + Slave1
  Slave1 + Slave2
```

跨机房部署,提升容灾能力。

### 8.2 监控和日志管理

**关键监控指标**:

**1. 性能指标**:
- QPS(每秒查询数)
- 响应时间(P50、P99、P999)
- 慢查询数量
- 网络吞吐量

**2. 资源指标**:
- 内存使用率
- CPU使用率
- 网络带宽
- 磁盘IO

**3. 可用性指标**:
- 主从延迟
- 集群节点状态
- 连接数
- 命中率

**Prometheus监控配置**:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'redis'
    static_configs:
      - targets: ['192.168.1.101:9121']  # redis_exporter
```

**Grafana监控大盘**:
- 导入Redis官方Dashboard(ID: 11835)
- 自定义告警规则

**日志配置**:

```conf
# redis.conf
loglevel notice
logfile /var/log/redis/redis.log

# 日志轮转(logrotate)
# /etc/logrotate.d/redis
/var/log/redis/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
```

### 8.3 备份和恢复策略

**备份策略**:

**1. RDB定时备份**:
```bash
# 每天凌晨2点全量备份
0 2 * * * redis-cli BGSAVE && \
  cp /var/lib/redis/dump.rdb /backup/dump-$(date +\%Y\%m\%d).rdb && \
  find /backup -name "dump-*.rdb" -mtime +7 -delete  # 删除7天前的备份
```

**2. AOF持续备份**:
```conf
appendonly yes
appendfsync everysec
```

**3. 远程备份**:
```bash
# 上传到对象存储(如S3)
aws s3 cp /backup/dump-20240101.rdb s3://redis-backup/
```

**恢复流程**:

**场景1: 数据误删除**
```bash
# 1. 停止Redis
redis-cli SHUTDOWN SAVE

# 2. 恢复RDB文件
cp /backup/dump-20240101.rdb /var/lib/redis/dump.rdb

# 3. 启动Redis
redis-server /etc/redis/redis.conf
```

**场景2: 整个实例宕机**
```bash
# 1. 在新服务器部署Redis
# 2. 恢复RDB文件
# 3. 如果有AOF,追加AOF文件
cat appendonly.aof >> /var/lib/redis/appendonly.aof
# 4. 启动Redis
```

**场景3: 集群节点故障**
```bash
# Sentinel自动故障转移,无需人工介入
# 修复故障节点后,重新加入集群:
redis-cli REPLICAOF 192.168.1.101 6379
```

### 8.4 常见故障处理

**故障1: 内存不足**

**现象**:
```
(error) OOM command not allowed when used memory > 'maxmemory'
```

**排查**:
```bash
INFO memory
# used_memory: 8GB
# maxmemory: 8GB
```

**解决方案**:
1. 临时扩容:`CONFIG SET maxmemory 10gb`
2. 清理过期键:`redis-cli --scan --pattern "temp:*" | xargs redis-cli DEL`
3. 调整淘汰策略:`CONFIG SET maxmemory-policy allkeys-lru`
4. 升级硬件或扩展集群

**故障2: 主从复制延迟**

**现象**:
从库数据落后主库数秒。

**排查**:
```bash
INFO replication
# master_repl_offset: 1000000
# slave_repl_offset: 900000  # 延迟100000字节
```

**原因**:
- 网络延迟
- 从库负载高
- 主库写入量大

**解决方案**:
1. 优化网络(同机房部署)
2. 从库只读,不执行慢查询
3. 增加从库数量,分散读压力
4. 使用集群模式分片

**故障3: 慢查询导致阻塞**

**现象**:
所有命令响应变慢。

**排查**:
```bash
SLOWLOG GET 10
# 1) 1) (integer) 123
#    2) (integer) 1672531200
#    3) (integer) 50000  # 耗时50ms
#    4) 1) "KEYS"
#       2) "user:*"
```

**解决方案**:
1. 禁用KEYS命令:`rename-command KEYS ""`
2. 优化慢查询(改用SCAN)
3. 拆分大集合

**故障4: 连接数耗尽**

**现象**:
```
(error) ERR max number of clients reached
```

**排查**:
```bash
INFO clients
# connected_clients: 10000
# CONFIG GET maxclients  # 10000
```

**解决方案**:
1. 临时扩容:`CONFIG SET maxclients 20000`
2. 检查连接泄漏(应用未释放连接)
3. 使用连接池

### 8.5 容量规划

**数据量预估**:

```python
# String类型
key_size = 20          # 键名平均长度
value_size = 100       # 值平均长度
metadata_size = 60     # 元数据(redisObject等)
total_per_key = key_size + value_size + metadata_size = 180字节

# 100万个键
total_memory = 100万 × 180字节 ≈ 172MB

# Hash类型(100个字段)
key_size = 20
field_size = 10 × 100  # 100个字段名
value_size = 20 × 100  # 100个字段值
metadata_size = 100
total_per_hash = 20 + 1000 + 2000 + 100 = 3120字节

# 10万个Hash
total_memory = 10万 × 3120字节 ≈ 298MB
```

**扩容时机**:

- 内存使用率 > 70%: 考虑扩容
- 内存使用率 > 85%: 必须扩容
- QPS > 单实例上限(8万): 增加分片

**扩容方案**:

**方案A: 垂直扩容(升级硬件)**
- 适合数据量增长,但QPS未达瓶颈
- 停机时间短

**方案B: 水平扩容(增加节点)**
- 适合QPS达到瓶颈
- 需要数据迁移

**集群扩容流程**:
```bash
# 1. 添加新节点
redis-cli --cluster add-node 192.168.1.107:6379 192.168.1.101:6379

# 2. 重新分配槽位
redis-cli --cluster reshard 192.168.1.101:6379

# 3. 添加从节点
redis-cli --cluster add-node 192.168.1.108:6379 192.168.1.101:6379 --cluster-slave --cluster-master-id <master-id>
```

---

## 九、最佳实践与设计模式

### 9.1 业界公认的最佳实践

**1. 键命名规范**

```
业务:对象:ID:属性

示例:
user:1001:profile         # 用户资料
order:2024:1001:status    # 订单状态
cache:product:1001        # 商品缓存
session:abc123            # 会话
lock:stock:1001           # 库存锁
```

**优点**:
- 清晰的层级结构
- 便于管理和维护
- 支持通配符查询(SCAN)

**2. 过期时间设置**

```bash
# 避免雪崩:加随机偏移
expire_time = base_time + random(0, 300)  # 基础时间±5分钟

# 示例
SET cache:product:1001 data EX (3600 + random(0, 300))
```

**3. 大key拆分**

```bash
# 错误:单个Hash存储100万字段
HSET user:all field1 value1 field2 value2 ...

# 正确:按ID分片,每个Hash最多1000字段
HSET user:1 field1 value1 field2 value2
HSET user:2 field1 value1 field2 value2
```

**4. 热key优化**

**方案A: 本地缓存**
```python
# 应用内缓存热key
local_cache = {}
def get_hot_key(key):
    if key in local_cache:
        return local_cache[key]
    value = redis.get(key)
    local_cache[key] = value
    return value
```

**方案B: 多副本**
```bash
# 将热key复制多份
SET hot_key:1 value
SET hot_key:2 value
SET hot_key:3 value

# 随机读取
random_suffix = random(1, 3)
GET hot_key:{random_suffix}
```

**5. 避免全量操作**

```bash
# 错误
KEYS user:*              # 阻塞
SMEMBERS big_set         # 一次返回百万元素

# 正确
SCAN 0 MATCH user:* COUNT 100  # 渐进式遍历
SSCAN big_set 0 COUNT 100      # 分批获取
```

### 9.2 常见设计模式

**1. 缓存更新模式**

**Cache-Aside(旁路缓存)**:
```python
def get_data(id):
    # 1. 查询缓存
    data = redis.get(f'cache:{id}')
    if data:
        return data

    # 2. 缓存未命中,查询数据库
    data = db.query(f'SELECT * FROM table WHERE id={id}')

    # 3. 写入缓存
    redis.setex(f'cache:{id}', 3600, data)
    return data

def update_data(id, new_data):
    # 1. 更新数据库
    db.update(f'UPDATE table SET data={new_data} WHERE id={id}')

    # 2. 删除缓存(而非更新)
    redis.delete(f'cache:{id}')
```

**Write-Through(写穿)**:
```python
def update_data(id, new_data):
    # 1. 先更新缓存
    redis.setex(f'cache:{id}', 3600, new_data)

    # 2. 再更新数据库
    db.update(f'UPDATE table SET data={new_data} WHERE id={id}')
```

**Write-Behind(写回)**:
```python
def update_data(id, new_data):
    # 1. 只更新缓存
    redis.setex(f'cache:{id}', 3600, new_data)

    # 2. 异步批量写入数据库
    queue.enqueue(id, new_data)
```

**2. 分布式ID生成器**

```lua
-- Lua脚本:雪花算法
local key = KEYS[1]
local timestamp = tonumber(ARGV[1])
local machine_id = tonumber(ARGV[2])

local last_timestamp = redis.call('hget', key, 'timestamp') or 0
local sequence = redis.call('hget', key, 'sequence') or 0

if timestamp == tonumber(last_timestamp) then
    sequence = (sequence + 1) % 4096
else
    sequence = 0
end

redis.call('hmset', key, 'timestamp', timestamp, 'sequence', sequence)
redis.call('expire', key, 3600)

-- 生成ID: timestamp(41bit) + machine_id(10bit) + sequence(12bit)
local id = (timestamp << 22) | (machine_id << 12) | sequence
return id
```

**3. 布隆过滤器(防止缓存穿透)**

```python
from pybloom_live import BloomFilter

# 创建布隆过滤器
bf = BloomFilter(capacity=1000000, error_rate=0.001)

# 预热:将所有有效ID加入布隆过滤器
for product_id in db.query('SELECT id FROM products'):
    bf.add(product_id)

def get_product(product_id):
    # 1. 布隆过滤器判断
    if product_id not in bf:
        return None  # 一定不存在,直接返回

    # 2. 查询缓存
    data = redis.get(f'product:{product_id}')
    if data:
        return data

    # 3. 查询数据库
    data = db.query(f'SELECT * FROM products WHERE id={product_id}')
    if data:
        redis.setex(f'product:{product_id}', 3600, data)
    return data
```

**4. 消息队列模式**

**可靠消息队列(ACK机制)**:
```python
# 生产者
def produce(queue, message):
    # 1. 消息加入待处理队列
    redis.lpush(f'queue:{queue}:pending', message)

# 消费者
def consume(queue):
    # 1. 从pending队列移动到processing队列
    message = redis.rpoplpush(
        f'queue:{queue}:pending',
        f'queue:{queue}:processing'
    )

    # 2. 处理消息
    try:
        process(message)
        # 3. 处理成功,从processing队列删除
        redis.lrem(f'queue:{queue}:processing', 1, message)
    except Exception as e:
        # 4. 处理失败,消息仍在processing队列,等待重试
        print(f'Error: {e}')

# 定时任务:将超时的消息从processing移回pending
def retry_failed():
    # 超过5分钟的消息视为失败
    timeout_messages = redis.lrange(f'queue:{queue}:processing', 0, -1)
    for message in timeout_messages:
        redis.lrem(f'queue:{queue}:processing', 1, message)
        redis.lpush(f'queue:{queue}:pending', message)
```

### 9.3 反模式(应该避免的做法)

**1. 使用KEYS命令**

```bash
# ❌ 错误:阻塞主线程
KEYS user:*

# ✅ 正确:使用SCAN
SCAN 0 MATCH user:* COUNT 100
```

**2. 单个大key**

```bash
# ❌ 错误:单个Hash存储百万字段
HSET big_hash field1 value1 field2 value2 ...

# ✅ 正确:拆分
HSET hash:1 field1 value1
HSET hash:2 field2 value2
```

**3. 不设置过期时间**

```bash
# ❌ 错误:永不过期,内存泄漏
SET cache:product:1001 data

# ✅ 正确:设置过期时间
SETEX cache:product:1001 3600 data
```

**4. 使用事务代替Lua脚本**

```bash
# ❌ 错误:MULTI/EXEC不保证原子性(其他客户端可能修改)
MULTI
GET counter
SET counter 101
EXEC

# ✅ 正确:Lua脚本原子执行
EVAL "local val = redis.call('GET', KEYS[1]); redis.call('SET', KEYS[1], val + 1)" 1 counter
```

---

## 十、学习路线

### 10.1 从入门到精通的完整学习路径

**第一阶段:入门(1-2周)**

**目标**:
- 理解Redis基本概念
- 掌握5种基本数据类型
- 能够独立安装部署

**学习内容**:
1. Redis简介和应用场景
2. 安装Redis(Docker或源码)
3. 基本命令:String、List、Hash、Set、ZSet
4. 过期时间和键管理
5. 简单缓存应用

**实践项目**:
- 实现用户信息缓存
- 实现文章点赞计数器
- 实现最新文章列表

**检验标准**:
能够回答:
- Redis是什么?解决什么问题?
- 5种数据类型的应用场景?
- 如何设置键的过期时间?

**第二阶段:进阶(2-4周)**

**目标**:
- 理解持久化机制
- 掌握主从复制
- 熟悉常见应用场景

**学习内容**:
1. RDB和AOF持久化
2. 主从复制原理和配置
3. 发布订阅
4. 事务和Lua脚本
5. 缓存设计模式(Cache-Aside等)
6. 分布式锁实现

**实践项目**:
- 实现商品秒杀系统
- 实现分布式Session
- 实现排行榜系统

**检验标准**:
- RDB和AOF的区别?
- 主从复制的流程?
- 如何实现分布式锁?

**第三阶段:高级(1-2个月)**

**目标**:
- 掌握高可用架构
- 理解集群分片原理
- 具备生产环境部署能力

**学习内容**:
1. Sentinel高可用方案
2. Cluster集群模式
3. 性能优化技巧
4. 监控和故障排查
5. 安全加固
6. 容量规划

**实践项目**:
- 搭建Redis Sentinel集群
- 搭建Redis Cluster集群
- 实现缓存雪崩/击穿/穿透防护

**检验标准**:
- Sentinel故障转移流程?
- Cluster槽位分配原理?
- 如何优化慢查询?

**第四阶段:精通(3-6个月)**

**目标**:
- 深入理解源码实现
- 能够解决复杂生产问题
- 具备架构设计能力

**学习内容**:
1. 源码分析(事件循环、数据结构)
2. 分布式场景高级应用
3. 大规模集群运维
4. 性能调优案例
5. 业界最佳实践

**实践项目**:
- 阅读Redis核心源码
- 实现自定义Redis模块
- 优化生产环境Redis集群

**检验标准**:
- SDS、跳表、ziplist实现原理?
- 单线程为何高性能?
- 如何设计百万QPS的Redis架构?

---

## 十一、学习资源推荐

### 11.1 官方文档和教程

**官方资源**:
- Redis官网: https://redis.io
- 官方文档: https://redis.io/docs
- 官方命令参考: https://redis.io/commands
- GitHub仓库: https://github.com/redis/redis

### 11.2 优质书籍推荐

**中文书籍**:
1. **《Redis设计与实现》**(黄健宏)
   - 深入讲解Redis内部实现
   - 适合进阶学习

2. **《Redis开发与运维》**(付磊、张益军)
   - 覆盖开发、运维、优化
   - 实战案例丰富

3. **《Redis实战》**(Josiah L. Carlson)
   - 实战项目导向
   - 适合入门

**英文书籍**:
1. **Redis in Action** (Josiah L. Carlson)
2. **The Little Redis Book** (Karl Seguin)

### 11.3 在线课程和视频

- 尚硅谷Redis教程(B站)
- 黑马程序员Redis实战(B站)
- Redis University(官方免费课程)

### 11.4 技术博客和文章

- Redis作者博客: http://antirez.com
- 美团技术团队Redis实践
- 阿里云Redis最佳实践

### 11.5 开源项目和示例

- **Redisson**: Java Redis客户端,实现了分布式锁、限流器等
- **redis-py**: Python Redis客户端
- **ioredis**: Node.js Redis客户端

### 11.6 社区和论坛

- Redis中文社区: http://redis.cn
- Stack Overflow Redis标签
- Redis GitHub Issues

---

## 十二、常见问题解答(FAQ)

### 12.1 新手常见问题

**Q1: Redis是数据库还是缓存?**

A: Redis既是数据库也是缓存。
- 作为缓存:提供高速数据访问,配合MySQL等使用
- 作为数据库:支持持久化,可作为主存储(适合特定场景)

**Q2: Redis为什么这么快?**

A: 主要原因:
1. 纯内存操作(内存访问速度是磁盘的10万倍)
2. 单线程模型(避免锁竞争和上下文切换)
3. IO多路复用(epoll高效处理网络IO)
4. 高效的数据结构(C语言实现,内存布局优化)

**Q3: Redis单线程会成为性能瓶颈吗?**

A: 大部分场景不会,因为:
- 单实例可达10万QPS,足够大多数应用
- 瓶颈通常在网络IO,而非CPU
- Redis 6.0引入多线程网络IO,进一步提升性能
- 需要更高性能时,使用集群模式水平扩展

**Q4: 什么时候用String,什么时候用Hash?**

A:
- **String**: 简单值(用户名、计数器、Session等)
- **Hash**: 对象(用户信息、商品属性等)

示例:
```bash
# String:每个属性独立存储,内存浪费
SET user:1001:name "Alice"
SET user:1001:age 25

# Hash:一个对象存储所有属性,节省内存
HSET user:1001 name "Alice" age 25
```

**Q5: 如何选择RDB还是AOF?**

A:
- **纯缓存**: RDB(性能好,允许丢数据)
- **重要数据**: AOF everysec(平衡性能和安全)
- **金融数据**: AOF always(最安全)
- **推荐**: RDB + AOF混合持久化

**Q6: Redis如何实现分布式锁?**

A: 使用`SET NX EX`命令:
```python
# 加锁
locked = redis.set('lock:resource', 'unique_value', nx=True, ex=10)

# 释放锁(Lua脚本保证原子性)
lua = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""
redis.eval(lua, 1, 'lock:resource', 'unique_value')
```

**Q7: Redis如何删除大key?**

A:
```bash
# 错误:直接删除会阻塞
DEL big_key

# 正确:渐进式删除
# Hash/Set/ZSet: 先删除部分元素,再删除key
HSCAN big_hash 0 COUNT 100  # 分批删除字段
HDEL big_hash field1 field2 ...

# Redis 4.0+: 异步删除
UNLINK big_key
```

**Q8: 缓存雪崩、击穿、穿透的区别?**

A:
- **雪崩**: 大量key同时失效,请求打到数据库
  - 解决:过期时间加随机值、多级缓存
- **击穿**: 热点key失效,大量请求打到数据库
  - 解决:互斥锁、热点key永不过期
- **穿透**: 查询不存在的数据,缓存和数据库都没有
  - 解决:布隆过滤器、缓存空值

**Q9: Redis如何实现消息队列?**

A: 多种方案:
1. **List**: LPUSH生产,BRPOP消费(简单,无ACK)
2. **Pub/Sub**: 实时推送(不持久化,消息可能丢失)
3. **Stream**: Redis 5.0+,消费组、ACK、持久化(推荐)

**Q10: Redis集群如何扩容?**

A:
```bash
# 1. 添加新节点
redis-cli --cluster add-node 新节点IP:端口 集群节点IP:端口

# 2. 重新分配槽位
redis-cli --cluster reshard 集群节点IP:端口

# 3. 添加从节点
redis-cli --cluster add-node 从节点IP:端口 集群节点IP:端口 --cluster-slave
```

### 12.2 进阶问题

**Q11: 为什么集群模式不支持多键操作?**

A: 因为多个key可能分布在不同节点上,无法保证事务性。

解决方案:使用Hash Tag
```bash
# {user123}是Hash Tag,只计算{}内的CRC16
MGET {user123}:name {user123}:age  # 确保在同一槽位
```

**Q12: 主从复制延迟如何解决?**

A:
1. **读主库**: 关键业务读主库,牺牲读写分离
2. **监控延迟**: 超过阈值时切换到主库
3. **异步通知**: 写入后通过消息队列异步通知,不依赖同步读
4. **优化网络**: 主从部署在同一机房

**Q13: Redis内存碎片如何处理?**

A:
```bash
# 1. 查看碎片率
INFO memory
# mem_fragmentation_ratio: 1.5  (>1.5需要处理)

# 2. 主动整理碎片(Redis 4.0+)
CONFIG SET activedefrag yes

# 3. 手动整理(重启)
redis-cli SHUTDOWN SAVE
redis-server /etc/redis/redis.conf
```

**Q14: Redis如何实现幂等性?**

A:
```python
# 方案1: 唯一ID
request_id = 'req:abc123'
if redis.setnx(request_id, '1'):
    redis.expire(request_id, 3600)
    # 执行业务逻辑
else:
    return "重复请求"

# 方案2: Lua脚本
lua = """
local exists = redis.call('exists', KEYS[1])
if exists == 0 then
    redis.call('set', KEYS[1], ARGV[1])
    redis.call('expire', KEYS[1], 3600)
    return 1
else
    return 0
end
"""
```

**Q15: Redis如何实现限流?**

A: 见5.4节"限流器"详细实现。

**Q16: 如何监控Redis性能?**

A:
1. 使用`INFO`命令查看实时指标
2. 部署redis_exporter + Prometheus + Grafana
3. 关键指标:
   - QPS(instantaneous_ops_per_sec)
   - 内存使用率(used_memory / maxmemory)
   - 慢查询数量
   - 主从延迟(master_repl_offset - slave_repl_offset)

**Q17: Redis如何实现排行榜?**

A: 见5.3节"排行榜系统"详细实现。

**Q18: Redis Cluster脑裂如何解决?**

A:
```conf
# 配置:至少N个从节点在线,主节点才接受写入
min-replicas-to-write 1
min-replicas-max-lag 10  # 从节点延迟不超过10秒
```

**Q19: Redis如何实现延迟队列?**

A: 见5.5节"延迟队列"详细实现。

**Q20: Redis如何保证数据一致性?**

A:
- **单机**: 单线程模型天然保证
- **主从**: 最终一致性(异步复制)
- **集群**: 最终一致性(AP模型,牺牲强一致性)
- **需要强一致性**: 使用Redis事务+WATCH或Lua脚本

---

## 十三、思考题与实战练习

### 13.1 思考题(由浅入深)

**基础题**:

1. Redis为什么采用单线程模型?有什么优缺点?
2. RDB和AOF持久化的区别是什么?如何选择?
3. 什么是缓存击穿?如何防止?
4. String和Hash存储对象有什么区别?
5. 如何查看Redis的内存使用情况?

**进阶题**:

6. 主从复制的完整流程是什么?
7. Sentinel如何实现自动故障转移?
8. Redis Cluster的槽位分配机制是什么?
9. 为什么ZSet使用跳表而不是红黑树?
10. 如何实现一个可靠的分布式锁?

**高级题**:

11. 如何设计一个支持百万QPS的Redis架构?
12. Redis单线程如何支持10万QPS?
13. 如何解决主从复制的延迟问题?
14. RedLock分布式锁有什么争议?
15. 如何优化Redis的内存使用?

### 13.2 实战项目练习

**项目1: 简易秒杀系统(入门)**

**目标**: 实现商品秒杀,防止超卖

**要求**:
- 使用Redis存储库存
- Lua脚本保证原子性
- 支持1000并发

**参考实现**: 见5.2节"分布式锁"

---

**项目2: 排行榜系统(进阶)**

**目标**: 实现实时游戏排行榜

**要求**:
- 支持实时更新积分
- 查询Top 100
- 查询玩家排名
- 支持百万玩家

**参考实现**: 见5.3节"排行榜系统"

---

**项目3: 分布式Session系统(进阶)**

**目标**: 实现跨服务器Session共享

**要求**:
- 用户登录后生成Session
- 支持多台应用服务器
- Session过期时间30分钟
- 用户活跃时自动续期

---

**项目4: 消息队列系统(高级)**

**目标**: 实现可靠消息队列

**要求**:
- 支持消息持久化
- 支持ACK机制
- 支持消息重试
- 支持消费组

**提示**: 使用Redis Stream或List+Lua脚本

---

**项目5: 缓存系统优化(高级)**

**目标**: 为电商网站设计缓存方案

**要求**:
- 商品详情缓存
- 防止缓存雪崩、击穿、穿透
- 支持缓存预热
- 监控缓存命中率

**思考**:
- 如何设置过期时间?
- 如何更新缓存?
- 如何处理热点数据?

---

### 13.3 面试高频问题

**架构设计类**:

1. 设计一个日活千万的签到系统(考察Bitmap)
2. 设计一个附近的人功能(考察Geo)
3. 设计一个UV统计系统(考察HyperLogLog)
4. 如何实现限流器?(考察滑动窗口)
5. 如何设计Redis高可用架构?(考察Sentinel/Cluster)

**原理深挖类**:

6. Redis单线程为何高性能?
7. 跳表的查找过程是怎样的?
8. SDS相比C字符串的优势?
9. Redis过期键删除策略?
10. AOF重写的原理?

**实战问题类**:

11. 如何排查Redis慢查询?
12. 如何处理Redis内存满了?
13. 主从复制延迟如何解决?
14. 如何保证缓存和数据库一致性?
15. 分布式锁如何防止死锁?

---

**全文完**

通过本文档的系统学习,你将掌握Redis从基础到高级的全部知识,具备在生产环境中设计、部署、优化Redis系统的能力。

建议学习路径:
1. 第一遍通读全文,建立知识框架
2. 第二遍精读重点章节,动手实践
3. 完成实战项目,巩固所学
4. 深入源码和论文,达到精通水平

**持续学习资源**:
- 关注Redis官方博客
- 阅读大厂技术博客(美团、阿里、京东)
- 参与开源项目
- 分享你的实践经验
