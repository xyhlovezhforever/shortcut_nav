# 后端面试题 - 数据库与SQL优化篇

## 一、数据库基础理论

### 1.1 关系型数据库核心概念

#### Q1: 什么是ACID特性?请详细解释每个特性及其应用场景

**答案**:

ACID是关系型数据库事务必须满足的四个特性:

**1. Atomicity (原子性)**
- **定义**: 事务中的所有操作要么全部成功,要么全部失败回滚
- **实现机制**:
  - Undo Log: 记录修改前的值,回滚时使用
  - 回滚段: 保存事务开始前的数据快照
- **应用场景**:
  ```
  转账操作:
  BEGIN TRANSACTION;
  UPDATE accounts SET balance = balance - 100 WHERE id = 1; -- A账户扣款
  UPDATE accounts SET balance = balance + 100 WHERE id = 2; -- B账户收款
  COMMIT;

  如果第二步失败,第一步必须回滚,否则会丢失100元
  ```

**2. Consistency (一致性)**
- **定义**: 事务执行前后,数据库从一个一致性状态转换到另一个一致性状态
- **约束条件**:
  - 主键约束
  - 外键约束
  - 唯一约束
  - 检查约束
  - 业务规则约束
- **应用场景**:
  ```
  库存扣减:
  - 约束1: 库存 >= 0 (CHECK约束)
  - 约束2: 订单商品必须在商品表存在 (外键约束)
  - 业务规则: 总库存 = 可用库存 + 锁定库存
  ```

**3. Isolation (隔离性)**
- **定义**: 并发事务之间互不干扰
- **隔离级别**:
  - READ UNCOMMITTED: 可能脏读
  - READ COMMITTED: 可能不可重复读
  - REPEATABLE READ: 可能幻读
  - SERIALIZABLE: 完全隔离,性能最差
- **应用场景**:
  ```
  电商秒杀:
  - 隔离级别太低: 可能出现超卖(脏读)
  - 隔离级别太高: 并发性能差,吞吐量低
  - 推荐: READ COMMITTED + 乐观锁
  ```

**4. Durability (持久性)**
- **定义**: 事务提交后,数据永久保存,即使系统崩溃也不会丢失
- **实现机制**:
  - Redo Log: 记录修改后的值
  - WAL (Write-Ahead Logging): 先写日志,再写数据
  - 双写缓冲: 防止页断裂
- **应用场景**:
  ```
  支付系统:
  - 用户支付成功后服务器宕机
  - 重启后必须能恢复支付记录
  - 否则会导致资金纠纷
  ```

#### Q2: 事务隔离级别有哪些?各自会产生什么问题?如何选择?

**答案**:

**四种隔离级别对比**:

| 隔离级别 | 脏读 | 不可重复读 | 幻读 | 实现方式 | 性能 |
|---------|------|-----------|------|---------|------|
| READ UNCOMMITTED | ✓ | ✓ | ✓ | 不加锁 | 最高 |
| READ COMMITTED | ✗ | ✓ | ✓ | 行级共享锁 | 较高 |
| REPEATABLE READ | ✗ | ✗ | ✓ | MVCC+行锁 | 中等 |
| SERIALIZABLE | ✗ | ✗ | ✗ | 表级锁/间隙锁 | 最低 |

**问题详解**:

**1. 脏读 (Dirty Read)**
```
时间线:
T1: BEGIN
T1: UPDATE users SET balance = 1000 WHERE id = 1
T2: BEGIN
T2: SELECT balance FROM users WHERE id = 1  -- 读到1000 (未提交数据)
T1: ROLLBACK  -- 回滚,balance实际还是原值
T2: 使用1000这个脏数据进行业务逻辑
```
**危害**: 读取到未提交的数据,该数据可能被回滚

**2. 不可重复读 (Non-Repeatable Read)**
```
时间线:
T1: BEGIN
T1: SELECT balance FROM users WHERE id = 1  -- 读到500
T2: BEGIN
T2: UPDATE users SET balance = 1000 WHERE id = 1
T2: COMMIT
T1: SELECT balance FROM users WHERE id = 1  -- 读到1000
T1: 两次读取结果不一致!
```
**危害**: 同一事务内多次读取同一行数据,结果不一致

**3. 幻读 (Phantom Read)**
```
时间线:
T1: BEGIN
T1: SELECT COUNT(*) FROM orders WHERE user_id = 1  -- 返回5条
T2: BEGIN
T2: INSERT INTO orders (user_id, amount) VALUES (1, 100)
T2: COMMIT
T1: SELECT COUNT(*) FROM orders WHERE user_id = 1  -- 返回6条
T1: 出现了"幻影"记录!
```
**危害**: 同一事务内多次查询,返回的记录数不一致

**如何选择隔离级别**:

**业务场景决定**:
```
1. 数据分析/报表系统
   → READ UNCOMMITTED
   → 理由: 对数据一致性要求不高,追求性能

2. 普通Web应用 (推荐)
   → READ COMMITTED (Oracle/PostgreSQL默认)
   → 理由: 平衡性能和一致性,避免脏读即可

3. 金融/支付系统
   → REPEATABLE READ (MySQL默认)
   → 理由: 需要保证事务内数据一致性

4. 库存/账户系统
   → SERIALIZABLE (谨慎使用)
   → 理由: 绝对不能出错,宁可牺牲性能
```

**MySQL的特殊优化**:
- MySQL的REPEATABLE READ通过Next-Key Lock避免了幻读
- 因此MySQL的RR级别既能防止幻读,性能又比SERIALIZABLE好
- 这是MySQL选择RR作为默认隔离级别的原因

#### Q3: 什么是MVCC?它如何解决并发问题?

**答案**:

**MVCC (Multi-Version Concurrency Control) - 多版本并发控制**

**核心思想**:
- 不加锁的情况下,通过保存数据的多个历史版本来实现并发控制
- 读操作不阻塞写操作,写操作也不阻塞读操作
- 大幅提升并发性能

**实现机制**:

**1. 隐藏字段**
每行记录包含三个隐藏字段:
```
DB_TRX_ID    : 6字节,最后修改该行的事务ID
DB_ROLL_PTR  : 7字节,回滚指针,指向Undo Log中的旧版本
DB_ROW_ID    : 6字节,行ID,如果没有主键则自动生成
```

**2. Undo Log版本链**
```
当前数据: id=1, name='张三', age=25, trx_id=100
    ↓ (回滚指针)
旧版本1: id=1, name='张三', age=24, trx_id=90
    ↓ (回滚指针)
旧版本2: id=1, name='张三', age=23, trx_id=80
```

**3. ReadView快照读**
事务开始时创建ReadView,记录:
- `m_ids`: 当前活跃事务ID列表
- `min_trx_id`: 最小活跃事务ID
- `max_trx_id`: 下一个将要分配的事务ID
- `creator_trx_id`: 创建该ReadView的事务ID

**4. 可见性判断算法**
```
对于某个数据版本的trx_id:

IF trx_id < min_trx_id:
    → 该版本在ReadView创建前已提交,可见

ELSE IF trx_id >= max_trx_id:
    → 该版本在ReadView创建后才开始,不可见

ELSE IF trx_id IN m_ids:
    → 该版本的事务还未提交,不可见
    → 沿着Undo Log链找更早的版本

ELSE:
    → 该版本已提交,可见
```

**实战案例**:

```
场景:电商订单查询系统

-- 事务1 (trx_id=100): 查询订单
BEGIN;
SELECT * FROM orders WHERE id = 1;
-- 创建ReadView: min_trx_id=100, max_trx_id=105, m_ids=[100,101,102]

-- 事务2 (trx_id=101): 修改订单状态
BEGIN;
UPDATE orders SET status = 'PAID' WHERE id = 1;  -- 未提交

-- 事务1再次查询
SELECT * FROM orders WHERE id = 1;
-- 判断: trx_id=101 在 m_ids 中,不可见
-- 通过Undo Log读取旧版本,status仍是'PENDING'
-- 实现了可重复读!

-- 事务2提交
COMMIT;

-- 事务1再次查询
SELECT * FROM orders WHERE id = 1;
-- ReadView不变,trx_id=101仍在m_ids中,仍读取旧版本
-- 这就是REPEATABLE READ的实现原理
```

**MVCC的优势**:
1. **读写不冲突**: SELECT不阻塞UPDATE,并发性能高
2. **无锁读**: 快照读不需要加锁,避免死锁
3. **历史版本**: 支持闪回查询,数据恢复

**MVCC的局限**:
1. **空间开销**: 需要存储多个版本,Undo Log占用空间
2. **只适用于RC和RR**: SERIALIZABLE仍需加锁
3. **当前读需加锁**: `SELECT ... FOR UPDATE`会加锁

### 1.2 索引原理与优化

#### Q4: B+树索引为什么比B树和红黑树更适合数据库?

**答案**:

**三种数据结构对比**:

**1. 红黑树 (Red-Black Tree)**
```
结构:
       10(黑)
      /      \
    5(红)    15(黑)
   /  \      /   \
 3(黑) 7(黑) 12(红) 18(红)

特点:
- 高度: O(log n)
- 每个节点存储一个键值对
- 磁盘IO次数 = 树高度

问题:
- 100万条数据,树高约20层
- 查询需要20次磁盘IO
- 每次IO耗时10ms,总耗时200ms
- 性能无法接受!
```

**2. B树 (B-Tree)**
```
结构: (每个节点存储数据)
       [10, 20, 30]
      /   |   |   \
  [5,7] [12,15] [25,28] [35,40]

特点:
- 多路平衡树,每个节点可存储多个键值对
- 叶子节点和非叶子节点都存储数据
- 高度低,减少磁盘IO

问题:
- 非叶子节点存储数据,导致每个节点能存储的键数量减少
- 树的高度增加
- 范围查询需要遍历多个分支,效率低
```

**3. B+树 (B+ Tree)**
```
结构: (非叶子节点只存索引,叶子节点存数据并连成链表)
         [10, 20, 30]
        /    |    |   \
    [5,7,10] → [12,15,20] → [25,28,30] → [35,40,45]

特点:
- 非叶子节点只存索引,不存数据
- 所有数据都在叶子节点
- 叶子节点用双向链表连接
- MySQL中InnoDB的页大小是16KB
```

**为什么B+树最适合数据库?**

**理由1: 树高度更低,磁盘IO次数少**
```
假设:
- 索引字段为INT (4字节)
- 指针大小为8字节
- 每个节点大小为16KB

B树计算:
- 每个节点存储: 键(4B) + 数据(1KB) + 指针(8B) ≈ 1012B
- 每个节点能存储: 16KB / 1012B ≈ 16个键值对
- 3层B树能存储: 16 × 16 × 16 = 4096条记录

B+树计算:
- 非叶子节点存储: 键(4B) + 指针(8B) = 12B
- 每个节点能存储: 16KB / 12B ≈ 1365个键
- 3层B+树能存储: 1365 × 1365 × 16 ≈ 2900万条记录!

结论:
- 同样高度,B+树能存储更多数据
- 或者说,相同数据量,B+树高度更低
- 树高度每降低1层,就减少1次磁盘IO
```

**理由2: 范围查询效率高**
```
查询: SELECT * FROM users WHERE age BETWEEN 20 AND 30

B树:
1. 找到age=20的节点
2. 中序遍历,需要回溯到父节点
3. 再下降到下一个节点
4. 重复步骤2-3直到age=30
→ 需要多次随机IO,效率低

B+树:
1. 找到age=20的叶子节点
2. 顺着双向链表向右遍历
3. 直到age=30
→ 只需1次定位 + 顺序IO,效率极高!
```

**理由3: 全表扫描效率高**
```
B树:
- 需要中序遍历整棵树
- 频繁回溯,随机IO

B+树:
- 直接从最左叶子节点开始
- 沿着链表顺序读取
- 利用操作系统的预读机制(read-ahead)
- 顺序IO,效率是随机IO的数百倍!
```

**理由4: 适合操作系统的页管理**
```
操作系统和数据库都以页(Page)为单位进行IO:
- Linux默认页大小: 4KB
- MySQL InnoDB页大小: 16KB
- 一次IO读取整页数据

B+树:
- 每个节点大小 = 1页 = 16KB
- 读取1个节点 = 1次IO
- 完美匹配!

红黑树:
- 每个节点只有几十字节
- 读取16KB却只用到几十字节
- 浪费IO带宽
```

**实际性能对比**:

```
场景: 1000万条用户记录

红黑树:
- 树高度: log₂(10000000) ≈ 24层
- 磁盘IO次数: 24次
- 查询耗时: 24 × 10ms = 240ms

B+树:
- 树高度: 3层 (计算如上)
- 磁盘IO次数: 3次 (根节点可能在内存中,实际2次)
- 查询耗时: 2 × 10ms = 20ms

性能提升: 240ms / 20ms = 12倍!
```

**总结**:
1. B+树高度低 → 减少磁盘IO
2. 叶子节点链表 → 范围查询快
3. 非叶子节点不存数据 → 每个节点存更多索引
4. 节点大小匹配页大小 → 充分利用IO

#### Q5: 联合索引的最左前缀原则是什么?为什么会有这个限制?

**答案**:

**最左前缀原则定义**:
对于联合索引 `INDEX(a, b, c)`,只有以下查询能使用该索引:
- `WHERE a = ?`
- `WHERE a = ? AND b = ?`
- `WHERE a = ? AND b = ? AND c = ?`

但以下查询无法使用索引:
- `WHERE b = ?` (缺少a)
- `WHERE c = ?` (缺少a和b)
- `WHERE b = ? AND c = ?` (缺少a)

**底层原理**:

**联合索引的存储结构**:
```
假设建立索引: INDEX idx_abc (a, b, c)

数据按(a, b, c)的字典序排列:
(a=1, b=1, c=1)
(a=1, b=1, c=2)
(a=1, b=2, c=1)
(a=1, b=2, c=2)
(a=2, b=1, c=1)
(a=2, b=1, c=2)
(a=2, b=2, c=1)
(a=2, b=2, c=2)

排序规则:
1. 先按a排序
2. a相同时,按b排序
3. a和b都相同时,按c排序
```

**为什么必须最左匹配?**

**情况1: WHERE a = 1**
```
查找过程:
1. 在索引中二分查找a=1的起始位置
2. 找到(a=1, b=1, c=1)
3. 继续向后扫描,直到a!=1为止
4. 返回所有a=1的记录

索引有效: ✓
原因: a是有序的,可以二分查找
```

**情况2: WHERE a = 1 AND b = 2**
```
查找过程:
1. 在索引中二分查找a=1的起始位置
2. 在a=1的范围内,b也是有序的
3. 继续二分查找b=2的位置
4. 找到(a=1, b=2, c=1)
5. 返回所有a=1且b=2的记录

索引有效: ✓
原因: 在a确定的前提下,b是有序的
```

**情况3: WHERE b = 2 (缺少a)**
```
问题分析:
- 索引中b的顺序: 1,1,2,2,1,1,2,2
- b是无序的! (只有在a确定时才有序)
- 无法使用二分查找
- 只能全索引扫描

索引无效: ✗
原因: b在全局范围内是无序的
```

**情况4: WHERE a = 1 AND c = 2 (缺少b)**
```
查找过程:
1. 在索引中二分查找a=1的起始位置 ✓
2. 在a=1的范围内,c的顺序: 1,2,1,2
3. c是无序的! (只有在a和b都确定时才有序)
4. 无法二分查找c
5. 需要扫描所有a=1的记录,逐个判断c

索引部分有效: △
- a使用了索引定位
- c只能通过过滤条件筛选
- MySQL称为"索引条件下推"(Index Condition Pushdown)
```

**实战案例**:

```sql
-- 建表
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(50),
    age INT,
    city VARCHAR(50),
    INDEX idx_name_age_city (name, age, city)
);

-- 插入100万条测试数据
-- name: 1000个不同姓名
-- age: 18-60岁
-- city: 100个城市

-- 案例1: 最左前缀完整匹配 ✓
EXPLAIN SELECT * FROM users
WHERE name = '张三' AND age = 25 AND city = '北京';
→ type: ref (使用索引)
→ key: idx_name_age_city
→ rows: 10 (预估扫描行数)

-- 案例2: 最左前缀部分匹配 ✓
EXPLAIN SELECT * FROM users
WHERE name = '张三' AND age = 25;
→ type: ref
→ key: idx_name_age_city
→ rows: 100

-- 案例3: 只有最左列 ✓
EXPLAIN SELECT * FROM users
WHERE name = '张三';
→ type: ref
→ key: idx_name_age_city
→ rows: 1000

-- 案例4: 缺少最左列 ✗
EXPLAIN SELECT * FROM users
WHERE age = 25 AND city = '北京';
→ type: ALL (全表扫描)
→ key: NULL (未使用索引)
→ rows: 1000000

-- 案例5: 跳过中间列 △
EXPLAIN SELECT * FROM users
WHERE name = '张三' AND city = '北京';
→ type: ref
→ key: idx_name_age_city
→ key_len: 53 (只用了name部分)
→ Extra: Using index condition (city作为过滤条件)
```

**常见误区**:

**误区1: 查询条件顺序必须和索引顺序一致**
```sql
-- 错误理解
WHERE age = 25 AND city = '北京' AND name = '张三'  -- 认为无法使用索引

-- 实际情况
MySQL查询优化器会自动调整顺序:
WHERE name = '张三' AND age = 25 AND city = '北京'
→ 可以正常使用索引!

结论: 查询条件的书写顺序不重要,MySQL会自动优化
```

**误区2: 范围查询后面的索引列都失效**
```sql
-- 场景
WHERE name = '张三' AND age > 25 AND city = '北京'

-- 实际情况
1. name使用索引定位 ✓
2. age使用索引范围扫描 ✓
3. city无法使用索引 ✗ (age是范围查询,city在age确定后才有序)

-- MySQL 5.6+优化
Extra: Using index condition
→ city作为索引条件下推,在存储引擎层过滤
→ 减少回表次数
```

**如何设计联合索引?**

**原则1: 区分度高的列放前面**
```sql
-- 场景: 查询某个城市的某个年龄段的用户

-- 方案1: INDEX(city, age)
city区分度: 100个城市 → 每个城市平均10000条记录
age区分度: 43个年龄 → 每个年龄平均23255条记录
→ WHERE city='北京' AND age=25 → 先过滤到10000条,再过滤到200条

-- 方案2: INDEX(age, city)
→ WHERE age=25 AND city='北京' → 先过滤到23255条,再过滤到200条

结论: 方案1更优,city区分度更高,应该放前面
```

**原则2: 等值查询的列放前面**
```sql
-- 场景: WHERE name='张三' AND age>25

-- 方案1: INDEX(name, age) ✓
→ name定位 + age范围扫描

-- 方案2: INDEX(age, name) ✗
→ age范围扫描 + name过滤(无法使用索引)

结论: 等值条件(=)放前面,范围条件(>, <, BETWEEN)放后面
```

**原则3: 覆盖常用查询组合**
```sql
-- 常用查询1: WHERE name='张三'
-- 常用查询2: WHERE name='张三' AND age=25
-- 常用查询3: WHERE name='张三' AND age=25 AND city='北京'

-- 最优索引: INDEX(name, age, city)
→ 一个索引覆盖所有查询!
```

### 1.3 SQL优化实战

#### Q6: 如何分析和优化慢SQL?请给出完整的优化流程

**答案**:

**完整的SQL优化流程**:

**第一步: 发现慢SQL**

**方法1: 慢查询日志**
```sql
-- MySQL配置
SET GLOBAL slow_query_log = 'ON';
SET GLOBAL long_query_time = 1;  -- 超过1秒记录
SET GLOBAL log_queries_not_using_indexes = 'ON';  -- 记录未使用索引的查询

-- 分析慢查询日志
mysqldumpslow -s t -t 10 /var/log/mysql/slow.log
→ 按时间排序,显示前10条最慢的SQL

输出示例:
Count: 1523 Time=2.35s (3579s) Lock=0.00s (0s) Rows=100.5 (153115)
SELECT * FROM orders WHERE user_id = N AND status = 'S'
```

**方法2: Performance Schema**
```sql
-- 查询执行时间最长的SQL
SELECT
    DIGEST_TEXT AS query,
    COUNT_STAR AS exec_count,
    AVG_TIMER_WAIT/1000000000000 AS avg_time_sec,
    SUM_ROWS_EXAMINED AS total_rows_scanned
FROM performance_schema.events_statements_summary_by_digest
ORDER BY AVG_TIMER_WAIT DESC
LIMIT 10;
```

**方法3: 应用监控 (APM)**
```
工具:
- Skywalking: 分布式链路追踪
- Prometheus + Grafana: 指标监控
- Alibaba Arthas: Java应用诊断

优势:
- 可视化慢SQL统计
- 自动告警
- 关联业务场景
```

**第二步: 分析SQL执行计划**

```sql
-- 问题SQL
EXPLAIN SELECT * FROM orders
WHERE user_id = 12345 AND status = 'PENDING' AND create_time > '2024-01-01';

-- 执行计划输出
+----+-------+--------+------+---------------+------+---------+------+--------+-------------+
| id | type  | table  | key  | key_len       | ref  | rows   | Extra                        |
+----+-------+--------+------+---------------+------+---------+------+--------+-------------+
|  1 | ALL   | orders | NULL | NULL          | NULL | 980000 | Using where                  |
+----+-------+--------+------+---------------+------+---------+------+--------+-------------+
```

**关键字段解读**:

**1. type (访问类型) - 最重要!**
```
性能从好到差:
system > const > eq_ref > ref > range > index > ALL

- system/const: 常量查询,最快
  SELECT * FROM orders WHERE id = 1;

- eq_ref: 唯一索引查询
  SELECT * FROM orders o JOIN users u ON o.user_id = u.id;

- ref: 非唯一索引查询
  SELECT * FROM orders WHERE user_id = 12345;

- range: 索引范围扫描
  SELECT * FROM orders WHERE id BETWEEN 1 AND 100;

- index: 全索引扫描
  SELECT id FROM orders;  -- 覆盖索引

- ALL: 全表扫描 (最差,必须优化!)
  SELECT * FROM orders WHERE status = 'PENDING';
```

**2. key (使用的索引)**
```
- NULL: 未使用索引,需优化!
- idx_user_id: 使用了idx_user_id索引
```

**3. rows (预估扫描行数)**
```
- rows=10: 很好
- rows=1000: 可以接受
- rows=100000: 需要优化
- rows=1000000: 必须优化!
```

**4. Extra (额外信息)**
```
好的情况:
- Using index: 覆盖索引,不需要回表
- Using index condition: 索引条件下推

需要优化:
- Using filesort: 需要额外排序,考虑增加索引
- Using temporary: 使用临时表,考虑优化GROUP BY或ORDER BY
- Using where: 服务器层过滤,数据量大时性能差
```

**第三步: 定位性能瓶颈**

**瓶颈1: 全表扫描**
```sql
-- 问题SQL
SELECT * FROM orders WHERE status = 'PENDING';

-- EXPLAIN结果
type: ALL
rows: 1000000
Extra: Using where

-- 原因分析
1. status字段没有索引
2. 需要扫描100万行数据
3. 大量随机IO读取数据页

-- 优化方案
CREATE INDEX idx_status ON orders(status);

-- 优化后EXPLAIN
type: ref
rows: 5000
key: idx_status
```

**瓶颈2: 索引失效**
```sql
-- 问题SQL
SELECT * FROM orders WHERE YEAR(create_time) = 2024;

-- EXPLAIN结果
type: ALL
key: NULL
rows: 1000000

-- 原因分析
1. create_time字段有索引
2. 但对索引列使用了函数YEAR()
3. 导致索引失效,全表扫描

-- 优化方案
改写SQL,避免对索引列使用函数:
SELECT * FROM orders
WHERE create_time >= '2024-01-01' AND create_time < '2025-01-01';

-- 优化后EXPLAIN
type: range
key: idx_create_time
rows: 120000
```

**瓶颈3: 回表次数过多**
```sql
-- 问题SQL
SELECT * FROM orders WHERE user_id = 12345;

-- 执行过程
1. 在idx_user_id索引中找到user_id=12345的记录ID: [1001, 1002, 1003...]
2. 根据ID回表查询完整数据 (回表1000次!)
3. 返回结果

-- 优化方案1: 覆盖索引
如果只需要id, user_id, amount字段:
CREATE INDEX idx_user_amount ON orders(user_id, amount);

SELECT id, user_id, amount FROM orders WHERE user_id = 12345;
→ 无需回表,直接从索引获取数据

-- 优化方案2: 延迟关联
先获取ID,再回表:
SELECT o.* FROM orders o
INNER JOIN (
    SELECT id FROM orders WHERE user_id = 12345 LIMIT 10
) t ON o.id = t.id;
→ 只回表10次,而不是1000次
```

**第四步: 实施优化方案**

**优化策略1: 添加索引**
```sql
-- 分析查询模式
查询1: WHERE user_id = ? AND status = ?
查询2: WHERE user_id = ? ORDER BY create_time DESC

-- 方案1: 单列索引 (不推荐)
CREATE INDEX idx_user_id ON orders(user_id);
CREATE INDEX idx_status ON orders(status);
→ 问题: MySQL只会选择一个索引,另一个条件需要回表后过滤

-- 方案2: 联合索引 (推荐)
CREATE INDEX idx_user_status_time ON orders(user_id, status, create_time);
→ 覆盖所有查询场景
```

**优化策略2: 改写SQL**
```sql
-- 问题SQL: 隐式类型转换
CREATE TABLE users (phone VARCHAR(20), INDEX(phone));
SELECT * FROM users WHERE phone = 13800138000;  -- phone是字符串,传入数字

-- EXPLAIN
type: ALL  -- 索引失效!
原因: VARCHAR类型与INT比较,MySQL会将phone转为数字
等价于: WHERE CAST(phone AS UNSIGNED) = 13800138000

-- 优化方案
SELECT * FROM users WHERE phone = '13800138000';  -- 使用字符串

-- 问题SQL: OR条件
SELECT * FROM orders WHERE user_id = 123 OR status = 'PENDING';

-- EXPLAIN
type: ALL  -- 无法同时使用两个索引

-- 优化方案: 使用UNION
SELECT * FROM orders WHERE user_id = 123
UNION
SELECT * FROM orders WHERE status = 'PENDING';
```

**优化策略3: 分页优化**
```sql
-- 问题SQL: 深分页
SELECT * FROM orders ORDER BY create_time DESC LIMIT 1000000, 10;

-- 问题分析
1. 需要扫描1000010行数据
2. 丢弃前1000000行
3. 返回10行
4. 性能极差!

-- 优化方案1: 延迟关联
SELECT o.* FROM orders o
INNER JOIN (
    SELECT id FROM orders ORDER BY create_time DESC LIMIT 1000000, 10
) t ON o.id = t.id;

-- 优化方案2: 游标分页
记录上次查询的最后一条记录的create_time:
SELECT * FROM orders
WHERE create_time < '2024-01-01 12:00:00'
ORDER BY create_time DESC
LIMIT 10;
```

**第五步: 验证优化效果**

```sql
-- 优化前
EXPLAIN SELECT * FROM orders WHERE user_id = 12345 AND status = 'PENDING';
type: ALL
rows: 1000000
time: 2.35s

-- 创建索引
CREATE INDEX idx_user_status ON orders(user_id, status);

-- 优化后
EXPLAIN SELECT * FROM orders WHERE user_id = 12345 AND status = 'PENDING';
type: ref
key: idx_user_status
rows: 50
time: 0.01s

-- 性能提升: 2.35s → 0.01s,提升235倍!
```

**第六步: 持续监控**

```sql
-- 监控索引使用情况
SELECT
    table_schema,
    table_name,
    index_name,
    rows_selected,
    rows_inserted,
    rows_updated,
    rows_deleted
FROM sys.schema_index_statistics
WHERE table_schema = 'your_database'
ORDER BY rows_selected DESC;

-- 发现未使用的索引
SELECT * FROM sys.schema_unused_indexes;

-- 删除无用索引
ALTER TABLE orders DROP INDEX idx_unused;
```

**优化案例总结**:

| 问题类型 | 典型表现 | 优化方案 | 效果 |
|---------|---------|---------|------|
| 全表扫描 | type=ALL, rows很大 | 添加索引 | 100倍+ |
| 索引失效 | key=NULL | 避免函数/隐式转换 | 50倍+ |
| 回表过多 | rows适中但慢 | 覆盖索引/延迟关联 | 10倍+ |
| 深分页 | LIMIT偏移量大 | 游标分页 | 100倍+ |
| 排序慢 | Using filesort | 索引包含排序字段 | 20倍+ |

**终极优化原则**:
1. **能用索引就用索引** - 减少扫描行数
2. **能覆盖索引就覆盖** - 避免回表
3. **能延迟关联就延迟** - 减少回表次数
4. **能批量就批量** - 减少网络往返
5. **能异步就异步** - 提升用户体验

---

## 二、数据库高级特性

### 2.1 事务与锁机制

#### Q7: 数据库的锁有哪些类型?什么时候会发生死锁?如何避免?

**答案**:

**锁的分类维度**:

**维度1: 按粒度分类**

```
表级锁 (Table Lock)
- 锁定整张表
- 开销小,加锁快
- 锁粒度大,并发度低
- 例如: LOCK TABLES orders WRITE;

行级锁 (Row Lock)
- 锁定单行记录
- 开销大,加锁慢
- 锁粒度小,并发度高
- InnoDB默认使用行级锁

页级锁 (Page Lock)
- 锁定数据页(16KB)
- 介于表锁和行锁之间
- BDB存储引擎使用
```

**维度2: 按用途分类**

```
共享锁 (Shared Lock, S锁, 读锁)
- 多个事务可以同时持有
- 阻止其他事务获取排他锁
- 语法: SELECT ... LOCK IN SHARE MODE;

排他锁 (Exclusive Lock, X锁, 写锁)
- 只有一个事务可以持有
- 阻止其他事务获取任何锁
- 语法: SELECT ... FOR UPDATE;
- 自动加锁: UPDATE, DELETE, INSERT
```

**维度3: 按算法分类 (InnoDB特有)**

```
记录锁 (Record Lock)
- 锁定单条索引记录
- 例如: WHERE id = 1

间隙锁 (Gap Lock)
- 锁定索引记录之间的间隙
- 防止幻读
- 例如: 锁定(5, 10)之间的间隙,防止插入6,7,8,9

临键锁 (Next-Key Lock)
- 记录锁 + 间隙锁
- 锁定索引记录及其前面的间隙
- MySQL默认锁算法
```

**锁的兼容性矩阵**:

```
        S锁    X锁
S锁     ✓      ✗
X锁     ✗      ✗

✓ = 兼容(可以同时持有)
✗ = 不兼容(需要等待)
```

**死锁产生的条件**:

```
必须同时满足4个条件:

1. 互斥: 资源不能共享,只能被一个事务占用
2. 占有且等待: 事务持有资源同时等待其他资源
3. 不可抢占: 资源不能被强制剥夺,只能主动释放
4. 循环等待: 形成事务间的循环等待链

打破任何一个条件,就能避免死锁
```

**经典死锁案例**:

**案例1: 交叉锁定**
```sql
-- 场景: 两个事务分别锁定不同记录,然后互相请求对方持有的锁

时间线:
T1: BEGIN;
T1: UPDATE orders SET amount = 100 WHERE id = 1;  -- 持有id=1的X锁

T2: BEGIN;
T2: UPDATE orders SET amount = 200 WHERE id = 2;  -- 持有id=2的X锁

T1: UPDATE orders SET amount = 150 WHERE id = 2;  -- 请求id=2的X锁,等待T2释放

T2: UPDATE orders SET amount = 250 WHERE id = 1;  -- 请求id=1的X锁,等待T1释放

→ 形成死锁!

MySQL检测到死锁,自动回滚其中一个事务:
ERROR 1213: Deadlock found when trying to get lock; try restarting transaction
```

**案例2: 索引失效导致的死锁**
```sql
-- 场景: 没有索引,导致锁定范围扩大

CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(50),
    phone VARCHAR(20)  -- 没有索引!
);

时间线:
T1: BEGIN;
T1: UPDATE users SET name = 'Alice' WHERE phone = '13800138000';
    -- phone没有索引,需要全表扫描
    -- InnoDB会锁定扫描过的所有行!

T2: BEGIN;
T2: UPDATE users SET name = 'Bob' WHERE phone = '13900139000';
    -- 同样全表扫描,请求T1持有的锁

→ 死锁!

解决方案:
CREATE INDEX idx_phone ON users(phone);
→ 现在只锁定匹配的行,不会死锁
```

**案例3: 间隙锁死锁**
```sql
-- 场景: REPEATABLE READ隔离级别下的间隙锁

CREATE TABLE orders (
    id INT PRIMARY KEY,
    user_id INT,
    INDEX(user_id)
);

-- 假设user_id索引中有: 5, 10, 15

时间线:
T1: BEGIN;
T1: SELECT * FROM orders WHERE user_id = 7 FOR UPDATE;
    -- user_id=7不存在
    -- 加间隙锁:(5, 10)

T2: BEGIN;
T2: SELECT * FROM orders WHERE user_id = 8 FOR UPDATE;
    -- user_id=8不存在
    -- 请求间隙锁:(5, 10),与T1冲突,等待

T1: INSERT INTO orders (id, user_id) VALUES (100, 8);
    -- 尝试在(5, 10)间隙插入
    -- T2持有间隙锁,T1等待

→ 死锁!

解决方案:
改为READ COMMITTED隔离级别(不使用间隙锁)
或者先判断是否存在,再决定是否插入
```

**如何避免死锁?**

**策略1: 固定加锁顺序**
```sql
-- 问题代码
-- 线程1: 先锁A,再锁B
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;

-- 线程2: 先锁B,再锁A
UPDATE accounts SET balance = balance - 100 WHERE id = 2;
UPDATE accounts SET balance = balance + 100 WHERE id = 1;

→ 可能死锁

-- 优化方案: 统一按ID从小到大锁定
IF id1 < id2 THEN
    UPDATE accounts SET balance = balance - 100 WHERE id = id1;
    UPDATE accounts SET balance = balance + 100 WHERE id = id2;
ELSE
    UPDATE accounts SET balance = balance - 100 WHERE id = id2;
    UPDATE accounts SET balance = balance + 100 WHERE id = id1;
END IF;

→ 永远先锁定ID小的记录,避免循环等待
```

**策略2: 添加合适的索引**
```sql
-- 问题: 无索引导致锁定范围扩大
UPDATE users SET status = 'ACTIVE' WHERE phone = '13800138000';
→ 全表扫描,锁定所有行

-- 优化: 添加索引
CREATE INDEX idx_phone ON users(phone);
→ 只锁定匹配的行
```

**策略3: 降低隔离级别**
```sql
-- REPEATABLE READ: 使用间隙锁,容易死锁
SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ;

-- READ COMMITTED: 不使用间隙锁,减少死锁
SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;

注意: RC级别会出现不可重复读,需根据业务权衡
```

**策略4: 使用乐观锁代替悲观锁**
```sql
-- 悲观锁: SELECT ... FOR UPDATE (容易死锁)
BEGIN;
SELECT * FROM products WHERE id = 1 FOR UPDATE;
-- 业务逻辑处理...
UPDATE products SET stock = stock - 1 WHERE id = 1;
COMMIT;

-- 乐观锁: 使用版本号 (不会死锁)
-- 1. 查询当前版本号
SELECT stock, version FROM products WHERE id = 1;
-- stock=10, version=5

-- 2. 更新时校验版本号
UPDATE products
SET stock = 9, version = 6
WHERE id = 1 AND version = 5;

-- 3. 判断affected_rows
IF affected_rows = 0 THEN
    -- 版本号变化,说明被其他事务修改,重试
ELSE
    -- 更新成功
END IF;
```

**策略5: 控制事务大小**
```sql
-- 问题: 大事务持有锁时间长
BEGIN;
-- 100条UPDATE语句
UPDATE ...;
UPDATE ...;
...
COMMIT;

-- 优化: 拆分成多个小事务
FOR i IN 1..100 LOOP
    BEGIN;
    UPDATE ...;
    COMMIT;
END LOOP;

→ 减少锁持有时间,降低死锁概率
```

**策略6: 设置锁超时时间**
```sql
-- MySQL配置
SET innodb_lock_wait_timeout = 10;  -- 等待锁超时时间10秒

-- 应用层重试逻辑
max_retries = 3
for attempt in range(max_retries):
    try:
        execute_transaction()
        break
    except DeadlockException:
        if attempt < max_retries - 1:
            time.sleep(random.uniform(0.1, 0.5))  -- 随机延迟重试
            continue
        else:
            raise
```

**死锁排查工具**:

```sql
-- 查看最近一次死锁信息
SHOW ENGINE INNODB STATUS\G

-- 输出示例
LATEST DETECTED DEADLOCK
------------------------
2024-01-15 10:30:25 0x7f8c9c123700
*** (1) TRANSACTION:
TRANSACTION 12345, ACTIVE 5 sec starting index read
mysql tables in use 1, locked 1
LOCK WAIT 2 lock struct(s), heap size 1136, 1 row lock(s)
MySQL thread id 10, OS thread handle 140241234567890, query id 500 localhost root updating
UPDATE orders SET amount = 100 WHERE id = 1

*** (1) WAITING FOR THIS LOCK TO BE GRANTED:
RECORD LOCKS space id 58 page no 3 n bits 72 index PRIMARY of table `test`.`orders`
trx id 12345 lock_mode X locks rec but not gap waiting

*** (2) TRANSACTION:
TRANSACTION 12346, ACTIVE 3 sec starting index read
mysql tables in use 1, locked 1
3 lock struct(s), heap size 1136, 2 row lock(s)
MySQL thread id 11, OS thread handle 140241234567891, query id 501 localhost root updating
UPDATE orders SET amount = 200 WHERE id = 2

*** (2) HOLDS THE LOCK(S):
RECORD LOCKS space id 58 page no 3 n bits 72 index PRIMARY of table `test`.`orders`
trx id 12346 lock_mode X locks rec but not gap

*** (2) WAITING FOR THIS LOCK TO BE GRANTED:
RECORD LOCKS space id 58 page no 3 n bits 72 index PRIMARY of table `test`.`orders`
trx id 12346 lock_mode X locks rec but not gap waiting

*** WE ROLL BACK TRANSACTION (1)
```

**总结**:
1. 死锁是并发系统的常见问题,无法完全避免
2. 通过合理设计(固定加锁顺序、添加索引、降低隔离级别)可以大幅减少
3. 应用层需要有重试机制应对死锁
4. 定期监控死锁日志,优化高频死锁场景

---

## 三、数据库设计与架构

### 3.1 数据库设计范式

#### Q8: 什么是数据库三大范式?反范式化设计的应用场景是什么?

**答案**:

**第一范式 (1NF): 列不可分**

**定义**: 表中每一列都是不可再分的原子值

**反例: 违反1NF**
```sql
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(50),
    phone VARCHAR(100)  -- 存储多个电话号码: "138xxx,139xxx,137xxx"
);

问题:
1. 无法通过SQL查询某个电话号码
2. 无法统计每个用户有几个电话
3. 更新电话号码需要字符串解析
```

**正确设计**
```sql
-- 方案1: 多列存储
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(50),
    phone1 VARCHAR(20),
    phone2 VARCHAR(20),
    phone3 VARCHAR(20)
);

-- 方案2: 独立电话表 (更好)
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(50)
);

CREATE TABLE user_phones (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    phone VARCHAR(20),
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

**第二范式 (2NF): 消除部分依赖**

**定义**: 在1NF基础上,非主键列必须完全依赖于主键,不能只依赖主键的一部分

**反例: 违反2NF**
```sql
-- 联合主键表
CREATE TABLE order_items (
    order_id INT,
    product_id INT,
    product_name VARCHAR(100),   -- 只依赖于product_id
    product_price DECIMAL(10,2), -- 只依赖于product_id
    quantity INT,                -- 依赖于(order_id, product_id)
    PRIMARY KEY (order_id, product_id)
);

问题:
1. product_name和product_price只依赖于product_id,不依赖order_id
2. 数据冗余: 同一商品在多个订单中,名称和价格重复存储
3. 更新异常: 商品改名需要更新所有订单
4. 删除异常: 删除最后一个订单,商品信息丢失
```

**正确设计**
```sql
-- 商品表
CREATE TABLE products (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    price DECIMAL(10,2)
);

-- 订单明细表
CREATE TABLE order_items (
    order_id INT,
    product_id INT,
    quantity INT,
    PRIMARY KEY (order_id, product_id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);
```

**第三范式 (3NF): 消除传递依赖**

**定义**: 在2NF基础上,非主键列之间不能存在依赖关系,都必须直接依赖于主键

**反例: 违反3NF**
```sql
CREATE TABLE orders (
    id INT PRIMARY KEY,
    user_id INT,
    user_name VARCHAR(50),      -- 依赖于user_id,而不是order_id
    user_phone VARCHAR(20),     -- 依赖于user_id
    product_id INT,
    product_name VARCHAR(100),  -- 依赖于product_id
    amount DECIMAL(10,2)
);

问题:
1. user_name依赖于user_id,user_id依赖于order_id → 传递依赖
2. 数据冗余: 同一用户的多个订单重复存储用户信息
3. 更新异常: 用户改名需要更新所有订单
```

**正确设计**
```sql
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(50),
    phone VARCHAR(20)
);

CREATE TABLE products (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    price DECIMAL(10,2)
);

CREATE TABLE orders (
    id INT PRIMARY KEY,
    user_id INT,
    product_id INT,
    amount DECIMAL(10,2),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);
```

**范式化的优缺点**:

**优点**:
1. **消除冗余**: 每条数据只存储一次
2. **数据一致性**: 修改一处,全局生效
3. **节省存储空间**: 无重复数据
4. **易于维护**: 结构清晰,职责单一

**缺点**:
1. **查询性能差**: 需要大量JOIN操作
2. **复杂度高**: 查询SQL复杂,开发成本高
3. **分库分表困难**: JOIN跨库性能极差

**反范式化设计 (Denormalization)**

**定义**: 故意引入冗余,减少JOIN,提升查询性能

**应用场景1: 高频查询优化**
```sql
-- 场景: 电商订单列表需要展示用户名和商品名

-- 范式化设计
SELECT o.id, u.name AS user_name, p.name AS product_name, o.amount
FROM orders o
JOIN users u ON o.user_id = u.id
JOIN products p ON o.product_id = p.id
WHERE o.status = 'PAID';

-- 性能问题:
1. 每次查询需要JOIN 3张表
2. 订单表1000万行,users 100万行,products 10万行
3. JOIN成本极高,查询耗时数秒

-- 反范式化设计
CREATE TABLE orders (
    id INT PRIMARY KEY,
    user_id INT,
    user_name VARCHAR(50),      -- 冗余字段
    product_id INT,
    product_name VARCHAR(100),  -- 冗余字段
    amount DECIMAL(10,2),
    status VARCHAR(20),
    INDEX(status)
);

-- 优化后查询
SELECT id, user_name, product_name, amount
FROM orders
WHERE status = 'PAID';

-- 性能提升: 3秒 → 0.01秒,提升300倍!

-- 代价: 用户改名或商品改名时,需要更新orders表
UPDATE orders SET user_name = 'New Name' WHERE user_id = 123;
```

**应用场景2: 统计字段冗余**
```sql
-- 场景: 论坛帖子表需要显示评论数和点赞数

-- 范式化设计
CREATE TABLE posts (
    id INT PRIMARY KEY,
    title VARCHAR(200),
    content TEXT
);

CREATE TABLE comments (
    id INT PRIMARY KEY,
    post_id INT,
    content TEXT
);

CREATE TABLE likes (
    id INT PRIMARY KEY,
    post_id INT,
    user_id INT
);

-- 查询评论数和点赞数
SELECT
    p.id,
    p.title,
    (SELECT COUNT(*) FROM comments WHERE post_id = p.id) AS comment_count,
    (SELECT COUNT(*) FROM likes WHERE post_id = p.id) AS like_count
FROM posts p
LIMIT 20;

-- 性能问题: 每次查询都需要COUNT,帖子多时性能极差

-- 反范式化设计
CREATE TABLE posts (
    id INT PRIMARY KEY,
    title VARCHAR(200),
    content TEXT,
    comment_count INT DEFAULT 0,  -- 冗余统计字段
    like_count INT DEFAULT 0      -- 冗余统计字段
);

-- 插入评论时更新统计
INSERT INTO comments (post_id, content) VALUES (1, 'Great post!');
UPDATE posts SET comment_count = comment_count + 1 WHERE id = 1;

-- 查询性能
SELECT id, title, comment_count, like_count
FROM posts
LIMIT 20;

-- 性能提升: 无需COUNT,直接读取字段
```

**应用场景3: 宽表设计 (数据仓库)**
```sql
-- 场景: 数据分析需要多维度查询订单

-- OLTP范式化设计
orders, users, products, categories, regions

-- OLAP宽表设计
CREATE TABLE fact_orders (
    order_id INT,
    order_date DATE,
    -- 用户维度
    user_id INT,
    user_name VARCHAR(50),
    user_age INT,
    user_gender VARCHAR(10),
    user_city VARCHAR(50),
    -- 商品维度
    product_id INT,
    product_name VARCHAR(100),
    product_category VARCHAR(50),
    product_brand VARCHAR(50),
    -- 金额指标
    amount DECIMAL(10,2),
    discount DECIMAL(10,2),
    profit DECIMAL(10,2)
);

-- 查询: 按城市、品类、月份统计销售额
SELECT
    user_city,
    product_category,
    DATE_FORMAT(order_date, '%Y-%m') AS month,
    SUM(amount) AS total_amount
FROM fact_orders
GROUP BY user_city, product_category, month;

-- 性能: 无需JOIN,扫描单表即可,速度极快!
```

**反范式化的代价和解决方案**:

**代价1: 数据一致性**
```
问题: 用户改名后,orders表的user_name不同步

解决方案:
1. 应用层保证: 更新users表时同步更新orders表
2. 数据库触发器:
   CREATE TRIGGER sync_user_name
   AFTER UPDATE ON users
   FOR EACH ROW
   UPDATE orders SET user_name = NEW.name WHERE user_id = NEW.id;

3. 定时同步脚本: 每天凌晨全量同步
4. 消息队列: 用户更新发消息,异步同步订单表
```

**代价2: 存储空间**
```
问题: 冗余字段占用存储空间

解决方案:
1. 成本降低: SSD价格下降,存储成本可接受
2. 数据压缩: InnoDB支持页压缩,减少空间占用
3. 归档历史数据: 老订单迁移到冷存储
```

**代价3: 写入性能下降**
```
问题: 更新用户名需要更新orders表所有记录

解决方案:
1. 异步更新: 不阻塞主业务,后台慢慢同步
2. 批量更新: 积累一批更新,一次性执行
3. 分批更新: 避免长事务,分批次提交
```

**如何选择?**

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| OLTP (在线交易) | 范式化 | 写多读少,保证一致性 |
| OLAP (数据分析) | 反范式化 | 读多写少,追求查询性能 |
| 用户系统 | 范式化 | 用户信息经常变更 |
| 订单系统 | 反范式化 | 订单一旦生成很少修改,查询频繁 |
| 日志系统 | 反范式化 | 只写不改,查询为主 |
| 金融系统 | 范式化 | 数据一致性要求极高 |

**混合策略 (推荐)**:
```
核心业务表: 范式化设计,保证数据一致性
统计字段: 反范式化,提升查询性能
历史数据: 迁移到OLAP宽表,支持复杂分析
```

**总结**:
- 范式化是理论基础,反范式化是实践优化
- 没有绝对的对错,根据业务场景权衡
- 遵循原则: **写优先用范式,读优先用反范式**

(未完待续...本文档包含数据库基础部分,后续还有缓存、分布式、高可用等章节)
