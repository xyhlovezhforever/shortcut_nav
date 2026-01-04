# Top5 高难度后端面试题 - 完整版（Part 2）

> 接续 Part 1，包含面试题 4 和 5 的完整答案

---

# 面试题 4：微服务架构下的性能优化

## 📋 题目描述

### 场景描述
一个复杂的电商系统，包含 20+ 微服务，存在以下问题：
- 接口响应慢（P99 > 2s）
- 服务间调用链路长（最多 8 层）
- 数据库慢查询频繁
- 系统吞吐量低

### 问题

#### 4.1 性能诊断
**如何定位性能瓶颈？**

请说明以下工具的使用：
- APM 监控（Skywalking、Pinpoint）
- 分布式追踪（Jaeger、Zipkin）
- JVM 性能分析（JProfiler、Arthas）
- 数据库性能分析（慢查询日志、EXPLAIN）

**关键指标**
- QPS、TPS
- RT（Response Time）：P50、P95、P99
- 错误率
- 资源使用率（CPU、内存、网络、磁盘 IO）

#### 4.2 数据库优化
**场景：订单查询接口慢**

```sql
-- 原始 SQL（耗时 3s）
SELECT o.*, u.username, p.product_name, p.price
FROM orders o
LEFT JOIN users u ON o.user_id = u.id
LEFT JOIN order_items oi ON o.id = oi.order_id
LEFT JOIN products p ON oi.product_id = p.id
WHERE o.create_time BETWEEN '2024-01-01' AND '2024-12-31'
  AND o.status = 'COMPLETED'
  AND u.city = 'Shanghai'
ORDER BY o.create_time DESC
LIMIT 10 OFFSET 1000
```

要求：
1. 分析慢的原因
2. 提供优化方案（索引、SQL 重写、分库分表）
3. 说明深分页问题及解决方案
4. 如何设计索引？原则是什么？
5. 如何避免索引失效？

#### 4.3 微服务调用优化
**问题：A → B → C → D 调用链路过长**

```java
// 请优化以下代码
public OrderDetailVO getOrderDetail(Long orderId) {
    // 串行调用，耗时 1500ms
    Order order = orderService.getOrder(orderId);           // 200ms
    User user = userService.getUser(order.getUserId());     // 300ms
    List<OrderItem> items = itemService.getItems(orderId);  // 400ms

    List<Product> products = new ArrayList<>();
    for (OrderItem item : items) {
        Product product = productService.getProduct(item.getProductId()); // 每次 100ms
        products.add(product);
    }

    return buildVO(order, user, items, products);
}
```

优化目标：将响应时间降低到 500ms 以内

#### 4.4 JVM 优化
**问题：服务频繁 Full GC，每次 GC 停顿 2s+**

---

## ✅ 答案解析

### 4.1 性能诊断

**APM 监控实践**

```java
// 使用 Skywalking 进行链路追踪
@RestController
public class OrderController {

    @Autowired
    private OrderService orderService;

    @GetMapping("/order/{id}")
    @Trace // Skywalking 注解
    public Result getOrderDetail(@PathVariable Long id) {
        // 自定义 Span
        ActiveSpan span = ContextManager.createLocalSpan("query-order-detail");
        try {
            span.tag("orderId", String.valueOf(id));

            OrderDetailVO detail = orderService.getOrderDetail(id);

            span.tag("result", "success");
            return Result.success(detail);

        } catch (Exception e) {
            span.tag("error", "true");
            span.log(e);
            throw e;
        } finally {
            ContextManager.stopSpan();
        }
    }
}

// 自定义 Metrics（Prometheus）
@Component
public class MetricsCollector {

    private final Counter orderCounter = Counter.build()
        .name("order_total")
        .help("Total orders")
        .labelNames("status")
        .register();

    private final Histogram orderLatency = Histogram.build()
        .name("order_latency_seconds")
        .help("Order processing latency")
        .buckets(0.01, 0.05, 0.1, 0.5, 1, 5)
        .register();

    public void recordOrder(String status, long startTime) {
        orderCounter.labels(status).inc();

        double duration = (System.currentTimeMillis() - startTime) / 1000.0;
        orderLatency.observe(duration);
    }
}
```

**JVM 性能分析（使用 Arthas）**

```bash
# 1. 查看最耗 CPU 的线程
$ thread -n 5

# 2. 反编译类，查看是否被 JIT 优化
$ jad com.example.OrderService

# 3. 查看方法调用耗时
$ trace com.example.OrderService getOrderDetail -n 5

# 4. 监控方法入参、返回值
$ watch com.example.OrderService getOrderDetail "{params,returnObj}" -x 3

# 5. 查看 GC 情况
$ dashboard

# 6. 生成 heap dump
$ heapdump /tmp/heap.hprof
```

---

### 4.2 数据库优化

**问题分析**

```sql
-- 使用 EXPLAIN 分析
EXPLAIN SELECT o.*, u.username, p.product_name, p.price
FROM orders o
LEFT JOIN users u ON o.user_id = u.id
LEFT JOIN order_items oi ON o.id = oi.order_id
LEFT JOIN products p ON oi.product_id = p.id
WHERE o.create_time BETWEEN '2024-01-01' AND '2024-12-31'
  AND o.status = 'COMPLETED'
  AND u.city = 'Shanghai'
ORDER BY o.create_time DESC
LIMIT 10 OFFSET 1000;

-- 输出分析
+----+-------------+-------+------+---------------+------+---------+------+------+----------+-------------+
| id | select_type | table | type | possible_keys | key  | key_len | ref  | rows | filtered | Extra       |
+----+-------------+-------+------+---------------+------+---------+------+------+----------+-------------+
|  1 | SIMPLE      | o     | ALL  | NULL          | NULL | NULL    | NULL | 1M   |    10.00 | Using where |
|  1 | SIMPLE      | u     | ref  | PRIMARY       | PRIMARY | 8    | o.user_id | 1 |   10.00 | Using where |
|  1 | SIMPLE      | oi    | ref  | order_id      | order_id | 8  | o.id | 5    |  100.00 | NULL        |
|  1 | SIMPLE      | p     | eq_ref | PRIMARY     | PRIMARY | 8    | oi.product_id | 1 | 100.00 | NULL |
+----+-------------+-------+------+---------------+------+---------+------+------+----------+-------------+

-- 问题：
-- 1. orders 表全表扫描（type=ALL），扫描 100 万行
-- 2. 没有使用索引（key=NULL）
-- 3. Using where 表示在 Server 层过滤，效率低
-- 4. 深分页（OFFSET 1000）导致大量数据扫描
```

**优化方案**

**1. 添加索引**

```sql
-- 创建复合索引（遵循最左匹配原则）
CREATE INDEX idx_orders_time_status ON orders(create_time, status);
CREATE INDEX idx_users_city ON users(city);

-- 创建覆盖索引（避免回表）
CREATE INDEX idx_orders_cover ON orders(create_time, status, id, order_no, amount, user_id);

-- 验证：再次 EXPLAIN
EXPLAIN SELECT ...;
-- 现在 type 从 ALL 变成 ref/range，rows 从 1M 降到 1000
```

**2. SQL 重写**

```sql
-- 优化后的 SQL（分步查询，避免大 JOIN）

-- Step 1: 查询订单 ID（索引覆盖，很快）
SELECT id FROM orders
WHERE create_time >= '2024-01-01'
  AND create_time < '2025-01-01'
  AND status = 'COMPLETED'
ORDER BY create_time DESC
LIMIT 10 OFFSET 1000;

-- Step 2: 根据 ID 查询详细信息（主键查询，很快）
SELECT o.*, u.username, p.product_name, p.price
FROM orders o
LEFT JOIN users u ON o.user_id = u.id
LEFT JOIN order_items oi ON o.id = oi.order_id
LEFT JOIN products p ON oi.product_id = p.id
WHERE o.id IN (1001, 1002, 1003, ...);
```

**3. 深分页优化**

```sql
-- 方案 1：游标分页（基于上一页最后一条记录）
SELECT * FROM orders
WHERE create_time < '2024-06-15 10:30:00' -- 上一页最后一条的时间
  AND status = 'COMPLETED'
ORDER BY create_time DESC
LIMIT 10;

-- 方案 2：延迟关联（先查 ID，再关联）
SELECT o.* FROM orders o
INNER JOIN (
    SELECT id FROM orders
    WHERE status = 'COMPLETED'
    ORDER BY create_time DESC
    LIMIT 10 OFFSET 1000
) AS t ON o.id = t.id;

-- 方案 3：使用 ES 做分页查询
// Java 代码
SearchRequest request = new SearchRequest("orders");
SearchSourceBuilder builder = new SearchSourceBuilder();
builder.query(QueryBuilders.boolQuery()
    .must(QueryBuilders.termQuery("status", "COMPLETED")))
    .sort("create_time", SortOrder.DESC)
    .from(1000)
    .size(10);
request.source(builder);
```

**4. 索引设计原则**

```sql
-- 原则 1：选择性高的列放前面
-- ❌ 错误
CREATE INDEX idx_bad ON orders(status, create_time); -- status 只有几个值，选择性低
-- ✅ 正确
CREATE INDEX idx_good ON orders(create_time, status); -- create_time 选择性高

-- 原则 2：最左匹配原则
-- 索引：idx(a, b, c)
-- ✅ 可用：WHERE a=1 AND b=2
-- ✅ 可用：WHERE a=1
-- ❌ 不可用：WHERE b=2（跳过了 a）
-- ❌ 不可用：WHERE c=3（跳过了 a、b）

-- 原则 3：避免过长的索引
-- ❌ 错误
CREATE INDEX idx_long ON users(username, email, phone, address, city, province);
-- ✅ 正确
CREATE INDEX idx_short ON users(email); -- 根据实际查询需求
```

**5. 索引失效场景**

```sql
-- 1. 在索引列上使用函数
-- ❌ 错误
SELECT * FROM orders WHERE DATE(create_time) = '2024-01-01';
-- ✅ 正确
SELECT * FROM orders WHERE create_time >= '2024-01-01' AND create_time < '2024-01-02';

-- 2. 隐式类型转换
-- ❌ 错误（order_no 是 VARCHAR）
SELECT * FROM orders WHERE order_no = 123456;
-- ✅ 正确
SELECT * FROM orders WHERE order_no = '123456';

-- 3. 前导模糊查询
-- ❌ 错误
SELECT * FROM users WHERE username LIKE '%zhang%';
-- ✅ 正确（如果可以）
SELECT * FROM users WHERE username LIKE 'zhang%';

-- 4. OR 条件
-- ❌ 错误（status 和 create_time 在不同索引）
SELECT * FROM orders WHERE status = 'COMPLETED' OR create_time < '2024-01-01';
-- ✅ 正确
SELECT * FROM orders WHERE status = 'COMPLETED'
UNION ALL
SELECT * FROM orders WHERE create_time < '2024-01-01';

-- 5. 不等于操作
-- ❌ 可能不用索引
SELECT * FROM orders WHERE status != 'CANCELLED';
-- ✅ 更好
SELECT * FROM orders WHERE status IN ('PENDING', 'COMPLETED', 'SHIPPED');
```

**6. 分库分表方案**

```java
// 使用 Sharding-JDBC 分库分表
@Configuration
public class ShardingConfig {

    @Bean
    public DataSource dataSource() throws SQLException {
        // 1. 配置真实数据源
        Map<String, DataSource> dataSourceMap = new HashMap<>();
        dataSourceMap.put("ds0", createDataSource("db0"));
        dataSourceMap.put("ds1", createDataSource("db1"));

        // 2. 配置分表规则
        TableRuleConfiguration orderTableRule = new TableRuleConfiguration(
            "orders",
            "ds${0..1}.orders_${0..15}" // 2 个库，每库 16 张表
        );

        // 3. 配置分库策略（按 user_id 哈希）
        orderTableRule.setDatabaseShardingStrategy(
            new InlineShardingStrategyConfiguration(
                "user_id",
                "ds${user_id % 2}"
            )
        );

        // 4. 配置分表策略（按 order_id 哈希）
        orderTableRule.setTableShardingStrategy(
            new InlineShardingStrategyConfiguration(
                "order_id",
                "orders_${order_id % 16}"
            )
        );

        // 5. 分布式 ID 生成（雪花算法）
        Properties props = new Properties();
        props.setProperty("worker.id", "1");
        orderTableRule.setKeyGeneratorConfig(
            new KeyGeneratorConfiguration("SNOWFLAKE", "order_id", props)
        );

        // 6. 创建分片数据源
        ShardingRuleConfiguration shardingConfig = new ShardingRuleConfiguration();
        shardingConfig.getTableRuleConfigs().add(orderTableRule);

        return ShardingDataSourceFactory.createDataSource(
            dataSourceMap,
            shardingConfig,
            new Properties()
        );
    }
}

// 雪花算法实现
public class SnowflakeIdGenerator {

    private final long epoch = 1704038400000L; // 2024-01-01
    private final long workerId;
    private final long datacenterId;
    private long sequence = 0L;
    private long lastTimestamp = -1L;

    // 各部分位数
    private final long workerIdBits = 10L;
    private final long datacenterIdBits = 5L;
    private final long sequenceBits = 12L;

    // 各部分偏移
    private final long workerIdShift = sequenceBits;
    private final long datacenterIdShift = sequenceBits + workerIdBits;
    private final long timestampLeftShift = sequenceBits + workerIdBits + datacenterIdBits;

    // 序列号掩码
    private final long sequenceMask = ~(-1L << sequenceBits);

    public SnowflakeIdGenerator(long workerId, long datacenterId) {
        this.workerId = workerId;
        this.datacenterId = datacenterId;
    }

    public synchronized long nextId() {
        long timestamp = System.currentTimeMillis();

        // 时钟回拨检测
        if (timestamp < lastTimestamp) {
            throw new RuntimeException("Clock moved backwards");
        }

        if (timestamp == lastTimestamp) {
            // 同一毫秒内，序列号 +1
            sequence = (sequence + 1) & sequenceMask;
            if (sequence == 0) {
                timestamp = waitNextMillis(lastTimestamp);
            }
        } else {
            sequence = 0L;
        }

        lastTimestamp = timestamp;

        // 组装 ID
        return ((timestamp - epoch) << timestampLeftShift)
            | (datacenterId << datacenterIdShift)
            | (workerId << workerIdShift)
            | sequence;
    }

    private long waitNextMillis(long lastTimestamp) {
        long timestamp = System.currentTimeMillis();
        while (timestamp <= lastTimestamp) {
            timestamp = System.currentTimeMillis();
        }
        return timestamp;
    }
}
```

---

### 4.3 微服务调用优化

**问题代码分析**

```java
// 原代码：串行调用，耗时 1500ms
public OrderDetailVO getOrderDetail(Long orderId) {
    Order order = orderService.getOrder(orderId);           // 200ms
    User user = userService.getUser(order.getUserId());     // 300ms
    List<OrderItem> items = itemService.getItems(orderId);  // 400ms

    List<Product> products = new ArrayList<>();
    for (OrderItem item : items) {
        // 假设有 6 个商品，每次 100ms，串行执行 = 600ms
        Product product = productService.getProduct(item.getProductId());
        products.add(product);
    }

    return buildVO(order, user, items, products);
}

// 问题：
// 1. 串行调用：200 + 300 + 400 + 600 = 1500ms
// 2. N+1 查询：查询商品时循环调用
// 3. 无缓存：每次都查数据库
```

**优化方案：并行调用 + 批量查询 + 缓存**

```java
@Service
public class OrderServiceOptimized {

    @Autowired
    private AsyncExecutor asyncExecutor;

    public OrderDetailVO getOrderDetail(Long orderId) {
        // 使用 CompletableFuture 并行调用

        // 1. 查询订单
        CompletableFuture<Order> orderFuture = CompletableFuture.supplyAsync(
            () -> orderService.getOrder(orderId),
            asyncExecutor
        );

        // 2. 查询订单项
        CompletableFuture<List<OrderItem>> itemsFuture = CompletableFuture.supplyAsync(
            () -> itemService.getItems(orderId),
            asyncExecutor
        );

        // 3. 等待订单查询完成，再查询用户（依赖订单数据）
        CompletableFuture<User> userFuture = orderFuture.thenApplyAsync(
            order -> userService.getUser(order.getUserId()),
            asyncExecutor
        );

        // 4. 等待订单项查询完成，再批量查询商品
        CompletableFuture<List<Product>> productsFuture = itemsFuture.thenApplyAsync(
            items -> {
                // 批量查询，避免 N+1 问题
                List<Long> productIds = items.stream()
                    .map(OrderItem::getProductId)
                    .collect(Collectors.toList());

                return productService.batchGetProducts(productIds); // 一次查询
            },
            asyncExecutor
        );

        // 5. 等待所有异步任务完成
        CompletableFuture<Void> allFutures = CompletableFuture.allOf(
            orderFuture, userFuture, itemsFuture, productsFuture
        );

        try {
            allFutures.get(1, TimeUnit.SECONDS); // 最多等待 1s

            // 6. 组装结果
            return buildVO(
                orderFuture.get(),
                userFuture.get(),
                itemsFuture.get(),
                productsFuture.get()
            );

        } catch (Exception e) {
            throw new BusinessException("查询订单详情失败", e);
        }
    }
}

// 性能提升：
// 并行后：max(200, 300, 400) + 100(批量查询) = 500ms
// 提升：1500ms → 500ms（3倍）
```

**多级缓存**

```java
@Service
public class ProductServiceCached {

    // L1 缓存：本地缓存（Caffeine）
    private final Cache<Long, Product> localCache = Caffeine.newBuilder()
        .maximumSize(10_000)
        .expireAfterWrite(5, TimeUnit.MINUTES)
        .recordStats()
        .build();

    // L2 缓存：Redis
    @Autowired
    private StringRedisTemplate redisTemplate;

    // L3：数据库
    @Autowired
    private ProductMapper productMapper;

    /**
     * 批量查询商品（带缓存）
     */
    public List<Product> batchGetProducts(List<Long> productIds) {
        Map<Long, Product> result = new HashMap<>();
        List<Long> cacheMissIds = new ArrayList<>();

        // 1. 查询本地缓存
        for (Long productId : productIds) {
            Product product = localCache.getIfPresent(productId);
            if (product != null) {
                result.put(productId, product);
            } else {
                cacheMissIds.add(productId);
            }
        }

        if (cacheMissIds.isEmpty()) {
            return new ArrayList<>(result.values());
        }

        // 2. 查询 Redis（批量）
        List<String> keys = cacheMissIds.stream()
            .map(id -> "product:" + id)
            .collect(Collectors.toList());

        List<String> values = redisTemplate.opsForValue().multiGet(keys);

        List<Long> redisMissIds = new ArrayList<>();
        for (int i = 0; i < cacheMissIds.size(); i++) {
            String json = values.get(i);
            if (json != null && !json.isEmpty()) {
                Product product = JSON.parseObject(json, Product.class);
                result.put(cacheMissIds.get(i), product);
                // 回写本地缓存
                localCache.put(cacheMissIds.get(i), product);
            } else {
                redisMissIds.add(cacheMissIds.get(i));
            }
        }

        if (redisMissIds.isEmpty()) {
            return new ArrayList<>(result.values());
        }

        // 3. 查询数据库（批量）
        List<Product> products = productMapper.selectByIds(redisMissIds);

        // 4. 回写缓存
        for (Product product : products) {
            result.put(product.getId(), product);

            // 写入本地缓存
            localCache.put(product.getId(), product);

            // 写入 Redis（异步）
            asyncExecutor.submit(() -> {
                redisTemplate.opsForValue().set(
                    "product:" + product.getId(),
                    JSON.toJSONString(product),
                    5,
                    TimeUnit.MINUTES
                );
            });
        }

        return new ArrayList<>(result.values());
    }

    /**
     * 缓存预热
     */
    @PostConstruct
    public void warmUp() {
        // 查询热点商品
        List<Product> hotProducts = productMapper.selectHotProducts(1000);

        // 加载到缓存
        for (Product product : hotProducts) {
            localCache.put(product.getId(), product);
            redisTemplate.opsForValue().set(
                "product:" + product.getId(),
                JSON.toJSONString(product),
                5,
                TimeUnit.MINUTES
            );
        }

        log.info("Cache warmed up with {} products", hotProducts.size());
    }

    /**
     * 缓存更新（商品信息变更时）
     */
    public void updateProduct(Product product) {
        // 1. 更新数据库
        productMapper.updateById(product);

        // 2. 删除缓存（而不是更新，避免并发问题）
        localCache.invalidate(product.getId());
        redisTemplate.delete("product:" + product.getId());

        // 3. 发送 MQ 消息，通知其他节点删除本地缓存
        mqTemplate.send("cache-invalidate", product.getId());
    }
}
```

---

### 4.4 JVM 优化

**问题诊断**

```bash
# 1. 查看 GC 日志
java -XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:gc.log ...

# 示例输出
2024-01-15T10:30:15.123+0800: [Full GC (Allocation Failure)
[PSYoungGen: 0K->0K(2097152K)]
[ParOldGen: 4194304K->4194300K(4194304K)]
4194304K->4194300K(6291456K),
[Metaspace: 102400K->102400K(1146880K)], 2.5 seconds]

# 分析：
# - Full GC 频繁（每分钟多次）
# - Old Gen 几乎满了（4G/4G）
# - 每次 GC 耗时 2.5s
# - 原因：老年代内存不足，可能存在内存泄漏
```

**Heap Dump 分析**

```bash
# 1. 生成 heap dump
jmap -dump:live,format=b,file=heap.hprof <pid>

# 2. 使用 MAT 分析
# 打开 Eclipse Memory Analyzer Tool
# File -> Open Heap Dump -> heap.hprof

# 3. 查看 Leak Suspects（内存泄漏嫌疑）
# MAT 会自动分析，列出可能的内存泄漏

# 示例发现：
# - java.util.HashMap 占用 2GB（50%）
# - 包含 500 万个 User 对象
# - 引用链：UserCache -> ConcurrentHashMap -> User[]
# - 原因：缓存未设置过期时间，无限增长
```

**优化方案**

```java
// 问题代码
public class UserCache {
    // ❌ 错误：无限增长的缓存
    private static final Map<Long, User> cache = new ConcurrentHashMap<>();

    public User getUser(Long userId) {
        return cache.computeIfAbsent(userId, id -> {
            return userMapper.selectById(id);
        });
    }
}

// 优化代码
public class UserCacheOptimized {
    // ✅ 正确：使用 Caffeine，自动淘汰
    private final Cache<Long, User> cache = Caffeine.newBuilder()
        .maximumSize(100_000)  // 最多 10 万条
        .expireAfterWrite(10, TimeUnit.MINUTES) // 10 分钟过期
        .expireAfterAccess(5, TimeUnit.MINUTES) // 5 分钟未访问过期
        .weakKeys() // 弱引用 key，帮助 GC
        .recordStats() // 记录统计信息
        .removalListener((key, value, cause) -> {
            log.info("Evicted: key={}, cause={}", key, cause);
        })
        .build();

    public User getUser(Long userId) {
        return cache.get(userId, id -> userMapper.selectById(id));
    }
}
```

**JVM 参数调优**

```bash
# 调优前（默认参数）
java -Xms4g -Xmx4g -jar app.jar

# 问题：
# - Young Gen 太小，导致频繁 Minor GC
# - Old Gen 很快被填满，导致 Full GC

# 调优后
java \
  # 堆大小
  -Xms8g -Xmx8g \              # 堆初始和最大值（相同避免扩容）
  -Xmn4g \                      # Young Gen 大小（堆的 50%）
  -XX:SurvivorRatio=8 \         # Eden:Survivor = 8:1:1

  # GC 收集器
  -XX:+UseG1GC \                # 使用 G1 收集器
  -XX:MaxGCPauseMillis=200 \    # 最大 GC 停顿 200ms
  -XX:G1HeapRegionSize=16m \    # Region 大小 16MB
  -XX:InitiatingHeapOccupancyPercent=45 \ # 堆占用 45% 时触发并发标记

  # GC 日志
  -XX:+PrintGCDetails \
  -XX:+PrintGCDateStamps \
  -XX:+PrintHeapAtGC \
  -XX:+PrintTenuringDistribution \
  -Xloggc:/var/log/gc.log \
  -XX:+UseGCLogFileRotation \
  -XX:NumberOfGCLogFiles=5 \
  -XX:GCLogFileSize=50M \

  # OOM 时生成 heap dump
  -XX:+HeapDumpOnOutOfMemoryError \
  -XX:HeapDumpPath=/var/log/heapdump.hprof \

  # 元空间
  -XX:MetaspaceSize=256m \
  -XX:MaxMetaspaceSize=512m \

  # 其他
  -XX:+DisableExplicitGC \      # 禁用 System.gc()
  -XX:+ParallelRefProcEnabled \ # 并行处理引用

  -jar app.jar
```

**GC 收集器选择**

| 收集器 | 适用场景 | STW 时间 | 吞吐量 | JDK 版本 |
|--------|---------|---------|--------|----------|
| Serial | 单核，小堆(<100MB) | 长 | 低 | 所有版本 |
| Parallel | 多核，注重吞吐量 | 较长 | 高 | 所有版本 |
| CMS | 多核，注重低延迟 | 短 | 中 | JDK 8 |
| G1 | 大堆(>4G)，可预测停顿 | 可控 | 中高 | JDK 8+ |
| ZGC | 超大堆(>100G)，超低延迟 | 极短 | 高 | JDK 11+ |

---

继续下一部分（面试题5）...
