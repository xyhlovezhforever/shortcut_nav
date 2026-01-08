# Top5 高难度后端面试题 - 完整版（含答案）

> 涵盖分布式系统、高并发、数据一致性、微服务架构等核心场景
>
> 难度级别：⭐⭐⭐⭐⭐
>
> 📖 本文档包含题目和详细答案解析

---

**目录导航**

- [面试题 1：分布式事务与最终一致性实现](#面试题-1分布式事务与最终一致性实现)
- [面试题 2：高并发秒杀系统设计](#面试题-2高并发秒杀系统设计)
- [面试题 3：大规模分布式系统的数据一致性](#面试题-3大规模分布式系统的数据一致性)
- [面试题 4：微服务架构下的性能优化](#面试题-4微服务架构下的性能优化)
- [面试题 5：海量数据处理与实时计算](#面试题-5海量数据处理与实时计算)

---

# 面试题 1：分布式事务与最终一致性实现

## 📋 题目描述

### 场景描述
设计一个电商订单系统，涉及以下微服务：
- 订单服务（Order Service）
- 库存服务（Inventory Service）
- 支付服务（Payment Service）
- 积分服务（Points Service）

用户下单流程需要：
1. 创建订单
2. 扣减库存
3. 完成支付
4. 增加积分

### 问题

#### 1.1 如何保证分布式事务的一致性？
请详细说明以下几种方案的实现细节、优缺点及适用场景：
- 2PC/3PC
- TCC（Try-Confirm-Cancel）
- SAGA 模式
- 本地消息表 + 定时任务
- 事务消息（如 RocketMQ）

#### 1.2 实现 TCC 模式的核心要点
```java
// 请设计并实现以下接口
public interface OrderTccService {
    // Try 阶段：预留资源
    boolean tryCreateOrder(OrderDTO order);

    // Confirm 阶段：确认提交
    boolean confirmCreateOrder(String orderId);

    // Cancel 阶段：回滚操作
    boolean cancelCreateOrder(String orderId);
}
```

要求：
- 如何处理网络超时导致的悬挂问题？
- 如何防止重复提交？
- 如何设计幂等性？
- 补偿机制如何实现？

#### 1.3 高并发场景下的挑战
当系统 QPS 达到 10万+ 时：
- 如何避免分布式事务成为性能瓶颈？
- 如何设计异步化方案？
- 如何处理消息积压？
- 如何保证消息的顺序性？

---

## ✅ 答案解析

### 1.1 分布式事务一致性方案详解

#### 方案一：2PC/3PC

**2PC（两阶段提交）原理**

```
阶段1：准备阶段（Prepare）
协调者 → 参与者：CanCommit?
参与者：执行事务但不提交，锁定资源
参与者 → 协调者：Yes/No

阶段2：提交阶段（Commit）
如果所有参与者都Yes：
    协调者 → 参与者：DoCommit
    参与者：提交事务，释放资源
否则：
    协调者 → 参与者：DoAbort
    参与者：回滚事务，释放资源
```

**2PC 的问题**
1. **同步阻塞**：所有参与者都是阻塞的，资源被锁定
2. **单点故障**：协调者宕机导致系统不可用
3. **数据不一致**：Phase2 如果部分参与者收到 Commit，部分未收到
4. **无法处理网络分区**

**3PC（三阶段提交）改进**
```
阶段1：CanCommit - 询问是否可以执行
阶段2：PreCommit - 预提交（写redo/undo日志）
阶段3：DoCommit - 真正提交
```

**优点**：引入超时机制，降低阻塞范围
**缺点**：网络分区时仍可能数据不一致

**适用场景**：强一致性要求，并发量不大，事务执行时间短

---

#### 方案二：TCC（Try-Confirm-Cancel）

**完整实现示例**

```java
@Service
public class OrderTccServiceImpl implements OrderTccService {

    @Autowired
    private OrderMapper orderMapper;

    @Autowired
    private RedisTemplate<String, String> redisTemplate;

    private static final String TCC_ORDER_PREFIX = "tcc:order:";

    /**
     * Try 阶段：预留资源，冻结库存
     */
    @Override
    @Transactional
    public boolean tryCreateOrder(OrderDTO orderDTO) {
        String orderId = orderDTO.getOrderId();

        // 1. 幂等性检查：防止重复提交
        String tryKey = TCC_ORDER_PREFIX + "try:" + orderId;
        Boolean setIfAbsent = redisTemplate.opsForValue()
            .setIfAbsent(tryKey, "1", 60, TimeUnit.SECONDS);
        if (Boolean.FALSE.equals(setIfAbsent)) {
            return true; // 已经执行过 Try
        }

        try {
            // 2. 创建订单记录（状态：TRYING）
            Order order = new Order();
            order.setOrderId(orderId);
            order.setUserId(orderDTO.getUserId());
            order.setStatus(OrderStatus.TRYING);
            order.setCreateTime(new Date());
            orderMapper.insert(order);

            // 3. 冻结库存（调用库存服务的 Try 接口）
            boolean stockResult = inventoryService.tryDeductStock(
                orderDTO.getProductId(),
                orderDTO.getQuantity()
            );
            if (!stockResult) {
                throw new BusinessException("库存不足");
            }

            // 4. 预扣款（调用支付服务的 Try 接口）
            boolean paymentResult = paymentService.tryFreeze(
                orderDTO.getUserId(),
                orderDTO.getAmount()
            );
            if (!paymentResult) {
                throw new BusinessException("余额不足");
            }

            // 5. 记录 TCC 上下文到 Redis
            TccContext context = new TccContext();
            context.setOrderId(orderId);
            context.setProductId(orderDTO.getProductId());
            context.setQuantity(orderDTO.getQuantity());
            context.setAmount(orderDTO.getAmount());
            context.setTryTime(System.currentTimeMillis());

            String contextKey = TCC_ORDER_PREFIX + "context:" + orderId;
            redisTemplate.opsForValue().set(
                contextKey,
                JSON.toJSONString(context),
                1,
                TimeUnit.HOURS
            );

            return true;

        } catch (Exception e) {
            log.error("Try create order failed, orderId={}", orderId, e);
            return false;
        }
    }

    /**
     * Confirm 阶段：确认提交，真正扣减资源
     */
    @Override
    @Transactional
    public boolean confirmCreateOrder(String orderId) {
        // 1. 幂等性检查
        String confirmKey = TCC_ORDER_PREFIX + "confirm:" + orderId;
        Boolean setIfAbsent = redisTemplate.opsForValue()
            .setIfAbsent(confirmKey, "1", 60, TimeUnit.SECONDS);
        if (Boolean.FALSE.equals(setIfAbsent)) {
            return true;
        }

        // 2. 获取 TCC 上下文
        String contextKey = TCC_ORDER_PREFIX + "context:" + orderId;
        String contextStr = redisTemplate.opsForValue().get(contextKey);
        if (StringUtils.isEmpty(contextStr)) {
            log.error("TCC context not found, orderId={}", orderId);
            return false;
        }
        TccContext context = JSON.parseObject(contextStr, TccContext.class);

        try {
            // 3. 更新订单状态为 CONFIRMED
            Order order = new Order();
            order.setOrderId(orderId);
            order.setStatus(OrderStatus.CONFIRMED);
            order.setConfirmTime(new Date());
            orderMapper.updateByOrderId(order);

            // 4. 确认扣减库存
            inventoryService.confirmDeductStock(
                context.getProductId(),
                context.getQuantity()
            );

            // 5. 确认扣款
            paymentService.confirmFreeze(
                context.getUserId(),
                context.getAmount()
            );

            // 6. 增加积分（异步，允许最终一致性）
            pointsService.addPoints(context.getUserId(), context.getAmount());

            // 7. 清理 TCC 上下文
            redisTemplate.delete(contextKey);
            redisTemplate.delete(TCC_ORDER_PREFIX + "try:" + orderId);

            return true;

        } catch (Exception e) {
            log.error("Confirm create order failed, orderId={}", orderId, e);
            return false;
        }
    }

    /**
     * Cancel 阶段：回滚，释放资源
     */
    @Override
    @Transactional
    public boolean cancelCreateOrder(String orderId) {
        // 1. 幂等性检查
        String cancelKey = TCC_ORDER_PREFIX + "cancel:" + orderId;
        Boolean setIfAbsent = redisTemplate.opsForValue()
            .setIfAbsent(cancelKey, "1", 60, TimeUnit.SECONDS);
        if (Boolean.FALSE.equals(setIfAbsent)) {
            return true;
        }

        // 2. 处理空回滚：Try 未执行，直接 Cancel
        String contextKey = TCC_ORDER_PREFIX + "context:" + orderId;
        String contextStr = redisTemplate.opsForValue().get(contextKey);
        if (StringUtils.isEmpty(contextStr)) {
            String tryKey = TCC_ORDER_PREFIX + "try:" + orderId;
            if (!redisTemplate.hasKey(tryKey)) {
                // Try 未执行，记录空回滚标记
                redisTemplate.opsForValue().set(cancelKey, "1", 1, TimeUnit.HOURS);
                return true;
            }
        }

        TccContext context = JSON.parseObject(contextStr, TccContext.class);

        try {
            // 3. 更新订单状态为 CANCELLED
            Order order = new Order();
            order.setOrderId(orderId);
            order.setStatus(OrderStatus.CANCELLED);
            order.setCancelTime(new Date());
            orderMapper.updateByOrderId(order);

            // 4. 释放冻结库存
            inventoryService.cancelDeductStock(
                context.getProductId(),
                context.getQuantity()
            );

            // 5. 释放冻结金额
            paymentService.cancelFreeze(
                context.getUserId(),
                context.getAmount()
            );

            // 6. 清理 TCC 上下文
            redisTemplate.delete(contextKey);
            redisTemplate.delete(TCC_ORDER_PREFIX + "try:" + orderId);

            return true;

        } catch (Exception e) {
            log.error("Cancel create order failed, orderId={}", orderId, e);
            return false;
        }
    }
}
```

**关键问题处理**

**1. 空回滚问题**
- **现象**：Try 因网络超时未执行，但 Cancel 先到达
- **解决**：Cancel 时检查 Try 是否执行，未执行则直接返回成功

**2. 悬挂问题**
- **现象**：Cancel 先执行完，Try 请求后到达
- **解决**：Try 时检查是否已执行过 Cancel，若是则拒绝执行

**3. 幂等性设计**
- 每个阶段使用 Redis SetNX 实现幂等
- Key 设置过期时间，防止内存泄漏

---

#### 方案三：SAGA 模式

**核心思想**：长事务拆分为多个本地事务，每个事务有对应的补偿操作

**命令协调（Orchestration）实现**

```java
@Service
public class OrderSagaOrchestrator {

    public void createOrder(OrderDTO orderDTO) {
        SagaDefinition saga = SagaDefinition.create()
            // Step 1: 创建订单
            .step()
                .invokeLocal(this::createOrder)
                .withCompensation(this::cancelOrder)
            // Step 2: 扣减库存
            .step()
                .invokeParticipant(inventoryService::deductStock)
                .withCompensation(inventoryService::restoreStock)
            // Step 3: 扣款
            .step()
                .invokeParticipant(paymentService::deduct)
                .withCompensation(paymentService::refund)
            // Step 4: 增加积分
            .step()
                .invokeParticipant(pointsService::addPoints)
                .withCompensation(pointsService::deductPoints)
            .build();

        // 执行 SAGA
        sagaExecutor.execute(saga, orderDTO);
    }
}
```

**适用场景**：长事务，允许最终一致性，业务流程复杂

---

#### 方案四：本地消息表 + 定时任务

```java
// 1. 订单服务：创建订单 + 写本地消息表（同一事务）
@Transactional
public void createOrder(OrderDTO orderDTO) {
    // 创建订单
    Order order = new Order();
    order.setOrderId(orderDTO.getOrderId());
    order.setStatus(OrderStatus.PENDING);
    orderMapper.insert(order);

    // 写本地消息表
    LocalMessage message = new LocalMessage();
    message.setMessageId(UUID.randomUUID().toString());
    message.setBusinessType("ORDER_CREATED");
    message.setBusinessKey(orderDTO.getOrderId());
    message.setPayload(JSON.toJSONString(orderDTO));
    message.setStatus(MessageStatus.PENDING);
    messageMapper.insert(message);
}

// 2. 定时任务：扫描消息表，发送消息
@Scheduled(fixedDelay = 1000)
public void scanAndSendMessages() {
    List<LocalMessage> messages = messageMapper.selectPendingMessages(100);

    for (LocalMessage message : messages) {
        try {
            mqTemplate.send("order-topic", message.getPayload());
            message.setStatus(MessageStatus.SENT);
            messageMapper.updateById(message);
        } catch (Exception e) {
            message.setRetryCount(message.getRetryCount() + 1);
            if (message.getRetryCount() >= 3) {
                message.setStatus(MessageStatus.FAILED);
            }
            messageMapper.updateById(message);
        }
    }
}
```

---

#### 方案五：事务消息（RocketMQ）

**完整实现**

```java
@Service
public class OrderServiceImpl {

    @Autowired
    private RocketMQTemplate rocketMQTemplate;

    /**
     * 创建订单（发送事务消息）
     */
    public void createOrder(OrderDTO orderDTO) {
        rocketMQTemplate.sendMessageInTransaction(
            "order-topic",
            MessageBuilder.withPayload(orderDTO).build(),
            orderDTO
        );
    }

    /**
     * 本地事务执行器
     */
    @RocketMQTransactionListener
    static class OrderTransactionListener implements RocketMQLocalTransactionListener {

        @Autowired
        private OrderMapper orderMapper;

        @Override
        public RocketMQLocalTransactionState executeLocalTransaction(
            Message msg, Object arg) {

            OrderDTO orderDTO = (OrderDTO) arg;
            String orderId = orderDTO.getOrderId();

            try {
                // 执行本地事务
                Order order = new Order();
                order.setOrderId(orderId);
                order.setStatus(OrderStatus.PENDING);
                orderMapper.insert(order);

                // 记录事务状态
                redisTemplate.opsForValue().set(
                    "tx:order:" + orderId,
                    "COMMIT",
                    1,
                    TimeUnit.HOURS
                );

                return RocketMQLocalTransactionState.COMMIT;

            } catch (Exception e) {
                redisTemplate.opsForValue().set(
                    "tx:order:" + orderId,
                    "ROLLBACK",
                    1,
                    TimeUnit.HOURS
                );
                return RocketMQLocalTransactionState.ROLLBACK;
            }
        }

        /**
         * 回查本地事务状态
         */
        @Override
        public RocketMQLocalTransactionState checkLocalTransaction(Message msg) {
            OrderDTO orderDTO = JSON.parseObject(
                new String(msg.getBody()),
                OrderDTO.class
            );
            String orderId = orderDTO.getOrderId();

            String txStatus = redisTemplate.opsForValue().get("tx:order:" + orderId);

            if ("COMMIT".equals(txStatus)) {
                return RocketMQLocalTransactionState.COMMIT;
            } else if ("ROLLBACK".equals(txStatus)) {
                return RocketMQLocalTransactionState.ROLLBACK;
            } else {
                return RocketMQLocalTransactionState.UNKNOWN;
            }
        }
    }
}
```

---

### 1.3 高并发场景优化

**优化方案**

**1. 异步化**
```java
// 同步 TCC（耗时 300ms）
orderTccService.tryCreateOrder(orderDTO);
orderTccService.confirmCreateOrder(orderId);

// 异步化改造（耗时 100ms）
boolean tryResult = orderTccService.tryCreateOrder(orderDTO);
if (tryResult) {
    asyncExecutor.submit(() -> {
        orderTccService.confirmCreateOrder(orderId);
    });
    return "订单创建中";
}
```

**2. 消息队列削峰**
```java
// 请求先入队
messageQueue.send("order-create-queue", orderDTO);

// 消费者批量处理
@RabbitListener(queues = "order-create-queue", concurrency = "10-50")
public void batchCreateOrders(List<OrderDTO> orders) {
    for (OrderDTO order : orders) {
        orderTccService.tryCreateOrder(order);
    }
}
```

**3. 分库分表**
```java
// 按 userId 哈希分库分表
int dbIndex = Math.abs(userId.hashCode()) % 16;  // 16 个数据库
int tableIndex = Math.abs(orderId.hashCode()) % 256;  // 每库 256 张表
```

---

# 面试题 2：高并发秒杀系统设计

## 📋 题目描述

### 场景描述
设计一个支持百万级并发的秒杀系统，商品数量有限（如100件），需要保证：
- 不超卖
- 不少卖
- 高可用
- 低延迟（P99 < 100ms）

### 问题

#### 2.1 整体架构设计
```
用户请求 → CDN → 网关 → 秒杀服务 → 缓存 → DB
```

请详细说明：
- 如何进行多层次的流量削峰？
- 如何设计前端限流（验证码、按钮置灰）？
- 如何设计后端限流（令牌桶、漏桶、滑动窗口）？
- 如何设计动静分离？

#### 2.2 库存扣减方案
请对比以下方案的优缺点：

**方案 A：数据库扣减**
```sql
UPDATE product
SET stock = stock - 1
WHERE id = ? AND stock > 0
```

**方案 B：Redis 原子扣减**
```lua
local stock = redis.call('GET', KEYS[1])
if tonumber(stock) > 0 then
    redis.call('DECR', KEYS[1])
    return 1
else
    return 0
end
```

**方案 C：Redis + 消息队列异步扣减**

#### 2.3 分布式锁实现
实现一个基于 Redis 的分布式锁，要求：
- 如何保证加锁和设置过期时间的原子性？
- 如何防止误删其他线程的锁？
- 如何实现可重入锁？
- Redlock 算法的原理和问题是什么？

#### 2.4 热点数据问题
- 如何发现热点数据？
- 如何进行热点数据的本地缓存？
- 如何处理缓存击穿、穿透、雪崩？

---

## ✅ 答案解析

### 2.1 整体架构设计

**完整架构图**

```
┌─────────────────────────────────────────────────┐
│                  用户层                          │
│  - 前端限流：按钮置灰、验证码                     │
│  - 静态资源：CDN 加速                            │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│                 接入层                           │
│  - Nginx：限流、负载均衡                        │
│  - 网关：统一鉴权、限流、熔断                    │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│                应用层                            │
│  - 秒杀服务：业务逻辑                           │
│  - 限流：Guava RateLimiter / Sentinel          │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│                缓存层                            │
│  - Redis：库存缓存、分布式锁                    │
│  - 本地缓存：Caffeine                           │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│               消息队列                           │
│  - Kafka / RocketMQ：异步削峰                   │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│                数据层                            │
│  - MySQL：订单数据持久化                        │
└─────────────────────────────────────────────────┘
```

**多层限流策略**

**1. 前端限流**
```javascript
let clicked = false;
function seckill() {
    if (clicked) return;
    clicked = true;

    setTimeout(() => clicked = false, 3000);
    fetch('/api/seckill', { method: 'POST' });
}
```

**2. Nginx 限流**
```nginx
http {
    limit_req_zone $binary_remote_addr zone=seckill:10m rate=10r/s;

    server {
        location /api/seckill {
            limit_req zone=seckill burst=20 nodelay;
            proxy_pass http://backend;
        }
    }
}
```

**3. 应用层限流**
```java
@RestController
public class SeckillController {

    private final RateLimiter rateLimiter = RateLimiter.create(1000);

    @PostMapping("/api/seckill")
    public Result seckill(@RequestBody SeckillRequest request) {
        if (!rateLimiter.tryAcquire(100, TimeUnit.MILLISECONDS)) {
            return Result.error("系统繁忙");
        }
        return seckillService.doSeckill(request);
    }
}
```

---

### 2.2 库存扣减方案对比

#### 方案 B：Redis 原子扣减（推荐）

```java
@Service
public class SeckillService {

    @Autowired
    private StringRedisTemplate redisTemplate;

    public Result seckill(Long userId, Long productId) {
        String stockKey = "seckill:stock:" + productId;
        String userKey = "seckill:user:" + productId + ":" + userId;

        // 检查用户是否已参与
        Boolean hasKey = redisTemplate.hasKey(userKey);
        if (Boolean.TRUE.equals(hasKey)) {
            return Result.error("您已经参与过秒杀");
        }

        // Lua 脚本原子扣减
        String luaScript =
            "local stock = redis.call('GET', KEYS[1]) " +
            "if not stock then return -1 end " +
            "if tonumber(stock) <= 0 then return 0 end " +
            "redis.call('DECR', KEYS[1]) " +
            "redis.call('SETEX', KEYS[2], 86400, '1') " +
            "return 1";

        Long result = redisTemplate.execute(
            new DefaultRedisScript<>(luaScript, Long.class),
            Arrays.asList(stockKey, userKey)
        );

        if (result == 1) {
            // 发送MQ异步创建订单
            rabbitTemplate.convertAndSend("seckill-queue",
                new SeckillMessage(userId, productId));
            return Result.success("秒杀成功");
        }

        return Result.error("商品已售罄");
    }
}
```

**性能对比**

| 方案 | QPS | P99延迟 | 优点 | 缺点 |
|------|-----|---------|------|------|
| 数据库扣减 | ~500 | 200ms | 强一致 | 性能差 |
| Redis扣减 | ~50,000 | 10ms | 高性能 | 需保证一致性 |
| Redis+MQ | ~100,000 | 5ms | 超高性能 | 最终一致性 |

---

### 2.3 分布式锁实现

**完整的 Redis 分布式锁**

```java
@Component
public class RedisDistributedLock {

    @Autowired
    private StringRedisTemplate redisTemplate;

    /**
     * 尝试获取锁
     */
    public boolean tryLock(String key, String value, long expireTime) {
        Boolean result = redisTemplate.opsForValue()
            .setIfAbsent(key, value, expireTime, TimeUnit.MILLISECONDS);
        return Boolean.TRUE.equals(result);
    }

    /**
     * 释放锁（Lua 脚本保证原子性）
     */
    public boolean unlock(String key, String value) {
        String luaScript =
            "if redis.call('GET', KEYS[1]) == ARGV[1] then " +
            "    return redis.call('DEL', KEYS[1]) " +
            "else " +
            "    return 0 " +
            "end";

        Long result = redisTemplate.execute(
            new DefaultRedisScript<>(luaScript, Long.class),
            Collections.singletonList(key),
            value
        );

        return result != null && result == 1;
    }

    /**
     * 可重入锁（使用 Redisson）
     */
    public void lockWithReentrant(String lockKey, Runnable task) {
        RLock lock = redissonClient.getLock(lockKey);
        try {
            boolean locked = lock.tryLock(10, 30, TimeUnit.SECONDS);
            if (locked) {
                task.run();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } finally {
            if (lock.isHeldByCurrentThread()) {
                lock.unlock();
            }
        }
    }
}
```

**Redlock 算法**

```java
public class RedlockImpl {

    private List<RedisClient> redisClients; // 多个独立的 Redis 实例

    public boolean tryLock(String key, String value, long expireTime) {
        int n = redisClients.size();
        int quorum = n / 2 + 1;  // 过半节点数

        List<RedisClient> lockedClients = new ArrayList<>();
        long startTime = System.currentTimeMillis();

        // 尝试在所有节点上加锁
        for (RedisClient client : redisClients) {
            try {
                boolean locked = client.setNX(key, value, expireTime);
                if (locked) {
                    lockedClients.add(client);
                }
            } catch (Exception e) {
                // 忽略异常
            }
        }

        long elapsedTime = System.currentTimeMillis() - startTime;

        // 检查是否获取了过半节点的锁
        if (lockedClients.size() >= quorum && elapsedTime < expireTime) {
            return true;
        }

        // 失败则释放所有锁
        for (RedisClient client : lockedClients) {
            try {
                client.del(key);
            } catch (Exception e) {
                // 忽略
            }
        }

        return false;
    }
}
```

---

### 2.4 热点数据处理

**多级缓存架构**

```java
@Service
public class ProductService {

    // L1 缓存：本地缓存
    private final Cache<String, Product> localCache = Caffeine.newBuilder()
        .maximumSize(10_000)
        .expireAfterWrite(10, TimeUnit.SECONDS)
        .build();

    // L2 缓存：Redis
    @Autowired
    private StringRedisTemplate redisTemplate;

    // L3：数据库
    @Autowired
    private ProductMapper productMapper;

    public Product getProduct(Long productId) {
        String key = "product:" + productId;

        // 1. 查询本地缓存
        Product product = localCache.getIfPresent(key);
        if (product != null) {
            return product;
        }

        // 2. 查询 Redis
        String json = redisTemplate.opsForValue().get(key);
        if (json != null) {
            product = JSON.parseObject(json, Product.class);
            localCache.put(key, product);
            return product;
        }

        // 3. 查询数据库
        product = productMapper.selectById(productId);
        if (product != null) {
            redisTemplate.opsForValue().set(key,
                JSON.toJSONString(product), 5, TimeUnit.MINUTES);
            localCache.put(key, product);
        }

        return product;
    }
}
```

**缓存问题处理**

**1. 缓存穿透（布隆过滤器）**

```java
@Component
public class BloomFilterCache {

    private final BloomFilter<Long> bloomFilter = BloomFilter.create(
        Funnels.longFunnel(),
        100_000_000, // 1亿元素
        0.01         // 1% 误判率
    );

    @PostConstruct
    public void init() {
        List<Long> productIds = productMapper.selectAllIds();
        productIds.forEach(bloomFilter::put);
    }

    public boolean mightContain(Long productId) {
        return bloomFilter.mightContain(productId);
    }
}
```

**2. 缓存击穿（互斥锁）**

```java
public Product getProductWithMutex(Long productId) {
    String key = "product:" + productId;
    String lockKey = "lock:" + key;

    // 查询缓存
    String json = redisTemplate.opsForValue().get(key);
    if (json != null) {
        return JSON.parseObject(json, Product.class);
    }

    // 获取锁
    String lockValue = UUID.randomUUID().toString();
    boolean locked = redisTemplate.opsForValue()
        .setIfAbsent(lockKey, lockValue, 10, TimeUnit.SECONDS);

    if (locked) {
        try {
            // 双重检查
            json = redisTemplate.opsForValue().get(key);
            if (json != null) {
                return JSON.parseObject(json, Product.class);
            }

            // 查询数据库
            Product product = productMapper.selectById(productId);
            if (product != null) {
                redisTemplate.opsForValue().set(key,
                    JSON.toJSONString(product), 5, TimeUnit.MINUTES);
            }
            return product;
        } finally {
            // 释放锁
            unlock(lockKey, lockValue);
        }
    }

    return null;
}
```

**3. 缓存雪崩（过期时间加随机值）**

```java
int expireTime = 300 + new Random().nextInt(60); // 5~6分钟随机
redisTemplate.opsForValue().set(key, value, expireTime, TimeUnit.SECONDS);
```

---

# 面试题 3：大规模分布式系统的数据一致性

## 📋 题目描述

### 场景描述
设计一个分布式缓存系统，类似于 Redis Cluster，需要支持：
- 数据分片（Sharding）
- 数据复制（Replication）
- 故障转移（Failover）
- 强一致性读写

### 问题

#### 3.1 一致性协议选型
请详细对比以下一致性协议：

**Raft 协议**
- Leader 选举过程
- 日志复制机制
- 安全性保证
- 性能特点

**Paxos 协议**
- Basic Paxos 和 Multi-Paxos 的区别
- Proposer、Acceptor、Learner 的角色
- 活锁问题如何解决

**ZAB 协议（ZooKeeper）**
- 与 Raft 的异同
- 崩溃恢复过程
- 消息广播机制

#### 3.2 数据分片策略
要求实现以下策略并说明优缺点：
- 哈希取模
- 一致性哈希（Consistent Hashing）
- 虚拟节点（Virtual Nodes）
- 带权重的一致性哈希

问题：
- 节点扩容时如何进行数据迁移？
- 如何保证迁移过程中的可用性？
- 如何处理数据倾斜？

#### 3.3 读写策略
**Quorum 机制**
- N = 副本总数
- W = 写成功副本数
- R = 读取副本数

要求：
- 说明 W + R > N 如何保证强一致性
- 如何权衡一致性和可用性（CAP 定理）
- 如何实现最终一致性（Gossip 协议）
- 如何处理脑裂（Split-Brain）问题

---

## ✅ 答案解析

### 3.1 一致性协议详解

#### Raft 协议完整实现

**Leader 选举过程**

```
1. 初始状态：所有节点都是 Follower
2. 选举超时：150~300ms 内未收到心跳，转为 Candidate
3. 发起选举：
   - Candidate 增加 currentTerm
   - 投票给自己
   - 并行向所有节点发送 RequestVote RPC
4. 投票规则：
   - 每个节点在同一 term 内只能投一票
   - 投票给 term 更大、日志更新的节点
5. 选举结果：
   - 获得过半票数 → 成为 Leader
   - 其他节点成为 Leader → 转为 Follower
   - 超时未选出 → 重新选举（term+1）
```

**完整代码实现**

```java
public class RaftNode {

    enum State { FOLLOWER, CANDIDATE, LEADER }

    private volatile State state = State.FOLLOWER;
    private volatile int currentTerm = 0;
    private volatile String votedFor = null;
    private List<String> peers;
    private volatile long lastHeartbeatTime;

    private final Random random = new Random();

    /**
     * 心跳/选举超时检测
     */
    @Scheduled(fixedDelay = 50)
    public void checkTimeout() {
        if (state == State.LEADER) {
            sendHeartbeat();
            return;
        }

        // 检查选举超时
        long electionTimeout = 150 + random.nextInt(150); // 150~300ms
        long elapsedTime = System.currentTimeMillis() - lastHeartbeatTime;

        if (elapsedTime > electionTimeout) {
            startElection();
        }
    }

    /**
     * 发起选举
     */
    public void startElection() {
        state = State.CANDIDATE;
        currentTerm++;
        votedFor = this.nodeId;
        lastHeartbeatTime = System.currentTimeMillis();

        log.info("Node {} starting election, term={}", nodeId, currentTerm);

        AtomicInteger voteCount = new AtomicInteger(1); // 投自己一票

        // 并行向所有节点请求投票
        for (String peer : peers) {
            executor.submit(() -> {
                VoteRequest request = new VoteRequest();
                request.setTerm(currentTerm);
                request.setCandidateId(nodeId);
                request.setLastLogIndex(getLastLogIndex());
                request.setLastLogTerm(getLastLogTerm());

                try {
                    VoteResponse response = rpcClient.requestVote(peer, request);

                    if (response.isVoteGranted()) {
                        int count = voteCount.incrementAndGet();

                        // 获得过半票数
                        if (count > peers.size() / 2 && state == State.CANDIDATE) {
                            becomeLeader();
                        }
                    } else if (response.getTerm() > currentTerm) {
                        stepDown(response.getTerm());
                    }
                } catch (Exception e) {
                    log.error("Request vote failed, peer={}", peer, e);
                }
            });
        }
    }

    /**
     * 处理投票请求
     */
    public VoteResponse handleVoteRequest(VoteRequest request) {
        VoteResponse response = new VoteResponse();
        response.setTerm(currentTerm);
        response.setVoteGranted(false);

        // 1. term 检查
        if (request.getTerm() < currentTerm) {
            return response;
        }

        if (request.getTerm() > currentTerm) {
            stepDown(request.getTerm());
        }

        // 2. 投票规则
        boolean canVote = (votedFor == null || votedFor.equals(request.getCandidateId()));
        boolean logUpToDate = isLogUpToDate(request.getLastLogIndex(), request.getLastLogTerm());

        if (canVote && logUpToDate) {
            votedFor = request.getCandidateId();
            lastHeartbeatTime = System.currentTimeMillis();
            response.setVoteGranted(true);
            log.info("Node {} voted for {} in term {}",
                nodeId, request.getCandidateId(), currentTerm);
        }

        return response;
    }

    /**
     * 判断日志是否足够新
     */
    private boolean isLogUpToDate(int lastLogIndex, int lastLogTerm) {
        int myLastLogTerm = getLastLogTerm();
        int myLastLogIndex = getLastLogIndex();

        return lastLogTerm > myLastLogTerm ||
               (lastLogTerm == myLastLogTerm && lastLogIndex >= myLastLogIndex);
    }

    /**
     * 成为 Leader
     */
    private void becomeLeader() {
        state = State.LEADER;
        log.info("Node {} became LEADER in term {}", nodeId, currentTerm);
        sendHeartbeat();
    }

    /**
     * 发送心跳
     */
    private void sendHeartbeat() {
        for (String peer : peers) {
            Heartbeat heartbeat = new Heartbeat();
            heartbeat.setTerm(currentTerm);
            heartbeat.setLeaderId(nodeId);

            executor.submit(() -> {
                try {
                    rpcClient.sendHeartbeat(peer, heartbeat);
                } catch (Exception e) {
                    log.error("Send heartbeat failed, peer={}", peer, e);
                }
            });
        }
    }

    /**
     * 处理心跳
     */
    public void handleHeartbeat(Heartbeat heartbeat) {
        if (heartbeat.getTerm() > currentTerm) {
            stepDown(heartbeat.getTerm());
        }

        if (heartbeat.getTerm() >= currentTerm) {
            state = State.FOLLOWER;
            votedFor = null;
            lastHeartbeatTime = System.currentTimeMillis();
        }
    }

    /**
     * 退位为 Follower
     */
    private void stepDown(int newTerm) {
        currentTerm = newTerm;
        state = State.FOLLOWER;
        votedFor = null;
        lastHeartbeatTime = System.currentTimeMillis();
        log.info("Node {} stepped down to FOLLOWER, term={}", nodeId, newTerm);
    }
}
```

**日志复制机制**

```java
public class RaftLog {

    /**
     * Leader 复制日志到 Follower
     */
    public boolean replicateLog(LogEntry entry) {
        if (state != State.LEADER) {
            return false;
        }

        // 1. 追加到本地日志
        log.addEntry(entry);

        AtomicInteger successCount = new AtomicInteger(1);
        CountDownLatch latch = new CountDownLatch(peers.size());

        // 2. 并行发送到所有 Follower
        for (String peer : peers) {
            executor.submit(() -> {
                try {
                    AppendEntriesRequest request = new AppendEntriesRequest();
                    request.setTerm(currentTerm);
                    request.setLeaderId(nodeId);
                    request.setEntries(Collections.singletonList(entry));
                    request.setPrevLogIndex(entry.getIndex() - 1);
                    request.setPrevLogTerm(log.getTerm(entry.getIndex() - 1));
                    request.setLeaderCommit(commitIndex);

                    AppendEntriesResponse response = rpcClient.appendEntries(peer, request);

                    if (response.isSuccess()) {
                        successCount.incrementAndGet();
                    }
                } finally {
                    latch.countDown();
                }
            });
        }

        try {
            latch.await(100, TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        // 3. 过半节点成功，提交日志
        if (successCount.get() > peers.size() / 2) {
            commitIndex = entry.getIndex();
            applyToStateMachine(entry);
            return true;
        }

        return false;
    }

    /**
     * Follower 处理日志复制请求
     */
    public AppendEntriesResponse handleAppendEntries(AppendEntriesRequest request) {
        AppendEntriesResponse response = new AppendEntriesResponse();
        response.setTerm(currentTerm);
        response.setSuccess(false);

        // 1. term 检查
        if (request.getTerm() < currentTerm) {
            return response;
        }

        if (request.getTerm() > currentTerm) {
            stepDown(request.getTerm());
        }

        lastHeartbeatTime = System.currentTimeMillis();

        // 2. 日志一致性检查
        if (!log.matchLog(request.getPrevLogIndex(), request.getPrevLogTerm())) {
            return response;
        }

        // 3. 追加日志
        for (LogEntry entry : request.getEntries()) {
            log.addEntry(entry);
        }

        // 4. 更新 commitIndex
        if (request.getLeaderCommit() > commitIndex) {
            commitIndex = Math.min(request.getLeaderCommit(), log.getLastIndex());
            applyCommittedLogs();
        }

        response.setSuccess(true);
        return response;
    }
}
```

---

### 3.2 数据分片策略

#### 一致性哈希实现

```java
public class ConsistentHash {

    private static final int VIRTUAL_NODES = 150;
    private final TreeMap<Long, String> ring = new TreeMap<>();
    private final List<String> realNodes = new ArrayList<>();

    /**
     * 添加节点
     */
    public void addNode(String node) {
        realNodes.add(node);

        // 添加虚拟节点
        for (int i = 0; i < VIRTUAL_NODES; i++) {
            String virtualNode = node + "#" + i;
            long hash = hash(virtualNode);
            ring.put(hash, node);
        }

        log.info("Added node: {}, total virtual nodes: {}", node, ring.size());
    }

    /**
     * 移除节点
     */
    public void removeNode(String node) {
        realNodes.remove(node);

        for (int i = 0; i < VIRTUAL_NODES; i++) {
            String virtualNode = node + "#" + i;
            long hash = hash(virtualNode);
            ring.remove(hash);
        }

        log.info("Removed node: {}, remaining virtual nodes: {}", node, ring.size());
    }

    /**
     * 获取数据应该存储的节点
     */
    public String getNode(String key) {
        if (ring.isEmpty()) {
            return null;
        }

        long hash = hash(key);

        // 顺时针找到第一个节点
        Map.Entry<Long, String> entry = ring.ceilingEntry(hash);
        if (entry == null) {
            entry = ring.firstEntry(); // 环形
        }

        return entry.getValue();
    }

    /**
     * MurmurHash3 算法
     */
    private long hash(String key) {
        ByteBuffer buf = ByteBuffer.wrap(key.getBytes());
        int seed = 0x1234ABCD;

        ByteOrder byteOrder = buf.order();
        buf.order(ByteOrder.LITTLE_ENDIAN);

        long m = 0xc6a4a7935bd1e995L;
        int r = 47;
        long h = seed ^ (buf.remaining() * m);

        long k;
        while (buf.remaining() >= 8) {
            k = buf.getLong();
            k *= m;
            k ^= k >>> r;
            k *= m;
            h ^= k;
            h *= m;
        }

        if (buf.remaining() > 0) {
            ByteBuffer finish = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            finish.put(buf).rewind();
            h ^= finish.getLong();
            h *= m;
        }

        h ^= h >>> r;
        h *= m;
        h ^= h >>> r;

        buf.order(byteOrder);
        return h;
    }

    /**
     * 数据迁移计划
     */
    public Map<String, List<String>> getDataMigrationPlan(String newNode) {
        Map<String, List<String>> migrationPlan = new HashMap<>();

        for (int i = 0; i < VIRTUAL_NODES; i++) {
            String virtualNode = newNode + "#" + i;
            long hash = hash(virtualNode);

            // 找到前一个节点
            Map.Entry<Long, String> prevEntry = ring.lowerEntry(hash);
            if (prevEntry == null) {
                prevEntry = ring.lastEntry();
            }

            String fromNode = prevEntry.getValue();
            if (!fromNode.equals(newNode)) {
                migrationPlan.computeIfAbsent(fromNode, k -> new ArrayList<>())
                    .add("hash_range_" + prevEntry.getKey() + "_" + hash);
            }
        }

        return migrationPlan;
    }
}

// 带权重的一致性哈希
public class WeightedConsistentHash extends ConsistentHash {

    /**
     * 添加带权重的节点
     */
    public void addNode(String node, int weight) {
        realNodes.add(node);

        // 根据权重调整虚拟节点数量
        int virtualNodes = VIRTUAL_NODES * weight;

        for (int i = 0; i < virtualNodes; i++) {
            String virtualNode = node + "#" + i;
            long hash = hash(virtualNode);
            ring.put(hash, node);
        }

        log.info("Added weighted node: {}, weight={}, virtual nodes: {}",
            node, weight, virtualNodes);
    }
}
```

---

### 3.3 读写策略（Quorum）

**Quorum 机制实现**

```java
@Service
public class QuorumStorage {

    private List<StorageNode> nodes;
    private final int N; // 副本总数
    private final int W; // 写成功数
    private final int R; // 读取副本数

    public QuorumStorage(List<StorageNode> nodes, int w, int r) {
        this.nodes = nodes;
        this.N = nodes.size();
        this.W = w;
        this.R = r;

        // W + R > N 保证强一致性
        if (W + R <= N) {
            throw new IllegalArgumentException(
                "W + R must be > N for strong consistency");
        }
    }

    /**
     * 写入数据（需要 W 个节点成功）
     */
    public boolean write(String key, String value) {
        AtomicInteger successCount = new AtomicInteger(0);
        CountDownLatch latch = new CountDownLatch(N);

        // 并发写入所有节点
        for (StorageNode node : nodes) {
            executor.submit(() -> {
                try {
                    boolean success = node.put(key, value);
                    if (success) {
                        successCount.incrementAndGet();
                    }
                } catch (Exception e) {
                    log.error("Write to node {} failed", node.getAddress(), e);
                } finally {
                    latch.countDown();
                }
            });
        }

        try {
            latch.await(1, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        return successCount.get() >= W;
    }

    /**
     * 读取数据（从 R 个节点读取，选择最新版本）
     */
    public String read(String key) {
        List<Future<VersionedValue>> futures = new ArrayList<>();

        // 并发读取 R 个节点
        for (int i = 0; i < R && i < nodes.size(); i++) {
            StorageNode node = nodes.get(i);
            futures.add(executor.submit(() -> node.get(key)));
        }

        // 收集结果
        List<VersionedValue> values = new ArrayList<>();
        for (Future<VersionedValue> future : futures) {
            try {
                VersionedValue value = future.get(100, TimeUnit.MILLISECONDS);
                if (value != null) {
                    values.add(value);
                }
            } catch (Exception e) {
                log.error("Read from node failed", e);
            }
        }

        // 选择版本最新的数据
        return values.stream()
            .max(Comparator.comparing(VersionedValue::getVersion))
            .map(VersionedValue::getValue)
            .orElse(null);
    }

    /**
     * Read Repair：读取时修复不一致数据
     */
    public String readWithRepair(String key) {
        List<Future<VersionedValue>> futures = new ArrayList<>();

        // 读取所有节点
        for (StorageNode node : nodes) {
            futures.add(executor.submit(() -> node.get(key)));
        }

        List<VersionedValue> values = new ArrayList<>();
        for (int i = 0; i < futures.size(); i++) {
            try {
                VersionedValue value = futures.get(i).get(100, TimeUnit.MILLISECONDS);
                if (value != null) {
                    values.add(value);
                }
            } catch (Exception e) {
                log.error("Read from node {} failed", nodes.get(i).getAddress(), e);
            }
        }

        // 找到最新版本
        VersionedValue latest = values.stream()
            .max(Comparator.comparing(VersionedValue::getVersion))
            .orElse(null);

        if (latest == null) {
            return null;
        }

        // Read Repair：将最新版本写入旧节点
        for (int i = 0; i < values.size(); i++) {
            if (values.get(i).getVersion() < latest.getVersion()) {
                StorageNode node = nodes.get(i);
                executor.submit(() -> {
                    node.put(key, latest.getValue(), latest.getVersion());
                });
            }
        }

        return latest.getValue();
    }
}
```

**不同配置的权衡**

| 配置 | 一致性 | 可用性 | 性能 | 适用场景 |
|------|-------|-------|------|---------|
| W=N, R=1 | 强一致 | 低 | 写慢读快 | 读多写少 |
| W=1, R=N | 最终一致 | 高 | 写快读慢 | 写多读少 |
| W=Q, R=Q (Q=N/2+1) | 强一致 | 中 | 均衡 | 通用场景 |
| W=1, R=1 | 最终一致 | 高 | 快 | 低一致性要求 |

**脑裂处理（Fencing Token）**

```java
public class SplitBrainResolver {

    /**
     * 使用 Fencing Token 防止脑裂
     */
    public void writeWithFencing(String key, String value) {
        // 从 ZooKeeper 获取单调递增的 token
        long fencingToken = zookeeperClient.getFencingToken();

        // 写入时携带 token
        for (StorageNode node : nodes) {
            node.putWithToken(key, value, fencingToken);
        }
    }

    /**
     * 存储节点：拒绝旧 token 的写入
     */
    public boolean putWithToken(String key, String value, long token) {
        Long currentToken = tokenMap.get(key);

        // 如果 token 更小，拒绝写入（说明是旧 Leader）
        if (currentToken != null && token < currentToken) {
            log.warn("Rejected write with old token: {} < {}", token, currentToken);
            return false;
        }

        // 更新 token 并写入数据
        tokenMap.put(key, token);
        dataMap.put(key, value);
        return true;
    }
}
```

---

**📌 完整的面试题 4 和 5 的答案请查看：**
- [Top5高难度后端面试题-完整版(含答案)-Part2.md](./Top5高难度后端面试题-完整版(含答案)-Part2.md)

---

## 综合评分标准

### 优秀（90-100分）
- 能够完整、清晰地阐述解决方案
- 深入理解底层原理，能说明各种方案的权衡
- 有实际项目经验，能举例说明踩过的坑
- 能从业务、技术、成本等多维度分析问题
- 能够提出创新性的优化思路

### 良好（75-89分）
- 能够给出正确的解决方案
- 理解核心原理，能说明主要优缺点
- 有一定的实践经验
- 能够回答大部分追问

### 及格（60-74分）
- 知道基本概念和常见方案
- 理解不够深入，无法说明细节
- 缺乏实际经验，回答偏理论
- 对追问回答不够准确

### 不及格（<60分）
- 基本概念模糊
- 无法给出可行的解决方案
- 缺乏系统性思考
- 对追问无法回答

---

## 附录：推荐学习资源

### 书籍
- 《设计数据密集型应用》（DDIA）
- 《分布式系统原理与范型》
- 《高性能MySQL》
- 《深入理解Java虚拟机》
- 《从Paxos到Zookeeper：分布式一致性原理与实践》

### 开源项目
- Apache Flink
- Apache Kafka
- Redis
- Etcd（Raft 实现）
- TiDB（分布式数据库）

### 实践建议
1. 搭建本地环境，实际运行和调试
2. 阅读优秀开源项目的源码
3. 总结实际项目中的问题和解决方案
4. 关注技术博客和论文（如 MIT 6.824）

---

**祝面试顺利！** 🎉
