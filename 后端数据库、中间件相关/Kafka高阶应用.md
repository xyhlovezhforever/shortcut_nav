# Kafka 高阶应用深度学习指南

## 一、历史背景与发展历程

### 1.1 Kafka 诞生的历史背景

**时间节点**:2010年,LinkedIn公司

**诞生背景**:
- LinkedIn面临着海量日志和用户行为数据的实时处理需求
- 传统消息队列(ActiveMQ、RabbitMQ)在大数据量场景下性能瓶颈明显
- 需要一个能够处理TB级数据、支持高吞吐量的消息系统
- 现有方案无法满足"既能做消息队列,又能做数据管道"的需求

**创始人**: Jay Kreps、Neha Narkhede、Jun Rao(三位LinkedIn工程师)

**命名由来**: 以作家Franz Kafka命名,因为"Kafka是为writing优化的系统"(双关:writing=写作/写入)

### 1.2 解决的核心问题

#### 问题1: 传统消息队列的吞吐量瓶颈

**场景**: 电商双11场景
```
传统方案(RabbitMQ):
- 单节点TPS: 1-2万/秒
- 面对百万级消息/秒无法支撑
- 磁盘写入成为瓶颈

Kafka方案:
- 单节点TPS: 10万+/秒
- 集群可达百万级TPS
- 顺序写磁盘 + 批量发送 + 零拷贝
```

#### 问题2: 日志收集系统的可靠性

**场景**: 应用日志收集
```
传统方案(Flume + HDFS):
- 数据丢失风险高
- 无法保证顺序
- 不支持多消费者

Kafka方案:
- 分区内严格有序
- 副本机制保证可靠性
- 消费者组支持多订阅
```

#### 问题3: 实时数据流处理

**场景**: 用户行为实时分析
```
传统方案(批处理):
- T+1天数据可用
- 无法实时决策

Kafka + Kafka Streams:
- 毫秒级延迟
- 实时推荐、实时风控
```

### 1.3 发展历程中的重要里程碑

#### 2011年: 开源发布
- 1月: Kafka 0.7版本开源
- 4月: 成为Apache孵化项目
- **影响**: 引起业界关注,Netflix、Twitter等公司开始试用

#### 2012年: 成为Apache顶级项目
- 10月: 从孵化器毕业
- 0.8版本: 引入副本机制(Replication)
- **影响**: 可靠性大幅提升,生产环境大规模采用

#### 2014年: Kafka Connect诞生
- 0.9版本: 引入Kafka Connect
- **突破**: 简化与其他系统的集成(数据库、HDFS、ES等)
- **案例**: Uber用Kafka Connect实现MySQL到HDFS的实时同步

#### 2016年: Kafka Streams发布
- 0.10版本: 原生流处理库Kafka Streams
- **意义**: 无需Spark/Flink即可进行流处理
- **应用**: LinkedIn用于实时推荐系统

#### 2017年: Exactly-Once语义
- 0.11版本: 支持精确一次语义(Exactly-Once Semantics)
- **突破**: 解决重复消费和消息丢失问题
- **场景**: 金融交易、订单处理等强一致性场景

#### 2019年: KRaft模式提出
- 提出移除ZooKeeper依赖的KRaft模式
- **目标**: 简化部署,提升性能
- **进展**: 2021年开始预览,2023年生产可用

#### 2023年: Kafka 3.5+
- KRaft模式生产就绪
- Tiered Storage(分层存储)
- 性能进一步优化

### 1.4 目前在业界的地位和影响力

#### 市场占有率
- **全球500强**: 80%以上使用Kafka
- **日消息量**: 全球每天处理超过7万亿条消息
- **云服务**: AWS MSK、Azure Event Hubs、Confluent Cloud

#### 典型应用案例

**LinkedIn**(诞生地):
- 每天处理7万亿条消息
- 峰值吞吐: 1400万消息/秒
- 用途: 活动追踪、日志聚合、流处理

**Netflix**:
- 每天处理8万亿条消息
- 用途: 实时监控、推荐系统、A/B测试

**Uber**:
- 每天处理1万亿条消息
- 用途: 实时定价、司机位置追踪、数据管道

**腾讯**:
- 万亿级消息量/天
- 用途: 微信消息推送、广告投放、用户画像

**字节跳动**:
- 抖音推荐系统的核心组件
- 实时特征计算、实时AB实验

#### 生态系统
- **Confluent**: 商业化公司(由Kafka创始人创立)
- **工具**: Kafka Manager、Kafka Eagle、Cruise Control
- **集成**: 200+连接器(Debezium、Maxwell、Canal等)

### 1.5 未来发展趋势和方向

#### 趋势1: 无ZooKeeper架构(KRaft)
```
当前痛点:
- ZooKeeper成为单点故障风险
- 运维复杂度高
- 元数据同步延迟

KRaft模式:
- 基于Raft协议
- 元数据存储在Kafka内部
- 部署更简单,性能更好
```

#### 趋势2: 云原生化
```
Serverless Kafka:
- 按需付费,无需管理集群
- 自动扩缩容
- Confluent Cloud、AWS MSK Serverless

Kubernetes化:
- Strimzi Operator
- 容器化部署
- 弹性伸缩
```

#### 趋势3: 存储与计算分离
```
Tiered Storage(分层存储):
- 热数据: 本地SSD
- 温数据: 对象存储(S3、OSS)
- 冷数据: 归档存储

优势:
- 降低存储成本60%+
- 支持更长的数据保留
- 无限扩展能力
```

#### 趋势4: 实时OLAP增强
```
Kafka + ClickHouse/Druid:
- 实时数据写入Kafka
- 流式导入OLAP引擎
- 秒级查询TB级数据

应用:
- 实时大屏
- 实时报表
- 实时监控
```

#### 趋势5: 边缘计算集成
```
IoT场景:
- 边缘侧Kafka Lite
- 设备数据就近处理
- 断网续传能力

5G + Kafka:
- 车联网实时决策
- 工业物联网
```

---

## 二、核心概念与设计理念

### 2.1 Kafka的核心设计理念

#### 理念1: 以日志为中心的设计(Log-Centric Architecture)

**核心思想**: Kafka将消息存储为"不可变的、顺序追加的日志"

**为什么这样设计?**
```
传统数据库: 随机写(B-Tree)
- 需要寻址、锁竞争
- 磁盘随机写: 100次/秒

Kafka: 顺序写(Append-Only Log)
- 无需寻址,直接追加
- 磁盘顺序写: 600MB/秒
- 性能接近内存

类比:
就像写日记,只能往后写,不能修改历史
这种限制换来了极致性能
```

**实际案例**:
```
场景: 电商订单日志
每秒10万订单 = 每秒10万次写入

传统方案(MySQL):
- 随机写入B+树
- 需要加锁、维护索引
- 单表几千TPS就是瓶颈

Kafka方案:
- 顺序追加到日志文件
- 无锁设计
- 单分区可达数万TPS
```

#### 理念2: 分区并行(Partition for Parallelism)

**核心思想**: 通过分区实现水平扩展和并行处理

**为什么需要分区?**
```
问题: 单个日志文件无法无限扩展
- 磁盘容量限制
- 单线程写入性能上限
- 消费者读取瓶颈

解决: 分区机制
- Topic分成多个Partition
- 每个Partition是一个独立的日志
- 分布在不同Broker上
- 多个消费者并行消费

类比:
像高速公路的多车道
每个车道独立运行,互不干扰
总吞吐量 = 单车道吞吐 × 车道数
```

**实际案例**:
```
场景: 用户行为日志收集
每秒100万条日志

设计:
Topic: user-behavior
Partition: 100个
每个Partition: 1万条/秒

消费者组:
100个消费者实例
每个实例处理1个Partition
并行度: 100
```

#### 理念3: 消费者拉取模式(Pull-Based Consumer)

**核心思想**: 消费者主动拉取消息,而非Broker推送

**为什么用Pull而不是Push?**
```
Push模式(RabbitMQ):
优势: 实时性好
劣势:
- Broker需要维护消费者状态
- 消费速度不一致时容易压垮消费者
- 难以实现批量消费

Pull模式(Kafka):
优势:
- 消费者自己控制速度
- 支持批量拉取,提高吞吐
- Broker无状态,简单高效
劣势:
- 需要轮询,可能浪费资源

Kafka优化:
- 长轮询(long polling)
- 有消息立即返回,无消息等待一段时间
- 兼顾实时性和资源消耗
```

**实际案例**:
```
场景: 大数据ETL任务
凌晨处理白天积累的数据

Pull模式优势:
- 消费者可以控制消费速度
- 批量拉取1000条,减少网络开销
- 处理完再拉取,不会被压垮

如果用Push:
- Broker不知道消费者处理能力
- 可能把消费者推爆
- 需要复杂的流控机制
```

#### 理念4: 零拷贝技术(Zero-Copy)

**核心思想**: 利用操作系统特性,减少数据在内存中的拷贝次数

**传统方式的问题**:
```
传统数据传输(4次拷贝):
1. 磁盘 → 内核缓冲区 (DMA拷贝)
2. 内核缓冲区 → 应用程序缓冲区 (CPU拷贝)
3. 应用程序缓冲区 → Socket缓冲区 (CPU拷贝)
4. Socket缓冲区 → 网卡 (DMA拷贝)

Kafka零拷贝(2次拷贝):
1. 磁盘 → 内核缓冲区 (DMA拷贝)
2. 内核缓冲区 → 网卡 (DMA拷贝)

使用技术: sendfile系统调用
性能提升: 2-3倍
```

**实际案例**:
```
场景: 消费者读取历史数据
需要读取1TB的日志文件

传统方式:
- 数据需要经过应用层
- CPU拷贝消耗大量资源
- 速度: 200MB/秒

Kafka零拷贝:
- 数据直接从磁盘到网卡
- CPU几乎不参与
- 速度: 600MB/秒+
```

#### 理念5: 批量处理(Batching)

**核心思想**: 尽可能批量处理消息,提高吞吐量

**批量设计体现在哪?**
```
生产者批量:
- 消息累积到一定大小再发送
- 减少网络请求次数
- 参数: batch.size, linger.ms

消费者批量:
- 一次拉取多条消息
- 批量反序列化、批量处理
- 参数: max.poll.records

磁盘批量:
- 批量刷盘
- 减少磁盘IO次数
- 参数: log.flush.interval.messages

压缩批量:
- 一批消息一起压缩
- 压缩率更高
- 参数: compression.type
```

**实际案例**:
```
场景: 日志收集系统
每条日志200字节,每秒1万条

方案1(逐条发送):
- 1万次网络请求/秒
- 网络开销巨大
- 吞吐量: 2MB/秒

方案2(批量发送,每批100条):
- 100次网络请求/秒
- 网络开销降低99%
- 吞吐量: 20MB/秒
```

### 2.2 架构设计的独特之处

#### 特点1: 分布式Commit Log

**设计思路**:
```
Kafka = 分布式的、可复制的、持久化的日志系统

核心组件:
┌─────────────────────────────────────┐
│  Topic: user-events                 │
├─────────────────────────────────────┤
│  Partition 0  │  Partition 1  │ ... │
├───────────────┼───────────────┼─────┤
│  Replica 0-0  │  Replica 1-0  │     │
│  Replica 0-1  │  Replica 1-1  │     │
│  Replica 0-2  │  Replica 1-2  │     │
└─────────────────────────────────────┘

每个Partition:
- 是一个有序的日志文件
- 有多个副本(Replica)
- 分布在不同Broker上
```

**独特之处**:
- **不是队列**: 消息不会被删除,只会过期
- **多订阅**: 多个消费者组可以独立消费同一份数据
- **回溯**: 可以重新消费历史数据

#### 特点2: ISR机制(In-Sync Replicas)

**设计思路**:
```
问题: 如何在一致性和可用性之间平衡?

传统方案:
- 所有副本都同步: 慢节点拖累整体(强一致性,低可用性)
- 只同步Leader: 数据可能丢失(高可用性,弱一致性)

Kafka ISR:
- 动态维护"同步中"的副本集合
- 只有ISR中的副本才能参与选举
- 落后太多的副本会被踢出ISR

ISR的判断标准:
replica.lag.time.max.ms = 10秒
超过10秒未同步 → 踢出ISR
```

**实际案例**:
```
场景: 3副本集群,一个副本所在机器网络故障

传统同步复制:
- 等待所有副本确认
- 故障副本导致写入阻塞
- 可用性下降

Kafka ISR:
1. 检测到副本落后
2. 将其踢出ISR (ISR: 3 → 2)
3. 只等待剩余2个副本确认
4. 写入继续,不受影响
5. 故障恢复后,重新加入ISR
```

#### 特点3: 分层存储架构

**设计思路**:
```
分段日志(Segment):
Partition不是一个大文件,而是多个Segment

user-events-0/
├── 00000000000000000000.log (1GB, 已关闭)
├── 00000000000001000000.log (1GB, 已关闭)
├── 00000000000002000000.log (1GB, 已关闭)
└── 00000000000003000000.log (写入中)

优势:
- 删除过期数据只需删除Segment文件
- 不影响当前写入
- 压缩策略更灵活
```

**索引机制**:
```
每个Segment对应两个索引文件:
- .index: 偏移量索引(稀疏索引)
- .timeindex: 时间戳索引

稀疏索引示例:
offset    position
0         0
1000      102400
2000      204800
...

查找offset=1500的消息:
1. 找到 offset=1000 的position=102400
2. 从102400开始顺序扫描到1500
```

#### 特点4: 消费者组协调机制

**设计思路**:
```
问题: 如何实现消费者负载均衡和故障转移?

Kafka方案: 消费者组(Consumer Group)

消费者组特性:
- 同一组内的消费者共享Partition
- 每个Partition只能被组内一个消费者消费
- 消费者数量 ≤ Partition数量最高效

再均衡(Rebalance):
- 消费者加入/退出时触发
- 重新分配Partition
- Group Coordinator协调
```

**实际案例**:
```
场景: 订单消息处理
Topic: orders (12个Partition)

初始状态:
消费者组: order-processors
消费者数: 4个
分配: 每个消费者处理3个Partition

Consumer1: P0, P1, P2
Consumer2: P3, P4, P5
Consumer3: P6, P7, P8
Consumer4: P9, P10, P11

扩容场景(增加2个消费者):
触发Rebalance
新分配: 每个消费者处理2个Partition

Consumer1: P0, P1
Consumer2: P2, P3
Consumer3: P4, P5
Consumer4: P6, P7
Consumer5: P8, P9
Consumer6: P10, P11
```

### 2.3 与同类工具的核心差异

#### Kafka vs RabbitMQ

| 维度 | Kafka | RabbitMQ |
|------|-------|----------|
| **设计目标** | 高吞吐量的日志系统 | 灵活的消息路由 |
| **消息模型** | 发布-订阅(基于日志) | 多种模式(直连、主题、RPC等) |
| **消息顺序** | 分区内严格有序 | 单队列有序,多队列无保证 |
| **吞吐量** | 百万级/秒 | 万级/秒 |
| **消息持久化** | 所有消息都持久化 | 可选持久化 |
| **消息删除** | 时间/大小策略,与消费无关 | 消费后即删除 |
| **消费模式** | Pull(消费者拉取) | Push(服务器推送) |
| **回溯消费** | 支持 | 不支持 |
| **适用场景** | 日志收集、流处理、大数据管道 | 任务队列、RPC、微服务通信 |

**选型建议**:
```
选Kafka:
✓ 日志/事件收集
✓ 大数据管道
✓ 流处理
✓ 需要消息回溯
✓ 吞吐量优先

选RabbitMQ:
✓ 微服务间通信
✓ 任务队列
✓ 需要复杂路由
✓ 低延迟优先
✓ 消息确认机制
```

#### Kafka vs Pulsar

| 维度 | Kafka | Pulsar |
|------|-------|---------|
| **架构** | 分区日志 | 分层架构(Broker+BookKeeper) |
| **存储** | Broker自带存储 | 存储计算分离 |
| **多租户** | 弱支持 | 原生支持 |
| **跨地域复制** | MirrorMaker(需额外部署) | 内置Geo-Replication |
| **消息确认** | 基于Offset | 基于消息ID,支持单条确认 |
| **延迟消息** | 不支持 | 原生支持 |
| **生态成熟度** | 非常成熟 | 相对较新 |

**选型建议**:
```
选Kafka:
✓ 生态成熟,工具丰富
✓ 成本敏感(不需要BookKeeper)
✓ 团队熟悉Kafka
✓ 已有Kafka基础设施

选Pulsar:
✓ 需要存储计算分离
✓ 多租户隔离要求高
✓ 跨地域部署
✓ 需要延迟消息功能
```

#### Kafka vs Event Hubs(Azure)

| 维度 | Kafka | Azure Event Hubs |
|------|-------|------------------|
| **部署方式** | 自建/托管(MSK、Confluent) | 完全托管 |
| **协议兼容** | Kafka协议 | Kafka协议 + AMQP/HTTP |
| **自动扩缩容** | 手动 | 自动 |
| **运维负担** | 需要运维团队 | 无需运维 |
| **成本** | 固定成本(包月) | 按量付费 |
| **定制化** | 高度可定制 | 有限定制 |

**选型建议**:
```
选自建Kafka:
✓ 有专业运维团队
✓ 需要深度定制
✓ 成本敏感(大规模场景)

选Event Hubs:
✓ 小团队,无专职运维
✓ Azure生态
✓ 快速上线需求
```

### 2.4 底层工作原理和机制

#### 原理1: 消息写入流程

```
完整写入流程:
1. Producer发送消息
2. 确定Partition(分区策略)
3. 确定Leader Broker
4. 写入Leader的本地日志
5. Follower从Leader拉取
6. 写入Follower的本地日志
7. Follower向Leader确认
8. Leader向Producer确认

时序图:
Producer          Leader            Follower1         Follower2
  │                │                   │                 │
  ├─send message──>│                   │                 │
  │                ├─append to log     │                 │
  │                │                   │                 │
  │                │<─fetch request────┤                 │
  │                │<─fetch request────┼─────────────────┤
  │                │                   │                 │
  │                ├─send messages────>│                 │
  │                ├─send messages─────┼────────────────>│
  │                │                   │                 │
  │                │                   ├─append to log   │
  │                │                   │                 ├─append to log
  │                │                   │                 │
  │                │<─ack──────────────┤                 │
  │                │<─ack──────────────┼─────────────────┤
  │                │                   │                 │
  │<─ack───────────┤                   │                 │
```

#### 原理2: 副本同步机制

**HW和LEO概念**:
```
LEO (Log End Offset): 日志末端偏移量
- 每个副本的最新写入位置

HW (High Watermark): 高水位线
- 所有ISR副本都已同步的位置
- 消费者只能读到HW之前的消息

示例:
Leader:  [0][1][2][3][4][5]     LEO=6, HW=4
Follower1:[0][1][2][3][4]       LEO=5
Follower2:[0][1][2][3]          LEO=4

HW = min(所有ISR的LEO) = 4
消费者只能读到offset=0,1,2,3的消息
```

**为什么要有HW?**
```
保证已提交消息不丢失:

场景: Leader宕机
- HW=4表示offset<4的消息已在所有ISR副本上
- 选举新Leader后,这些消息肯定存在
- offset>=4的消息可能丢失,但未对消费者可见

如果没有HW:
- 消费者可能读到offset=5的消息
- Leader宕机,新Leader没有offset=5
- 消费者看到的消息"消失"了
```

#### 原理3: 分区Leader选举

**选举触发条件**:
1. Leader所在Broker宕机
2. Leader主动卸任(Broker关闭)
3. 分区重新分配

**选举流程**:
```
1. Controller检测到Leader宕机
2. 从ISR中选择一个副本作为新Leader
   选择规则: ISR中第一个可用的副本
3. 更新元数据
4. 通知所有Broker新的Leader信息
5. 生产者和消费者切换到新Leader

特殊情况(ISR为空):
- unclean.leader.election.enable=true
  → 允许从非ISR副本中选举,可能丢数据
- unclean.leader.election.enable=false
  → 等待ISR恢复,保证不丢数据但降低可用性
```

**实际案例**:
```
场景: 3副本集群,ISR=[0,1,2],副本0是Leader

情况1: 副本0所在Broker断电
1. Controller检测到副本0不可达
2. ISR=[1,2],选择副本1为新Leader
3. 更新元数据,通知所有客户端
4. 客户端重连副本1继续工作

情况2: ISR全部宕机
如果unclean.leader.election.enable=false:
- 分区不可用,等待ISR恢复
- 保证数据不丢失

如果unclean.leader.election.enable=true:
- 从副本2选举Leader
- 可能丢失未同步的数据
- 优先保证可用性
```

#### 原理4: 消费者Offset管理

**Offset存储演进**:
```
0.9版本之前: ZooKeeper
- 存储在ZK的/consumers/[group]/offsets/[topic]/[partition]
- 问题: ZK不适合频繁写入,成为瓶颈

0.9版本之后: Kafka内部Topic(__consumer_offsets)
- 50个分区的内部Topic
- 使用Compact策略,只保留最新Offset
- 性能大幅提升
```

**Offset提交方式**:
```
自动提交:
enable.auto.commit=true
auto.commit.interval.ms=5000

优点: 简单
缺点: 可能重复消费或丢失消息

手动提交(同步):
consumer.commitSync()

优点: 精确控制
缺点: 阻塞,性能较低

手动提交(异步):
consumer.commitAsync(callback)

优点: 不阻塞,高性能
缺点: 提交失败可能导致重复消费
```

**实际案例**:
```
场景: 订单处理系统,要求精确一次处理

方案1(错误): 自动提交
消费消息 → 处理订单 → 自动提交Offset

问题:
- 处理失败,Offset已提交 → 消息丢失

方案2(正确): 手动提交+幂等处理
消费消息 → 处理订单(写数据库) → 手动提交Offset

优化: 事务写入
beginTransaction()
  处理订单(写数据库)
  提交Offset
commitTransaction()

保证:
- 数据库写入和Offset提交原子性
- 重启后从正确位置继续
```

### 2.5 关键概念和术语解释

#### Broker
```
定义: Kafka集群中的服务器节点

职责:
- 接收生产者的消息
- 存储消息到磁盘
- 响应消费者的拉取请求
- 参与副本复制

类比: 仓库的一个库房
```

#### Topic
```
定义: 消息的逻辑分类

特性:
- 一个Topic可以有多个Partition
- 订阅Topic即可接收该类消息

类比: 报纸的不同版面(体育、财经、娱乐)
```

#### Partition
```
定义: Topic的物理分片

特性:
- 有序的消息序列
- 分布在不同Broker上
- 是并行处理的基本单元

类比: 书籍的章节,每章可以独立阅读
```

#### Offset
```
定义: 消息在Partition中的位置标识

特性:
- 64位整数,从0开始递增
- 分区内唯一,单调递增
- 消费者通过Offset定位消息

类比: 书籍的页码
```

#### Consumer Group
```
定义: 共享消费进度的消费者集合

特性:
- 组内每个Partition只分配给一个消费者
- 不同组独立消费,互不影响
- 实现负载均衡和故障转移

类比: 一个部门的员工共同完成一项工作
```

#### Replica
```
定义: 分区的副本

类型:
- Leader: 处理读写请求
- Follower: 复制Leader数据,作为备份

特性:
- 每个Partition有N个Replica (replication-factor=N)
- 分布在不同Broker上

类比: 文件的多个备份
```

#### ISR (In-Sync Replicas)
```
定义: 与Leader保持同步的副本集合

判断标准:
- Follower的LEO与Leader的差距 < replica.lag.time.max.ms

动态变化:
- Follower落后 → 踢出ISR
- Follower追上 → 加入ISR

作用:
- Leader选举的候选池
- acks=all时的确认范围
```

#### Controller
```
定义: Kafka集群的管理节点

职责:
- 管理Partition的Leader选举
- 监控Broker上线/下线
- 更新集群元数据
- 分区重新分配

选举:
- 第一个启动的Broker成为Controller
- Controller宕机,重新选举
```

#### Coordinator
```
定义: 管理消费者组的Broker

职责:
- 处理消费者加入/退出
- 触发Rebalance
- 管理Offset提交

确定方式:
- hash(group.id) % 50 → 分区号
- __consumer_offsets该分区的Leader即为Coordinator
```

---

## 三、基础知识

### 3.1 安装部署的完整流程

#### 方式1: 单机部署(开发测试)

**环境准备**:
```bash
# 系统要求
操作系统: Linux (CentOS 7/Ubuntu 20.04+)
JDK: 11或17 (推荐)
内存: 最小4GB,推荐8GB+
磁盘: 最小50GB,推荐SSD

# 安装JDK
sudo yum install java-11-openjdk-devel  # CentOS
sudo apt install openjdk-11-jdk         # Ubuntu

# 验证
java -version
```

**下载Kafka**:
```bash
# 下载
wget https://downloads.apache.org/kafka/3.6.1/kafka_2.13-3.6.1.tgz

# 解压
tar -xzf kafka_2.13-3.6.1.tgz
cd kafka_2.13-3.6.1

# 设置环境变量
echo 'export KAFKA_HOME=/path/to/kafka_2.13-3.6.1' >> ~/.bashrc
echo 'export PATH=$PATH:$KAFKA_HOME/bin' >> ~/.bashrc
source ~/.bashrc
```

**启动ZooKeeper**:
```bash
# 启动内置ZooKeeper
bin/zookeeper-server-start.sh -daemon config/zookeeper.properties

# 验证
echo stat | nc localhost 2181
```

**配置Kafka**:
```properties
# config/server.properties

# Broker ID (集群中唯一)
broker.id=0

# 监听地址
listeners=PLAINTEXT://localhost:9092
advertised.listeners=PLAINTEXT://localhost:9092

# 日志目录
log.dirs=/data/kafka-logs

# ZooKeeper连接
zookeeper.connect=localhost:2181

# 分区数(默认)
num.partitions=3

# 副本数(默认)
default.replication.factor=1

# 日志保留时间(7天)
log.retention.hours=168

# 日志段文件大小(1GB)
log.segment.bytes=1073741824
```

**启动Kafka**:
```bash
# 启动
bin/kafka-server-start.sh -daemon config/server.properties

# 验证
jps  # 应该看到Kafka进程

# 查看日志
tail -f logs/server.log
```

**测试**:
```bash
# 创建Topic
bin/kafka-topics.sh --create \
  --bootstrap-server localhost:9092 \
  --topic test \
  --partitions 3 \
  --replication-factor 1

# 查看Topic
bin/kafka-topics.sh --list --bootstrap-server localhost:9092

# 生产消息
bin/kafka-console-producer.sh \
  --bootstrap-server localhost:9092 \
  --topic test
> Hello Kafka
> Test Message

# 消费消息(新开终端)
bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic test \
  --from-beginning
```

#### 方式2: 集群部署(生产环境)

**架构规划**:
```
3节点Kafka集群 + 3节点ZooKeeper集群

节点规划:
kafka1: 192.168.1.101
kafka2: 192.168.1.102
kafka3: 192.168.1.103

ZooKeeper:
zk1: 192.168.1.101:2181
zk2: 192.168.1.102:2181
zk3: 192.168.1.103:2181
```

**配置ZooKeeper集群**:
```properties
# config/zookeeper.properties (所有节点)

dataDir=/data/zookeeper
clientPort=2181

# 集群配置
server.1=192.168.1.101:2888:3888
server.2=192.168.1.102:2888:3888
server.3=192.168.1.103:2888:3888

# 每个节点创建myid文件
# kafka1:
echo "1" > /data/zookeeper/myid

# kafka2:
echo "2" > /data/zookeeper/myid

# kafka3:
echo "3" > /data/zookeeper/myid
```

**启动ZooKeeper集群**:
```bash
# 在每个节点执行
bin/zookeeper-server-start.sh -daemon config/zookeeper.properties

# 验证
echo stat | nc localhost 2181
# 应该看到 Mode: follower 或 Mode: leader
```

**配置Kafka集群**:
```properties
# config/server.properties (kafka1)

broker.id=1  # kafka2改为2, kafka3改为3
listeners=PLAINTEXT://192.168.1.101:9092
advertised.listeners=PLAINTEXT://192.168.1.101:9092

log.dirs=/data/kafka-logs

# ZooKeeper集群地址
zookeeper.connect=192.168.1.101:2181,192.168.1.102:2181,192.168.1.103:2181

num.partitions=3
default.replication.factor=3  # 改为3副本

# 网络配置
num.network.threads=8
num.io.threads=16
socket.send.buffer.bytes=102400
socket.receive.buffer.bytes=102400
socket.request.max.bytes=104857600

# 副本配置
min.insync.replicas=2
unclean.leader.election.enable=false
```

**启动Kafka集群**:
```bash
# 在每个节点执行
bin/kafka-server-start.sh -daemon config/server.properties

# 验证集群状态
bin/kafka-broker-api-versions.sh --bootstrap-server kafka1:9092,kafka2:9092,kafka3:9092
```

**创建高可用Topic**:
```bash
bin/kafka-topics.sh --create \
  --bootstrap-server kafka1:9092,kafka2:9092,kafka3:9092 \
  --topic orders \
  --partitions 12 \
  --replication-factor 3 \
  --config min.insync.replicas=2

# 查看详情
bin/kafka-topics.sh --describe \
  --bootstrap-server kafka1:9092 \
  --topic orders
```

#### 方式3: Docker部署(快速体验)

**Docker Compose配置**:
```yaml
# docker-compose.yml
version: '3'

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"
    volumes:
      - zk-data:/var/lib/zookeeper/data
      - zk-logs:/var/lib/zookeeper/log

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
    volumes:
      - kafka-data:/var/lib/kafka/data

volumes:
  zk-data:
  zk-logs:
  kafka-data:
```

**启动**:
```bash
docker-compose up -d

# 查看日志
docker-compose logs -f kafka

# 进入容器测试
docker exec -it <kafka-container-id> bash
kafka-topics --list --bootstrap-server localhost:9092
```

#### 方式4: Kubernetes部署(云原生)

**使用Strimzi Operator**:
```bash
# 安装Strimzi Operator
kubectl create namespace kafka
kubectl create -f 'https://strimzi.io/install/latest?namespace=kafka' -n kafka

# 等待Operator启动
kubectl get pod -n kafka --watch
```

**部署Kafka集群**:
```yaml
# kafka-cluster.yaml
apiVersion: kafka.strimzi.io/v1beta2
kind: Kafka
metadata:
  name: my-cluster
  namespace: kafka
spec:
  kafka:
    version: 3.6.1
    replicas: 3
    listeners:
      - name: plain
        port: 9092
        type: internal
        tls: false
      - name: tls
        port: 9093
        type: internal
        tls: true
    config:
      offsets.topic.replication.factor: 3
      transaction.state.log.replication.factor: 3
      transaction.state.log.min.isr: 2
      default.replication.factor: 3
      min.insync.replicas: 2
    storage:
      type: jbod
      volumes:
      - id: 0
        type: persistent-claim
        size: 100Gi
        deleteClaim: false
  zookeeper:
    replicas: 3
    storage:
      type: persistent-claim
      size: 10Gi
      deleteClaim: false
  entityOperator:
    topicOperator: {}
    userOperator: {}
```

**应用配置**:
```bash
kubectl apply -f kafka-cluster.yaml -n kafka

# 查看状态
kubectl get kafka -n kafka
kubectl get pod -n kafka
```

**创建Topic**:
```yaml
# topic.yaml
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaTopic
metadata:
  name: my-topic
  namespace: kafka
  labels:
    strimzi.io/cluster: my-cluster
spec:
  partitions: 12
  replicas: 3
  config:
    retention.ms: 604800000  # 7天
    segment.bytes: 1073741824
```

```bash
kubectl apply -f topic.yaml -n kafka
```

### 3.2 基本配置和环境设置

#### Broker核心配置

**必配参数**:
```properties
# Broker标识
broker.id=0  # 集群中唯一

# 监听地址
listeners=PLAINTEXT://:9092
advertised.listeners=PLAINTEXT://192.168.1.100:9092

# 日志目录(可配置多个,逗号分隔)
log.dirs=/data1/kafka-logs,/data2/kafka-logs,/data3/kafka-logs

# ZooKeeper连接
zookeeper.connect=zk1:2181,zk2:2181,zk3:2181/kafka
```

**性能调优参数**:
```properties
# 网络线程数(处理网络请求)
num.network.threads=8  # 建议CPU核数

# IO线程数(处理磁盘读写)
num.io.threads=16  # 建议CPU核数的2倍

# Socket缓冲区
socket.send.buffer.bytes=102400      # 100KB
socket.receive.buffer.bytes=102400   # 100KB
socket.request.max.bytes=104857600   # 100MB

# 副本拉取线程数
num.replica.fetchers=4
```

**存储配置**:
```properties
# 日志保留策略
log.retention.hours=168              # 7天
log.retention.bytes=1073741824000    # 1TB

# 日志段文件大小
log.segment.bytes=1073741824         # 1GB

# 日志段滚动时间
log.roll.hours=168                   # 7天

# 日志清理策略
log.cleanup.policy=delete            # delete或compact

# 日志清理间隔
log.retention.check.interval.ms=300000  # 5分钟
```

**副本配置**:
```properties
# 默认副本数
default.replication.factor=3

# 最小同步副本数
min.insync.replicas=2

# 副本落后判断阈值
replica.lag.time.max.ms=10000        # 10秒

# 不允许非ISR副本选举为Leader
unclean.leader.election.enable=false
```

#### 生产者配置

**基础配置**:
```java
Properties props = new Properties();

// Broker地址
props.put("bootstrap.servers", "kafka1:9092,kafka2:9092,kafka3:9092");

// 序列化器
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");

// ACK级别
props.put("acks", "all");  // 0, 1, all

// 重试次数
props.put("retries", 3);
props.put("retry.backoff.ms", 100);
```

**性能优化配置**:
```java
// 批量发送
props.put("batch.size", 16384);      // 16KB
props.put("linger.ms", 10);          // 等待10ms凑批

// 压缩
props.put("compression.type", "lz4"); // none, gzip, snappy, lz4, zstd

// 缓冲区
props.put("buffer.memory", 33554432);  // 32MB

// 最大请求大小
props.put("max.request.size", 1048576); // 1MB

// 并发请求数
props.put("max.in.flight.requests.per.connection", 5);
```

**可靠性配置**:
```java
// 幂等性(防止重复)
props.put("enable.idempotence", true);

// 事务ID(用于事务)
props.put("transactional.id", "my-transactional-id");

// 请求超时
props.put("request.timeout.ms", 30000);  // 30秒
```

#### 消费者配置

**基础配置**:
```java
Properties props = new Properties();

// Broker地址
props.put("bootstrap.servers", "kafka1:9092,kafka2:9092,kafka3:9092");

// 消费者组ID
props.put("group.id", "my-consumer-group");

// 反序列化器
props.put("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
props.put("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
```

**Offset管理配置**:
```java
// 自动提交
props.put("enable.auto.commit", "true");
props.put("auto.commit.interval.ms", "1000");

// 消费起始位置
props.put("auto.offset.reset", "earliest");  // earliest, latest, none

// 或手动提交
props.put("enable.auto.commit", "false");
```

**性能优化配置**:
```java
// 一次拉取的最大记录数
props.put("max.poll.records", 500);

// 拉取超时
props.put("max.poll.interval.ms", 300000);  // 5分钟

// 单次拉取最小/最大字节数
props.put("fetch.min.bytes", 1);
props.put("fetch.max.bytes", 52428800);     // 50MB

// 单个分区拉取最大字节数
props.put("max.partition.fetch.bytes", 1048576);  // 1MB
```

**会话配置**:
```java
// 会话超时(心跳)
props.put("session.timeout.ms", 10000);      // 10秒
props.put("heartbeat.interval.ms", 3000);    // 3秒
```

### 3.3 核心功能的基础使用

#### 功能1: Topic管理

**创建Topic**:
```bash
kafka-topics.sh --create \
  --bootstrap-server localhost:9092 \
  --topic my-topic \
  --partitions 3 \
  --replication-factor 2 \
  --config retention.ms=86400000 \     # 1天
  --config segment.bytes=1073741824    # 1GB
```

**查看Topic列表**:
```bash
kafka-topics.sh --list --bootstrap-server localhost:9092
```

**查看Topic详情**:
```bash
kafka-topics.sh --describe \
  --bootstrap-server localhost:9092 \
  --topic my-topic

# 输出示例:
# Topic: my-topic	PartitionCount: 3	ReplicationFactor: 2
# 	Topic: my-topic	Partition: 0	Leader: 1	Replicas: 1,2	Isr: 1,2
# 	Topic: my-topic	Partition: 1	Leader: 2	Replicas: 2,0	Isr: 2,0
# 	Topic: my-topic	Partition: 2	Leader: 0	Replicas: 0,1	Isr: 0,1
```

**修改Topic**:
```bash
# 增加分区(只能增加不能减少)
kafka-topics.sh --alter \
  --bootstrap-server localhost:9092 \
  --topic my-topic \
  --partitions 5

# 修改配置
kafka-configs.sh --alter \
  --bootstrap-server localhost:9092 \
  --entity-type topics \
  --entity-name my-topic \
  --add-config retention.ms=604800000  # 改为7天
```

**删除Topic**:
```bash
kafka-topics.sh --delete \
  --bootstrap-server localhost:9092 \
  --topic my-topic
```

#### 功能2: 生产消息

**命令行生产**:
```bash
kafka-console-producer.sh \
  --bootstrap-server localhost:9092 \
  --topic my-topic
> Message 1
> Message 2
```

**Java生产者**:
```java
import org.apache.kafka.clients.producer.*;
import java.util.Properties;

public class SimpleProducer {
    public static void main(String[] args) {
        Properties props = new Properties();
        props.put("bootstrap.servers", "localhost:9092");
        props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        props.put("acks", "all");

        KafkaProducer<String, String> producer = new KafkaProducer<>(props);

        // 异步发送
        for (int i = 0; i < 100; i++) {
            ProducerRecord<String, String> record =
                new ProducerRecord<>("my-topic", "key-" + i, "value-" + i);

            producer.send(record, new Callback() {
                @Override
                public void onCompletion(RecordMetadata metadata, Exception exception) {
                    if (exception == null) {
                        System.out.printf("Sent to partition %d, offset %d%n",
                            metadata.partition(), metadata.offset());
                    } else {
                        exception.printStackTrace();
                    }
                }
            });
        }

        producer.close();
    }
}
```

**指定分区发送**:
```java
// 方式1: 指定分区号
ProducerRecord<String, String> record =
    new ProducerRecord<>("my-topic", 0, "key", "value");

// 方式2: 指定Key(根据Key的hash分区)
ProducerRecord<String, String> record =
    new ProducerRecord<>("my-topic", "user123", "value");
```

#### 功能3: 消费消息

**命令行消费**:
```bash
# 从最新位置开始消费
kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic my-topic

# 从头开始消费
kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic my-topic \
  --from-beginning

# 显示Key和Offset
kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic my-topic \
  --property print.key=true \
  --property print.offset=true \
  --from-beginning
```

**Java消费者(自动提交)**:
```java
import org.apache.kafka.clients.consumer.*;
import java.time.Duration;
import java.util.Arrays;
import java.util.Properties;

public class SimpleConsumer {
    public static void main(String[] args) {
        Properties props = new Properties();
        props.put("bootstrap.servers", "localhost:9092");
        props.put("group.id", "my-consumer-group");
        props.put("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
        props.put("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
        props.put("enable.auto.commit", "true");
        props.put("auto.commit.interval.ms", "1000");

        KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
        consumer.subscribe(Arrays.asList("my-topic"));

        try {
            while (true) {
                ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
                for (ConsumerRecord<String, String> record : records) {
                    System.out.printf("partition=%d, offset=%d, key=%s, value=%s%n",
                        record.partition(), record.offset(), record.key(), record.value());
                }
            }
        } finally {
            consumer.close();
        }
    }
}
```

**Java消费者(手动提交)**:
```java
props.put("enable.auto.commit", "false");

KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
consumer.subscribe(Arrays.asList("my-topic"));

try {
    while (true) {
        ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));

        for (ConsumerRecord<String, String> record : records) {
            // 处理消息
            processRecord(record);
        }

        // 同步提交所有offset
        consumer.commitSync();
    }
} finally {
    consumer.close();
}
```

### 3.4 常用命令/API详解

#### 消费者组管理命令

**查看所有消费者组**:
```bash
kafka-consumer-groups.sh --list --bootstrap-server localhost:9092
```

**查看消费者组详情**:
```bash
kafka-consumer-groups.sh --describe \
  --bootstrap-server localhost:9092 \
  --group my-consumer-group

# 输出示例:
# GROUP           TOPIC      PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG
# my-group        my-topic   0          100             150             50
# my-group        my-topic   1          200             200             0
# my-group        my-topic   2          300             350             50
```

**重置Offset**:
```bash
# 重置到最早
kafka-consumer-groups.sh --reset-offsets \
  --bootstrap-server localhost:9092 \
  --group my-consumer-group \
  --topic my-topic \
  --to-earliest \
  --execute

# 重置到指定位置
kafka-consumer-groups.sh --reset-offsets \
  --bootstrap-server localhost:9092 \
  --group my-consumer-group \
  --topic my-topic:0 \
  --to-offset 100 \
  --execute

# 重置到指定时间
kafka-consumer-groups.sh --reset-offsets \
  --bootstrap-server localhost:9092 \
  --group my-consumer-group \
  --topic my-topic \
  --to-datetime 2024-01-01T00:00:00.000 \
  --execute

# 向前/向后移动
kafka-consumer-groups.sh --reset-offsets \
  --bootstrap-server localhost:9092 \
  --group my-consumer-group \
  --topic my-topic \
  --shift-by -100 \  # 向前100条
  --execute
```

**删除消费者组**:
```bash
kafka-consumer-groups.sh --delete \
  --bootstrap-server localhost:9092 \
  --group my-consumer-group
```

#### 性能测试命令

**生产者性能测试**:
```bash
kafka-producer-perf-test.sh \
  --topic test-topic \
  --num-records 1000000 \
  --record-size 1024 \
  --throughput 10000 \
  --producer-props bootstrap.servers=localhost:9092

# 输出示例:
# 999999 records sent, 9999.9 records/sec (9.77 MB/sec), 2.5 ms avg latency
```

**消费者性能测试**:
```bash
kafka-consumer-perf-test.sh \
  --bootstrap-server localhost:9092 \
  --topic test-topic \
  --messages 1000000 \
  --threads 1

# 输出示例:
# data.consumed.in.MB=976.5625, MB.sec=19.5313, data.consumed.in.nMsg=1000000
```

#### 配置管理命令

**查看配置**:
```bash
# 查看Topic配置
kafka-configs.sh --describe \
  --bootstrap-server localhost:9092 \
  --entity-type topics \
  --entity-name my-topic

# 查看Broker配置
kafka-configs.sh --describe \
  --bootstrap-server localhost:9092 \
  --entity-type brokers \
  --entity-name 0
```

**动态修改配置**:
```bash
# 修改Topic配置
kafka-configs.sh --alter \
  --bootstrap-server localhost:9092 \
  --entity-type topics \
  --entity-name my-topic \
  --add-config retention.ms=86400000,max.message.bytes=2097152

# 删除配置
kafka-configs.sh --alter \
  --bootstrap-server localhost:9092 \
  --entity-type topics \
  --entity-name my-topic \
  --delete-config retention.ms
```

### 3.5 新手必须掌握的基础概念

#### 概念1: 消息的有序性

**保证规则**:
```
1. 同一Partition内的消息严格有序
2. 不同Partition之间的消息无序
3. 指定相同Key的消息会进入同一Partition

实际应用:
场景: 订单状态更新
要求: 同一订单的状态变更必须有序

解决方案:
- 使用订单ID作为Key
- 相同订单ID的消息进入同一Partition
- 保证顺序: 创建 → 支付 → 发货 → 完成
```

**代码示例**:
```java
// 发送订单消息,使用订单ID作为Key
String orderId = "ORDER123";
ProducerRecord<String, String> record = new ProducerRecord<>(
    "orders",
    orderId,  // Key: 订单ID
    orderJson // Value: 订单详情
);
producer.send(record);

// 相同orderId的消息进入同一Partition,保证有序
```

#### 概念2: 消费者的并行度

**核心原理**:
```
并行度 = min(消费者数量, Partition数量)

情况1: 消费者数 < Partition数
- 每个消费者处理多个Partition
- 未充分利用消费者

情况2: 消费者数 = Partition数 (推荐)
- 一对一映射
- 最佳并行度

情况3: 消费者数 > Partition数
- 多余的消费者空闲
- 浪费资源
```

**实际案例**:
```
Topic: orders, Partition: 12

配置1(低并行):
消费者: 3个
分配: 每个消费者处理4个Partition
并行度: 3
吞吐量: 30万条/秒

配置2(最佳并行):
消费者: 12个
分配: 每个消费者处理1个Partition
并行度: 12
吞吐量: 120万条/秒

配置3(过度配置):
消费者: 20个
分配: 12个消费者工作,8个空闲
并行度: 12 (没有提升)
吞吐量: 120万条/秒 (浪费资源)
```

#### 概念3: Offset的含义

**理解Offset**:
```
Offset是消息在Partition中的位置标识

特点:
- 64位整数,从0开始
- 单调递增,永不重复
- 每个Partition独立计数

类比: 书籍的页码
- 第1页 = Offset 0
- 第2页 = Offset 1
- 可以跳到任意页(Offset)阅读
```

**Offset管理**:
```java
// 获取当前消费位置
consumer.position(new TopicPartition("my-topic", 0));

// 跳到指定位置
consumer.seek(new TopicPartition("my-topic", 0), 100);

// 跳到开始
consumer.seekToBeginning(Arrays.asList(new TopicPartition("my-topic", 0)));

// 跳到末尾
consumer.seekToEnd(Arrays.asList(new TopicPartition("my-topic", 0)));
```

#### 概念4: Rebalance(再均衡)

**触发时机**:
```
1. 消费者加入组
2. 消费者离开组(主动关闭或崩溃)
3. Topic的Partition数量变化
4. 消费者订阅的Topic变化
```

**Rebalance过程**:
```
1. 所有消费者停止消费
2. 释放当前Partition
3. 重新分配Partition
4. 恢复消费

影响:
- 短暂的消费暂停(STW)
- 可能导致重复消费
```

**如何避免不必要的Rebalance**:
```java
// 增加会话超时时间
props.put("session.timeout.ms", "30000");  // 30秒

// 减少心跳间隔
props.put("heartbeat.interval.ms", "3000"); // 3秒

// 增加处理超时时间
props.put("max.poll.interval.ms", "600000"); // 10分钟

// 确保及时提交Offset
consumer.commitSync();
```

### 3.6 最简单的"Hello World"级别示例

**完整示例: 消息发送与接收**

```java
// ========== 生产者 ==========
import org.apache.kafka.clients.producer.*;
import java.util.Properties;

public class HelloKafkaProducer {
    public static void main(String[] args) {
        // 1. 配置
        Properties props = new Properties();
        props.put("bootstrap.servers", "localhost:9092");
        props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");

        // 2. 创建生产者
        KafkaProducer<String, String> producer = new KafkaProducer<>(props);

        // 3. 发送消息
        try {
            for (int i = 0; i < 10; i++) {
                String key = "key-" + i;
                String value = "Hello Kafka " + i;

                ProducerRecord<String, String> record =
                    new ProducerRecord<>("hello-topic", key, value);

                producer.send(record, (metadata, exception) -> {
                    if (exception == null) {
                        System.out.printf("发送成功: partition=%d, offset=%d%n",
                            metadata.partition(), metadata.offset());
                    } else {
                        System.err.println("发送失败: " + exception.getMessage());
                    }
                });
            }
        } finally {
            // 4. 关闭生产者
            producer.close();
        }
    }
}

// ========== 消费者 ==========
import org.apache.kafka.clients.consumer.*;
import java.time.Duration;
import java.util.Arrays;
import java.util.Properties;

public class HelloKafkaConsumer {
    public static void main(String[] args) {
        // 1. 配置
        Properties props = new Properties();
        props.put("bootstrap.servers", "localhost:9092");
        props.put("group.id", "hello-group");
        props.put("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
        props.put("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
        props.put("auto.offset.reset", "earliest");

        // 2. 创建消费者
        KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);

        // 3. 订阅Topic
        consumer.subscribe(Arrays.asList("hello-topic"));

        // 4. 消费消息
        try {
            while (true) {
                ConsumerRecords<String, String> records =
                    consumer.poll(Duration.ofMillis(100));

                for (ConsumerRecord<String, String> record : records) {
                    System.out.printf("接收消息: partition=%d, offset=%d, key=%s, value=%s%n",
                        record.partition(), record.offset(), record.key(), record.value());
                }
            }
        } finally {
            // 5. 关闭消费者
            consumer.close();
        }
    }
}
```

**运行步骤**:
```bash
# 1. 创建Topic
kafka-topics.sh --create \
  --bootstrap-server localhost:9092 \
  --topic hello-topic \
  --partitions 3 \
  --replication-factor 1

# 2. 运行消费者(先启动)
java HelloKafkaConsumer

# 3. 运行生产者(另开终端)
java HelloKafkaProducer

# 4. 观察输出
# 生产者输出:
# 发送成功: partition=0, offset=0
# 发送成功: partition=1, offset=0
# ...

# 消费者输出:
# 接收消息: partition=0, offset=0, key=key-0, value=Hello Kafka 0
# 接收消息: partition=1, offset=0, key=key-1, value=Hello Kafka 1
# ...
```

---

**本节小结**:
- Kafka诞生于2010年LinkedIn,解决了传统消息队列的吞吐量瓶颈
- 核心理念包括日志为中心、分区并行、Pull模式、零拷贝、批量处理
- 独特的ISR机制平衡了一致性和可用性
- 基础使用包括Topic管理、消息生产和消费
- 理解Offset、Partition、Rebalance等核心概念是掌握Kafka的基础

下一章将深入讲解Kafka的核心功能详解,包括分区策略、副本机制、事务等高级特性。

