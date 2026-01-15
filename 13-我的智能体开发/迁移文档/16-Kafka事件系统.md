# Kafka事件系统

> **文档版本**: v1.0
> **创建日期**: 2026-01-05
> **适用场景**: 任务事件生产、消费与审计日志

---

## 1. 功能概述

Kafka事件系统提供**事件驱动的任务生命周期管理**能力：

- **事件生产**：发送任务创建、完成、失败等事件到Kafka
- **事件消费**：监听和处理来自其他服务的任务事件
- **审计日志**：记录任务操作的合规性日志
- **HTTP代理**：通过HTTP API访问Kafka Service（简化集成）

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kafka事件系统架构                             │
│                                                                  │
│  ┌──────────────────┐                  ┌──────────────────┐     │
│  │ Task Orchestration│                 │  Kafka Service   │     │
│  │     Service      │                  │  (HTTP API)      │     │
│  │                  │                  │                  │     │
│  │ ┌──────────────┐ │   HTTP/JSON      │ ┌──────────────┐ │     │
│  │ │TaskEventProducer│ ─────────────> │ │  Broker API  │ │     │
│  │ └──────────────┘ │                  │ └──────────────┘ │     │
│  │                  │                  │        │         │     │
│  │ ┌──────────────┐ │                  │        ▼         │     │
│  │ │TaskEventConsumer│ <────────────── │ ┌──────────────┐ │     │
│  │ └──────────────┘ │                  │ │ Kafka Cluster │ │     │
│  │                  │                  │ └──────────────┘ │     │
│  │ ┌──────────────┐ │                  │                  │     │
│  │ │ AuditLogger  │ │ (rdkafka直连)   │                  │     │
│  │ └──────────────┘ │ ─────────────────┼──────────>       │     │
│  │                  │                  │                  │     │
│  └──────────────────┘                  └──────────────────┘     │
│                                                                  │
│  Topic列表：                                                     │
│  • task.created   - 任务创建事件                                │
│  • task.completed - 任务完成事件                                │
│  • task.failed    - 任务失败事件                                │
│  • task.audit     - 审计日志事件                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心模块

### 2.1 模块结构

```rust
//! Kafka Service 集成模块

pub mod audit;       // 审计日志（直连Kafka）
pub mod client_trait; // 客户端trait定义
pub mod consumer;    // 事件消费者
pub mod grpc_client; // HTTP客户端实现
pub mod producer;    // 事件生产者

// 重新导出常用的类型
pub use audit::AuditLogger;
pub use client_trait::KafkaServiceClient;
pub use consumer::TaskEventConsumer;
pub use grpc_client::KafkaServiceHttpClient;
pub use producer::TaskEventProducer;
```

---

## 3. KafkaServiceClient - 客户端Trait

### 3.1 Trait定义

```rust
/// Kafka Service 客户端 trait
#[async_trait]
pub trait KafkaServiceClient: Send + Sync + Clone {
    /// 创建客户端实例
    async fn new(config: KafkaServiceConfig) -> Result<Self>;

    /// 健康检查
    async fn health_check(&self) -> Result<HealthCheckResponse>;

    /// 获取集群信息
    async fn get_cluster_info(&self) -> Result<GetClusterInfoResponse>;

    /// 发送单条消息
    async fn send_message(
        &self,
        topic: &str,
        key: Option<&str>,
        value: &[u8],
        headers: Option<HashMap<String, String>>,
    ) -> Result<SendMessageResponse>;

    /// 批量发送消息
    async fn batch_send_message(
        &self,
        topic: &str,
        messages: Vec<KafkaMessage>,
    ) -> Result<BatchSendMessageResponse>;

    /// 消费消息
    async fn consume_messages(
        &self,
        topic: &str,
        consumer_group: &str,
        max_messages: Option<i32>,
        timeout_seconds: Option<i32>,
    ) -> Result<ConsumeMessagesResponse>;
}
```

### 3.2 响应结构

```rust
/// 健康检查响应
pub struct HealthCheckResponse {
    pub healthy: bool,
    pub message: String,
    pub timestamp: String,
}

/// 集群信息响应
pub struct GetClusterInfoResponse {
    pub cluster_id: String,
    pub brokers: Vec<BrokerInfo>,
    pub topics: Vec<TopicInfo>,
}

/// 发送消息响应
pub struct SendMessageResponse {
    pub success: bool,
    pub message: String,
    pub partition: Option<i32>,
    pub offset: Option<i64>,
}

/// 批量发送响应
pub struct BatchSendMessageResponse {
    pub success: bool,
    pub message: String,
    pub successful_count: i32,
    pub failed_count: i32,
}

/// 消费消息响应
pub struct ConsumeMessagesResponse {
    pub messages: Vec<KafkaMessage>,
    pub has_more: bool,
}

/// Kafka消息
pub struct KafkaMessage {
    pub key: Option<String>,
    pub value: Vec<u8>,
    pub headers: HashMap<String, String>,
}
```

---

## 4. KafkaServiceHttpClient - HTTP客户端

### 4.1 客户端结构

```rust
/// Kafka Service HTTP 客户端
#[derive(Clone)]
pub struct KafkaServiceHttpClient {
    /// HTTP 客户端
    http_client: HttpClient,
    /// 配置
    config: KafkaServiceConfig,
}

/// HTTP 客户端内部结构
#[derive(Clone)]
struct HttpClient {
    client: Client,
    base_url: String,
    timeout: Duration,
}
```

### 4.2 客户端创建

```rust
#[async_trait]
impl KafkaServiceClient for KafkaServiceHttpClient {
    /// 创建新的 HTTP 客户端
    async fn new(config: KafkaServiceConfig) -> Result<Self> {
        // 使用第一个 broker 作为基础 URL
        let endpoint = if !config.brokers.is_empty() {
            &config.brokers[0]
        } else {
            "localhost:9092"
        };

        info!(endpoint = %endpoint, "初始化 Kafka Service HTTP 客户端");

        let http_client = HttpClient::new(&config)?;

        info!("Kafka Service HTTP 客户端初始化成功");

        Ok(Self {
            http_client,
            config,
        })
    }
}
```

### 4.3 消息发送

```rust
impl KafkaServiceHttpClient {
    /// 发送单条消息
    async fn send_message(
        &self,
        topic: &str,
        key: Option<&str>,
        value: &[u8],
        headers: Option<HashMap<String, String>>,
    ) -> Result<SendMessageResponse> {
        debug!(topic = topic, key = ?key, "发送消息");

        let request = SendMessageRequestHttp {
            topic: topic.to_string(),
            key: key.map(|k| k.to_string()),
            value: value.to_vec(),
            headers: headers.unwrap_or_default(),
        };

        let response: SendMessageResponseHttp =
            self.http_client.post("/send-message", &request).await?;

        Ok(SendMessageResponse {
            success: response.success,
            message: response.message,
            partition: response.partition,
            offset: response.offset,
        })
    }
}
```

---

## 5. TaskEventProducer - 事件生产者

### 5.1 生产者结构

```rust
/// 任务事件生产者
pub struct TaskEventProducer {
    /// Kafka 客户端
    kafka_client: KafkaServiceHttpClient,
}

impl TaskEventProducer {
    /// 创建新的任务事件生产者
    pub async fn new(kafka_client: KafkaServiceHttpClient) -> Result<Self> {
        Ok(Self { kafka_client })
    }
}
```

### 5.2 任务创建事件

```rust
impl TaskEventProducer {
    /// 发送任务创建事件
    pub async fn send_task_created_event(
        &self,
        task_id: &str,
        task_data: &Value,
    ) -> Result<()> {
        let topic = "task.created";
        let message = serde_json::to_string(task_data)
            .map_err(|e| ServiceError::SerializationError(e.to_string()))?;

        info!(task_id = task_id, "发送任务创建事件");

        let response = self.kafka_client
            .send_message(topic, Some(task_id), message.as_bytes(), None)
            .await?;

        if response.success {
            info!(
                task_id = task_id,
                partition = ?response.partition,
                offset = ?response.offset,
                "任务创建事件发送成功"
            );
        } else {
            error!(
                task_id = task_id,
                error = %response.message,
                "任务创建事件发送失败"
            );
        }

        Ok(())
    }
}
```

### 5.3 任务完成事件

```rust
impl TaskEventProducer {
    /// 发送任务完成事件
    pub async fn send_task_completed_event(
        &self,
        task_id: &str,
        result: &Value,
    ) -> Result<()> {
        let topic = "task.completed";
        let message = serde_json::to_string(result)
            .map_err(|e| ServiceError::SerializationError(e.to_string()))?;

        info!(task_id = task_id, "发送任务完成事件");

        let response = self.kafka_client
            .send_message(topic, Some(task_id), message.as_bytes(), None)
            .await?;

        if response.success {
            info!(
                task_id = task_id,
                partition = ?response.partition,
                offset = ?response.offset,
                "任务完成事件发送成功"
            );
        }

        Ok(())
    }
}
```

### 5.4 任务失败事件

```rust
impl TaskEventProducer {
    /// 发送任务失败事件
    pub async fn send_task_failed_event(
        &self,
        task_id: &str,
        error_message: &str,
    ) -> Result<()> {
        let topic = "task.failed";
        let message = format!(
            "{{\"taskId\":\"{}\",\"error\":\"{}\"}}",
            task_id, error_message
        );

        info!(task_id = task_id, "发送任务失败事件");

        let response = self.kafka_client
            .send_message(topic, Some(task_id), message.as_bytes(), None)
            .await?;

        if response.success {
            info!(task_id = task_id, "任务失败事件发送成功");
        }

        Ok(())
    }
}
```

### 5.5 通用事件发送

```rust
impl TaskEventProducer {
    /// 发送通用任务事件
    pub async fn send_task_event(
        &self,
        topic: &str,
        task_id: &str,
        event_data: &Value,
    ) -> Result<()> {
        let message = serde_json::to_string(event_data)
            .map_err(|e| ServiceError::SerializationError(e.to_string()))?;

        debug!(task_id = task_id, topic = topic, "发送任务事件");

        let response = self.kafka_client
            .send_message(topic, Some(task_id), message.as_bytes(), None)
            .await?;

        if response.success {
            debug!(
                task_id = task_id,
                topic = topic,
                partition = ?response.partition,
                offset = ?response.offset,
                "任务事件发送成功"
            );
        }

        Ok(())
    }
}
```

---

## 6. TaskEventConsumer - 事件消费者

### 6.1 消费者结构

```rust
/// 任务事件消费者
#[derive(Clone)]
pub struct TaskEventConsumer {
    /// Kafka 客户端
    kafka_client: KafkaServiceHttpClient,
    /// 消费者组 ID
    consumer_group: String,
}

impl TaskEventConsumer {
    /// 创建新的任务事件消费者
    pub async fn new(
        kafka_client: KafkaServiceHttpClient,
        consumer_group: String,
    ) -> Result<Self> {
        Ok(Self {
            kafka_client,
            consumer_group,
        })
    }
}
```

### 6.2 任务创建事件监听

```rust
impl TaskEventConsumer {
    /// 启动任务创建事件监听器
    pub async fn start_task_created_listener(&self) -> Result<()> {
        let topic = "task.created";
        info!(
            topic = topic,
            consumer_group = &self.consumer_group,
            "启动任务创建事件监听器"
        );

        loop {
            match self.kafka_client
                .consume_messages(topic, &self.consumer_group, Some(10), Some(30))
                .await
            {
                Ok(response) => {
                    for message in response.messages {
                        if let Some(key) = &message.key {
                            match self.handle_task_created(key, &message.value).await {
                                Ok(_) => {
                                    info!(task_id = key, "任务创建事件处理成功");
                                }
                                Err(e) => {
                                    error!(
                                        task_id = key,
                                        error = %e,
                                        "任务创建事件处理失败"
                                    );
                                }
                            }
                        }
                    }

                    if !response.has_more {
                        // 暂停一段时间再继续消费
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                }
                Err(e) => {
                    error!(topic = topic, error = %e, "消费任务创建事件失败");
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
            }
        }
    }
}
```

### 6.3 事件处理

```rust
impl TaskEventConsumer {
    /// 处理任务创建事件
    async fn handle_task_created(&self, task_id: &str, message: &[u8]) -> Result<()> {
        let message_str = String::from_utf8_lossy(message);
        debug!(task_id = task_id, message = %message_str, "收到任务创建事件");

        // 解析消息内容并执行任务创建逻辑
        info!(task_id = task_id, "处理任务创建事件");

        Ok(())
    }

    /// 处理任务完成事件
    async fn handle_task_completed(&self, task_id: &str, message: &[u8]) -> Result<()> {
        let message_str = String::from_utf8_lossy(message);
        debug!(task_id = task_id, message = %message_str, "收到任务完成事件");

        info!(task_id = task_id, "处理任务完成事件");

        Ok(())
    }

    /// 处理任务失败事件
    async fn handle_task_failed(&self, task_id: &str, message: &[u8]) -> Result<()> {
        let message_str = String::from_utf8_lossy(message);
        debug!(task_id = task_id, message = %message_str, "收到任务失败事件");

        info!(task_id = task_id, "处理任务失败事件");

        Ok(())
    }
}
```

---

## 7. AuditLogger - 审计日志

### 7.1 审计日志器（直连Kafka）

```rust
#[cfg(feature = "kafka")]
pub struct AuditLogger {
    producer: FutureProducer,
    topic: String,
}

#[cfg(feature = "kafka")]
impl AuditLogger {
    pub fn new(config: &KafkaServiceConfig) -> Result<Self> {
        // 检查是否启用 Kafka
        if !config.enabled {
            return Err(anyhow!("Kafka is not enabled"));
        }

        let producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", config.brokers.join(","))
            .set("compression.type", &config.compression)
            .set("message.timeout.ms", "5000")
            .create()?;

        info!(
            brokers = %config.brokers.join(","),
            topic = %config.topic,
            "Kafka审计日志初始化成功"
        );

        Ok(Self {
            producer,
            topic: config.topic.clone(),
        })
    }
}
```

### 7.2 记录任务提交

```rust
impl AuditLogger {
    /// 记录任务提交审计日志
    pub async fn log_task_submission(
        &self,
        task_id: &str,
        task_description: &str,
        user_id: Option<&str>,
        duration_ms: u64,
        success: bool,
    ) -> Result<()> {
        let log = json!({
            "event_type": "task_submission",
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "task_id": task_id,
            "task_description": task_description,
            "user_id": user_id,
            "duration_ms": duration_ms,
            "success": success,
        });

        let payload = serde_json::to_string(&log)?;

        self.producer
            .send(
                FutureRecord::to(&self.topic)
                    .payload(&payload)
                    .key(task_id),
                Duration::from_secs(0),
            )
            .await
            .map_err(|(e, _)| anyhow!("Kafka发送失败: {}", e))?;

        Ok(())
    }
}
```

### 7.3 记录任务完成

```rust
impl AuditLogger {
    /// 记录任务完成审计日志
    pub async fn log_task_completion(
        &self,
        task_id: &str,
        task_description: &str,
        user_id: Option<&str>,
        duration_ms: u64,
        success: bool,
    ) -> Result<()> {
        let log = json!({
            "event_type": "task_completion",
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "task_id": task_id,
            "task_description": task_description,
            "user_id": user_id,
            "duration_ms": duration_ms,
            "success": success,
        });

        let payload = serde_json::to_string(&log)?;

        self.producer
            .send(
                FutureRecord::to(&self.topic)
                    .payload(&payload)
                    .key(task_id),
                Duration::from_secs(0),
            )
            .await
            .map_err(|(e, _)| anyhow!("Kafka发送失败: {}", e))?;

        Ok(())
    }
}
```

### 7.4 Feature条件编译

```rust
// 不使用Kafka feature时的空实现
#[cfg(not(feature = "kafka"))]
pub struct AuditLogger;

#[cfg(not(feature = "kafka"))]
impl AuditLogger {
    pub fn new(_config: &KafkaServiceConfig) -> Result<Self> {
        Ok(Self)
    }

    pub async fn log_task_submission(
        &self,
        _task_id: &str,
        _task_description: &str,
        _user_id: Option<&str>,
        _duration_ms: u64,
        _success: bool,
    ) -> Result<()> {
        Ok(())
    }

    // 其他方法同样为空实现...
}
```

---

## 8. 配置选项

### 8.1 TOML配置

```toml
# config.toml

[kafka]
# 是否启用Kafka
enabled = true

# Kafka Broker地址列表
# 如果使用HTTP代理，填写Kafka Service的HTTP地址
brokers = ["http://192.168.0.141:8080"]

# 默认Topic
topic = "task.audit"

# 压缩类型
compression = "gzip"

# 超时时间（秒）
timeout_secs = 30

# 消费者组ID
consumer_group = "task-orchestration-group"
```

### 8.2 环境变量

```bash
# 启用Kafka
export KAFKA_ENABLED=true

# Broker地址
export KAFKA_BROKERS=http://192.168.0.141:8080

# 审计日志Topic
export KAFKA_AUDIT_TOPIC=task.audit

# 消费者组
export KAFKA_CONSUMER_GROUP=task-orchestration-group
```

### 8.3 Cargo.toml Feature

```toml
[features]
default = []
kafka = ["rdkafka"]

[dependencies]
# Kafka 支持（可选）
rdkafka = { version = "0.36", optional = true, features = ["cmake-build"] }
```

---

## 9. 使用示例

### 9.1 发送任务事件

```rust
async fn handle_task_completion(
    producer: &TaskEventProducer,
    task_id: &str,
    result: &serde_json::Value,
) -> Result<()> {
    // 发送任务完成事件
    producer.send_task_completed_event(task_id, result).await?;

    info!(task_id = task_id, "任务完成事件已发送");

    Ok(())
}
```

### 9.2 启动事件消费者

```rust
async fn start_event_consumers(
    kafka_client: KafkaServiceHttpClient,
) -> Result<()> {
    let consumer = TaskEventConsumer::new(
        kafka_client,
        "task-orchestration-group".to_string(),
    ).await?;

    // 启动多个监听器
    tokio::spawn(async move {
        let consumer_clone = consumer.clone();
        consumer_clone.start_task_created_listener().await
    });

    tokio::spawn(async move {
        let consumer_clone = consumer.clone();
        consumer_clone.start_task_completed_listener().await
    });

    tokio::spawn(async move {
        consumer.start_task_failed_listener().await
    });

    Ok(())
}
```

### 9.3 记录审计日志

```rust
async fn log_task_audit(
    audit_logger: &AuditLogger,
    task_id: &str,
    task_description: &str,
    duration_ms: u64,
    success: bool,
) -> Result<()> {
    audit_logger.log_task_completion(
        task_id,
        task_description,
        Some("user_001"),
        duration_ms,
        success,
    ).await?;

    info!(task_id = task_id, "审计日志已记录");

    Ok(())
}
```

---

## 10. Topic设计

### 10.1 Topic列表

| Topic | 描述 | 生产者 | 消费者 |
|-------|------|--------|--------|
| task.created | 任务创建事件 | Task Service | 外部系统 |
| task.completed | 任务完成事件 | Task Service | 外部系统 |
| task.failed | 任务失败事件 | Task Service | 外部系统 |
| task.audit | 审计日志 | Task Service | 审计系统 |

### 10.2 消息格式

```json
// task.created
{
  "taskId": "task_001",
  "taskDescription": "查询设备状态",
  "metadata": {...},
  "timestamp": "2026-01-05T10:30:00Z"
}

// task.completed
{
  "taskId": "task_001",
  "result": {...},
  "duration_ms": 5000,
  "timestamp": "2026-01-05T10:30:05Z"
}

// task.failed
{
  "taskId": "task_001",
  "error": "工具执行超时",
  "timestamp": "2026-01-05T10:30:03Z"
}

// task.audit
{
  "event_type": "task_completion",
  "timestamp": "2026-01-05T10:30:05Z",
  "task_id": "task_001",
  "task_description": "查询设备状态",
  "user_id": "user_001",
  "duration_ms": 5000,
  "success": true
}
```

---

## 11. 迁移实现清单

### 11.1 客户端Trait

- [ ] 实现 `KafkaServiceClient` trait
- [ ] 定义响应结构体
- [ ] 定义消息结构体

### 11.2 HTTP客户端

- [ ] 实现 `KafkaServiceHttpClient`
- [ ] 实现健康检查
- [ ] 实现消息发送
- [ ] 实现消息消费

### 11.3 事件生产者

- [ ] 实现 `TaskEventProducer`
- [ ] 实现任务创建事件
- [ ] 实现任务完成事件
- [ ] 实现任务失败事件

### 11.4 事件消费者

- [ ] 实现 `TaskEventConsumer`
- [ ] 实现事件监听循环
- [ ] 实现事件处理方法

### 11.5 审计日志

- [ ] 实现 `AuditLogger`（可选kafka feature）
- [ ] 实现任务提交日志
- [ ] 实现任务完成日志
- [ ] 实现条件编译空实现

---

## 12. 相关文档

- [08-API与服务接口](./08-API与服务接口.md) - 服务接口定义
- [09-配置与部署指南](./09-配置与部署指南.md) - 配置说明
- [15-服务发现与注册](./15-服务发现与注册.md) - 服务发现机制

---

**文档维护者**: Task Orchestration Team
**最后更新**: 2026-01-05
