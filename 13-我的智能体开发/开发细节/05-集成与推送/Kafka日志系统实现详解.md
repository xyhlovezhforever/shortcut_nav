# Kafka 日志系统实现详解

> 文档版本：v1.6
> 最后更新：2026-01-15
> 相关模块：logging, kafka

---

## 📋 目录

- [1. 概述](#1-概述)
- [2. 日志架构](#2-日志架构)
- [3. 核心组件](#3-核心组件)
- [4. 日志发送点](#4-日志发送点)
- [5. 配置管理](#5-配置管理)
- [6. 数据结构](#6-数据结构)
- [7. 实现细节](#7-实现细节)
- [8. 使用示例](#8-使用示例)

---

## 1. 概述

### 1.1 功能定位

任务编排服务的 Kafka 日志系统负责将关键执行事件发送到 Kafka 消息队列，支持：

- **步骤规划日志**：记录每个执行步骤的工具信息和依赖关系
- **工具执行日志**：记录工具执行的成功/失败结果
- **任务状态日志**：记录任务的完成/失败状态
- **审计日志**：记录任务的提交、完成、取消等操作

### 1.2 设计目标

- ✅ **统一配置**：所有 Kafka 主题从配置文件读取
- ✅ **主题分离**：审计日志与事件通知使用不同主题
- ✅ **容错机制**：Kafka 不可用时自动降级到控制台输出
- ✅ **结构化日志**：使用统一的 JSON 格式
- ✅ **分类标记**：通过 `category` 字段方便日志过滤
- ✅ **智能分区**：使用 `task-orchestration-service-{module}-{session_id}` 格式的 key，确保同一任务的日志分配到同一分区

---

## 2. 日志架构

### 2.1 主题划分

| 主题类型 | 主题名称 | 配置路径 | 用途 |
|---------|---------|---------|------|
| **审计日志** | `task-audit-log` | `kafka_service.topic` | 详细执行日志（步骤规划、工具执行、任务状态） |
| **任务创建事件** | `task.created` | `kafka_service.event_topics.task_created` | 任务创建通知（事件驱动） |
| **任务完成事件** | `task.completed` | `kafka_service.event_topics.task_completed` | 任务完成通知（事件驱动） |
| **任务失败事件** | `task.failed` | `kafka_service.event_topics.task_failed` | 任务失败通知（事件驱动） |

### 2.2 日志分类

通过 `category` 字段进行分类：

```rust
pub enum LogCategory {
    StepPlanned,      // 步骤规划
    ToolExecution,    // 工具执行
    TaskCompleted,    // 任务完成
    TaskFailed,       // 任务失败
}
```

---

## 3. 核心组件

### 3.1 KafkaLogger

**文件位置**: `src/logging/kafka_logger.rs`

**职责**：
- 发送步骤规划日志
- 发送工具执行结果日志
- 发送任务成功/失败日志

**关键实现**：

```rust
pub struct KafkaLogger {
    #[cfg(feature = "kafka")]
    producer: Option<FutureProducer>,
    service_name: String,
    topic: String,  // 从配置文件读取
}

impl KafkaLogger {
    pub async fn new(config: &AppConfig) -> Result<Self> {
        Ok(Self {
            producer,
            service_name: "task-orchestration-service".to_string(),
            topic: config.kafka_service.topic.clone(), // ✅ 从配置读取
        })
    }
}
```

**关键方法**：
- `log()` - 发送通用日志
- `log_tool_execution_result()` - 发送工具执行结果
- `log_tool_selection()` - 发送工具选择记录
- `log_llm_interaction()` - 发送 LLM 交互记录

---

### 3.2 AuditLogger

**文件位置**: `src/kafka/audit.rs`

**职责**：
- 发送任务提交审计日志
- 发送任务完成审计日志
- 发送任务取消审计日志

**关键实现**：

```rust
pub struct AuditLogger {
    producer: FutureProducer,
    topic: String,  // 从配置文件读取
}

impl AuditLogger {
    pub fn new(config: &KafkaServiceConfig) -> Result<Self> {
        Ok(Self {
            producer,
            topic: config.topic.clone(), // ✅ 从配置读取
        })
    }
}
```

---

### 3.3 TaskEventProducer

**文件位置**: `src/kafka/producer.rs`

**职责**：
- 发送任务创建事件
- 发送任务完成事件
- 发送任务失败事件

**关键实现**（2026-01-13 更新）：

```rust
pub struct TaskEventProducer {
    kafka_client: KafkaServiceHttpClient,
    event_topics: TaskEventTopics,  // ✅ 从配置读取
}

impl TaskEventProducer {
    pub async fn new(
        kafka_client: KafkaServiceHttpClient,
        event_topics: TaskEventTopics,
    ) -> Result<Self> {
        Ok(Self { kafka_client, event_topics })
    }

    pub async fn send_task_created_event(&self, task_id: &str, task_data: &Value) -> Result<()> {
        let topic = &self.event_topics.task_created; // ✅ 从配置读取
        // ...
    }
}
```

---

## 4. 日志发送点

### 4.1 步骤规划日志

**触发点**: `Planner::generate_plan_with_sender()` 方法返回前
**文件位置**: `src/core/planner.rs:955-973`

```rust
for (i, step) in plan.steps.iter().enumerate() {
    // 📤 发送步骤规划信息到 Kafka
    let mut fields = HashMap::new();
    fields.insert("plan_id".to_string(), plan.plan_id.clone());
    fields.insert("step_index".to_string(), (i + 1).to_string());
    fields.insert("step_id".to_string(), step.step_id.clone());
    fields.insert("step_name".to_string(), step.name.clone());
    fields.insert("tool_id".to_string(), step.tool.clone());
    fields.insert("dependencies".to_string(), format!("{:?}", step.dependencies));
    fields.insert("category".to_string(), "step_planned".to_string());

    let _ = self.kafka_logger.info("planner", &format!("步骤已规划: {}", step.name), fields).await;
}
```

**日志包含字段**：
- `plan_id`: 计划ID
- `step_index`: 步骤序号
- `step_id`: 步骤ID
- `step_name`: 步骤名称
- `tool_id`: 使用的工具ID
- `dependencies`: 步骤依赖关系
- `category`: "step_planned"

---

### 4.2 工具执行日志

**触发点**: `Executor::execute_step_with_event_sender()` 执行完成后
**文件位置**: `src/core/executor.rs:1222-1238` (成功), `1307-1319` (失败), `1349-1361` (异常)

#### 4.2.1 执行成功

```rust
if response.status.eq_ignore_ascii_case("success") {
    let mut exec_fields = HashMap::new();
    exec_fields.insert("plan_id".to_string(), plan_id.to_string());
    exec_fields.insert("step_name".to_string(), step.name.clone());

    let _ = self.kafka_logger.log_tool_execution_result(
        "executor",
        &step.step_id,
        tool_id,
        true,                    // is_success
        Some(output.as_str()),   // 执行结果
        None,                    // 无错误
        execution_time,
        exec_fields,
    ).await;
}
```

**日志包含字段**：
- `category`: "tool_execution"
- `step_id`: 步骤ID
- `tool_id`: 工具ID
- `status`: "success"
- `execution_time_ms`: 执行耗时（毫秒）
- `output_preview`: 输出预览（截断到512字符）
- `plan_id`: 计划ID
- `step_name`: 步骤名称

#### 4.2.2 执行失败

```rust
// 工具返回失败状态
let mut exec_fields = HashMap::new();
exec_fields.insert("plan_id".to_string(), plan_id.to_string());
exec_fields.insert("step_name".to_string(), step.name.clone());

let _ = self.kafka_logger.log_tool_execution_result(
    "executor",
    &step.step_id,
    tool_id,
    false,                        // is_success
    None,
    Some(error_message.as_str()), // 错误信息
    execution_time,
    exec_fields,
).await;
```

#### 4.2.3 工具调用失败（如"查找工具失败"）

```rust
Ok(Err(e)) => {
    let error_detail = format!("工具调用失败: {}", e);
    let mut exec_fields = HashMap::new();
    exec_fields.insert("plan_id".to_string(), plan_id.to_string());
    exec_fields.insert("step_name".to_string(), step.name.clone());

    let _ = self.kafka_logger.log_tool_execution_result(
        "executor",
        &step.step_id,
        tool_id,
        false,
        None,
        Some(error_detail.as_str()), // 包含 "查找工具失败" 等错误
        execution_time,
        exec_fields,
    ).await;
}
```

#### 4.2.4 执行超时

```rust
Err(_) => {
    let timeout_error = "步骤执行超时".to_string();
    let mut exec_fields = HashMap::new();
    exec_fields.insert("plan_id".to_string(), plan_id.to_string());
    exec_fields.insert("step_name".to_string(), step.name.clone());

    let _ = self.kafka_logger.log_tool_execution_result(
        "executor",
        &step.step_id,
        tool_id,
        false,
        None,
        Some(timeout_error.as_str()),
        execution_time,
        exec_fields,
    ).await;
}
```

---

### 4.3 任务完成日志

**触发点**: `Orchestrator::orchestrate()` 任务成功时
**文件位置**: `src/core/orchestrator.rs:775-789`

```rust
Ok(output) => {
    task.complete(output.clone());
    self.state_manager.update_task(task.clone());

    // 📤 记录任务成功到 Kafka
    let mut fields = HashMap::new();
    fields.insert("task_id".to_string(), task.task_id.clone());
    fields.insert("total_rounds".to_string(), task.current_round.to_string());
    if let Some(duration) = task.duration_secs() {
        fields.insert("total_duration_secs".to_string(), duration.to_string());
    }
    if let Some(score) = task.reflection_rounds.last().and_then(|r| r.score) {
        fields.insert("final_score".to_string(), score.to_string());
    }
    fields.insert("category".to_string(), "task_completed".to_string());

    let _ = self.kafka_logger.info("orchestrator", "任务执行成功", fields).await;
}
```

**日志包含字段**：
- `task_id`: 任务ID
- `total_rounds`: 总反思轮次
- `total_duration_secs`: 总耗时（秒）
- `final_score`: 最终评分
- `category`: "task_completed"

---

### 4.4 任务失败日志

**触发点**: `Orchestrator::orchestrate()` 任务失败时
**文件位置**: `src/core/orchestrator.rs:834-849`

```rust
Err(e) => {
    task.fail(e.to_string());
    self.state_manager.update_task(task.clone());

    // 📤 记录错误到 Kafka（增强版）
    let mut fields = HashMap::new();
    fields.insert("task_id".to_string(), task.task_id.clone());
    fields.insert("error".to_string(), e.to_string());
    fields.insert("total_rounds".to_string(), task.current_round.to_string());
    if let Some(duration) = task.duration_secs() {
        fields.insert("total_duration_secs".to_string(), duration.to_string());
    }
    if let Some(score) = task.reflection_rounds.last().and_then(|r| r.score) {
        fields.insert("final_score".to_string(), score.to_string());
    }
    fields.insert("category".to_string(), "task_failed".to_string());

    let _ = self.kafka_logger.error("orchestrator", "任务执行失败", fields).await;
}
```

**日志包含字段**：
- `task_id`: 任务ID
- `error`: 失败原因
- `total_rounds`: 总反思轮次
- `total_duration_secs`: 总耗时（秒）
- `final_score`: 最终评分（如有）
- `category`: "task_failed"

---

## 5. 配置管理

### 5.1 配置结构

**文件位置**: `src/config/mod.rs:208-242`

```rust
/// Kafka Service 配置
pub struct KafkaServiceConfig {
    /// 是否启用
    pub enabled: bool,
    /// Kafka brokers 地址列表
    pub brokers: Vec<String>,
    /// 主题名称（审计日志主题）
    pub topic: String,
    /// 任务事件主题配置
    #[serde(default = "default_task_event_topics")]
    pub event_topics: TaskEventTopics,
    /// 压缩类型
    pub compression: String,
    /// 连接超时时间（秒）
    pub connect_timeout_secs: u64,
    /// 请求超时时间（秒）
    pub timeout_secs: u64,
    /// 最大重试次数
    pub max_retries: u32,
}

/// 任务事件主题配置
pub struct TaskEventTopics {
    /// 任务创建事件主题
    pub task_created: String,
    /// 任务完成事件主题
    pub task_completed: String,
    /// 任务失败事件主题
    pub task_failed: String,
}
```

### 5.2 默认值

```rust
fn default_task_event_topics() -> TaskEventTopics {
    TaskEventTopics {
        task_created: "task.created".to_string(),
        task_completed: "task.completed".to_string(),
        task_failed: "task.failed".to_string(),
    }
}
```

### 5.3 配置文件示例

**文件位置**: `config.dev.toml`

```toml
[kafka_service]
# 是否启用
enabled = true
# Kafka brokers 地址列表（直接连接 Kafka broker）
brokers = ["192.168.0.141:9092"]
# 主题名称（审计日志主题）
topic = "task-audit-log"
# 压缩类型
compression = "none"
# 连接超时时间（秒）
connect_timeout_secs = 5
# 请求超时时间（秒）
timeout_secs = 60
# 最大重试次数
max_retries = 3

# 任务事件主题配置
[kafka_service.event_topics]
task_created = "task.created"
task_completed = "task.completed"
task_failed = "task.failed"
```

---

## 6. 数据结构

### 6.1 日志消息格式

所有 Kafka 日志使用统一的 JSON 格式：

```json
{
  "timestamp": "2026-01-13T10:30:45.123Z",
  "level": "INFO" | "WARN" | "ERROR",
  "service_name": "task-orchestration-service",
  "module": "planner" | "executor" | "orchestrator",
  "message": "步骤已规划: 查询用户信息",
  "fields": {
    "category": "step_planned",
    "plan_id": "plan_abc123",
    "step_id": "step_1",
    "tool_id": "database_query",
    ...
  }
}
```

### 6.2 日志级别

```rust
pub enum LogLevel {
    Trace,   // 跟踪级别
    Debug,   // 调试级别
    Info,    // 信息级别
    Warn,    // 警告级别
    Error,   // 错误级别
}
```

### 6.3 日志分类（category 字段）

| Category | 含义 | 模块 | 日志级别 |
|----------|------|------|----------|
| `step_planned` | 步骤规划完成 | planner | INFO |
| `tool_execution` | 工具执行结果 | executor | INFO/WARN |
| `planning_start` | 开始规划阶段 | orchestrator | INFO |
| `planning_failed` | 规划阶段失败 | orchestrator | ERROR |
| `evaluation_start` | 开始评估阶段 | orchestrator | INFO |
| `evaluation_completed` | 评估阶段完成 | evaluator | INFO |
| `reflection_start` | 开始反思阶段 | orchestrator | INFO |
| `reflection_completed` | 反思阶段完成 | reflector | INFO |
| `next_round` | 准备进入下一轮规划 | orchestrator | INFO |
| `task_cancelled` | 任务已取消 | orchestrator | WARN |
| `step_execution_error` | 步骤执行异常 | orchestrator | ERROR |
| `step_execution_success` | 步骤执行完成 | orchestrator | INFO |
| `parameter_retry` | 参数调整重试 | orchestrator | INFO |
| `alternative_tool_retry` | 备选工具重试 | orchestrator | INFO |
| `waiting_for_user` | 任务暂停等待用户介入 | orchestrator | INFO |
| `task_replan_start` | 开始任务重新规划 | orchestrator | INFO |
| `single_step_repair_success` | 单步修复成功 | orchestrator | INFO |
| `single_step_repair_failed` | 单步修复失败 | orchestrator | INFO |
| `replan_success` | 重新规划成功 | orchestrator | INFO |
| `replan_failed` | 重新规划失败 | orchestrator | ERROR |
| `step_reflection_complete` | 步骤级反思执行完成 | orchestrator | INFO |
| `task_completed` | 任务成功完成 | orchestrator | INFO |
| `task_failed` | 任务执行失败 | orchestrator | ERROR |

### 6.4 Kafka 消息 Key 格式

**设计原则**：使用智能 key 生成策略，确保同一任务的日志分配到同一 Kafka 分区，便于按会话ID维度查询和分析。

**Key 格式**：
```
task-orchestration-service-{module}-{session_id}
```

- `{module}`：功能模块名称，如 `orchestrator`、`planner`、`executor`、`evaluator`、`reflector` 等
- `{session_id}`：会话ID，通常等于任务ID（task_id），用于聚合同一任务的所有日志

**生成逻辑**（`src/logging/kafka_logger.rs:140-147`）：

```rust
// 生成 key: task-orchestration-service-{module}-{session_id}
// 如果没有 session_id，则降级为原格式
let key = if let Some(session_id) = fields.get("session_id") {
    format!("{}-{}-{}", self.service_name, module, session_id)
} else {
    // 降级为原格式（兼容旧代码）
    format!("{}-{}", self.service_name, module)
};
```

**Key 示例**：
- `task-orchestration-service-orchestrator-121b6d55-bb06-4198-8d6f-c70f1082ec0a`
- `task-orchestration-service-planner-121b6d55-bb06-4198-8d6f-c70f1082ec0a`
- `task-orchestration-service-executor-121b6d55-bb06-4198-8d6f-c70f1082ec0a`
- `task-orchestration-service-planner_tool_selection-121b6d55-bb06-4198-8d6f-c70f1082ec0a`
- `task-orchestration-service-executor`（降级格式，用于没有 session_id 的日志）

**优势**：
1. **同任务聚合**：同一 session_id 的所有日志会路由到同一分区，便于消费者按任务聚合
2. **模块清晰**：module 字段在 key 中，便于快速识别日志来源模块
3. **向后兼容**：没有 session_id 时自动降级为旧格式
4. **负载均衡**：不同 session_id 会分配到不同分区，实现负载均衡
5. **用户可控**：用户可以在提交任务时指定 session_id，方便按业务需求聚合日志

**session_id 自动注入**（`src/state/task_state.rs:124-149`）：

为了确保所有日志都包含 session_id，在 `Task::new` 时会自动将 session_id 添加到 metadata 中：

```rust
pub fn new(description: String, max_rounds: u32, session_id: Option<String>) -> Self {
    let now = Utc::now();
    let task_id = Uuid::new_v4().to_string();
    // 如果用户没有提供 session_id，则自动生成一个
    let session_id = session_id.unwrap_or_else(|| Uuid::new_v4().to_string());

    let mut metadata = HashMap::new();
    // 将 session_id 添加到 metadata 中，用于 Kafka 日志的 key 生成
    metadata.insert("session_id".to_string(), session_id.clone());

    Self {
        task_id,
        session_id,
        description,
        // ...
        metadata,
        // ...
    }
}
```

**用户如何指定 session_id**（`proto/task_orchestrator_service.proto:35-38`）：

用户在提交任务时可以通过 gRPC 请求传递可选的 `session_id` 字段：

```protobuf
message SubmitTaskRequest {
    string task_description = 1;
    optional uint32 max_rounds = 2;
    map<string, string> metadata = 3;
    optional TaskContext context = 4;

    // 会话ID（可选）
    // 用于 Kafka 日志的分区 key 生成，便于按会话ID聚合查询
    // 如果不提供，系统会自动生成一个唯一的会话ID
    optional string session_id = 5;
}
```

---

## 7. 实现细节

### 7.1 容错机制

当 Kafka 不可用时，自动降级到控制台输出：

```rust
#[cfg(feature = "kafka")]
if let Some(ref producer) = self.producer {
    match producer.send(record, Duration::from_secs(0)).await {
        Ok(_) => {
            info!("日志消息发送成功: {}", message);
        }
        Err((e, _)) => {
            // Kafka 发送失败，降级到控制台输出
            self.fallback_log(&log_message);
            warn!("日志消息发送异常: {}，已降级到控制台输出", e);
        }
    }
} else {
    // Kafka 生产者不可用，直接使用备选输出
    self.fallback_log(&log_message);
}
```

### 7.2 字段截断

为避免日志过大，对长字段进行截断（限制512字符）：

```rust
fn truncate_value(value: &str, limit: usize) -> String {
    if value.chars().count() <= limit {
        return value.to_string();
    }
    let truncated: String = value.chars().take(limit).collect();
    format!("{truncated}...<truncated>")
}
```

### 7.3 异步发送

所有日志发送都是异步的，不会阻塞主流程：

```rust
let _ = self.kafka_logger.info("planner", "步骤已规划", fields).await;
```

使用 `let _ =` 忽略发送结果，确保日志发送失败不影响主流程。

---

## 8. 使用示例

### 8.1 发送步骤规划日志

```rust
let mut fields = HashMap::new();
fields.insert("plan_id".to_string(), plan.plan_id.clone());
fields.insert("step_id".to_string(), step.step_id.clone());
fields.insert("tool_id".to_string(), step.tool.clone());
fields.insert("category".to_string(), "step_planned".to_string());

kafka_logger.info("planner", "步骤已规划", fields).await?;
```

### 8.2 发送工具执行结果

```rust
let mut fields = HashMap::new();
fields.insert("plan_id".to_string(), plan_id.to_string());
fields.insert("step_name".to_string(), step.name.clone());

kafka_logger.log_tool_execution_result(
    "executor",
    &step.step_id,
    &tool_id,
    true,                   // 成功
    Some(&output),          // 输出
    None,                   // 无错误
    execution_time_ms,
    fields,
).await?;
```

### 8.3 发送任务完成日志

```rust
let mut fields = HashMap::new();
fields.insert("task_id".to_string(), task_id.clone());
fields.insert("total_rounds".to_string(), rounds.to_string());
fields.insert("category".to_string(), "task_completed".to_string());

kafka_logger.info("orchestrator", "任务执行成功", fields).await?;
```

### 8.4 Kafka 消费者示例

```bash
# 消费审计日志
kafka-console-consumer --bootstrap-server 192.168.0.141:9092 \
  --topic task-audit-log \
  --from-beginning

# 消费任务事件
kafka-console-consumer --bootstrap-server 192.168.0.141:9092 \
  --topic task.created \
  --from-beginning
```

### 8.5 使用 jq 过滤日志

```bash
# 只查看步骤规划日志
kafka-console-consumer --bootstrap-server 192.168.0.141:9092 \
  --topic task-audit-log \
  --from-beginning | jq 'select(.fields.category == "step_planned")'

# 只查看工具执行失败日志
kafka-console-consumer --bootstrap-server 192.168.0.141:9092 \
  --topic task-audit-log \
  --from-beginning | jq 'select(.fields.category == "tool_execution" and .fields.status == "failure")'
```

---

## 附录：代码文件映射

| 组件 | 文件位置 | 行数 |
|------|---------|------|
| KafkaLogger | `src/logging/kafka_logger.rs` | ~381行 |
| AuditLogger | `src/kafka/audit.rs` | ~189行 |
| TaskEventProducer | `src/kafka/producer.rs` | ~140行 |
| 配置结构 | `src/config/mod.rs` | 208-254行 |
| Planner 日志 | `src/core/planner.rs` | 955-973行 |
| Executor 日志 | `src/core/executor.rs` | 1222-1238, 1307-1361行 |
| Orchestrator 日志 | `src/core/orchestrator.rs` | 775-789, 834-849行 |

---

## 更新日志

### v1.6 (2026-01-15)

**🔄 字段重命名：`log_id` 改为 `session_id`：**

本次更新将所有 `log_id` 相关的字段和注释统一重命名为 `session_id`，更准确地反映其语义含义。

**重命名原因：**
1. `log_id` 这个名称容易让人误解为"日志的ID"，但实际上它是用于标识一个会话或任务的唯一标识符
2. `session_id` 更准确地表达了该字段的作用：标识同一个会话/任务的所有日志
3. 使字段命名更加语义化，提高代码可读性

**修改范围：**

1. **Proto 定义** (`proto/task_orchestrator_service.proto:35-38`)
   - ✅ 字段名：`log_id` → `session_id`
   - ✅ 注释：`日志ID` → `会话ID`

2. **Task 结构** (`src/state/task_state.rs:88`)
   - ✅ 字段名：`pub log_id: String` → `pub session_id: String`
   - ✅ 所有方法参数和注释同步更新

3. **gRPC 服务器** (`src/grpc/server.rs:196`)
   - ✅ 变量名：`log_id` → `session_id`

4. **核心模块** (orchestrator, planner, executor, evaluator, reflector)
   - ✅ 所有 metadata 获取：`metadata.get("log_id")` → `metadata.get("session_id")`
   - ✅ 所有 fields 插入：`fields.insert("log_id", ...)` → `fields.insert("session_id", ...)`
   - ✅ 约 100+ 处引用全部更新

5. **Kafka Logger** (`src/logging/kafka_logger.rs:142`)
   - ✅ Key 生成逻辑：`fields.get("log_id")` → `fields.get("session_id")`
   - ✅ 注释：`log_id` → `session_id`

**Kafka Key 格式保持不变：**
```
task-orchestration-service-{module}-{session_id}
```

**向后兼容性：**
- 现有的 Kafka 消费者需要更新，从 `fields.log_id` 改为读取 `fields.session_id`
- Proto 定义的字段序号（field number = 5）保持不变，二进制兼容
- 降级策略仍然有效：没有 session_id 时使用旧格式

**影响说明：**
- 这是一个 **破坏性变更**，需要同步更新所有消费 Kafka 日志的下游服务
- 建议在更新服务后，同时更新文档和 API 说明
- 所有新任务的日志将使用 `session_id` 字段

---

### v1.5 (2026-01-14)

**🔄 Key 格式从 `{category}` 改为 `{module}`：**

本次更新将 Kafka 日志 key 格式从 `task-orchestration-service-{category}-{session_id}` 修改为 `task-orchestration-service-{module}-{session_id}`，使 key 格式更加符合 `微服务名称-功能模块-会话ID` 的设计规范。

**修改原因：**
1. 原格式使用 `category` 作为 key 的第二段，但 `category` 是日志分类（如 `step_planned`、`tool_execution`），而非功能模块
2. 新格式使用 `module` 作为 key 的第二段，表示日志来源的功能模块（如 `orchestrator`、`planner`、`executor`）
3. 这样更便于按功能模块维度进行日志聚合和分析

**修复内容：**

1. **KafkaLogger** (`src/logging/kafka_logger.rs:140-147`)
   - ✅ Key 生成逻辑从依赖 `category` + `session_id` 改为只依赖 `session_id`
   - ✅ Key 格式：`{service_name}-{module}-{session_id}`
   - ✅ 降级格式：`{service_name}-{module}`（当没有 session_id 时）

2. **Orchestrator** (`src/core/orchestrator.rs:758-759`)
   - ✅ 在任务创建后将 `task_id` 注入到 `metadata["session_id"]`
   - ✅ 确保后续所有模块（planner、executor 等）都能从 metadata 获取到 session_id

**修复效果：**

修复前（不一致的 key 格式）：
```
❌ task-orchestration-service-executor                    （缺少 session_id）
❌ task-orchestration-service-orchestrator.executor       （用了点号）
❌ task-orchestration-service-planner                     （缺少 session_id）
❌ task-orchestration-service-planner_tool_selection      （用了下划线）
❌ 121b6d55-bb06-4198-8d6f-c70f1082ec0a                   （只有 session_id）
```

修复后（统一的 key 格式）：
```
✅ task-orchestration-service-orchestrator-121b6d55-bb06-4198-8d6f-c70f1082ec0a
✅ task-orchestration-service-planner-121b6d55-bb06-4198-8d6f-c70f1082ec0a
✅ task-orchestration-service-executor-121b6d55-bb06-4198-8d6f-c70f1082ec0a
✅ task-orchestration-service-planner_tool_selection-121b6d55-bb06-4198-8d6f-c70f1082ec0a
✅ task-orchestration-service-evaluator-121b6d55-bb06-4198-8d6f-c70f1082ec0a
✅ task-orchestration-service-reflector-121b6d55-bb06-4198-8d6f-c70f1082ec0a
```

**影响范围：**
- 所有新任务的日志都将使用统一的 `微服务名称-功能模块-会话ID` 格式
- 便于按功能模块维度进行日志筛选和分析
- `category` 字段仍保留在日志 JSON 的 `fields` 中，用于日志分类过滤

---

### v1.4 (2026-01-14)

**🔧 彻底修复 Kafka 日志 Key 格式不一致问题：**

本次更新彻底解决了日志 key 降级格式问题，确保所有日志都使用统一的 `task-orchestration-service-{category}-{session_id}` 格式。

**问题根源分析：**
1. 部分日志调用缺少 `category` 字段，导致降级为 `task-orchestration-service-{module}` 格式
2. 部分日志调用缺少 `session_id` 字段，导致降级为 `task-orchestration-service-{module}` 格式
3. `execute_plan` 旧方法使用 `HashMap` 而非 `ExecutionContext`，无法传递 session_id

**修复内容：**

1. **Orchestrator 模块** (`src/core/orchestrator.rs`)
   - ✅ 第 1196 行：执行阶段失败日志添加 `category: "execution_failed"`

2. **Executor 模块** (`src/core/executor.rs`)
   - ✅ 重构 `execute_plan` 方法使用 `ExecutionContext`（第 759-888 行）
   - ✅ 确保所有步骤执行日志都能获取 session_id
   - ✅ 修复旧版本方法导致的 session_id 丢失问题

3. **Planner 模块** (`src/core/planner.rs`)
   - ✅ 第 738 行：`generate_plan_with_sender` 的 LLM 交互日志添加 session_id
   - ✅ 第 1673 行：`replan_task` 的 LLM 交互日志添加 session_id
   - ✅ 第 1699 行：`replan_task` 的工具选择日志添加 session_id
   - ✅ 第 1832 行：`replan_single_step` 的 LLM 交互日志添加 session_id
   - ✅ 第 1857 行：`replan_single_step` 的工具选择日志添加 session_id
   - ✅ 修改 `replan_single_step` 函数签名，添加 `metadata` 参数以传递 session_id

**验证结果：**

经过全面检查，所有 15 处 Kafka 日志调用点都已确认包含 `session_id` 和 `category`：

| 文件 | 行号 | 方法类型 | session_id | category |
|------|------|---------|--------|----------|
| orchestrator.rs | 1137 | `.error()` | ✅ | ✅ planning_failed |
| orchestrator.rs | 1199 | `.error()` | ✅ | ✅ execution_failed |
| executor.rs | 1181 | `.info()` | ✅ | ✅ step_execution_start |
| executor.rs | 1248 | `.log_tool_execution_result()` | ✅ | ✅ (内置) |
| executor.rs | 1339 | `.log_tool_execution_result()` | ✅ | ✅ (内置) |
| executor.rs | 1387 | `.log_tool_execution_result()` | ✅ | ✅ (内置) |
| executor.rs | 1429 | `.log_tool_execution_result()` | ✅ | ✅ (内置) |
| planner.rs | 778 | `.log_llm_interaction()` | ✅ | ✅ (内置) |
| planner.rs | 940 | `.log_tool_selection()` | ✅ | ✅ (内置) |
| planner.rs | 979 | `.info()` | ✅ | ✅ step_planned |
| planner.rs | 1685 | `.log_llm_interaction()` | ✅ | ✅ (内置) |
| planner.rs | 1712 | `.log_tool_selection()` | ✅ | ✅ (内置) |
| planner.rs | 1843 | `.log_llm_interaction()` | ✅ | ✅ (内置) |
| planner.rs | 1868 | `.log_tool_selection()` | ✅ | ✅ (内置) |
| planner.rs | 2110 | `.log_llm_interaction()` | ✅ | ✅ (内置) |
| planner.rs | 2171 | `.log_tool_selection()` | ✅ | ✅ (内置) |

---

### v1.3 (2026-01-14)

**评估与反思阶段 Kafka 日志支持：**
1. ✅ 在 Evaluator 中添加评估完成日志（`evaluation_completed`）
2. ✅ 在 Reflector 中添加反思完成日志（`reflection_completed`）
3. ✅ 在 Orchestrator 中添加评估和反思开始日志（`evaluation_start`, `reflection_start`）
4. ✅ 所有评估和反思日志都包含 session_id，确保可追踪

**修复 Executor 和 Planner 模块 session_id 缺失问题：**
1. ✅ Executor 工具执行日志（4处）现在从 `execution_context` 获取 session_id
   - 工具执行成功：`src/core/executor.rs:1218-1226`
   - 工具返回失败：`src/core/executor.rs:1310-1318`
   - 工具调用异常：`src/core/executor.rs:1358-1366`
   - 步骤执行超时：`src/core/executor.rs:1400-1408`
2. ✅ Planner 工具选择日志现在从 `metadata` 获取 session_id：`src/core/planner.rs:924-932`

**实现细节：**
- Evaluator 日志：`src/core/evaluator.rs:333-346` - 评估完成时发送
  - 包含字段：session_id, evaluation_id, overall_score, is_successful, plan_id, category
- Reflector 日志：`src/core/reflector.rs:487-501` - 反思完成时发送
  - 包含字段：session_id, reflection_id, should_replan, current_round, max_rounds, root_causes_count, category
- Executor 日志修复：从 `execution_context.get_initial_metadata()` 提取 session_id
- Planner 日志修复：从 `metadata` 参数提取 session_id
- Orchestrator 集成：
  - `execute_evaluation_phase`: 添加 metadata 参数，发送 evaluation_start 日志
  - `execute_reflection_phase`: 添加 metadata 参数，发送 reflection_start 日志
- 主程序更新：`main.rs` 和 `main_grpc.rs` - 创建 Evaluator 和 Reflector 时传入 kafka_logger

**新增日志类别：**
- `evaluation_start`: 开始评估阶段（orchestrator）
- `evaluation_completed`: 评估阶段完成（evaluator）
- `reflection_start`: 开始反思阶段（orchestrator）
- `reflection_completed`: 反思阶段完成（reflector）

**问题修复：**
- 修复了部分日志使用降级 Kafka key 格式的问题（`task-orchestration-service-{module}`）
- 现在所有日志都使用完整格式：`task-orchestration-service-{category}-{session_id}`
- 确保同一任务的所有日志（包括 planner、executor、evaluator、reflector、orchestrator）都能通过 session_id 聚合查询

---

### v1.2 (2026-01-14)

**session_id 支持（重大更新）：**
1. ✅ 用户可在提交任务时指定可选的 `session_id` 字段
2. ✅ 如果不指定，系统自动生成唯一的会话ID
3. ✅ Kafka key 格式从 `{service}-{category}-{task_id}` 改为 `{service}-{category}-{session_id}`
4. ✅ 支持用户自定义日志聚合维度

**实现细节：**
- Proto 定义：`proto/task_orchestrator_service.proto` - 添加 `optional string session_id = 5`
- Task 结构：`src/state/task_state.rs` - 添加 `pub session_id: String` 字段
- gRPC 处理：`src/grpc/server.rs` - 从请求中获取并传递 session_id
- Kafka key 生成：`src/logging/kafka_logger.rs:140-150` - 使用 session_id 替代 task_id
- 所有日志调用：planner.rs (1处) + orchestrator.rs (17处) - 使用 session_id

**向后兼容：**
- 不传 session_id 时自动生成，无需修改现有客户端
- 降级策略保证旧日志系统仍然可用

---

### v1.1 (2026-01-14)

**Key 格式优化：**
1. ✅ 实现智能 key 生成策略：`task-orchestration-service-{category}-{session_id}`
2. ✅ 确保同一任务的所有日志路由到同一 Kafka 分区
3. ✅ 自动降级支持：无 session_id 或 category 时使用旧格式
4. ✅ Task::new 自动注入 session_id 到 metadata

**日志分类完善：**
1. ✅ 为所有 Orchestrator 日志添加 category 字段
2. ✅ 新增 17 个 category 类型覆盖完整任务生命周期
3. ✅ Planner 日志支持从 metadata 获取 session_id

**代码位置：**
- KafkaLogger key 生成逻辑：`src/logging/kafka_logger.rs:140-150`
- Task metadata 自动注入：`src/state/task_state.rs:124-149`
- Orchestrator 日志增强：`src/core/orchestrator.rs` 18 处更新
- Planner 日志增强：`src/core/planner.rs:958-973`

---

### v1.0 (2026-01-13)

**新增功能：**
1. ✅ 在 Planner 中添加步骤规划信息发送
2. ✅ 确认 Executor 中步骤执行结果发送（已存在）
3. ✅ 在 Orchestrator 中添加任务成功/失败日志

**配置优化：**
1. ✅ KafkaLogger 使用配置文件中的 topic
2. ✅ TaskEventProducer 使用配置文件中的 event_topics
3. ✅ 新增 TaskEventTopics 配置结构
4. ✅ 更新配置文件支持多主题配置

**改进：**
- 所有 Kafka 主题配置统一从配置文件读取
- 支持审计日志与事件主题分离
- 提供默认主题配置，确保向后兼容

---

**相关文档：**
- [用户模式下服务端事件推送机制详解](./6-user模式下服务端事件推送机制详解.md)
- [项目配置文档](../一些项目基础配置相关文档/)
