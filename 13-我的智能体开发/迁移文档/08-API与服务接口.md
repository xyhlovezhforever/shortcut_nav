# 08-API与服务接口

## 概述

任务编排服务提供两种协议的服务接口：
- **HTTP REST API**: 基于 Axum 框架，提供简单的任务管理接口
- **gRPC 服务**: 基于 Tonic 框架，提供流式监控和完整的任务编排能力

---

## HTTP REST API

### 核心文件

- `src/api/handlers.rs` - API 处理器
- `src/api/models.rs` - 数据模型
- `src/api/health.rs` - 健康检查
- `src/main.rs` - HTTP 服务启动

### 路由配置

```rust
// main.rs

let app = Router::new()
    // 健康检查
    .route("/health", get(health_check))
    .route("/info", get(service_info))
    // 任务管理
    .route("/api/v1/tasks", post(create_task))
    .route("/api/v1/tasks/:task_id", get(get_task_status))
    .route("/api/v1/tasks/:task_id/result", get(get_task_result))
    // 注入依赖
    .with_state(AppState {
        orchestrator: orchestrator.clone(),
    })
    // 添加请求追踪层
    .layer(TraceLayer::new_for_http());
```

### 接口定义

#### 1. 健康检查

**GET** `/health`

```rust
/// 健康检查响应
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HealthResponse {
    /// 服务状态
    pub status: String,
    /// 版本
    pub version: String,
    /// 时间戳
    pub timestamp: String,
}
```

**响应示例**:
```json
{
    "status": "healthy",
    "version": "0.1.0",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

---

#### 2. 创建任务

**POST** `/api/v1/tasks`

**请求模型**:
```rust
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CreateTaskRequest {
    /// 任务描述
    pub task_description: String,
    /// 最大反思轮次（可选，默认使用配置中的值）
    #[serde(default)]
    pub max_rounds: Option<u32>,
    /// 任务元数据（可选，支持任意自定义参数）
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}
```

**请求示例**:
```json
{
    "task_description": "计算1+2+3的结果",
    "max_rounds": 3,
    "metadata": {
        "priority": "high",
        "user_id": "12345"
    }
}
```

**响应模型**:
```rust
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CreateTaskResponse {
    /// 任务 ID
    pub task_id: String,
    /// 任务描述
    pub task_description: String,
    /// 最大轮次
    pub max_rounds: u32,
    /// 创建时间
    pub created_at: String,
    /// 元数据
    pub metadata: HashMap<String, String>,
}
```

**响应示例**:
```json
{
    "task_id": "task_abc123",
    "task_description": "计算1+2+3的结果",
    "max_rounds": 3,
    "created_at": "2024-01-15T10:30:00Z",
    "metadata": {
        "priority": "high",
        "user_id": "12345"
    }
}
```

---

#### 3. 获取任务状态

**GET** `/api/v1/tasks/:task_id`

**响应模型**:
```rust
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TaskStatusResponse {
    /// 任务 ID
    pub task_id: String,
    /// 任务描述
    pub description: String,
    /// 状态: Pending, Planning, Executing, Evaluating, Reflecting, Completed, Failed, Cancelled
    pub status: String,
    /// 当前轮次
    pub current_round: u32,
    /// 最大轮次
    pub max_rounds: u32,
    /// 创建时间
    pub created_at: String,
    /// 更新时间
    pub updated_at: String,
    /// 完成时间
    pub completed_at: Option<String>,
    /// 元数据
    pub metadata: HashMap<String, String>,
}
```

**响应示例**:
```json
{
    "task_id": "task_abc123",
    "description": "计算1+2+3的结果",
    "status": "Executing",
    "current_round": 1,
    "max_rounds": 3,
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:05Z",
    "completed_at": null,
    "metadata": {}
}
```

---

#### 4. 获取任务结果

**GET** `/api/v1/tasks/:task_id/result`

**响应模型**:
```rust
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TaskResultResponse {
    /// 任务 ID
    pub task_id: String,
    /// 任务描述
    pub task_description: String,
    /// 是否成功
    pub is_success: bool,
    /// 总轮次
    pub total_rounds: u32,
    /// 最终评分
    pub final_score: Option<f32>,
    /// 最终结果
    pub final_output: String,
    /// 总耗时（秒）
    pub total_duration_secs: Option<i64>,
    /// 元数据
    pub metadata: HashMap<String, String>,
}
```

**响应示例**:
```json
{
    "task_id": "task_abc123",
    "task_description": "计算1+2+3的结果",
    "is_success": true,
    "total_rounds": 1,
    "final_score": 95.0,
    "final_output": "6",
    "total_duration_secs": 5,
    "metadata": {}
}
```

---

### 处理器实现

```rust
// handlers.rs

/// 应用状态
#[derive(Clone)]
pub struct AppState {
    pub orchestrator: Arc<Orchestrator>,
}

/// 创建任务并执行
pub async fn create_task(
    State(state): State<AppState>,
    Json(req): Json<CreateTaskRequest>,
) -> Result<Json<CreateTaskResponse>, ServiceError> {
    // 验证请求
    if req.task_description.trim().is_empty() {
        return Err(ServiceError::ValidationError(
            "任务描述不能为空".to_string(),
        ));
    }

    // 在后台执行任务（异步）
    let orchestrator = state.orchestrator.clone();
    let task_description = req.task_description.clone();
    let metadata = req.metadata.clone();

    tokio::spawn(async move {
        match orchestrator.orchestrate_with_metadata(task_description, metadata).await {
            Ok(result) => {
                info!(task_id = result.task_id, is_success = result.is_success, "任务执行完成");
            }
            Err(e) => {
                warn!(error = %e, "任务执行失败");
            }
        }
    });

    // 立即返回任务ID
    let tasks = state.orchestrator.list_tasks();
    let task = tasks.last()
        .ok_or_else(|| ServiceError::InternalError("无法获取任务信息".to_string()))?;

    Ok(Json(CreateTaskResponse {
        task_id: task.task_id.clone(),
        task_description: task.description.clone(),
        max_rounds: task.max_rounds,
        created_at: task.created_at.to_rfc3339(),
        metadata: task.metadata.clone(),
    }))
}
```

---

## gRPC 服务

### 核心文件

- `proto/task_orchestrator_service.proto` - 协议定义
- `src/grpc/server.rs` - gRPC 服务实现
- `src/grpc/mod.rs` - 模块导出

### 服务定义

```protobuf
// proto/task_orchestrator_service.proto

service TaskOrchestratorService {
    // 提交任务
    rpc SubmitTask(SubmitTaskRequest) returns (SubmitTaskResponse);

    // 取消任务
    rpc CancelTask(CancelTaskRequest) returns (CancelTaskResponse);

    // 流式监控任务执行
    rpc StreamTaskExecution(StreamTaskExecutionRequest) returns (stream TaskExecutionEvent);
}
```

---

### 消息定义

#### 1. 提交任务

```protobuf
message SubmitTaskRequest {
    // 任务描述
    string task_description = 1;

    // 最大反思轮次（可选）
    optional uint32 max_rounds = 2;

    // 任务元数据（可选，支持任意自定义参数）
    map<string, string> metadata = 3;

    // 任务上下文信息（可选）
    optional TaskContext context = 4;
}

// 任务上下文信息
message TaskContext {
    // 对话历史或背景信息
    repeated string conversation_history = 1;

    // 相关文档/文件内容
    repeated ContextDocument documents = 2;

    // 结构化的附加信息
    map<string, string> additional_info = 3;

    // 用户偏好或要求
    repeated string user_preferences = 4;
}

// 上下文文档
message ContextDocument {
    // 文档名称/标识
    string name = 1;

    // 文档内容
    string content = 2;

    // 文档类型（code, json, yaml, markdown, text等）
    optional string doc_type = 3;

    // 文档描述
    optional string description = 4;
}

message SubmitTaskResponse {
    string task_id = 1;
    string task_description = 2;
    uint32 max_rounds = 3;
    string created_at = 4;
}
```

---

#### 2. 取消任务

```protobuf
message CancelTaskRequest {
    string task_id = 1;
    optional string reason = 2;
}

message CancelTaskResponse {
    string task_id = 1;
    bool success = 2;
    string message = 3;
}
```

---

#### 3. 流式监控事件

```protobuf
message StreamTaskExecutionRequest {
    string task_id = 1;
}

message TaskExecutionEvent {
    string task_id = 1;
    EventType event_type = 2;
    string timestamp = 3;

    oneof event_data {
        TaskStatusUpdate status_update = 4;
        PlanGenerated plan_generated = 5;
        StepStarted step_started = 6;
        StepCompleted step_completed = 7;
        StepFailed step_failed = 8;
        EvaluationCompleted evaluation_completed = 9;
        ReflectionCompleted reflection_completed = 10;
        TaskCompleted task_completed = 11;
        TaskFailed task_failed = 12;
        TaskExperience task_experience = 13;
        LlmStreamChunk llm_stream_chunk = 14;
        TaskResult task_result = 15;
        TaskWait task_wait = 16;
        ContextEngineeringEvent context_engineering = 17;
    }
}

enum EventType {
    EVENT_TYPE_UNSPECIFIED = 0;
    EVENT_TYPE_STATUS_UPDATE = 1;
    EVENT_TYPE_PLAN_GENERATED = 2;
    EVENT_TYPE_STEP_STARTED = 3;
    EVENT_TYPE_STEP_COMPLETED = 4;
    EVENT_TYPE_STEP_FAILED = 5;
    EVENT_TYPE_EVALUATION_COMPLETED = 6;
    EVENT_TYPE_REFLECTION_COMPLETED = 7;
    EVENT_TYPE_TASK_COMPLETED = 8;
    EVENT_TYPE_TASK_FAILED = 9;
    EVENT_TYPE_TASK_EXPERIENCE = 10;
    EVENT_TYPE_LLM_STREAM_CHUNK = 11;
    EVENT_TYPE_TASK_RESULT = 12;
    EVENT_TYPE_TASK_WAIT = 13;
    EVENT_TYPE_CONTEXT_ENGINEERING = 14;
}
```

---

### 事件类型详解

#### StatusUpdate - 状态更新

```protobuf
message TaskStatusUpdate {
    string status = 1;      // Pending, Planning, Executing, Evaluating, Reflecting, Completed, Failed
    string message = 2;     // 状态描述消息
    uint32 current_round = 3;  // 当前轮次
}
```

#### PlanGenerated - 计划生成

```protobuf
message PlanGenerated {
    ExecutionPlan plan = 1;
}

message ExecutionPlan {
    string plan_id = 1;
    string description = 2;
    repeated PlanStep steps = 3;
    float estimated_duration = 4;
}

message PlanStep {
    string step_id = 1;
    string name = 2;
    string tool = 3;
    string parameters = 4;
    repeated string dependencies = 5;
    string description = 6;
    repeated PlanAction actions = 7;  // 步骤内并行操作
}

message PlanAction {
    string action_id = 1;
    string name = 2;
    string tool = 3;
    string parameters = 4;
    repeated string dependencies = 5;
    string description = 6;
}
```

#### StepStarted/Completed/Failed - 步骤执行

```protobuf
message StepStarted {
    string step_id = 1;
    string step_name = 2;
    string tool_id = 3;
    optional string tool_parameters = 4;
}

message StepCompleted {
    StepResult result = 1;
}

message StepResult {
    string step_id = 1;
    string step_name = 2;
    bool success = 3;
    string output = 4;
    optional string error = 5;
    float duration_secs = 6;
    string tool_id = 7;
    optional string tool_parameters = 8;
}

message StepFailed {
    string step_id = 1;
    string step_name = 2;
    string tool_id = 3;
    string error = 4;
    float duration_secs = 5;
}
```

#### EvaluationCompleted - 评估完成

```protobuf
message EvaluationCompleted {
    EvaluationResult result = 1;
}

message EvaluationResult {
    float overall_score = 1;                    // 总分
    map<string, float> dimension_scores = 2;    // 各维度评分
    uint32 success_count = 3;                   // 成功步骤数
    uint32 failure_count = 4;                   // 失败步骤数
    repeated string suggestions = 5;            // 改进建议
    string analysis = 6;                        // 详细分析
}
```

#### ReflectionCompleted - 反思完成

```protobuf
message ReflectionCompleted {
    ReflectionResult result = 1;
}

message ReflectionResult {
    bool should_replan = 1;                                  // 是否需要重新规划
    repeated string root_causes = 2;                         // 根本原因
    repeated OptimizationSuggestion optimization_suggestions = 3;  // 优化建议
    repeated string lessons_learned = 4;                     // 经验教训
    string improvement_direction = 5;                        // 改进方向
}

message OptimizationSuggestion {
    string suggestion_type = 1;
    optional string target_step = 2;
    string description = 3;
    uint32 priority = 4;
}
```

#### LlmStreamChunk - LLM流式响应

```protobuf
message LlmStreamChunk {
    enum ChunkType {
        CHUNK_TYPE_START = 0;   // 开始流式传输
        CHUNK_TYPE_CHUNK = 1;   // 内容数据块
        CHUNK_TYPE_DONE = 2;    // 完成传输
        CHUNK_TYPE_ERROR = 3;   // 发生错误
    }

    ChunkType chunk_type = 1;
    string phase = 2;                  // 阶段标识（规划/评估/反思）
    optional string content = 3;       // 内容片段
    optional string model = 4;         // 使用的模型
    optional string error_message = 5; // 错误信息
}
```

#### ContextEngineeringEvent - 上下文工程事件

```protobuf
message ContextEngineeringEvent {
    string task_description = 1;
    repeated SuccessfulStepInfo successful_steps = 2;
    repeated FailedStepInfo failed_steps = 3;
}

message SuccessfulStepInfo {
    string step_id = 1;
    string step_name = 2;
    string step_description = 3;
    string tool_id = 4;
    string tool_parameters = 5;
    string tool_output = 6;
    repeated string dependencies = 7;
    map<string, string> extracted_fields = 8;
}

message FailedStepInfo {
    string step_id = 1;
    string step_name = 2;
    string step_description = 3;
    string tool_id = 4;
    string tool_parameters = 5;
    string error_message = 6;
    string reflection_and_action = 7;
    repeated string dependencies = 8;
}
```

---

### gRPC 服务实现

```rust
// grpc/server.rs

/// Orchestrator 服务实现
pub struct OrchestratorServiceImpl {
    /// 编排器
    orchestrator: Arc<Orchestrator>,
    /// 任务状态管理器
    state_manager: Arc<TaskStateManager>,
    /// Kafka 审计日志记录器
    audit_logger: Option<Arc<AuditLogger>>,
    /// 消息格式化器
    message_formatter: Arc<MessageFormatter>,
    /// AI 报告生成器
    ai_reporter: Option<Arc<AIReporter>>,
    /// 响应模式
    response_mode: ResponseMode,
    /// 调试配置
    debug_config: DebugConfig,
}

impl OrchestratorServiceImpl {
    pub fn new(
        orchestrator: Arc<Orchestrator>,
        state_manager: Arc<TaskStateManager>,
        response_mode: ResponseMode,
        llm_client: Arc<UnifiedLlmClient>,
        debug_config: DebugConfig,
    ) -> Self {
        let message_formatter = if response_mode == ResponseMode::User {
            Arc::new(MessageFormatter::new_with_llm(response_mode, llm_client))
        } else {
            Arc::new(MessageFormatter::new(response_mode))
        };

        Self {
            orchestrator,
            state_manager,
            audit_logger: None,
            message_formatter,
            ai_reporter: None,
            response_mode,
            debug_config,
        }
    }

    pub fn with_audit_logger(mut self, audit_logger: Arc<AuditLogger>) -> Self {
        self.audit_logger = Some(audit_logger);
        self
    }

    /// 发送状态更新事件
    async fn send_status_update(
        tx: &mpsc::Sender<Result<TaskExecutionEvent, Status>>,
        task_id: String,
        status: String,
        message: String,
        current_round: u32,
    ) -> Result<(), ()> {
        let event = TaskExecutionEvent {
            task_id,
            event_type: EventType::StatusUpdate as i32,
            timestamp: Utc::now().to_rfc3339(),
            event_data: Some(EventData::StatusUpdate(TaskStatusUpdate {
                status,
                message,
                current_round,
            })),
        };

        tx.send(Ok(event)).await.map_err(|_| ())
    }
}

#[tonic::async_trait]
impl TaskOrchestratorService for OrchestratorServiceImpl {
    /// 提交任务
    async fn submit_task(
        &self,
        request: Request<SubmitTaskRequest>,
    ) -> Result<Response<SubmitTaskResponse>, Status> {
        let req = request.into_inner();

        // 创建任务
        let max_rounds = req.max_rounds.unwrap_or(3);
        let task = self.state_manager.create_task_with_metadata(
            req.task_description.clone(),
            max_rounds,
            req.metadata.clone(),
        );

        // 在后台执行任务
        let orchestrator = self.orchestrator.clone();
        let task_id = task.task_id.clone();
        let task_description = task.description.clone();
        let metadata = req.metadata.clone();
        let context = req.context.clone();

        tokio::spawn(async move {
            // 等待流式监控设置好事件发送器
            tokio::time::sleep(Duration::from_millis(100)).await;

            match orchestrator
                .orchestrate_with_id_and_metadata(
                    Some(task_id.clone()),
                    task_description,
                    metadata,
                    context
                )
                .await
            {
                Ok(result) => {
                    info!(task_id = result.task_id, is_success = result.is_success, "任务执行完成");
                }
                Err(e) => {
                    error!(task_id = task_id, error = %e, "任务执行失败");
                }
            }
        });

        Ok(Response::new(SubmitTaskResponse {
            task_id: task.task_id,
            task_description: task.description,
            max_rounds: task.max_rounds,
            created_at: task.created_at.to_rfc3339(),
        }))
    }

    /// 取消任务
    async fn cancel_task(
        &self,
        request: Request<CancelTaskRequest>,
    ) -> Result<Response<CancelTaskResponse>, Status> {
        let req = request.into_inner();

        let mut task = self.state_manager
            .get_task(&req.task_id)
            .ok_or_else(|| Status::not_found("任务不存在"))?;

        task.update_status(TaskStatus::Cancelled);
        if let Some(reason) = &req.reason {
            task.final_result = Some(reason.clone());
        }
        self.state_manager.update_task(task);

        Ok(Response::new(CancelTaskResponse {
            task_id: req.task_id,
            success: true,
            message: "任务已取消".to_string(),
        }))
    }

    type StreamTaskExecutionStream = ReceiverStream<Result<TaskExecutionEvent, Status>>;

    /// 流式监控任务执行
    async fn stream_task_execution(
        &self,
        request: Request<StreamTaskExecutionRequest>,
    ) -> Result<Response<Self::StreamTaskExecutionStream>, Status> {
        let req = request.into_inner();
        let task_id = req.task_id;

        // 创建异步通道
        let (tx, rx) = mpsc::channel(100);

        // 创建事件发送器
        let event_sender = Arc::new(StreamEventSender {
            tx: tx.clone(),
            task_id: task_id.clone(),
            state_manager: self.state_manager.clone(),
            message_formatter: self.message_formatter.clone(),
            ai_reporter: self.ai_reporter.clone(),
            response_mode: self.response_mode,
            debug_config: self.debug_config.clone(),
        });

        // 为特定任务设置事件发送器
        self.orchestrator.set_task_event_sender(task_id.clone(), event_sender);

        // 在后台任务中监控任务执行状态
        let state_manager = self.state_manager.clone();
        let message_formatter = self.message_formatter.clone();

        tokio::spawn(async move {
            let mut last_status = String::new();

            loop {
                if let Some(task) = state_manager.get_task(&task_id) {
                    let status_str = format!("{:?}", task.status);

                    if status_str != last_status {
                        let message = message_formatter.format_task_status(&status_str).await;
                        if Self::send_status_update(&tx, task_id.clone(), status_str.clone(), message, task.current_round)
                            .await.is_err()
                        {
                            break;
                        }
                        last_status = status_str;
                    }

                    // 任务完成或失败时退出
                    if matches!(task.status, TaskStatus::Completed | TaskStatus::Failed | TaskStatus::Cancelled) {
                        break;
                    }
                }

                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}
```

---

## 事件发送器

### EventSender Trait

```rust
// core/orchestrator.rs

/// 事件发送器 trait
#[async_trait]
pub trait EventSender: Send + Sync {
    /// 发送规划开始事件
    async fn send_planning_started(&self) -> Result<()>;

    /// 发送计划生成事件
    async fn send_plan_generated(&self, plan: &ExecutionPlan) -> Result<()>;

    /// 发送步骤开始事件
    async fn send_step_started(&self, step: &PlanStep) -> Result<()>;

    /// 发送步骤完成事件
    async fn send_step_completed(&self, result: &StepResult) -> Result<()>;

    /// 发送步骤失败事件
    async fn send_step_failed(&self, step: &PlanStep, error: &str) -> Result<()>;

    /// 发送评估完成事件
    async fn send_evaluation_completed(&self, result: &EvaluationResult) -> Result<()>;

    /// 发送反思完成事件
    async fn send_reflection_completed(&self, result: &ReflectionResult) -> Result<()>;

    /// 发送任务完成事件
    async fn send_task_completed(&self, rounds: u32) -> Result<()>;

    /// 发送任务失败事件
    async fn send_task_failed(&self, error: &str, rounds: u32) -> Result<()>;

    /// 发送LLM流式chunk
    async fn send_llm_stream_chunk(&self, chunk: LlmStreamChunk) -> Result<()>;

    /// 发送等待事件
    async fn send_task_wait(&self, phase: &str) -> Result<()>;

    /// 发送上下文工程事件
    async fn send_context_engineering_event(&self, event: ContextEngineeringEvent) -> Result<()>;
}
```

### StreamEventSender 实现

```rust
// grpc/server.rs

pub struct StreamEventSender {
    tx: mpsc::Sender<Result<TaskExecutionEvent, Status>>,
    task_id: String,
    state_manager: Arc<TaskStateManager>,
    message_formatter: Arc<MessageFormatter>,
    ai_reporter: Option<Arc<AIReporter>>,
    response_mode: ResponseMode,
    debug_config: DebugConfig,
}

#[async_trait]
impl EventSender for StreamEventSender {
    async fn send_plan_generated(&self, plan: &ExecutionPlan) -> Result<()> {
        let event = TaskExecutionEvent {
            task_id: self.task_id.clone(),
            event_type: EventType::PlanGenerated as i32,
            timestamp: Utc::now().to_rfc3339(),
            event_data: Some(EventData::PlanGenerated(PlanGenerated {
                plan: Some(plan.clone()),
            })),
        };

        self.tx.send(Ok(event)).await.map_err(|e| anyhow::anyhow!("{}", e))
    }

    async fn send_step_completed(&self, result: &StepResult) -> Result<()> {
        let event = TaskExecutionEvent {
            task_id: self.task_id.clone(),
            event_type: EventType::StepCompleted as i32,
            timestamp: Utc::now().to_rfc3339(),
            event_data: Some(EventData::StepCompleted(StepCompleted {
                result: Some(result.clone()),
            })),
        };

        self.tx.send(Ok(event)).await.map_err(|e| anyhow::anyhow!("{}", e))
    }

    // ... 其他方法实现
}
```

---

## 错误处理

### ServiceError 定义

```rust
// utils/error.rs

#[derive(Debug)]
pub enum ServiceError {
    /// 验证错误
    ValidationError(String),
    /// 任务未找到
    TaskNotFound(String),
    /// 内部错误
    InternalError(String),
    /// 执行错误
    ExecutionError(String),
    /// 超时错误
    TimeoutError(String),
}

impl IntoResponse for ServiceError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            ServiceError::ValidationError(msg) => (StatusCode::BAD_REQUEST, msg),
            ServiceError::TaskNotFound(id) => (StatusCode::NOT_FOUND, format!("任务 {} 不存在", id)),
            ServiceError::InternalError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            ServiceError::ExecutionError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            ServiceError::TimeoutError(msg) => (StatusCode::GATEWAY_TIMEOUT, msg),
        };

        let body = Json(json!({
            "error": error_message
        }));

        (status, body).into_response()
    }
}
```

---

## 服务启动流程

```rust
// main.rs

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 加载配置
    let config = AppConfig::load()?;

    // 2. 初始化日志系统
    logging::init_logging(config.logging.clone())?;

    // 3. 初始化组件
    let state_manager = Arc::new(TaskStateManager::new());
    let llm_client = Arc::new(UnifiedLlmClient::new(config.llm.clone()).await?);
    let tool_client = Arc::new(UnifiedToolServiceClient::new(config.tool_service.clone()).await?);
    let kafka_logger = Arc::new(KafkaLogger::new(&config).await?);

    // 4. 创建核心组件
    let planner = Arc::new(Planner::new(/* ... */));
    let executor = Arc::new(Executor::new(/* ... */));
    let evaluator = Arc::new(Evaluator::new(/* ... */));
    let reflector = Arc::new(Reflector::new(/* ... */));

    // 5. 创建编排器
    let orchestrator = Arc::new(Orchestrator::new(
        planner, executor, evaluator, reflector,
        state_manager.clone(),
        orchestrator_config,
        kafka_logger,
        Arc::new(config.clone()),
    ));

    // 6. 创建HTTP路由
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/api/v1/tasks", post(create_task))
        .route("/api/v1/tasks/:task_id", get(get_task_status))
        .route("/api/v1/tasks/:task_id/result", get(get_task_result))
        .with_state(AppState { orchestrator: orchestrator.clone() })
        .layer(TraceLayer::new_for_http());

    // 7. 启动HTTP服务器
    let listener = TcpListener::bind(config.server_addr()).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}
```

---

## 客户端使用示例

### HTTP 客户端

```rust
use reqwest::Client;

async fn create_and_monitor_task() -> Result<()> {
    let client = Client::new();

    // 创建任务
    let response = client
        .post("http://localhost:8084/api/v1/tasks")
        .json(&json!({
            "task_description": "计算 5 的平方",
            "max_rounds": 3
        }))
        .send()
        .await?;

    let task: CreateTaskResponse = response.json().await?;
    let task_id = task.task_id;

    // 轮询状态
    loop {
        let status_response = client
            .get(&format!("http://localhost:8084/api/v1/tasks/{}", task_id))
            .send()
            .await?;

        let status: TaskStatusResponse = status_response.json().await?;

        if status.status == "Completed" || status.status == "Failed" {
            break;
        }

        tokio::time::sleep(Duration::from_secs(1)).await;
    }

    // 获取结果
    let result_response = client
        .get(&format!("http://localhost:8084/api/v1/tasks/{}/result", task_id))
        .send()
        .await?;

    let result: TaskResultResponse = result_response.json().await?;
    println!("任务结果: {}", result.final_output);

    Ok(())
}
```

### gRPC 客户端

```rust
use tonic::transport::Channel;

async fn stream_task_execution() -> Result<()> {
    let channel = Channel::from_static("http://localhost:50051")
        .connect()
        .await?;

    let mut client = TaskOrchestratorServiceClient::new(channel);

    // 提交任务
    let submit_response = client
        .submit_task(SubmitTaskRequest {
            task_description: "计算 5 的平方".to_string(),
            max_rounds: Some(3),
            metadata: HashMap::new(),
            context: None,
        })
        .await?;

    let task_id = submit_response.into_inner().task_id;

    // 流式监控
    let mut stream = client
        .stream_task_execution(StreamTaskExecutionRequest {
            task_id: task_id.clone(),
        })
        .await?
        .into_inner();

    while let Some(event) = stream.message().await? {
        match event.event_data {
            Some(EventData::StatusUpdate(update)) => {
                println!("[状态] {} - {}", update.status, update.message);
            }
            Some(EventData::PlanGenerated(plan)) => {
                println!("[计划] 生成了 {} 个步骤", plan.plan.unwrap().steps.len());
            }
            Some(EventData::StepCompleted(completed)) => {
                let result = completed.result.unwrap();
                println!("[步骤完成] {} - {}", result.step_id, result.output);
            }
            Some(EventData::TaskCompleted(completed)) => {
                println!("[任务完成] 共 {} 轮", completed.rounds);
                break;
            }
            Some(EventData::TaskFailed(failed)) => {
                println!("[任务失败] {}", failed.error);
                break;
            }
            _ => {}
        }
    }

    Ok(())
}
```

---

## 关键代码索引

| 功能 | 文件位置 | 说明 |
|------|---------|------|
| HTTP路由配置 | `src/main.rs` | Axum路由定义 |
| REST API处理器 | `src/api/handlers.rs` | 任务CRUD操作 |
| API数据模型 | `src/api/models.rs` | 请求/响应结构 |
| gRPC协议定义 | `proto/task_orchestrator_service.proto` | 服务和消息定义 |
| gRPC服务实现 | `src/grpc/server.rs` | Tonic服务实现 |
| 事件发送器 | `src/core/orchestrator.rs` | EventSender trait |
| 错误处理 | `src/utils/error.rs` | ServiceError定义 |
