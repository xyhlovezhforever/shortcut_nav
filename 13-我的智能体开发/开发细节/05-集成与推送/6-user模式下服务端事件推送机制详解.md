# User模式下服务端事件推送机制详解

> 本文档详细说明在 `RESPONSE_MODE=user` 模式下，服务端在什么时机推送哪些事件，事件信息的内容格式，以及对应的代码实现位置。

## 目录

- [事件概述](#事件概述)
- [事件类型详解](#事件类型详解)
  - [1. TaskWait 事件](#1-taskwait-事件)
  - [2. StatusUpdate 事件](#2-statusupdate-事件)
  - [3. TaskResult 事件](#3-taskresult-事件)
  - [4. TaskCompleted 事件](#4-taskcompleted-事件)
  - [5. TaskFailed 事件](#5-taskfailed-事件)
- [事件发送时序图](#事件发送时序图)
- [代码实现位置索引](#代码实现位置索引)

---

## 事件概述

在User模式下，服务端只推送用户关心的核心事件，隐藏技术细节。所有事件通过gRPC流式响应发送给客户端。

**核心设计原则：**
- ✅ **TaskWait事件**：在等待操作完成的间隙发送，让用户知道系统正在工作
- ✅ **StatusUpdate事件**：操作完成后发送，告知用户进展
- ✅ **列表化展示**：步骤完成消息以列表形式详细说明执行结果

---

## 事件类型详解

### 1. TaskWait 事件

**作用：** 在操作间隙发送，告知用户系统正在处理中，避免用户感觉系统"卡住"。

#### 1.1 工具筛选阶段

**发送时机：** AI Reporter 生成工具筛选消息**之后**

**事件内容：**
```rust
TaskWait {
    phase: "工具筛选",
    message: "收到您的「{任务描述}」任务！正在为您筛选最佳工具..." // AI生成
}
```

**代码位置：**
- 发送位置：[src/grpc/server.rs:597-617](../src/grpc/server.rs#L597-L617)
- AI生成：[src/ai_reporter.rs:151-195](../src/ai_reporter.rs#L151-L195) - `generate_llm_call_started_message()`

**流程说明：**
1. 调用 AI Reporter 生成友好的任务确认消息
2. 发送 TASK_WAIT 事件（包含AI生成的消息）
3. 发送 StatusUpdate 事件（包含相同消息）

---

#### 1.2 任务规划阶段

**发送时机：** 调用 LLM 生成规划方案时（发送请求后，等待响应期间）

**事件内容：**
```rust
TaskWait {
    phase: "任务规划",
    message: "" // 空字符串，由StatusUpdate补充信息
}
```

**代码位置：**
- Planner调用：[src/core/planner.rs:137](../src/core/planner.rs#L137)（非流式）
- Planner调用：[src/core/planner.rs:175](../src/core/planner.rs#L175)（流式）

---

#### 1.3 工具执行阶段

**发送时机：** 在执行工具**之前**

**事件内容：**
```rust
TaskWait {
    phase: "工具执行",
    message: "正在执行工具: {步骤名称}"
}
```

**代码位置：**
- 发送位置：[src/core/executor.rs:967](../src/core/executor.rs#L967)

**示例输出：**
```
正在执行工具: 获取连接信息 (工具执行)
```

---

#### 1.4 进度分析阶段 ✨ 新增

**发送时机：** AI Reporter 生成步骤完成进度报告**之前**

**事件内容：**
```rust
TaskWait {
    phase: "进度分析",
    message: "正在分析「{步骤名称}」的执行结果..."
}
```

**代码位置：**
- 发送位置：[src/core/orchestrator.rs:2304-2307](../src/core/orchestrator.rs#L2304-L2307)
- AI生成报告：[src/ai_reporter.rs:77-95](../src/ai_reporter.rs#L77-L95) - `generate_step_completion_message()`

**流程说明：**
1. 步骤执行完成，获取执行结果
2. 发送 TASK_WAIT 事件（告知用户正在分析结果）
3. 调用 AI Reporter 分析执行结果并生成进度报告（耗时操作）
4. 发送 StatusUpdate 事件（包含生成的进度报告）

---

#### 1.5 评估阶段

**发送时机：** 调用 LLM 评估执行结果时

**事件内容：**
```rust
TaskWait {
    phase: "评估",
    message: ""
}
```

**代码位置：**
- Evaluator调用：[src/core/evaluator.rs:163](../src/core/evaluator.rs#L163)（非流式）
- Evaluator调用：[src/core/evaluator.rs:190](../src/core/evaluator.rs#L190)（流式）

---

#### 1.6 反思阶段

**发送时机：** 调用 LLM 进行反思分析时

**事件内容：**
```rust
TaskWait {
    phase: "反思",
    message: ""
}
```

**代码位置：**
- Reflector调用：[src/core/reflector.rs:242](../src/core/reflector.rs#L242)（非流式）
- Reflector调用：[src/core/reflector.rs:275](../src/core/reflector.rs#L275)（流式）

---

#### 1.7 建议行动阶段

**发送时机：** 根据反思结果执行建议行动时

**事件内容：**
```rust
TaskWait {
    phase: "建议行动",
    message: "{根据行动类型生成的消息}"
}
```

**消息内容根据建议行动类型：**
- `RetryWithAdjustedParams`: "正在根据建议重试该步骤..."
- `RetryWithAlternativeTool`: "正在使用备选工具重试该步骤..."
- `ReplanEntireTask`: "正在重新规划整个任务..."
- `StopExecution`: "任务无法继续执行，准备终止..."

**代码位置：**
- 发送位置：[src/grpc/server.rs:1193-1208](../src/grpc/server.rs#L1193-L1208)

---

### 2. StatusUpdate 事件

**作用：** 操作完成后发送，告知用户具体进展和结果。

#### 2.1 工具筛选完成

**发送时机：** AI Reporter 生成工具筛选消息后

**事件内容：**
```rust
StatusUpdate {
    status: "llm_call_started",
    message: "收到您的「{任务描述}」任务！正在为您筛选最佳工具...",
    current_round: 0
}
```

**代码位置：**
- 发送位置：[src/grpc/server.rs:637-648](../src/grpc/server.rs#L637-L648)

---

#### 2.2 步骤完成进度报告 ✨ 核心事件

**发送时机：** 步骤执行完成后，AI Reporter 生成进度报告

**事件内容：**
```rust
StatusUpdate {
    status: "",
    message: "{AI生成的进度报告，列表格式}",
    current_round: 0
}
```

**消息格式示例：**
```
✅ 搞定！「获取连接信息」已完成（2/4）
  • 成功获取了15个设备的连接配置
  • 包含5台冷水机组和10台水泵的通信参数
  • 所有设备连接状态正常
接下来将处理拓扑结构。
```

**代码位置：**
- 发送位置：[src/grpc/server.rs:1247-1279](../src/grpc/server.rs#L1247-L1279) - `send_progress_update()`
- 调用入口：[src/core/orchestrator.rs:2341](../src/core/orchestrator.rs#L2341)
- AI生成逻辑：[src/ai_reporter.rs:77-95](../src/ai_reporter.rs#L77-L95)
- 提示词模板：[src/prompt_templates.rs:95-171](../src/prompt_templates.rs#L95-L171)

**关键特性：**
- ✅ 分析步骤输出的JSON结果（`success`、`msg`、`result`、`data`等字段）
- ✅ 以列表形式（用 `•` 符号）详细展示执行结果
- ✅ 用自然语言描述完成了什么
- ✅ 字数限制：120字

---

#### 2.3 任务完成消息

**发送时机：** 任务成功完成所有步骤

**事件内容：**
```rust
StatusUpdate {
    status: "task_completed",
    message: "{任务完成的友好消息}",
    current_round: 0
}
```

**代码位置：**
- 消息格式化：[src/message_formatter.rs](../src/message_formatter.rs) - `format_task_completed()`

---

### 3. TaskResult 事件

**发送时机：** 任务成功完成后，返回最终结果数据

**事件内容：**
```rust
TaskResult {
    result_json: "{包含任务执行结果的JSON字符串}"
}
```

**用户模式显示：**
直接显示结果JSON，带换行

**代码位置：**
- 客户端处理：[examples/plc_test.rs:544-556](../examples/plc_test.rs#L544-L556)

---

### 4. TaskCompleted 事件

**发送时机：** 任务成功完成所有步骤

**事件内容：**
```rust
TaskCompleted {
    rounds: {执行轮次}
}
```

**用户模式显示：**
不显示（避免重复，StatusUpdate 已包含完成信息）

**代码位置：**
- 客户端处理：[examples/plc_test.rs:558-568](../examples/plc_test.rs#L558-L568)

---

### 5. TaskFailed 事件

**发送时机：** 任务执行失败，达到最大重试次数或遇到不可恢复错误

**事件内容：**
```rust
TaskFailed {
    error: "{友好的错误描述}"
}
```

**用户模式显示：**
显示错误信息，提供易懂的失败原因

**示例：**
```
任务执行失败: 连接超时，请检查网络设置
```

**代码位置：**
- 客户端处理：[examples/plc_test.rs:532-548](../examples/plc_test.rs#L532-L548)

---

## 事件发送时序图

### 典型成功流程

```
用户提交任务
    ↓
【工具筛选阶段】
    ├─ TaskWait: "收到您的任务！正在为您筛选最佳工具..."
    ├─ StatusUpdate: "收到您的任务！正在为您筛选最佳工具..."
    ↓
【任务规划阶段】
    ├─ TaskWait: "" (phase: "任务规划")
    ├─ StatusUpdate: "✓ 计划制定完成，共3个步骤"
    ↓
【步骤1执行】
    ├─ StatusUpdate: "正在执行步骤1: 连接PLC控制器"
    ├─ TaskWait: "正在执行工具: 连接PLC控制器"
    ├─ (工具执行中...)
    ├─ TaskWait: "正在分析「连接PLC控制器」的执行结果..."
    ├─ (AI Reporter生成进度报告...)
    ├─ StatusUpdate: "✅ 完成了！「连接PLC控制器」已完成（1/3）
    │                  • 成功连接到设备
    │                  • IP地址: 192.168.1.100
    │                接下来将获取实时数据。"
    ↓
【步骤2执行】
    ├─ StatusUpdate: "正在执行步骤2: 获取实时数据"
    ├─ TaskWait: "正在执行工具: 获取实时数据"
    ├─ (工具执行中...)
    ├─ TaskWait: "正在分析「获取实时数据」的执行结果..."
    ├─ StatusUpdate: "✅ 搞定！「获取实时数据」已完成（2/3）
    │                  • 成功获取15个数据点
    │                  • 数据时间戳: 2025-12-12 10:30:00
    │                接下来将处理数据。"
    ↓
【步骤3执行】
    ├─ StatusUpdate: "正在执行步骤3: 处理数据"
    ├─ TaskWait: "正在执行工具: 处理数据"
    ├─ (工具执行中...)
    ├─ TaskWait: "正在分析「处理数据」的执行结果..."
    ├─ StatusUpdate: "✅ 完成了！「处理数据」已完成（3/3）
    │                  • 数据处理完成
    │                  • 生成报表3份"
    ↓
【评估阶段】
    ├─ TaskWait: "" (phase: "评估")
    ├─ StatusUpdate: "✓ 任务执行成功"
    ↓
【任务完成】
    ├─ TaskResult: {"success": true, "data": {...}}
    └─ TaskCompleted: (不显示)
```

---

## 代码实现位置索引

### 事件发送核心文件

| 文件 | 主要功能 | 关键函数 |
|------|---------|---------|
| [src/grpc/server.rs](../src/grpc/server.rs) | gRPC事件发送实现 | `send_task_wait()`, `send_progress_update()`, `send_llm_call_started()` |
| [src/core/orchestrator.rs](../src/core/orchestrator.rs) | 任务编排逻辑 | `generate_and_send_progress_report()`, `execute_with_step_level_reflection()` |
| [src/core/executor.rs](../src/core/executor.rs) | 工具执行 | `execute_single_step_with_overrides()` |
| [src/core/planner.rs](../src/core/planner.rs) | 任务规划 | `call_llm_with_events()` |
| [src/core/evaluator.rs](../src/core/evaluator.rs) | 结果评估 | `call_llm_with_events()` |
| [src/core/reflector.rs](../src/core/reflector.rs) | 反思分析 | `call_llm_with_events()` |
| [src/ai_reporter.rs](../src/ai_reporter.rs) | AI消息生成 | `generate_step_completion_message()`, `generate_llm_call_started_message()` |
| [src/prompt_templates.rs](../src/prompt_templates.rs) | 提示词模板 | `execution::STEP_COMPLETED` |
| [src/message_formatter.rs](../src/message_formatter.rs) | 消息格式化 | `format_llm_call_started()`, `format_task_completed()` |

### EventSender Trait实现

**Trait定义：** [src/core/orchestrator.rs:58-175](../src/core/orchestrator.rs#L58-L175)

**实现类：**
- `GrpcEventSender` - [src/grpc/server.rs:519-1380](../src/grpc/server.rs#L519-L1380)
- `NoOpEventSender` - [src/core/orchestrator.rs:144-175](../src/core/orchestrator.rs#L144-L175)

---

## 配置说明

### 启用User模式

**方式1：环境变量**
```bash
export RESPONSE_MODE=user
```

**方式2：配置文件**
```toml
# config.dev.toml
[response]
mode = "user"
```

### 客户端检测

```rust
fn is_user_mode() -> bool {
    std::env::var("RESPONSE_MODE")
        .unwrap_or_else(|_| String::from("developer"))
        .to_lowercase() == "user"
}
```

---

## 最佳实践

### 1. 事件设计原则

- ✅ **及时性**：TaskWait在操作开始前发送，让用户知道系统正在工作
- ✅ **完整性**：StatusUpdate在操作完成后发送，提供详细结果
- ✅ **友好性**：所有消息都经过AI润色，用自然语言表达
- ✅ **列表化**：步骤结果以列表形式展示，信息清晰

### 2. 添加新的等待事件

如果需要在新的耗时操作前发送TASK_WAIT，参考以下模式：

```rust
// 1. 在操作前发送TASK_WAIT
event_sender.send_task_wait(
    "操作阶段名称",
    "正在进行某项操作..."
);

// 2. 执行耗时操作
let result = perform_long_running_operation().await;

// 3. 操作完成后发送StatusUpdate
event_sender.send_event(TaskExecutionEvent {
    event_type: EventType::StatusUpdate,
    event_data: Some(EventData::StatusUpdate(
        TaskStatusUpdate {
            status: "operation_completed",
            message: "操作完成！",
            current_round: 0,
        }
    )),
    ...
}).await;
```

### 3. AI Reporter使用建议

**传递完整的步骤输出：**
```rust
ai_reporter.generate_step_completion_message(
    &plan.description,
    &step_name,
    &step_result.output,  // ✅ 传递完整输出，让AI分析
    step_index,
    total_steps,
    next_step_name,
).await
```

**提示词要求列表化输出：**
- 使用 `•` 符号
- 每条信息独立一行
- 提取关键数据
- 避免技术术语

---

## 常见问题

### Q1: 为什么有些TASK_WAIT消息是空字符串？

A: 在规划、评估、反思等阶段，TASK_WAIT主要用于告知客户端"正在思考中"的状态，具体的进度信息会通过StatusUpdate事件补充。空字符串可以减少冗余信息。

### Q2: TASK_WAIT和StatusUpdate有什么区别？

A:
- **TASK_WAIT**：操作**开始前**发送，表示"正在进行中"
- **StatusUpdate**：操作**完成后**发送，表示"已完成"

### Q3: 为什么步骤完成消息要用列表格式？

A: 列表格式可以清晰地展示步骤执行的具体结果，比如：
- 创建了哪些对象
- 获取了哪些数据
- 处理了多少内容

这比笼统的"任务完成"更有信息量。

### Q4: 如何调试事件发送？

A:
1. 查看服务端日志：`tracing::debug!("发送事件: {:?}", event);`
2. 查看客户端输出：在`is_user_mode()`分支添加调试输出
3. 使用`test_log.txt`记录完整的事件流

---

## 更新日志

### 2025-12-12
- ✅ 新增：进度分析阶段的TASK_WAIT事件
- ✅ 优化：步骤完成消息改为列表格式展示
- ✅ 增强：传递完整的步骤输出给AI Reporter分析
- ✅ 文档：创建本详细文档

---

## 相关文档

- [用户(User)端实际推送的事件说明](../Core_document/用户向页面上推送消息/用户(user)端实际推送的事件.md)
- [客户端测试时可设置不同的模式打印输出](../examples/客户端测试时可设置不同的模式打印输出.txt)
- [Message Formatter源码](../src/message_formatter.rs)
- [AI Reporter源码](../src/ai_reporter.rs)

---

**文档维护：** 请在修改事件推送逻辑时同步更新本文档。
