# 17 - Plan-with-Files 模式：执行历史作为单一真理来源

> **文档版本**: v1.0
> **创建日期**: 2026-01-14
> **迁移优先级**: ⭐⭐⭐⭐ (高)

---

## 1. 概述

### 1.1 什么是 Plan-with-Files 模式

Plan-with-Files 模式是受 [Claude Code (Manus)](https://github.com/anthropics/claude-code) 的**上下文工程原则**启发而设计的任务编排范式。

**核心思想**: 使用结构化文件作为**单一真理来源（Single Source of Truth）**，而非依赖对话历史。

### 1.2 为什么需要这个模式

在传统对话式任务编排中，大模型需要在长对话历史中反复查找已执行步骤的信息。这存在以下问题：

| 问题 | 影响 |
|------|------|
| ❌ **信息分散** | 成功和失败的步骤散落在多轮对话中 |
| ❌ **难以定位** | 大模型需要在长对话中反复搜索 |
| ❌ **上下文丢失** | 对话轮次过多时，早期信息可能被遗忘 |
| ❌ **缺乏结构** | 非结构化对话不利于系统化分析 |

Plan-with-Files 模式通过将执行历史格式化为结构化文件解决了这些问题：

| 优势 | 说明 |
|------|------|
| ✅ **信息集中** | 所有执行历史在一个文件中清晰呈现 |
| ✅ **快速定位** | 结构化格式便于大模型快速理解 |
| ✅ **统计可见** | 执行统计摘要帮助大模型评估问题严重程度 |
| ✅ **高度结构化** | 便于系统化分析和决策 |

---

## 2. 核心实现

### 2.1 ContextEngineeringEventBuilder 结构

**文件位置**: `src/core/orchestrator.rs`

```rust
pub struct ContextEngineeringEventBuilder {
    task_description: String,
    pub successful_steps: Vec<SuccessfulStepData>,
    pub failed_steps: Vec<FailedStepData>,

    // ⭐ 统计字段（v1.1 新增）
    current_round: u32,              // 当前反思轮次
    total_step_retries: u32,         // 累计步骤重试次数
    total_task_replans: u32,         // 累计任务重规划次数
}
```

### 2.2 成功步骤数据结构

```rust
pub struct SuccessfulStepData {
    pub step_id: String,
    pub step_name: String,
    pub description: String,
    pub tool_id: String,
    pub parameters: String,          // JSON格式参数
    pub output: String,              // JSON格式输出
    pub dependencies: Vec<String>,   // 依赖的步骤ID列表
    pub extracted_fields: Vec<ExtractedField>,
}

pub struct ExtractedField {
    pub field_name: String,
    pub field_value: String,
}
```

### 2.3 失败步骤数据结构

```rust
pub struct FailedStepData {
    pub step_id: String,
    pub step_name: String,
    pub description: String,
    pub tool_id: String,
    pub parameters: String,
    pub error: String,
    pub reflection_action: String,  // 反思建议的行动
}
```

---

## 3. 执行历史格式

### 3.1 完整示例

```markdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 执行统计摘要
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • 当前反思轮次: 第 2 轮
  • 累计步骤重试次数: 3 次
  • 累计任务重规划次数: 1 次
  • 成功步骤数: 4
  • 失败步骤数: 2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ 成功执行的步骤 (4个)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Step 1] 获取历史负荷数据
  📝 描述: 从数据库获取过去30天的负荷数据
  🔧 工具: get_load_data_tool (ID: tool_001)
  📥 输入参数:
    {
      "region": "华东区域",
      "time_range": "30天"
    }
  📤 输出结果:
    {
      "data_file": "load_data_20260101_20260131.csv",
      "records": 720
    }
  🔗 依赖步骤: 无
  📦 提取字段:
    - data_file: load_data_20260101_20260131.csv

───────────────────────────────────

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ 失败的步骤 (2个)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Step 3] 负荷预测 ❌
  📝 描述: 使用LSTM模型进行负荷预测
  🔧 工具: lstm_forecast_tool (ID: tool_003)
  📥 输入参数:
    {
      "data_file": "{{step_2.cleaned_file}}",
      "forecast_horizon": "7天"
    }
  ❌ 错误信息:
    模型文件加载失败: model_lstm.pkl not found

  💡 反思建议:
    建议行动: RetryWithAdjustedParams
    失败原因分类: 参数错误
    具体建议:
    1. 检查模型文件路径是否正确
    2. 确认模型文件是否存在
    3. 可能需要先训练模型再进行预测
```

### 3.2 格式说明

#### 执行统计摘要

包含以下关键统计信息：

- **当前反思轮次**: 表示当前是第几轮反思/重规划
- **累计步骤重试次数**: 所有步骤的累计重试次数
- **累计任务重规划次数**: 任务级别的重规划次数
- **成功/失败步骤数**: 当前执行历史中的步骤统计

#### 成功步骤格式

- **步骤标识**: `[Step N] 步骤名称`
- **描述**: 步骤的详细说明
- **工具**: 使用的工具ID和名称
- **输入参数**: JSON格式，包含参数引用（如 `{{step_1.data_file}}`）
- **输出结果**: JSON格式的执行结果
- **依赖步骤**: 列出所有依赖的步骤ID
- **提取字段**: 可被后续步骤引用的输出字段

#### 失败步骤格式

- **步骤标识**: `[Step N] 步骤名称 ❌`
- **基本信息**: 同成功步骤
- **错误信息**: 详细的错误描述
- **反思建议**: 包含建议行动、失败原因分类、具体建议

---

## 4. 核心方法

### 4.1 format_to_file()

```rust
impl ContextEngineeringEventBuilder {
    pub fn format_to_file(&self) -> String {
        let mut output = String::new();

        // 1. 执行统计摘要
        output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        output.push_str("📊 执行统计摘要\n");
        output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        output.push_str(&format!("  • 当前反思轮次: 第 {} 轮\n", self.current_round));
        output.push_str(&format!("  • 累计步骤重试次数: {} 次\n", self.total_step_retries));
        output.push_str(&format!("  • 累计任务重规划次数: {} 次\n", self.total_task_replans));
        output.push_str(&format!("  • 成功步骤数: {}\n", self.successful_steps.len()));
        output.push_str(&format!("  • 失败步骤数: {}\n\n", self.failed_steps.len()));

        // 2. 成功步骤格式化
        if !self.successful_steps.is_empty() {
            output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
            output.push_str(&format!("✅ 成功执行的步骤 ({}个)\n", self.successful_steps.len()));
            output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

            for step in &self.successful_steps {
                output.push_str(&format!("[{}] {}\n", step.step_id, step.step_name));
                output.push_str(&format!("  📝 描述: {}\n", step.description));
                output.push_str(&format!("  🔧 工具: {} (ID: {})\n", step.step_name, step.tool_id));
                output.push_str(&format!("  📥 输入参数:\n    {}\n", step.parameters));
                output.push_str(&format!("  📤 输出结果:\n    {}\n", step.output));

                if !step.dependencies.is_empty() {
                    output.push_str(&format!("  🔗 依赖步骤: {}\n", step.dependencies.join(", ")));
                } else {
                    output.push_str("  🔗 依赖步骤: 无\n");
                }

                if !step.extracted_fields.is_empty() {
                    output.push_str("  📦 提取字段:\n");
                    for field in &step.extracted_fields {
                        output.push_str(&format!("    - {}: {}\n", field.field_name, field.field_value));
                    }
                }

                output.push_str("\n───────────────────────────────────\n\n");
            }
        }

        // 3. 失败步骤格式化
        if !self.failed_steps.is_empty() {
            output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
            output.push_str(&format!("❌ 失败的步骤 ({}个)\n", self.failed_steps.len()));
            output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

            for step in &self.failed_steps {
                output.push_str(&format!("[{}] {} ❌\n", step.step_id, step.step_name));
                output.push_str(&format!("  📝 描述: {}\n", step.description));
                output.push_str(&format!("  🔧 工具: (ID: {})\n", step.tool_id));
                output.push_str(&format!("  📥 输入参数:\n    {}\n", step.parameters));
                output.push_str(&format!("  ❌ 错误信息:\n    {}\n", step.error));
                output.push_str(&format!("\n  💡 反思建议:\n    {}\n", step.reflection_action));
                output.push_str("\n───────────────────────────────────\n\n");
            }
        }

        output
    }
}
```

### 4.2 统计信息更新方法

```rust
impl ContextEngineeringEventBuilder {
    /// 整体更新统计信息
    pub fn update_statistics(&mut self,
        current_round: u32,
        total_step_retries: u32,
        total_task_replans: u32
    ) {
        self.current_round = current_round;
        self.total_step_retries = total_step_retries;
        self.total_task_replans = total_task_replans;
    }

    /// 增加步骤重试次数
    pub fn increment_step_retries(&mut self) {
        self.total_step_retries += 1;
    }

    /// 增加任务重规划次数
    pub fn increment_task_replans(&mut self) {
        self.total_task_replans += 1;
    }

    /// 设置当前轮次
    pub fn set_current_round(&mut self, round: u32) {
        self.current_round = round;
    }
}
```

---

## 5. 使用场景

### 5.1 反思阶段传递执行历史

```rust
// src/core/orchestrator.rs
// 在反思阶段使用执行历史

// 1. 生成执行历史文件内容
let execution_history = task.context_event_builder.format_to_file();

// 2. 构建反思上下文
let reflection_context = ReflectionContext {
    task_type: Some(task_type.clone()),
    user_context: Some(execution_history),  // ⭐ 作为用户上下文传递
    execution_history: vec![],  // 可以为空，信息都在 user_context 中
};

// 3. 调用反思器
let reflection_result = reflector.reflect(&reflection_context, &task).await?;
```

### 5.2 何时启用 Plan-with-Files

```rust
/// 判断是否应该使用 plan-with-files 模式
fn should_use_plan_with_files(task: &Task) -> bool {
    // 1. 已经有执行历史（非首次规划）
    let has_execution_history = !task.context_event_builder.successful_steps.is_empty()
                             || !task.context_event_builder.failed_steps.is_empty();

    // 2. 已经进行过至少一轮反思
    let has_reflection_rounds = task.current_round > 1;

    // 3. 步骤重试次数 >= 2 或任务重规划次数 >= 1
    let has_retries = task.context_event_builder.total_step_retries >= 2
                   || task.context_event_builder.total_task_replans >= 1;

    has_execution_history && (has_reflection_rounds || has_retries)
}
```

### 5.3 统计信息的更新时机

```rust
// 在步骤重试时更新
task.context_event_builder.increment_step_retries();

// 在任务重规划时更新
task.context_event_builder.increment_task_replans();

// 在每轮开始时更新轮次
task.context_event_builder.set_current_round(task.current_round);

// 或者整体更新
task.context_event_builder.update_statistics(
    task.current_round,
    task.total_step_retries,
    task.total_task_replans
);
```

---

## 6. 与传统模式对比

| 对比维度 | 传统模式（对话驱动） | Plan-with-Files 模式（文件驱动） |
|---------|-------------------|-------------------------------|
| **信息组织** | 分散在多轮对话中 | 集中在单一执行历史文件 |
| **信息查找** | 需要反复搜索对话历史 | 一次性呈现全部关键信息 |
| **统计可见性** | 无统计摘要 | 提供执行统计摘要 |
| **决策依据** | 依赖对话上下文 | 依赖结构化执行历史 + 统计信息 |
| **可维护性** | 对话历史难以复盘 | 执行历史清晰可追溯 |
| **大模型理解难度** | 较高（需要综合多轮对话） | 较低（结构化文件一目了然） |
| **Token消耗** | 包含所有对话内容 | 仅包含关键执行信息 |
| **适用场景** | 简单任务、短流程 | 复杂任务、多轮反思 |

---

## 7. 迁移检查项

### 7.1 核心结构
- [ ] 定义 `ContextEngineeringEventBuilder` 结构
- [ ] 添加统计字段 (`current_round`, `total_step_retries`, `total_task_replans`)
- [ ] 定义 `SuccessfulStepData` 结构
- [ ] 定义 `FailedStepData` 结构
- [ ] 定义 `ExtractedField` 结构

### 7.2 格式化方法
- [ ] 实现 `format_to_file()` 方法
- [ ] 实现执行统计摘要格式化
- [ ] 实现成功步骤格式化
- [ ] 实现失败步骤格式化

### 7.3 统计信息管理
- [ ] 实现 `update_statistics()` 方法
- [ ] 实现 `increment_step_retries()` 方法
- [ ] 实现 `increment_task_replans()` 方法
- [ ] 实现 `set_current_round()` 方法

### 7.4 集成
- [ ] 在反思阶段传递执行历史
- [ ] 实现 `should_use_plan_with_files()` 判断逻辑
- [ ] 在适当时机更新统计信息

---

## 下一步

阅读 [18-移除轮数限制机制.md](./18-移除轮数限制机制.md) 了解智能轮数管理的详细设计。
