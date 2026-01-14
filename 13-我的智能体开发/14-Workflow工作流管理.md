# Workflow工作流管理

> **文档版本**: v1.0
> **创建日期**: 2026-01-05
> **适用场景**: 标准流程定义与任务匹配

---

## 1. 功能概述

Workflow工作流管理模块提供**标准化任务流程**的定义、加载和匹配功能：

- **流程定义**：以TOML配置文件定义标准业务流程
- **关键词匹配**：根据任务类型和描述智能匹配标准流程
- **流程引导**：为LLM提供标准流程参考，提升规划准确性
- **工具参数集成**：自动关联工具的输入输出参数信息

```
┌─────────────────────────────────────────────────────────────────┐
│                    工作流匹配与使用流程                          │
│                                                                  │
│  ┌──────────────┐                                               │
│  │  任务描述     │                                               │
│  │ "进行负荷预测" │                                               │
│  └──────┬───────┘                                               │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────┐                   │
│  │         TaskWorkflowManager              │                   │
│  │  ┌────────────────────────────────────┐  │                   │
│  │  │  match_workflow()                  │  │                   │
│  │  │  • 遍历所有流程                     │  │                   │
│  │  │  • 计算关键词匹配分数               │  │                   │
│  │  │  • 返回最佳匹配                     │  │                   │
│  │  └────────────────────────────────────┘  │                   │
│  └──────┬───────────────────────────────────┘                   │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────┐   ┌──────────────────┐                    │
│  │  匹配成功         │   │   匹配失败        │                    │
│  │  返回标准流程     │   │   LLM自行规划     │                    │
│  └────────┬─────────┘   └──────────────────┘                    │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────────────────────────────┐                   │
│  │  format_for_llm_with_tools()             │                   │
│  │  生成LLM可读的流程描述                    │                   │
│  │  包含工具参数信息                         │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心数据结构

### 2.1 WorkflowStep - 流程步骤

```rust
/// 任务流程步骤
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStep {
    /// 步骤名称
    pub name: String,

    /// 工具ID（对应Tool Service中的工具）
    pub tool_id: String,

    /// 步骤描述
    pub description: String,
}
```

### 2.2 TaskWorkflow - 任务流程

```rust
/// 任务流程定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskWorkflow {
    /// 任务名称
    pub name: String,

    /// 任务描述
    pub description: String,

    /// 关键词列表（用于匹配）
    pub keywords: Vec<String>,

    /// 工具列表（用于工具筛选阶段优先选择）
    #[serde(default)]
    pub tool_list: Option<Vec<String>>,

    /// 标准流程步骤
    pub steps: Vec<WorkflowStep>,

    /// 流程注意事项
    #[serde(default)]
    pub notes: Option<String>,

    /// 工具类别提示（用于工具筛选阶段）
    #[serde(default)]
    pub tool_categories: Option<Vec<String>>,
}
```

### 2.3 WorkflowsConfig - 配置根结构

```rust
/// 任务流程配置（整个配置文件的根结构）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowsConfig {
    /// 所有任务流程
    pub workflows: HashMap<String, TaskWorkflow>,
}
```

---

## 3. 配置文件格式

### 3.1 TOML配置示例

```toml
# workflows.toml

[workflows.load_prediction]
name = "负荷预测"
description = "负荷预测完整流程，包括数据查询、模型加载和预测执行"
keywords = ["负荷预测", "负荷", "预测", "电力预测"]
tool_list = ["data_query", "model_loader", "prediction_executor"]
tool_categories = ["数据查询", "模型管理", "预测执行"]
notes = """
⚠️ 注意事项：
• 必须先查询历史数据才能进行预测
• 确保模型已正确加载
• 预测结果需要验证合理性
"""

[[workflows.load_prediction.steps]]
name = "查询历史负荷数据"
tool_id = "data_query"
description = "从数据库查询指定时间范围的历史负荷数据"

[[workflows.load_prediction.steps]]
name = "加载预测模型"
tool_id = "model_loader"
description = "加载负荷预测模型，准备进行预测计算"

[[workflows.load_prediction.steps]]
name = "执行负荷预测"
tool_id = "prediction_executor"
description = "使用模型对历史数据进行负荷预测，生成预测结果"

[workflows.auto_modeling]
name = "自动建模"
description = "自动建模流程，包括数据准备、模型训练和评估"
keywords = ["自动建模", "建模", "模型训练", "机器学习"]
tool_list = ["data_prepare", "model_trainer", "model_evaluator"]

[[workflows.auto_modeling.steps]]
name = "准备训练数据"
tool_id = "data_prepare"
description = "准备和预处理训练所需的数据集"

[[workflows.auto_modeling.steps]]
name = "训练模型"
tool_id = "model_trainer"
description = "使用准备好的数据训练机器学习模型"

[[workflows.auto_modeling.steps]]
name = "评估模型性能"
tool_id = "model_evaluator"
description = "评估训练完成的模型性能指标"
```

---

## 4. TaskWorkflowManager实现

### 4.1 管理器结构

```rust
/// 任务流程管理器
pub struct TaskWorkflowManager {
    /// 所有流程的映射
    workflows: Arc<HashMap<String, TaskWorkflow>>,
}

impl TaskWorkflowManager {
    /// 从配置文件加载任务流程
    pub fn from_file(config_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        info!("加载任务流程配置: {}", config_path);

        let content = std::fs::read_to_string(config_path)?;
        let config: WorkflowsConfig = toml::from_str(&content)?;

        info!("成功加载 {} 个任务流程", config.workflows.len());
        for (id, workflow) in &config.workflows {
            debug!(
                workflow_id = id,
                name = %workflow.name,
                steps_count = workflow.steps.len(),
                keywords_count = workflow.keywords.len(),
                "任务流程详情"
            );
        }

        Ok(Self {
            workflows: Arc::new(config.workflows),
        })
    }
}
```

### 4.2 关键词匹配算法

```rust
impl TaskWorkflowManager {
    /// 根据任务类型匹配标准流程
    ///
    /// # 参数
    /// - `task_type`: 任务类型（从工具筛选阶段获得）
    /// - `task_description`: 任务描述（用于辅助匹配）
    ///
    /// # 返回
    /// - Some(workflow): 匹配到的标准流程
    /// - None: 未匹配到任何流程
    pub fn match_workflow(
        &self,
        task_type: &str,
        task_description: &str,
    ) -> Option<TaskWorkflow> {
        debug!(
            task_type = task_type,
            task_description = task_description,
            "尝试匹配任务流程"
        );

        // 1. 将任务类型和描述转为小写便于匹配
        let task_type_lower = task_type.to_lowercase();
        let task_desc_lower = task_description.to_lowercase();

        // 2. 遍历所有流程，计算匹配分数
        let mut best_match: Option<(String, f32, &TaskWorkflow)> = None;

        for (id, workflow) in self.workflows.iter() {
            let mut score = 0.0f32;
            let mut matched_keywords = Vec::new();

            // 计算关键词匹配分数
            for keyword in &workflow.keywords {
                let keyword_lower = keyword.to_lowercase();

                // 任务类型完全匹配：+10分
                if task_type_lower == keyword_lower {
                    score += 10.0;
                    matched_keywords.push(keyword.clone());
                }
                // 任务类型包含关键词：+5分
                else if task_type_lower.contains(&keyword_lower) {
                    score += 5.0;
                    matched_keywords.push(keyword.clone());
                }
                // 任务描述包含关键词：+3分
                else if task_desc_lower.contains(&keyword_lower) {
                    score += 3.0;
                    matched_keywords.push(keyword.clone());
                }
            }

            // 更新最佳匹配
            if score > 0.0 {
                if best_match.is_none() || score > best_match.as_ref().unwrap().1 {
                    debug!(
                        workflow_id = id,
                        workflow_name = %workflow.name,
                        score = score,
                        matched_keywords = ?matched_keywords,
                        "找到更高分的匹配流程"
                    );
                    best_match = Some((id.clone(), score, workflow));
                }
            }
        }

        // 3. 返回匹配结果
        if let Some((id, score, workflow)) = best_match {
            info!(
                workflow_id = id,
                workflow_name = %workflow.name,
                match_score = score,
                "✅ 匹配到标准任务流程"
            );
            Some(workflow.clone())
        } else {
            info!("❌ 未匹配到标准任务流程，将由LLM自行规划");
            None
        }
    }
}
```

### 4.3 匹配分数规则

| 匹配类型 | 分数 | 说明 |
|---------|------|------|
| 任务类型完全匹配 | +10分 | `task_type == keyword` |
| 任务类型包含关键词 | +5分 | `task_type.contains(keyword)` |
| 任务描述包含关键词 | +3分 | `task_description.contains(keyword)` |

---

## 5. LLM格式化输出

### 5.1 带工具参数的格式化

```rust
impl TaskWorkflow {
    /// 格式化为可读文本，用于传递给LLM（包含工具参数信息）
    ///
    /// # 参数
    /// - `available_tools`: 可用的工具列表，用于查找工具的参数信息
    pub fn format_for_llm_with_tools(&self, available_tools: &[ToolInfo]) -> String {
        let mut text = String::new();

        text.push_str(&format!("【标准流程】{}\n\n", self.name));
        text.push_str(&format!("📝 流程描述：{}\n\n", self.description));

        text.push_str("📋 标准步骤（按顺序，包含工具参数）：\n\n");
        for (i, step) in self.steps.iter().enumerate() {
            text.push_str(&format!(
                "{}. {} (tool_id: {})\n",
                i + 1, step.name, step.tool_id
            ));
            text.push_str(&format!("   说明: {}\n", step.description));

            // 查找对应的工具信息
            if let Some(tool) = available_tools.iter().find(|t| t.id == step.tool_id) {
                // 显示输入参数
                if let Some(input_params) = &tool.input_params {
                    text.push_str("   📥 输入参数:\n");
                    // 解析并格式化参数
                    self.format_params(&mut text, input_params);
                }

                // 显示输出参数
                if let Some(output_params) = &tool.output_params {
                    text.push_str("   📤 输出参数:\n");
                    self.format_params(&mut text, output_params);
                }
            } else {
                text.push_str(&format!(
                    "   ⚠️ 警告: 未找到工具 {} 的详细信息\n",
                    step.tool_id
                ));
            }

            text.push_str("\n");
        }

        // 添加注意事项
        if let Some(notes) = &self.notes {
            text.push_str(notes);
            text.push_str("\n");
        }

        // 添加参数配置指导
        text.push_str("\n⚠️ 【重要】参数配置和依赖关系指导：\n");
        text.push_str("  • 必须根据上述工具的输入参数要求填充 parameters 字段\n");
        text.push_str("  • 如果某个参数需要前置步骤的输出，引用前置步骤\n");
        text.push_str("  • 设置正确的 dependencies 字段，确保有数据依赖的步骤按顺序执行\n");
        text.push_str("  • 步骤顺序、个数、tool_id 必须与上述标准流程完全一致\n");

        text
    }
}
```

### 5.2 输出示例

```
【标准流程】负荷预测

📝 流程描述：负荷预测完整流程，包括数据查询、模型加载和预测执行

📋 标准步骤（按顺序，包含工具参数）：

1. 查询历史负荷数据 (tool_id: data_query)
   说明: 从数据库查询指定时间范围的历史负荷数据
   📥 输入参数:
      - start_time: datetime (查询开始时间)
      - end_time: datetime (查询结束时间)
      - data_type: string (数据类型)
   📤 输出参数:
      - data: array (历史负荷数据)
      - count: integer (数据条数)

2. 加载预测模型 (tool_id: model_loader)
   说明: 加载负荷预测模型，准备进行预测计算
   📥 输入参数:
      - model_id: string (模型标识)
   📤 输出参数:
      - model_info: object (模型信息)

3. 执行负荷预测 (tool_id: prediction_executor)
   说明: 使用模型对历史数据进行负荷预测，生成预测结果
   📥 输入参数:
      - model_info: object (模型信息，来自step_2)
      - data: array (历史数据，来自step_1)
   📤 输出参数:
      - predictions: array (预测结果)

⚠️ 注意事项：
• 必须先查询历史数据才能进行预测
• 确保模型已正确加载
• 预测结果需要验证合理性

⚠️ 【重要】参数配置和依赖关系指导：
  • 必须根据上述工具的输入参数要求填充 parameters 字段
  • 如果某个参数需要前置步骤的输出，引用前置步骤
  • 设置正确的 dependencies 字段，确保有数据依赖的步骤按顺序执行
  • 步骤顺序、个数、tool_id 必须与上述标准流程完全一致
```

---

## 6. 辅助方法

### 6.1 获取流程列表

```rust
impl TaskWorkflowManager {
    /// 获取所有可用的流程列表（用于日志和调试）
    pub fn list_workflows(&self) -> Vec<String> {
        self.workflows
            .iter()
            .map(|(id, wf)| format!("{}: {} ({} 步骤)", id, wf.name, wf.steps.len()))
            .collect()
    }
}
```

### 6.2 根据ID获取流程

```rust
impl TaskWorkflowManager {
    /// 根据流程ID获取标准流程
    ///
    /// # 参数
    /// - `workflow_id`: 流程ID
    ///
    /// # 返回
    /// - Some(workflow): 匹配到的标准流程
    /// - None: 未找到该流程ID
    pub fn get_workflow(&self, workflow_id: &str) -> Option<TaskWorkflow> {
        self.workflows.get(workflow_id).cloned()
    }
}
```

### 6.3 格式化流程摘要

```rust
impl TaskWorkflowManager {
    /// 格式化所有流程为简化信息（用于LLM筛选）
    pub fn format_workflows_summary(&self) -> String {
        let mut summary = String::new();

        for (i, (id, workflow)) in self.workflows.iter().enumerate() {
            summary.push_str(&format!(
                "{}. ID: {} | 名称: {} | 描述: {}\n",
                i + 1, id, workflow.name, workflow.description
            ));
        }

        summary
    }
}
```

### 6.4 根据名称查找ID

```rust
impl TaskWorkflowManager {
    /// 根据工作流名称查找工作流ID
    pub fn find_workflow_id_by_name(&self, workflow_name: &str) -> Option<String> {
        self.workflows.iter()
            .find(|(_, wf)| wf.name == workflow_name)
            .map(|(id, _)| id.clone())
    }
}
```

---

## 7. 与Planner集成

### 7.1 规划阶段使用

```rust
impl Planner {
    /// 创建执行计划（集成工作流）
    pub async fn create_plan(
        &self,
        context: &ExecutionContext,
        available_tools: &[ToolInfo],
    ) -> Result<ExecutionPlan> {
        // 1. 尝试匹配标准工作流
        let workflow = self.workflow_manager.match_workflow(
            &context.task_type,
            &context.task_description,
        );

        // 2. 构建规划提示词
        let prompt = if let Some(wf) = workflow {
            // 有标准流程：提供流程参考
            format!(
                "{}\n\n{}\n\n请参考上述标准流程进行规划。",
                self.base_prompt,
                wf.format_for_llm_with_tools(available_tools)
            )
        } else {
            // 无标准流程：LLM自行规划
            format!(
                "{}\n\n请根据任务需求自行规划执行步骤。",
                self.base_prompt
            )
        };

        // 3. 调用LLM生成计划
        self.llm_client.generate_plan(&prompt).await
    }
}
```

---

## 8. 配置选项

### 8.1 配置文件路径

```toml
# config.toml

[workflow]
# 工作流配置文件路径
config_path = "./config/workflows.toml"

# 是否启用工作流匹配
enabled = true

# 最低匹配分数阈值（低于此分数视为未匹配）
min_match_score = 3.0
```

### 8.2 环境变量

```bash
# 工作流配置文件路径
export WORKFLOW_CONFIG_PATH=./config/workflows.toml

# 启用工作流匹配
export WORKFLOW_ENABLED=true
```

---

## 9. 迁移实现清单

### 9.1 数据结构

- [ ] 实现 `WorkflowStep` 结构
- [ ] 实现 `TaskWorkflow` 结构
- [ ] 实现 `WorkflowsConfig` 结构

### 9.2 管理器实现

- [ ] 实现 `TaskWorkflowManager`
- [ ] 实现 `from_file` 配置加载
- [ ] 实现 `match_workflow` 关键词匹配
- [ ] 实现 `format_for_llm_with_tools` 格式化

### 9.3 集成实现

- [ ] 创建 `workflows.toml` 配置文件
- [ ] 在Planner中集成工作流匹配
- [ ] 添加工作流匹配日志

---

## 10. 相关文档

- [03-核心模块实现详解](./03-核心模块实现详解.md) - Planner实现
- [05-LLM交互与提示词系统](./05-LLM交互与提示词系统.md) - 提示词设计
- [09-配置与部署指南](./09-配置与部署指南.md) - 配置文件说明

---

**文档维护者**: Task Orchestration Team
**最后更新**: 2026-01-05
