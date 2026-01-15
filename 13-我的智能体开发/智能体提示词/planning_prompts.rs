//! 规划提示词管理模块
//!
//! 提供场景感知的规划提示词模板管理
//! 支持根据任务类型（task_type）动态选择规划指导方向

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::scene_guidance::SceneManager;

/// 规划上下文信息（预留，用于未来扩展）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanningContext {
    /// 任务类型（如：自动建模、负荷预测、数据分析、数学计算等）
    pub task_type: Option<String>,
    /// 任务描述
    pub task_description: String,
    /// 可用工具数量
    pub available_tools_count: usize,
}

impl PlanningContext {
    /// 创建新的规划上下文
    pub fn new(task_type: Option<String>, task_description: String, available_tools_count: usize) -> Self {
        Self {
            task_type,
            task_description,
            available_tools_count,
        }
    }
}

// ==================== 基础规划提示词模板 ====================

/// 规划阶段系统提示词（阶段2：任务分解）
/// 注意：此提示词用于两阶段规划的第二阶段，输入的工具列表已经过第一阶段筛选
/// ⚠️ 这是基础模板，不应该被修改，所有场景专属指导通过追加的方式实现
pub const PLANNING_SYSTEM_PROMPT: &str = r#"你是任务规划专家，将任务分解为可执行步骤。

【核心职责】
综合工具列表和上下文（对话历史、文档、元数据、用户偏好），生成完整执行计划。

【质量要求】
完整准确、独立可执行、充分利用上下文、参数配置精准

【步骤粒度】
✓ 多对象分步：添加A和B → step_1:添加A, step_2:添加B
✗ 过度拆分：禁止添加验证/确认/检查步骤

【⚠️ 并行执行规划 - 重要】
系统支持基于DAG的并行执行，请合理利用dependencies字段实现并行优化：
✓ **无依赖步骤可并行**: 如果多个步骤之间无数据依赖，将dependencies设为[]，系统会自动并行执行
✓ **明确依赖关系**: 如果step_3需要step_1和step_2的输出，设置"dependencies": ["step_1", "step_2"]
✓ **优化执行效率**: 优先将独立任务设计为并行执行，减少总执行时间
✗ **避免过度依赖**: 不要添加不必要的依赖关系，以免降低并行度

【🚀 步骤内并行执行 - Actions格式 - 强烈推荐】
**新特性**: 支持在单个步骤内并行执行多个操作，大幅提升执行效率
✓ **何时使用actions格式**（优先考虑）:
  - ⭐ 当需要执行多个相同类型的操作时（如：添加多个角色、创建多个对象、删除多个项目）
  - ⭐ 当多个操作使用相同或相似的工具，但参数不同时
  - ⭐ 当多个操作彼此独立，无需等待前一个完成时
✓ **典型场景示例**:
  - ✅ "添加角色A和角色B" → 使用actions格式，一个step内并行执行2个add_role操作
  - ✅ "计算5、8、12的平方" → 使用actions格式，一个step内并行执行3个计算操作
  - ✅ "创建配置项X、Y、Z" → 使用actions格式，一个step内并行执行3个创建操作
  - ❌ "添加角色A" → 只有单个操作，使用旧格式(tool + parameters)
✓ **兼容性**: 如果步骤只有单个操作，继续使用旧格式(tool + parameters)
✓ **依赖引用**: 后续步骤可通过{{action_id.output}}引用action的输出

Actions格式示例1 - 并行计算：
```json
{
  "step_id": "step_1",
  "step_name": "并行计算三个数的平方",
  "actions": [
    {
      "action_id": "action_1_1",
      "name": "计算5的平方",
      "tool_id": "js_engine",
      "parameters": {"code": "5**2"},
      "dependencies": [],
      "expected_output": "25"
    },
    {
      "action_id": "action_1_2",
      "name": "计算8的平方",
      "tool_id": "js_engine",
      "parameters": {"code": "8**2"},
      "dependencies": [],
      "expected_output": "64"
    },
    {
      "action_id": "action_1_3",
      "name": "计算12的平方",
      "tool_id": "js_engine",
      "parameters": {"code": "12**2"},
      "dependencies": [],
      "expected_output": "144"
    }
  ],
  "dependencies": []
}
```

Actions格式示例2 - 并行添加角色：
```json
{
  "step_id": "step_2",
  "step_name": "批量添加角色",
  "actions": [
    {
      "action_id": "action_2_1",
      "name": "添加角色huarun_test_1",
      "tool_id": "add_role",
      "parameters": {"role_name": "huarun_test_1"},
      "dependencies": [],
      "expected_output": "角色添加成功"
    },
    {
      "action_id": "action_2_2",
      "name": "添加角色test_2",
      "tool_id": "add_role",
      "parameters": {"role_name": "test_2"},
      "dependencies": [],
      "expected_output": "角色添加成功"
    }
  ],
  "dependencies": ["step_1"]
}
```

后续步骤引用action输出示例：
```json
{
  "step_id": "step_3",
  "step_name": "汇总结果",
  "tool_id": "js_engine",
  "parameters": {
    "code": "{{action_1_1.output}} + {{action_1_2.output}}"
  },
  "dependencies": ["step_1"]
}
```

示例并行场景：
- 多个独立计算(加法、乘法) → dependencies均为[]，可并行
- 数据收集+数据验证 → 可并行执行，dependencies均为[]
- 结果汇总 → 依赖前面所有步骤，dependencies: ["step_1", "step_2", ...]

【参数优先级】
P1:用户偏好(必须遵守) > P2:上下文文档 > P3:元数据 > P4:任务描述 > P5:前置输出

【⚠️ 工作流匹配规则 - 重要】
⚠️ **关键约束：必须严格基于任务描述中的明确意图进行工作流匹配**
- 如果任务描述中已明确标注了意图（格式："意图: XXX, 用户输入: ..."），则**只能**匹配与该意图完全一致或高度相关的工作流
- 禁止匹配与明确意图无关的工作流，即使任务描述中包含其他关键词
- 例如：任务描述为"意图: PLC控制器, 用户输入: 创建脚本..."时，**禁止**匹配"自然语言建模"等无关工作流，即使描述中包含"创建"等词

如果传递来的标准工作流与任务意图完全一致：
- 规划的步骤顺序必须与工作流定义的步骤顺序完全一致
- 规划的步骤个数必须与工作流定义的步骤个数完全一致
- 每个步骤的tool_id必须严格按照工作流定义的tool_id
- 步骤名称(step_name)应与工作流定义的步骤名称保持一致
- 只能在parameters字段中根据任务描述填充具体参数值

示例：如果匹配到工作流定义为：
[
  { name = "获取模型信息", tool_id = "get_model_info" },
  { name = "获取连接信息", tool_id = "get_connections_info" },
  { name = "处理拓扑结构", tool_id = "process_topology" },
  { name = "执行仿真建模", tool_id = "exec_simulation_modeling" }
]
则生成的计划必须为这4个步骤，顺序完全一致，tool_id完全一致。

【关键约束】
- 仅用提供的工具，ID完全匹配
- description严格基于任务描述，不扩展
- dependencies引用已定义步骤
- 数据流一致：输入来源与输出用途对应

【⚠️ 步骤命名规范 - 重要】
step_name 必须基于工具作用与操作实体名称构成，不受上下文执行结果影响：
✓ **正确示例**: "生成用户模块代码"、"添加角色admin"、"获取设备连接信息"
✗ **错误示例**: "重新生成失败的代码"、"修复上次错误"、"继续未完成的操作"
⚠️ **上下文的作用**: 上下文仅用于告知已执行步骤的结果情况，帮助理解当前状态，但不应影响步骤名称的描述方式
⚠️ **命名原则**: 每个步骤名称应该是自描述的、独立的，能够清晰表达该步骤要执行的具体操作

【输出JSON】
{
  "plan_id": "plan_<uuid>",
  "description": "基于任务描述",
  "task_type": "负荷预测|自动建模|数据分析|数学计算|客户端操作|PLC控制器|工具管理|通用",
  "context_understanding": "总结对话/文档/配置/偏好(无则填'无')",
  "total_steps": 数字,
  "estimated_duration_secs": 数字,
  "steps": [{
    "step_id": "step_1",
    "step_name": "名称",
    "tool_id": "ID",
    "parameters": {},
    "dependencies": [],
    "expected_output": "输出",
    "data_input_source": "用户输入|step_X输出|元数据|上下文",
    "data_output_usage": "供step_X使用|最终结果|中间状态"
  }]
}"#;

/// 规划阶段用户提示词模板
/// ⚠️ 这是基础模板，不应该被修改
pub const PLANNING_USER_TEMPLATE: &str = r#"
【任务描述】
{task_description}
【元数据】
{metadata}
【上下文】
{context}
【可用工具】
{available_tools}
【内置工作流】
{workflow_hint}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 【场景专属规划指导 - 最高优先级】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{scene_guidance}

【要求】
⭐⭐⭐ 0. **如果上方有场景专属指导，必须严格遵守！优先级高于下方所有通用要求和示例！**
1. 填context_understanding（总结对话/文档/配置/偏好）
2. 按P1-P5优先级配置参数
3. ⭐ 多个相同类型操作优先使用actions格式（如：添加多个角色、创建多个对象）
4. 多对象分步，禁止过度拆分
5. 输出JSON

示例1(步骤内并行 - actions格式)：
```json
{
  "plan_id": "plan_001",
  "description": "批量添加角色",
  "context_understanding": "需要添加两个角色，可在一个步骤内并行执行",
  "total_steps": 1,
  "estimated_duration_secs": 30,
  "steps": [
    {
      "step_id": "step_1",
      "step_name": "批量添加角色",
      "actions": [
        {
          "action_id": "action_1_1",
          "name": "添加角色huarun_test_1",
          "tool_id": "add_role",
          "parameters": {"role_name": "huarun_test_1"},
          "dependencies": [],
          "expected_output": "角色添加成功"
        },
        {
          "action_id": "action_1_2",
          "name": "添加角色test_2",
          "tool_id": "add_role",
          "parameters": {"role_name": "test_2"},
          "dependencies": [],
          "expected_output": "角色添加成功"
        }
      ],
      "dependencies": []
    }
  ]
}
```

示例2(步骤间并行)：
```json
{
  "plan_id": "plan_001",
  "description": "并行计算A+B和A*B，然后汇总结果",
  "task_type": "数学计算",
  "context_understanding": "两个独立计算可并行执行，然后汇总",
  "total_steps": 3,
  "estimated_duration_secs": 60,
  "steps": [
    {
      "step_id": "step_1",
      "step_name": "计算A+B",
      "tool_id": "calculator",
      "parameters": {"operation": "add", "a": 10, "b": 20},
      "dependencies": [],
      "expected_output": "30",
      "data_input_source": "用户输入",
      "data_output_usage": "供step_3使用"
    },
    {
      "step_id": "step_2",
      "step_name": "计算A*B",
      "tool_id": "calculator",
      "parameters": {"operation": "multiply", "a": 10, "b": 20},
      "dependencies": [],
      "expected_output": "200",
      "data_input_source": "用户输入",
      "data_output_usage": "供step_3使用"
    },
    {
      "step_id": "step_3",
      "step_name": "汇总结果",
      "tool_id": "result_merger",
      "parameters": {"sources": ["step_1", "step_2"]},
      "dependencies": ["step_1", "step_2"],
      "expected_output": "汇总报告: A+B=30, A*B=200",
      "data_input_source": "step_1和step_2输出",
      "data_output_usage": "最终结果"
    }
  ]
}
```

示例(多步)：
```json
{
  "plan_id": "plan_002",
  "description": "添加设备A和B并配置通信",
  "task_type": "自动建模",
  "context_understanding": "添加PLC和传感器，文档要求PLC优先，偏好超时5000ms",
  "total_steps": 3,
  "estimated_duration_secs": 90,
  "steps": [
    {
      "step_id": "step_1",
      "step_name": "添加PLC",
      "tool_id": "add_device",
      "parameters": {"device_name": "设备A", "device_type": "PLC"},
      "dependencies": [],
      "expected_output": "设备A的ID",
      "data_input_source": "用户输入",
      "data_output_usage": "供step_3使用"
    },
    {
      "step_id": "step_2",
      "step_name": "添加传感器",
      "tool_id": "add_device",
      "parameters": {"device_name": "设备B", "device_type": "传感器"},
      "dependencies": ["step_1"],
      "expected_output": "设备B的ID",
      "data_input_source": "用户输入",
      "data_output_usage": "供step_3使用"
    },
    {
      "step_id": "step_3",
      "step_name": "配置通信",
      "tool_id": "configure_comm",
      "parameters": {"device_ids": ["step_1输出","step_2输出"], "timeout_ms": 5000},
      "dependencies": ["step_1", "step_2"],
      "expected_output": "配置结果",
      "data_input_source": "step_1输出和step_2输出",
      "data_output_usage": "最终结果"
    }
  ]
}
```

生成计划："#;

// ==================== 规划提示词构建器 ====================

/// 规划提示词构建器
/// 根据任务类型动态选择规划指导策略
pub struct PlanningPromptBuilder {
    /// 统一的场景管理器
    scene_manager: SceneManager,
}

impl Default for PlanningPromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PlanningPromptBuilder {
    /// 创建新的规划提示词构建器
    pub fn new() -> Self {
        Self {
            scene_manager: SceneManager::new(),
        }
    }

    /// 根据任务类型获取场景专属的规划指导
    ///
    /// # 参数
    /// - `task_type`: 任务类型（���工具筛选阶段获得）
    ///
    /// # 返回
    /// - 对应场景的规划指导字符串（如果未找到则返回通用场景指导）
    pub fn get_scene_guidance(&self, task_type: Option<&str>) -> &'static str {
        self.scene_manager.get_planning_guidance(task_type)
    }

    /// 构建增强的规划提示词（带场景指导）
    ///
    /// # 参数
    /// - `task_description`: 任务描述
    /// - `available_tools`: 可用工具列表
    /// - `metadata`: 元数据
    /// - `context`: 任务上下文（可选）
    /// - `workflow_hint`: 工作流提示（可选）
    /// - `task_type`: 任务类型（用于选择场景指导）
    ///
    /// # 返回
    /// - (系统提示词, 用户提示词)
    pub fn build_enhanced_planning_prompt(
        &self,
        task_description: &str,
        available_tools: &str,
        metadata: &HashMap<String, String>,
        context: Option<&crate::grpc::orchestrator::TaskContext>,
        workflow_hint: Option<&str>,
        task_type: Option<&str>,
    ) -> (String, String) {
        // 1. 获取场景专属规划指导
        eprintln!("🔍 DEBUG: task_type = {:?}", task_type);
        let scene_guidance = self.get_scene_guidance(task_type);
        eprintln!("🔍 DEBUG: scene_guidance length = {}, is_empty = {}", scene_guidance.len(), scene_guidance.is_empty());

        // 2. 系统提示词使用基础模板
        let system_prompt = PLANNING_SYSTEM_PROMPT.to_string();

        // 3. 构建用户提示词（将场景指导插入用户提示词）
        let user_prompt = Self::build_planning_user_prompt(
            task_description,
            available_tools,
            metadata,
            context,
            workflow_hint,
            scene_guidance,
        );

        (system_prompt, user_prompt)
    }

    /// 构建规划用户提示词
    /// 内部方法，用于格式化用户提示词模板
    fn build_planning_user_prompt(
        task_description: &str,
        available_tools: &str,
        metadata: &HashMap<String, String>,
        context: Option<&crate::grpc::orchestrator::TaskContext>,
        workflow_hint: Option<&str>,
        scene_guidance: &str,
    ) -> String {
        // 格式化元数据
        let metadata_str = if metadata.is_empty() {
            "无".to_string()
        } else {
            metadata
                .iter()
                .map(|(k, v)| format!("  - {}: {}", k, v))
                .collect::<Vec<_>>()
                .join("\n")
        };

        // 格式化上下文信息
        let context_str = Self::format_task_context(context);

        // 格式化工作流程提示
        let workflow_str = if let Some(hint) = workflow_hint {
            format!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n【匹配的标准任务流程】\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n{}\n", hint)
        } else {
            String::new()
        };

        // 格式化场景指导（模板中已经有分隔线和标题，这里只需要返回内容）
        let scene_guidance_str = if !scene_guidance.is_empty() && scene_guidance.trim() != "" {
            scene_guidance.to_string()
        } else {
            "无场景专属指导，使用通用规划策略。".to_string()
        };

        PLANNING_USER_TEMPLATE
            .replace("{task_description}", task_description)
            .replace("{available_tools}", available_tools)
            .replace("{metadata}", &metadata_str)
            .replace("{context}", &context_str)
            .replace("{workflow_hint}", &workflow_str)
            .replace("{scene_guidance}", &scene_guidance_str)
    }

    /// 格式化任务上下文信息
    fn format_task_context(context: Option<&crate::grpc::orchestrator::TaskContext>) -> String {
        if let Some(ctx) = context {
            let mut context_parts = Vec::new();

            // 对话历史
            if !ctx.conversation_history.is_empty() {
                let history = ctx.conversation_history.iter()
                    .enumerate()
                    .map(|(i, msg)| format!("    [{}] {}", i + 1, msg))
                    .collect::<Vec<_>>()
                    .join("\n");
                context_parts.push(format!("【对话历史】\n{}", history));
            }

            // 相关文档
            if !ctx.documents.is_empty() {
                let docs = ctx.documents.iter()
                    .map(|doc| {
                        let doc_type = doc.doc_type.as_deref().unwrap_or("未知");
                        let desc = doc.description.as_deref().unwrap_or("");
                        format!(
                            "  - 文档: {}\n    类型: {}\n    描述: {}\n    内容:\n{}\n",
                            doc.name,
                            doc_type,
                            desc,
                            Self::indent_text(&doc.content, 6)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                context_parts.push(format!("【相关文档】\n{}", docs));
            }

            // 附加信息
            if !ctx.additional_info.is_empty() {
                let info = ctx.additional_info.iter()
                    .map(|(k, v)| format!("  - {}: {}", k, v))
                    .collect::<Vec<_>>()
                    .join("\n");
                context_parts.push(format!("【附加信息】\n{}", info));
            }

            // 用户偏好
            if !ctx.user_preferences.is_empty() {
                let prefs = ctx.user_preferences.iter()
                    .enumerate()
                    .map(|(i, pref)| format!("  {}. {}", i + 1, pref))
                    .collect::<Vec<_>>()
                    .join("\n");
                context_parts.push(format!("【用户偏好/要求】\n{}", prefs));
            }

            if context_parts.is_empty() {
                "无".to_string()
            } else {
                format!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n【任务上下文信息】\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n{}\n", context_parts.join("\n\n"))
            }
        } else {
            "无".to_string()
        }
    }

    /// 缩进文本
    fn indent_text(text: &str, spaces: usize) -> String {
        let indent = " ".repeat(spaces);
        text.lines()
            .map(|line| format!("{}{}", indent, line))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// 获取支持的任务类型列表
    pub fn supported_task_types(&self) -> Vec<String> {
        self.scene_manager
            .all_task_type_names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planning_prompt_builder() {
        let builder = PlanningPromptBuilder::new();
        let metadata = HashMap::new();

        // 测试客户端管理场景
        let (system_client, user_client) = builder.build_enhanced_planning_prompt(
            "添加角色huarun_test_1和test_2",
            "工具列表",
            &metadata,
            None,
            None,
            Some("客户端管理"),
        );
        // 基础模板应该始终存在于系统提示词
        assert!(system_client.contains("任务规划专家"));
        assert!(user_client.contains("添加角色huarun_test_1和test_2"));

        // 测试PLC控制器场景
        let (system_plc, user_plc) = builder.build_enhanced_planning_prompt(
            "连接PLC设备并读取数据",
            "工具列表",
            &metadata,
            None,
            None,
            Some("PLC控制器"),
        );
        assert!(system_plc.contains("任务规划专家"));
        assert!(user_plc.contains("连接PLC设备并读取数据"));

        // 测试通用场景（默认）
        let (system_general, user_general) = builder.build_enhanced_planning_prompt(
            "通用任务",
            "工具列表",
            &metadata,
            None,
            None,
            None,
        );
        // 基础模板应该始终存在于系统提示词
        assert!(system_general.contains("任务规划专家"));
        assert!(system_general.contains("并行执行规划"));
        assert!(user_general.contains("通用任务"));
    }

    #[test]
    fn test_scene_guidance_selection() {
        let builder = PlanningPromptBuilder::new();

        // 测试精确匹配 - 工具管理
        let tool_mgmt_guidance = builder.get_scene_guidance(Some("工具管理"));
        assert!(tool_mgmt_guidance.contains("创建工具的特殊规则"));
        assert!(tool_mgmt_guidance.contains("只规划一个步骤"));

        // 测试工具管理英文别名
        let tool_mgmt_en = builder.get_scene_guidance(Some("tool_management"));
        assert!(tool_mgmt_en.contains("创建工具的特殊规则"));

        // 测试模糊匹配 - PLC控制器
        let plc_guidance = builder.get_scene_guidance(Some("PLC控制器"));
        // PLC场景当前为空字符串，只检查返回值不为null
        let _ = plc_guidance;

        // 测试默认返回（通用场景为空字符串）
        let unknown_guidance = builder.get_scene_guidance(Some("未知类型"));
        assert_eq!(unknown_guidance, "");

        let none_guidance = builder.get_scene_guidance(None);
        assert_eq!(none_guidance, "");
    }

    #[test]
    fn test_base_template_presence() {
        // 测试基础模板常量存在且包含关键内容
        assert!(PLANNING_SYSTEM_PROMPT.contains("任务规划专家"));
        assert!(PLANNING_SYSTEM_PROMPT.contains("并行执行规划"));
        assert!(PLANNING_SYSTEM_PROMPT.contains("Actions格式"));
        assert!(PLANNING_SYSTEM_PROMPT.contains("工作流匹配规则"));

        assert!(PLANNING_USER_TEMPLATE.contains("{task_description}"));
        assert!(PLANNING_USER_TEMPLATE.contains("{available_tools}"));
        assert!(PLANNING_USER_TEMPLATE.contains("{workflow_hint}"));
        assert!(PLANNING_USER_TEMPLATE.contains("{scene_guidance}"));
    }

    #[test]
    fn test_tool_management_guidance() {
        let builder = PlanningPromptBuilder::new();
        let metadata = HashMap::new();

        // 测试工具管理场景（中文）
        let (_system_tool_mgmt, user_tool_mgmt) = builder.build_enhanced_planning_prompt(
            "创建计算器工具",
            "工具列表",
            &metadata,
            None,
            None,
            Some("工具管理"),
        );

        // 验证用户提示词包含工具管理指导
        assert!(user_tool_mgmt.contains("创建工具的特殊规则"));
        assert!(user_tool_mgmt.contains("只规划一个步骤"));

        // 验证用户提示词包含任务描述
        assert!(user_tool_mgmt.contains("创建计算器工具"));

        // 测试工具管理场景（英文）
        let (_system_tool_mgmt_en, user_tool_mgmt_en) = builder.build_enhanced_planning_prompt(
            "创建数据验证工具",
            "工具列表",
            &metadata,
            None,
            None,
            Some("tool_management"),
        );

        assert!(user_tool_mgmt_en.contains("创建工具的特殊规则"));
        assert!(user_tool_mgmt_en.contains("创建数据验证工具"));
    }
}
