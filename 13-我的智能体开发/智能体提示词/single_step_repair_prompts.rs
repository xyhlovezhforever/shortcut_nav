/// 单步修复提示词模块
///
/// 此模块负责生成单步修复场景的提示词
/// 当单个步骤执行失败但被判定为可恢复时，使用此模块生成修复后的步骤定义

use crate::llm::SceneManager;

// ==================== 单步修复提示词模板 ====================

/// 单步修复系统提示词（基础模板）
const SINGLE_STEP_REPAIR_SYSTEM_PROMPT: &str = r#"你是一个专业的步骤修复专家，专注于修复失败的任务步骤。

你的任务是基于给定的失败步骤和错误信息，提出一个改进的步骤定义。

重要原则：
1. 只修复当前步骤，不涉及其他步骤
2. 尽可能复用原步骤的逻辑，只调整必要部分
3. 在可用工具列表中选择最合适的工具
4. 考虑错误原因并提出针对性的改进

🚨 **避免重复修复 - 极其重要**：
- ⚠️ **必须查看执行历史**：仔细检查执行历史中是否已经尝试过相同的修复方案
- ❌ **禁止重复失败的方案**：如果历史中某个工具或参数组合已经失败过，绝对不要再次使用
- ✅ **寻找新的解决方案**：必须提供与历史失败尝试不同的工具、参数或方法
- ✅ **分析失败模式**：从历史失败中学习，理解为什么之前的方案不工作
- ⚠️ **如果无法找到新方案**：在 repair_reason 中明确说明"已尝试所有可行方案，建议重新规划整个任务"

你必须遵循以下 JSON Schema 生成修复后的步骤：
{
  "step_id": "string (保持原ID)",
  "step_name": "string (步骤名称，必须有意义)",
  "tool_id": "string (从可用工具列表中选择)",
  "parameters": {"key": "value"},
  "dependencies": ["step_id1", "step_id2"],
  "expected_output": "string",
  "repair_reason": "string (说明这个修复如何解决原问题，以及为什么与历史失败方案不同)"
}"#;

/// 单步修复用户提示词模板
const SINGLE_STEP_REPAIR_USER_TEMPLATE: &str = r#"{history_section}
原失败步骤：
- 步骤ID: {step_id}
- 步骤名称: {step_name}
- 工具ID: {tool_id}
- 参数: {parameters}
- 错误信息: {error_message}

可用工具列表：
{tools_info}

请提出一个改进的步骤定义，该步骤应该能够修复上述错误。
关键要求：
1. 步骤ID必须保持为 '{step_id}'
2. step_name 字段必须有明确的、有意义的名称
3. 选择最合适的工具来解决这个问题
4. 调整参数以避免导致失败的同样问题
5. **必须避免重复执行历史中已失败的方案**
6. 在 repair_reason 中解释这个修复如何帮助解决问题，以及为什么与历史失败方案不同"#;

/// 执行历史部分模板
const EXECUTION_HISTORY_SECTION_TEMPLATE: &str = r#"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【执行历史】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{execution_history}

⚠️ **重要提醒**：
- 上述执行历史显示了之前尝试过的所有方案及其结果
- 你**必须避免**使用历史中已经失败过的工具或参数组合
- 如果某个工具在历史中失败过，请选择其他工具或调整参数
- 如果所有可行方案都已尝试过，请在 repair_reason 中说明

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"#;

// ==================== 单步修复提示词构建器 ====================

/// 单步修复提示词构建器
pub struct SingleStepRepairPromptBuilder {
    /// 统一的场景管理器
    scene_manager: SceneManager,
}

impl Default for SingleStepRepairPromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SingleStepRepairPromptBuilder {
    /// 创建新的单步修复提示词构建器
    pub fn new() -> Self {
        Self {
            scene_manager: SceneManager::new(),
        }
    }

    /// 构建单步修复提示词（带场景指导和执行历史）
    ///
    /// # 参数
    /// - `step_id`: 步骤ID
    /// - `step_name`: 步骤名称
    /// - `tool_id`: 工具ID
    /// - `parameters`: 参数（格式化为字符串）
    /// - `error_message`: 错误消息
    /// - `tools_info`: 可用工具列表说明
    /// - `task_type`: 任务类型（可选）
    /// - `execution_history`: 执行历史（可选）
    ///
    /// # 返回
    /// (system_prompt, user_prompt)
    pub fn build_prompt(
        &self,
        step_id: &str,
        step_name: &str,
        tool_id: &str,
        parameters: &str,
        error_message: &str,
        tools_info: &str,
        task_type: Option<&str>,
        execution_history: Option<&str>,
    ) -> (String, String) {
        // 1. 获取场景特定的单步修复指导
        let scene_guidance = self.scene_manager.get_single_step_repair_guidance(task_type);

        // 2. 构建系统提示词
        let system_prompt = if !scene_guidance.trim().is_empty() {
            format!("{}\n\n{}", SINGLE_STEP_REPAIR_SYSTEM_PROMPT, scene_guidance)
        } else {
            SINGLE_STEP_REPAIR_SYSTEM_PROMPT.to_string()
        };

        // 3. 格式化执行历史
        let history_section = if let Some(history) = execution_history {
            if !history.trim().is_empty() {
                EXECUTION_HISTORY_SECTION_TEMPLATE.replace("{execution_history}", history)
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        // 4. 组装用户提示词
        let user_prompt = SINGLE_STEP_REPAIR_USER_TEMPLATE
            .replace("{history_section}", &history_section)
            .replace("{step_id}", step_id)
            .replace("{step_name}", step_name)
            .replace("{tool_id}", tool_id)
            .replace("{parameters}", parameters)
            .replace("{error_message}", error_message)
            .replace("{tools_info}", tools_info);

        // 5. 返回系统提示词和用户提示词
        (system_prompt, user_prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_builder_without_history() {
        let builder = SingleStepRepairPromptBuilder::new();
        let (system_prompt, user_prompt) = builder.build_prompt(
            "step_1",
            "测试步骤",
            "test_tool",
            r#"{"param": "value"}"#,
            "工具执行失败",
            "1. tool_a\n2. tool_b",
            None,
            None,
        );

        assert!(system_prompt.contains("步骤修复专家"));
        assert!(system_prompt.contains("避免重复修复"));
        assert!(system_prompt.contains("JSON Schema"));
        assert!(user_prompt.contains("step_1"));
        assert!(user_prompt.contains("测试步骤"));
        assert!(user_prompt.contains("test_tool"));
        assert!(user_prompt.contains("工具执行失败"));
    }

    #[test]
    fn test_prompt_builder_with_execution_history() {
        let builder = SingleStepRepairPromptBuilder::new();
        let history = "步骤1: 使用 tool_x 失败\n步骤2: 使用 tool_y 失败";
        let (system_prompt, user_prompt) = builder.build_prompt(
            "step_1",
            "测试步骤",
            "test_tool",
            r#"{"param": "value"}"#,
            "工具执行失败",
            "1. tool_a\n2. tool_b",
            None,
            Some(history),
        );

        assert!(system_prompt.contains("步骤修复专家"));
        assert!(user_prompt.contains("执行历史"));
        assert!(user_prompt.contains("tool_x"));
        assert!(user_prompt.contains("tool_y"));
        assert!(user_prompt.contains("必须避免"));
    }

    #[test]
    fn test_prompt_builder_with_scene_guidance() {
        let builder = SingleStepRepairPromptBuilder::new();
        let (system_prompt, _user_prompt) = builder.build_prompt(
            "step_1",
            "测试步骤",
            "test_tool",
            r#"{"param": "value"}"#,
            "工具执行失败",
            "1. tool_a\n2. tool_b",
            Some("工具管理"), // task_type
            None,
        );

        // 应该包含工具管理场景的指导
        assert!(system_prompt.contains("步骤修复专家"));
    }
}
