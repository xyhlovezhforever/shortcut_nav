//! LLM 提示词模板模块
//!
//! 包含用于 Reflective Planning 各个阶段的生产级提示词模板
//!
//! # 重要提示
//!
//! ## 提示词模块化迁移记录
//!
//! ### 2025-12-29: 规划与重新规划提示词迁移
//! - `PLANNING_SYSTEM_PROMPT` 和 `PLANNING_USER_TEMPLATE` 已迁移到 `planning_prompts.rs`
//! - `REPLANNING_SYSTEM_PROMPT` 和 `REPLANNING_USER_TEMPLATE` 已迁移到 `replanning_prompts.rs`
//!
//! ### 2025-12-30: 筛选与评估提示词迁移
//! - `TOOL_SELECTION_SYSTEM_PROMPT` 和 `TOOL_SELECTION_USER_TEMPLATE` 已迁移到 `selection_prompts.rs`
//! - `EVALUATION_SYSTEM_PROMPT` 和 `EVALUATION_USER_TEMPLATE` 已迁移到 `evaluation_prompts.rs`
//! - 用户消息模板已迁移到 `message_templates.rs`
//!
//! ## 迁移原因
//! - 实现场景感知的提示词管理，提高模块内聚性
//! - 每个模块专注于一个职责，便于维护和测试
//! - 为保持向后兼容，从新模块 re-export 这些常量

// Re-export 已迁移的常量以保持向后兼容
pub use crate::llm::planning_prompts::{
    PLANNING_SYSTEM_PROMPT,
    PLANNING_USER_TEMPLATE,
};
pub use crate::llm::replanning_prompts::{
    REPLANNING_SYSTEM_PROMPT,
    REPLANNING_USER_TEMPLATE,
};
pub use crate::llm::selection_prompts::{
    TOOL_SELECTION_SYSTEM_PROMPT,
    TOOL_SELECTION_USER_TEMPLATE,
};
pub use crate::llm::evaluation_prompts::{
    EVALUATION_SYSTEM_PROMPT,
    EVALUATION_USER_TEMPLATE,
};
pub use crate::llm::message_templates::{
    OUTPUT_STYLE_TEMPLATE,
    TOOL_SELECTION_MESSAGE_TEMPLATE,
    TASK_PLANNING_MESSAGE_TEMPLATE,
    REPLANNING_MESSAGE_TEMPLATE,
    CONTINUE_PLANNING_MESSAGE_TEMPLATE,
    STEP_COMPLETED_MESSAGE_TEMPLATE,
    EVALUATION_RESULT_MESSAGE_TEMPLATE,
    TASK_COMPLETION_MESSAGE_TEMPLATE,
    MESSAGE_POLISH_SYSTEM_PROMPT,
    MESSAGE_POLISH_USER_TEMPLATE,
};

/// 提示词模板类型
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    /// 系统提示词
    pub system: String,
    /// 用户提示词模板
    pub user_template: String,
}
// ==================== 筛选提示词已迁移 ====================
// TOOL_SELECTION_SYSTEM_PROMPT 和 TOOL_SELECTION_USER_TEMPLATE 已迁移到 selection_prompts.rs
// 请使用 selection_prompts::SelectionPromptBuilder

// ==================== 规划提示词已迁移 ====================
// PLANNING_SYSTEM_PROMPT 和 PLANNING_USER_TEMPLATE 已迁移到 planning_prompts.rs
// 请使用 planning_prompts::PlanningPromptBuilder
 
 
 
 
// ==================== 评估提示词已迁移 ====================
// EVALUATION_SYSTEM_PROMPT 和 EVALUATION_USER_TEMPLATE 已迁移到 evaluation_prompts.rs
// 请使用 evaluation_prompts::EvaluationPromptBuilder

// ==================== 重新规划提示词已迁移 ====================
// REPLANNING_SYSTEM_PROMPT 和 REPLANNING_USER_TEMPLATE 已迁移到 replanning_prompts.rs
// 请使用 replanning_prompts::ReplanningPromptBuilder



// ==================== 用户友好消息模板已迁移 ====================
// 所有消息模板已迁移到 message_templates.rs
// 请使用 message_templates::MessageTemplateBuilder



/// 提示词构建器
pub struct PromptBuilder;
 
impl PromptBuilder {
    /// 构建规划提示词
    pub fn build_planning_prompt(
        task_description: &str,
        available_tools: &str,
        reflection_history: Option<&str>,
        metadata: &std::collections::HashMap<String, String>,
        context: Option<&crate::grpc::orchestrator::TaskContext>,
        workflow_hint: Option<&str>,
    ) -> (String, String) {
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
 
        let user_prompt = PLANNING_USER_TEMPLATE
            .replace("{task_description}", task_description)
            .replace("{available_tools}", available_tools)
            .replace("{reflection_history}", reflection_history.unwrap_or("无"))
            .replace("{metadata}", &metadata_str)
            .replace("{context}", &context_str)
            .replace("{workflow_hint}", &workflow_str);
 
        (PLANNING_SYSTEM_PROMPT.to_string(), user_prompt)
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
 
    /// 构建评估提示词
    pub fn build_evaluation_prompt(
        task_description: &str,
        execution_plan: &str,
        execution_results: &str,
    ) -> (String, String) {
        // 委托给新模块的构建器
        let builder = crate::llm::evaluation_prompts::EvaluationPromptBuilder::new();
        builder.build_evaluation_prompt(
            task_description,
            execution_plan,
            execution_results,
            None, // 默认不传递任务类型
        )
    }

    /// 构建重新规划提示词（根据失败原因重新规划任务）
    pub fn build_replanning_prompt(
        failure_reason: &str,
        available_tools: &str,
        metadata: &std::collections::HashMap<String, String>,
        workflow_hint: Option<&str>,
    ) -> (String, String) {
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

        // 格式化工作流程提示
        let workflow_str = if let Some(hint) = workflow_hint {
            format!("\n\n【匹配的标准任务流程】\n{}\n", hint)
        } else {
            String::new()
        };

        let user_prompt = REPLANNING_USER_TEMPLATE
            .replace("{failure_reason}", failure_reason)
            .replace("{available_tools}", available_tools)
            .replace("{metadata}", &metadata_str)
            .replace("{workflow_hint}", &workflow_str);

        (REPLANNING_SYSTEM_PROMPT.to_string(), user_prompt)
    }

    /// 构建工具选择提示词（两阶段规划的第一阶段）
    pub fn build_tool_selection_prompt(
        task_description: &str,
        available_tools: &str,
        tool_count: usize,
        metadata: &std::collections::HashMap<String, String>,
        selection_threshold: f32,
    ) -> (String, String) {
        // 委托给新模块的构建器
        let builder = crate::llm::selection_prompts::SelectionPromptBuilder::new();
        builder.build_tool_selection_prompt(
            task_description,
            available_tools,
            tool_count,
            metadata,
            selection_threshold,
            None, // 默认不传递任务类型
        )
    }
}
 
#[cfg(test)]
mod tests {
    use super::*;
 
    #[test]
    fn test_build_planning_prompt() {
        let metadata = std::collections::HashMap::new();
        let (system, user) =
            PromptBuilder::build_planning_prompt("测试任务", "工具1, 工具2", Some("反思内容"), &metadata, None, None);
 
        assert!(system.contains("任务规划专家"));
        assert!(user.contains("测试任务"));
        assert!(user.contains("工具1"));
        assert!(user.contains("反思内容"));
    }
 
    #[test]
    fn test_build_evaluation_prompt() {
        let (system, user) =
            PromptBuilder::build_evaluation_prompt("测试任务", "计划内容", "执行结果");
 
        assert!(system.contains("评估专家"));
        assert!(user.contains("测试任务"));
    }
}
