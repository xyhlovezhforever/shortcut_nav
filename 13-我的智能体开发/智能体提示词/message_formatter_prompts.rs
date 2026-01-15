/// 消息格式化器提示词模块
///
/// 此模块负责生成用户友好的进度报告和消息格式化相关的提示词

// ============================================================================
// 进度报告提示词模板
// ============================================================================

/// 进度报告系统提示词模板
const PROGRESS_REPORT_SYSTEM_TEMPLATE: &str = r#"你是一个智能任务助手，正在为用户执行任务。请生成一个简洁的进度报告。"#;

/// 进度报告任务信息模板
const PROGRESS_REPORT_TASK_INFO_TEMPLATE: &str = r#"任务总体信息：
- 任务描述：{task_description}
- 总步骤数：{total_steps}
- 已完成步骤数：{completed_step_index}
- 剩余步骤数：{remaining_steps}"#;

/// 进度报告已完成步骤模板
const PROGRESS_REPORT_COMPLETED_STEP_TEMPLATE: &str = r#"刚完成的步骤：{last_step_name}"#;

/// 进度报告要求模板
const PROGRESS_REPORT_REQUIREMENTS_TEMPLATE: &str = r#"请生成一个简洁、自然的进度报告，要求：
1. 用1句话说明刚完成了什么，当前完成度
2. 如果还有剩余步骤，简要提及接下来要做什么
3. 语气要积极、鼓励，使用"太好了"、"完美"等词汇
4. 不要使用技术术语，用简单易懂的语言
5. 字数控制在40字以内，直接输出进度报告，不要有其他内容"#;

/// 进度报告示例模板
const PROGRESS_REPORT_EXAMPLE_TEMPLATE: &str = r#"示例格式：
"太好了，「步骤名」成功完成（X/总数），接下来我将为您完成「下一步骤」""#;

// ============================================================================
// 模板填充函数
// ============================================================================

/// 填充任务信息模板
fn fill_task_info(
    task_description: &str,
    total_steps: usize,
    completed_step_index: usize,
    remaining_steps: usize,
) -> String {
    PROGRESS_REPORT_TASK_INFO_TEMPLATE
        .replace("{task_description}", task_description)
        .replace("{total_steps}", &total_steps.to_string())
        .replace("{completed_step_index}", &completed_step_index.to_string())
        .replace("{remaining_steps}", &remaining_steps.to_string())
}

/// 填充已完成步骤模板
fn fill_completed_step(last_step_name: &str) -> String {
    PROGRESS_REPORT_COMPLETED_STEP_TEMPLATE.replace("{last_step_name}", last_step_name)
}

// ============================================================================
// 公共 API
// ============================================================================

/// 构建进度报告的提示词
///
/// # 参数
/// - `task_description`: 任务描述
/// - `total_steps`: 总步骤数
/// - `completed_step_index`: 已完成步骤数
/// - `remaining_steps`: 剩余步骤数
/// - `last_step_name`: 刚完成的步骤名称
/// - `next_step_info`: 下一步骤信息
///
/// # 返回
/// 格式化后的进度报告提示词
pub fn build_progress_report_prompt(
    task_description: &str,
    total_steps: usize,
    completed_step_index: usize,
    remaining_steps: usize,
    last_step_name: &str,
    next_step_info: &str,
) -> String {
    let sections = vec![
        PROGRESS_REPORT_SYSTEM_TEMPLATE.to_string(),
        String::new(),
        fill_task_info(task_description, total_steps, completed_step_index, remaining_steps),
        String::new(),
        fill_completed_step(last_step_name),
        String::new(),
        next_step_info.to_string(),
        String::new(),
        PROGRESS_REPORT_REQUIREMENTS_TEMPLATE.to_string(),
        String::new(),
        PROGRESS_REPORT_EXAMPLE_TEMPLATE.to_string(),
    ];

    sections.join("\n")
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_task_info() {
        let result = fill_task_info("测试任务", 5, 2, 3);
        assert!(result.contains("测试任务"));
        assert!(result.contains("5"));
        assert!(result.contains("2"));
        assert!(result.contains("3"));
    }

    #[test]
    fn test_fill_completed_step() {
        let result = fill_completed_step("数据处理");
        assert!(result.contains("数据处理"));
    }

    #[test]
    fn test_build_progress_report_prompt() {
        let prompt = build_progress_report_prompt(
            "测试任务",
            5,
            2,
            3,
            "数据处理",
            "下一步骤：结果分析",
        );

        // 验证系统提示词
        assert!(prompt.contains("智能任务助手"));

        // 验证任务信息
        assert!(prompt.contains("测试任务"));
        assert!(prompt.contains("5"));
        assert!(prompt.contains("2"));
        assert!(prompt.contains("3"));

        // 验证已完成步骤
        assert!(prompt.contains("数据处理"));

        // 验证下一步信息
        assert!(prompt.contains("下一步骤：结果分析"));

        // 验证要求部分
        assert!(prompt.contains("简洁、自然"));
        assert!(prompt.contains("40字以内"));

        // 验证示例部分
        assert!(prompt.contains("示例格式"));
    }

    #[test]
    fn test_progress_report_structure() {
        let prompt = build_progress_report_prompt(
            "任务A",
            10,
            5,
            5,
            "步骤5",
            "下一步骤：步骤6",
        );

        // 验证各部分按顺序出现
        let system_pos = prompt.find("智能任务助手").unwrap();
        let task_info_pos = prompt.find("任务总体信息").unwrap();
        let completed_pos = prompt.find("刚完成的步骤").unwrap();
        let requirements_pos = prompt.find("请生成一个简洁").unwrap();
        let example_pos = prompt.find("示例格式").unwrap();

        assert!(system_pos < task_info_pos);
        assert!(task_info_pos < completed_pos);
        assert!(completed_pos < requirements_pos);
        assert!(requirements_pos < example_pos);
    }
}
