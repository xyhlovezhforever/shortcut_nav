/// AI 报告器提示词模块
///
/// 此模块负责生成 AI 报告器相关的提示词
/// 用于任务编排助手的各类输出生成

/// AI 报告器的系统提示词
pub const AI_REPORTER_SYSTEM_PROMPT: &str =
    "你是一个专业的任务编排助手。请严格按照要求生成输出，不要添加额外的解释或前缀。";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ai_reporter_system_prompt() {
        assert!(AI_REPORTER_SYSTEM_PROMPT.contains("任务编排助手"));
        assert!(AI_REPORTER_SYSTEM_PROMPT.contains("严格按照要求"));
    }
}
