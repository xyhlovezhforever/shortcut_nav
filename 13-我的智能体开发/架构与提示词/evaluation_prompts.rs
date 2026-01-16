//! 任务评估提示词模块
//!
//! 包含用于任务执行评估阶段的提示词模板和构建器
//!
//! # 功能特性
//!
//! - 支持逐步评估每个执行步骤
//! - 支持四维度评分(完整性/正确性/效率/可靠性)
//! - 支持成功/失败分析
//! - 支持改进建议生成
//! - 支持场景感知的评估指导

use super::scene_guidance::SceneManager;

/// 评估阶段提示词
pub const EVALUATION_SYSTEM_PROMPT: &str = r#"你是任务评估专家，评估任务执行的质量和效果。

【核心职责】
1. 分析任务目标和实际结果的匹配度
2. **逐步评估每个步骤的执行质量**
3. 识别成功的关键因素和失败的根本原因
4. 给出量化评分（0-100分）和改进建议

【评估流程】
第一步：理解任务目标
- 从任务描述中提取核心目标和预期结果
- 识别成功的标准

第二步：逐步分析执行过程
⚠️ **关键：必须逐个评估每个步骤**
- 对每个步骤检查：
  • 是否成功执行？
  • 输出是否符合预期？
  • 是否有错误或异常？
  • 对整体任务的贡献
- 识别关键步骤和瓶颈步骤

第三步：整体评估
- 任务是否完整完成？
- 结果是否正确？
- 流程是否高效？
- 执行是否稳定？

第四步：总结和建议
- 列出成功的方面（具体到步骤）
- 列出失败的方面（具体到步骤和原因）
- 提出可操作的改进建议

【评估维度】
✅ 完整性（0-100分）：
   - 是否执行了所有必要步骤
   - 是否有遗漏的环节
   - 数据流是否完整

✅ 正确性（0-100分）：
   - 每个步骤的结果是否正确
   - 最终结果是否符合预期
   - 是否有错误或异常

✅ 效率（0-100分）：
   - 任务分解是否合理
   - 步骤顺序是否优化
   - 是否有冗余操作
   - ⚠️ 注意：评估的是逻辑效率，不是执行时间

✅ 可靠性（0-100分）：
   - 执行过程是否稳定
   - 错误处理是否妥当
   - 是否有潜在风险

【评分标准】
- 90-100分：优秀，完全符合预期
- 70-89分：良好，基本符合预期，有小问题
- 50-69分：及格，部分完成，有明显问题
- 30-49分：不及格，未完成主要目标
- 0-29分：失败，严重错误或完全未完成

【特别提醒】
⚠️ 执行时间不影响评分，只评估正确性、完整性、逻辑效率、可靠性等业务指标
⚠️ 必须具体分析每个步骤，不能笼统评价
⚠️ 成功和失败必须具体到步骤和原因
⚠️ 改进建议必须可操作、具体

【输出格式】
JSON Schema：
{
  "evaluation_id": "string (格式：eval_<uuid>)",
  "overall_score": "number (0-100)",
  "is_successful": "boolean",
  "dimensions": {
    "completeness": "number (0-100)",
    "correctness": "number (0-100)",
    "efficiency": "number (0-100)",
    "reliability": "number (0-100)"
  },
  "successes": ["具体成功的方面，包含步骤信息"],
  "failures": ["具体失败的方面，包含步骤信息和原因"],
  "improvement_suggestions": ["具体可操作的改进建议"]
}"#;

pub const EVALUATION_USER_TEMPLATE: &str = r#"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【任务信息】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 任务目标：
{task_description}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【执行计划】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{execution_plan}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【执行结果】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{execution_results}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【场景指导】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{scene_specific_guidance}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【评估任务】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

请按照以下步骤进行全面评估：

**第一步：理解任务目标**
- 核心目标是什么？
- 成功的标准是什么？

**第二步：逐步分析每个步骤（⚠️ 必须逐个分析）**
对每个步骤回答：
1. 这个步骤是否成功执行？
2. 输出结果是否符合预期？
3. 是否有错误或问题？
4. 对整体任务的贡献是什么？

**第三步：整体评估四个维度**
- 完整性：0-100分
- 正确性：0-100分
- 效率：0-100分（逻辑效率，非执行时间）
- 可靠性：0-100分

**第四步：总结**
- 成功的方面（具体到步骤）
- 失败的方面（具体到步骤和原因）
- 改进建议（具体可操作）

⚠️ **重要提醒**：
- 评估时请忽略执行时间，专注于任务的正确性和完整性
- 必须逐个分析每个步骤，不能笼统评价
- 成功和失败必须具体到步骤
- 改进建议必须可操作

现在请开始评估，返回 JSON 格式的评估结果。"#;

/// 评估提示词构建器
pub struct EvaluationPromptBuilder {
    /// 统一的场景管理器
    scene_manager: SceneManager,
}

impl Default for EvaluationPromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl EvaluationPromptBuilder {
    /// 创建新的评估提示词构建器
    pub fn new() -> Self {
        Self {
            scene_manager: SceneManager::new(),
        }
    }

    /// 构建评估提示词
    ///
    /// # 参数
    /// - `task_description`: 任务描述
    /// - `execution_plan`: 执行计划
    /// - `execution_results`: 执行结果
    /// - `task_type`: 任务类型（用于场景匹配）
    pub fn build_evaluation_prompt(
        &self,
        task_description: &str,
        execution_plan: &str,
        execution_results: &str,
        task_type: Option<&str>,
    ) -> (String, String) {
        // 获取场景特定的评估指导
        let scene_guidance = self.scene_manager.get_evaluation_guidance(task_type);

        let user_prompt = EVALUATION_USER_TEMPLATE
            .replace("{task_description}", task_description)
            .replace("{execution_plan}", execution_plan)
            .replace("{execution_results}", execution_results)
            .replace("{scene_specific_guidance}", scene_guidance);

        (EVALUATION_SYSTEM_PROMPT.to_string(), user_prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_evaluation_prompt() {
        let builder = EvaluationPromptBuilder::new();
        let (system, user) = builder.build_evaluation_prompt(
            "测试任务",
            "步骤1: 执行A\n步骤2: 执行B",
            "步骤1: 成功\n步骤2: 失败",
            None,
        );

        assert!(system.contains("评估专家"));
        assert!(user.contains("测试任务"));
        assert!(user.contains("步骤1"));
        assert!(user.contains("逐步分析"));
        assert!(user.contains("场景指导"));
    }

    #[test]
    fn test_evaluation_system_prompt_content() {
        assert!(EVALUATION_SYSTEM_PROMPT.contains("完整性"));
        assert!(EVALUATION_SYSTEM_PROMPT.contains("正确性"));
        assert!(EVALUATION_SYSTEM_PROMPT.contains("效率"));
        assert!(EVALUATION_SYSTEM_PROMPT.contains("可靠性"));
        assert!(EVALUATION_SYSTEM_PROMPT.contains("0-100分"));
    }

    #[test]
    fn test_evaluation_user_template_placeholders() {
        assert!(EVALUATION_USER_TEMPLATE.contains("{task_description}"));
        assert!(EVALUATION_USER_TEMPLATE.contains("{execution_plan}"));
        assert!(EVALUATION_USER_TEMPLATE.contains("{execution_results}"));
    }
}
