//! 任务后评估提示词模块
//!
//! 在任务执行完成后，对整个任务进行全局性的评估分析
//! 用于积累经验、优化后续任务执行策略
//!
//! # 功能特性
//!
//! - 从全局视角评估整个任务的执行质量
//! - 提取可复用的成功模式和失败教训
//! - 分析任务分解策略的合理性
//! - 评估工具选择的适配性
//! - 生成可供后续任务参考的经验总结

use serde::{Deserialize, Serialize};

/// 任务后评估系统提示词
pub const POST_TASK_EVALUATION_SYSTEM_PROMPT: &str = r#"你是任务执行经验提炼专家，专注于从已完成任务中提取【最有价值的一条经验】。

【核心目标】
从本次任务执行中提炼出**唯一一条最核心、最有指导意义**的经验，可以是解决思路或反思建议。

【关键要求】
⚠️ **只生成1条经验** - 不要贪多，只选最突出、最有效的那一条
⚠️ **质量优于数量** - 宁可一条精华，不要三条凑数

【经验提炼原则】

📌 **具体化原则**
- 不要笼统的描述，要具体到"做什么"、"怎么做"、"为什么这样做"
- 经验必须可以直接理解和应用，而非需要再次思考
- 反思必须明确指出"问题是什么"、"原因是什么"、"应该怎么改进"

📌 **场景绑定原则**
- 每条经验必须绑定具体的任务和意图
- 明确说明在什么场景下使用什么方法
- 便于后续相似任务参考和学习

📌 **价值优先原则**
- solving_ideas（解决思路）：当任务成功或找到有效方法时选择
- reflection_suggestions（反思建议）：当任务失败或发现系统性问题时选择
- 只选择对后续任务最有帮助的那一个类别

【经验提取格式】

💡 **解决思路（solving_ideas）**
记录内容：
- 我遇到了什么问题
- 我的解决思路是什么
- 我采用了什么方法/工具
- 为什么这个方法有效
- 具体的实施步骤

🤔 **反思建议（reflection_suggestions）**
记录内容：
- 关于这个任务的整体反思
- 发现的问题和不足
- 可以改进的地方
- 下次应该注意什么
- 对类似任务的建议

【评估维度】（简化打分，重点在经验提取）

- 任务完成度（0-100）：是否达成核心目标
- 执行效率（0-100）：是否有冗余步骤或无效尝试
- 工具使用（0-100）：工具选择和使用是否得当

【输出要求】
1. 经验必须【具体、可理解、可参考】
2. 每条经验都要有【真实案例支撑】
3. 经验描述要【自然流畅、易于阅读】
4. 避免空泛的总结，聚焦于【有价值的思路和反思】"#;

/// 任务后评估用户提示词模板
pub const POST_TASK_EVALUATION_USER_TEMPLATE: &str = r#"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【任务基本信息】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 任务描述：
{task_description}

🎯 任务类型：{task_type}

⏱️  总执行轮次：{total_rounds} 轮
✅ 最终状态：{final_status}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【任务上下文】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{task_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【完整执行历史】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{execution_history}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【各轮次统计】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{round_statistics}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【关键事件摘要】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{key_events}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【经验提炼任务】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

请基于以上执行历史，提炼出【解决思路】和【反思建议】：

**第一步：快速评估**
对任务完成度、执行效率、工具使用三个维度打分（0-100）

**第二步：提炼最有价值的经验（只选1条）**
从以下两个维度中选择最有价值的那一条：

选项A：解决思路（solving_ideas）
- 明确遇到了什么问题
- 说明采用了什么解决思路
- 描述具体的实施方法和步骤
- 解释为什么这个方法有效

选项B：反思建议（reflection_suggestions）
- 任务执行过程中的思考和感悟
- 发现的问题和可以改进的地方
- 对后续类似任务的建议
- 需要注意的关键点

**评选标准：选择对后续任务最有指导意义、最具操作性的那一条**
- 如果任务中有明确的解决方案且成功，优先选择 solving_ideas
- 如果任务失败或发现系统性问题，优先选择 reflection_suggestions
- 只输出1条经验，不要贪多

**第三步：使用指定的任务意图和任务描述**
- 本次任务的意图类型为：{user_intention}
- 本次任务的原始描述为：{task_description}
请在生成的经验中：
1. intention 字段使用上述意图类型
2. task 字段使用上述任务描述（不包含意图前缀，只保留任务描述部分）

现在请开始提炼，返回 JSON 格式的结果（按照下方的 schema）。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【输出 JSON Schema】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{
  "evaluation_id": "string (格式：post_eval_<uuid>)",
  "task_id": "string",
  "scores": {
    "task_completion": "number (0-100) 任务完成度",
    "execution_efficiency": "number (0-100) 执行效率",
    "tool_usage": "number (0-100) 工具使用得当程度"
  },
  "intention": "string (任务意图类型，如：plc控制器、自动建模、工具管理、测试验证等，使用上面提供的意图类型)",
  "category": "string (类别：solving_ideas 或 reflection_suggestions，只选一个最有价值的)",
  "task": "string (用户提交的原始任务描述，不包含意图前缀，直接使用上面提供的任务描述)",
  "experience": "string (经验内容，详细描述解决思路或反思建议，必须具体、完整、可操作)",
  "quick_summary": "string (一句话总结本次任务最重要的经验)"
}

【经验内容示例】

solving_ideas 示例：
"我解决了一个xxx问题，遇到的具体情况是xxx，我的解决思路是xxx，采用了xxx方法/工具，具体步骤是xxx，这个方法有效的原因是xxx。"

reflection_suggestions 示例：
"关于这个任务我的反思是：在执行过程中发现了xxx问题，主要原因是xxx，可以改进的地方是xxx，下次遇到类似任务应该注意xxx，我的建议是xxx。"

【注意事项】
1. 每个任务只生成 1 条经验（从 solving_ideas 或 reflection_suggestions 中选择最有价值的那一条）
2. 优先选择对后续任务最有指导意义的经验（solving_ideas 优先，但如果反思更有价值则选择 reflection_suggestions）
3. intention 字段必须使用上面提供的意图类型，不要自行判断
4. task 字段必须使用上面提供的原始任务描述，不要自行总结或修改
5. experience 字段要自然流畅，像是在讲述一个完整的故事，包含问题、思路、方法和原因
6. 避免空洞的描述，经验必须具体、可操作、有实际案例支撑"#;

/// 任务后评估结果（经验提炼）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostTaskEvaluation {
    /// 评估 ID
    pub evaluation_id: String,
    /// 任务 ID
    pub task_id: String,
    /// 评分
    pub scores: EvaluationScores,
    /// 单个经验对象（最新格式：直接作为字段）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experience: Option<Experience>,
    /// 可复用的最佳实践（旧格式，保留兼容性）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub best_practices: Vec<BestPractice>,
    /// 必须规避的反模式（旧格式，保留兼容性）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub anti_patterns: Vec<AntiPattern>,
    /// 工具使用备忘录（旧格式，保留兼容性）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_memos: Vec<ToolMemo>,
    /// 经验列表（旧数组格式，保留兼容性）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub experiences: Vec<Experience>,
    /// 一句话总结
    pub quick_summary: String,
}

/// 评估评分
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationScores {
    /// 任务完成度 (0-100)
    pub task_completion: f32,
    /// 执行效率 (0-100)
    pub execution_efficiency: f32,
    /// 工具使用得当程度 (0-100)
    pub tool_usage: f32,
}

/// 可复用的最佳实践
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestPractice {
    /// 触发场景：什么情况下使用这条经验
    pub scenario: String,
    /// 具体做法：详细的执行步骤/命令/参数
    pub action: String,
    /// 原因：为什么这样做有效
    pub reason: String,
    /// 案例：本次任务中的实际案例
    pub example: String,
}

/// 必须规避的反模式
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntiPattern {
    /// 触发场景：什么情况下容易犯这个错误
    pub scenario: String,
    /// 禁止做法：明确禁止的操作
    pub forbidden: String,
    /// 后果：这样做会导致什么问题
    pub consequence: String,
    /// 正确做法：应该怎么做
    pub correct_way: String,
    /// 案例：本次任务中的实际教训
    pub example: String,
}

/// 工具使用备忘录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolMemo {
    /// 工具ID/名称
    pub tool_id: String,
    /// 最佳用法：推荐的参数配置和调用方式
    pub best_usage: String,
    /// 常见陷阱：容易犯的错误
    pub pitfalls: String,
    /// 适用场景：什么时候用/不用这个工具
    pub applicable_scenarios: String,
}

/// 经验条目（新格式）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    /// 意图类型（如：plc控制器、自动建模等）
    pub intention: String,
    /// 类别（solving_ideas 或 reflection_suggestions）
    pub category: String,
    /// 任务描述
    pub task: String,
    /// 经验内容
    pub experience: String,
}

impl PostTaskEvaluation {
    /// 将评估结果格式化为 Markdown 格式
    pub fn to_markdown(&self, task_description: &str) -> String {
        let mut md = String::new();

        // 标题和基本信息
        md.push_str(&format!("# 任务评估报告\n\n"));
        md.push_str(&format!("**任务ID**: `{}`\n\n", self.task_id));
        md.push_str(&format!("**任务描述**: {}\n\n", task_description));
        md.push_str(&format!("**评估ID**: `{}`\n\n", self.evaluation_id));

        // 评分
        md.push_str("## 评分\n\n");
        md.push_str(&format!("- **任务完成度**: {:.1}/100\n", self.scores.task_completion));
        md.push_str(&format!("- **执行效率**: {:.1}/100\n", self.scores.execution_efficiency));
        md.push_str(&format!("- **工具使用**: {:.1}/100\n\n", self.scores.tool_usage));

        // 一句话总结
        md.push_str("## 总结\n\n");
        md.push_str(&format!("{}\n\n", self.quick_summary));

        // 经验记录（新格式：单个对象）
        if let Some(exp) = &self.experience {
            md.push_str("## 经验记录\n\n");

            let category_label = match exp.category.as_str() {
                "solving_ideas" => "💡 解决思路",
                "reflection_suggestions" => "🤔 反思建议",
                _ => "📝 经验",
            };

            md.push_str(&format!("### {} ({})\n\n", category_label, exp.intention));
            md.push_str(&format!("**任务**: {}\n\n", exp.task));
            md.push_str(&format!("{}\n\n", exp.experience));
        }
        // 兼容旧格式：经验列表（数组格式）
        else if !self.experiences.is_empty() {
            md.push_str("## 经验记录\n\n");

            // 按类别分组
            let mut solving_ideas: Vec<&Experience> = Vec::new();
            let mut reflection_suggestions: Vec<&Experience> = Vec::new();

            for exp in &self.experiences {
                match exp.category.as_str() {
                    "solving_ideas" => solving_ideas.push(exp),
                    "reflection_suggestions" => reflection_suggestions.push(exp),
                    _ => {}
                }
            }

            // 解决思路
            if !solving_ideas.is_empty() {
                md.push_str("### 💡 解决思路\n\n");
                for (idx, exp) in solving_ideas.iter().enumerate() {
                    md.push_str(&format!("#### {}.  {} ({})\n\n", idx + 1, exp.task, exp.intention));
                    md.push_str(&format!("{}\n\n", exp.experience));
                }
            }

            // 反思建议
            if !reflection_suggestions.is_empty() {
                md.push_str("### 🤔 反思建议\n\n");
                for (idx, exp) in reflection_suggestions.iter().enumerate() {
                    md.push_str(&format!("#### {}. {} ({})\n\n", idx + 1, exp.task, exp.intention));
                    md.push_str(&format!("{}\n\n", exp.experience));
                }
            }
        }

        // 分隔线
        md.push_str("\n---\n\n");

        // 附录：原始 JSON 数据
        md.push_str("## 📄 原始 JSON 数据\n\n");
        md.push_str("```json\n");

        // 序列化为格式化的 JSON
        match serde_json::to_string_pretty(self) {
            Ok(json_str) => {
                md.push_str(&json_str);
            }
            Err(e) => {
                md.push_str(&format!("// JSON 序列化失败: {}\n", e));
            }
        }

        md.push_str("\n```\n\n");
        md.push_str("---\n\n");

        md
    }
}

/// 任务后评估提示词构建器
pub struct PostTaskEvaluationPromptBuilder;

impl PostTaskEvaluationPromptBuilder {
    /// 构建任务后评估提示词
    ///
    /// # 参数
    /// - `task_id`: 任务ID
    /// - `task_description`: 任务描述
    /// - `task_type`: 任务类型
    /// - `task_context`: 任务上下文
    /// - `execution_history`: 执行历史
    /// - `total_rounds`: 总执行轮次
    /// - `final_status`: 最终状态
    /// - `round_statistics`: 各轮次统计信息
    /// - `key_events`: 关键事件摘要
    /// - `user_intention`: 用户意图（从 metadata 中获取）
    pub fn build_prompt(
        _task_id: &str,
        task_description: &str,
        task_type: Option<&str>,
        task_context: Option<&str>,
        execution_history: &str,
        total_rounds: u32,
        final_status: &str,
        round_statistics: &str,
        key_events: &str,
        user_intention: Option<&str>,
    ) -> (String, String) {
        let task_type_str = task_type.unwrap_or("未指定");
        let context_str = task_context.unwrap_or("（无额外上下文）");
        let intention_str = user_intention.unwrap_or("未指定");

        let user_prompt = POST_TASK_EVALUATION_USER_TEMPLATE
            .replace("{task_description}", task_description)
            .replace("{task_type}", task_type_str)
            .replace("{task_context}", context_str)
            .replace("{execution_history}", execution_history)
            .replace("{total_rounds}", &total_rounds.to_string())
            .replace("{final_status}", final_status)
            .replace("{round_statistics}", round_statistics)
            .replace("{key_events}", key_events)
            .replace("{user_intention}", intention_str);

        (POST_TASK_EVALUATION_SYSTEM_PROMPT.to_string(), user_prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_prompt() {
        let (system, user) = PostTaskEvaluationPromptBuilder::build_prompt(
            "task_123",
            "测试任务",
            Some("自动建模"),
            Some("测试上下文"),
            "执行历史...",
            3,
            "成功",
            "轮次统计...",
            "关键事件...",
        );

        assert!(system.contains("任务执行经验提炼专家"));
        assert!(system.contains("可直接复用的经验"));
        assert!(system.contains("必须规避的教训"));
        assert!(user.contains("测试任务"));
        assert!(user.contains("自动建模"));
        assert!(user.contains("3 轮"));
        assert!(user.contains("成功"));
    }

    #[test]
    fn test_prompt_structure() {
        // 验证系统提示词包含核心概念
        assert!(POST_TASK_EVALUATION_SYSTEM_PROMPT.contains("Best Practices"));
        assert!(POST_TASK_EVALUATION_SYSTEM_PROMPT.contains("Anti-Patterns"));
        assert!(POST_TASK_EVALUATION_SYSTEM_PROMPT.contains("工具使用备忘录"));
        assert!(POST_TASK_EVALUATION_SYSTEM_PROMPT.contains("具体化原则"));
        assert!(POST_TASK_EVALUATION_SYSTEM_PROMPT.contains("场景绑定原则"));

        // 验证用户模板包含必要占位符
        assert!(POST_TASK_EVALUATION_USER_TEMPLATE.contains("{task_description}"));
        assert!(POST_TASK_EVALUATION_USER_TEMPLATE.contains("{execution_history}"));
        assert!(POST_TASK_EVALUATION_USER_TEMPLATE.contains("best_practices"));
        assert!(POST_TASK_EVALUATION_USER_TEMPLATE.contains("anti_patterns"));
        assert!(POST_TASK_EVALUATION_USER_TEMPLATE.contains("tool_memos"));
    }
}
