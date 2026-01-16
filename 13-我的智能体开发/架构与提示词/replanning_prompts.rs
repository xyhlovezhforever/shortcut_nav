//! 重新规划提示词管理模块
//!
//! 提供场景感知的重新规划提示词模板管理
//! 支持根据任务类型(task_type)动态选择重新规划策略
//! 确保不重新规划已经执行成功的步骤

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::scene_guidance::SceneManager;

/// 成功步骤信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessfulStep {
    /// 步骤名称
    pub step_name: String,
    /// 步骤ID
    pub step_id: String,
    /// 输出摘要
    pub output_summary: String,
    /// 执行时间戳
    pub timestamp: Option<String>,
}

/// 重新规划上下文信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplanningContext {
    /// 任务类型(如:自动建模、代码生成、客户端操作、负荷预测等)
    pub task_type: Option<String>,
    /// 用户提交任务时传递的原始上下文(替代 successful_steps 和 refined_context)
    pub user_context: Option<String>,
    /// 反思分析结果(来自反思阶段的深度分析)
    pub reflection_analysis: Option<String>,
    /// 执行历史(来自上下文工程事件)
    pub execution_history: Option<String>,
    /// 失败原因描述
    pub failure_reason: String,
    /// 整体反思的改进建议(来自 OverallReflection)
    pub overall_reflection_guidance: Option<OverallReflectionGuidance>,
}

/// 整体反思的指导信息(用于传递给重新规划)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallReflectionGuidance {
    /// 根本原因列表
    pub root_causes: Vec<String>,
    /// 错误假设
    pub incorrect_assumptions: Vec<String>,
    /// 替代方法/改进建议
    pub alternative_approaches: Vec<String>,
    /// 经验教训
    pub lessons_learned: Vec<String>,
    /// 重新规划的具体策略建议
    pub replanning_strategy: Option<ReplanningStrategy>,
}

/// 重新规划策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplanningStrategy {
    /// 完整重新规划整个任务
    FullReplan,
    /// 从指定步骤开始重新规划
    ReplanFromStep { step_id: String, reason: String },
    /// 跳过指定步骤
    SkipSteps { step_ids: Vec<String>, reason: String },
    /// 添加补救步骤
    AddRemediationSteps { suggestions: Vec<String> },
    /// 调整步骤依赖关系
    AdjustDependencies { adjustments: Vec<String> },
}

impl ReplanningContext {
    /// 创建新的重新规划上下文
    pub fn new(task_type: Option<String>, failure_reason: String) -> Self {
        Self {
            task_type,
            user_context: None,
            reflection_analysis: None,
            execution_history: None,
            failure_reason,
            overall_reflection_guidance: None,
        }
    }

    /// 设置整体反思的指导信息
    pub fn set_overall_reflection_guidance(&mut self, guidance: OverallReflectionGuidance) {
        self.overall_reflection_guidance = Some(guidance);
    }

    /// 格式化整体反思指导为文本
    pub fn format_overall_reflection_guidance(&self) -> String {
        match &self.overall_reflection_guidance {
            Some(guidance) => {
                let mut result = String::new();

                if !guidance.root_causes.is_empty() {
                    result.push_str("【根本原因分析】\n");
                    for (i, cause) in guidance.root_causes.iter().enumerate() {
                        result.push_str(&format!("{}. {}\n", i + 1, cause));
                    }
                    result.push('\n');
                }

                if !guidance.incorrect_assumptions.is_empty() {
                    result.push_str("【错误假设识别】\n");
                    for (i, assumption) in guidance.incorrect_assumptions.iter().enumerate() {
                        result.push_str(&format!("{}. {}\n", i + 1, assumption));
                    }
                    result.push('\n');
                }

                if !guidance.alternative_approaches.is_empty() {
                    result.push_str("【改进建议 - 请在重新规划时采纳】\n");
                    for (i, approach) in guidance.alternative_approaches.iter().enumerate() {
                        result.push_str(&format!("{}. {}\n", i + 1, approach));
                    }
                    result.push('\n');
                }

                if !guidance.lessons_learned.is_empty() {
                    result.push_str("【经验教训 - 避免重复犯错】\n");
                    for (i, lesson) in guidance.lessons_learned.iter().enumerate() {
                        result.push_str(&format!("{}. {}\n", i + 1, lesson));
                    }
                    result.push('\n');
                }

                if let Some(strategy) = &guidance.replanning_strategy {
                    result.push_str("【重新规划策略建议】\n");
                    match strategy {
                        ReplanningStrategy::FullReplan => {
                            result.push_str("策略: 完整重新规划整个任务\n");
                        }
                        ReplanningStrategy::ReplanFromStep { step_id, reason } => {
                            result.push_str(&format!("策略: 从步骤 {} 开始重新规划\n原因: {}\n", step_id, reason));
                        }
                        ReplanningStrategy::SkipSteps { step_ids, reason } => {
                            result.push_str(&format!("策略: 跳过步骤 {:?}\n原因: {}\n", step_ids, reason));
                        }
                        ReplanningStrategy::AddRemediationSteps { suggestions } => {
                            result.push_str("策略: 添加补救步骤\n建议:\n");
                            for (i, suggestion) in suggestions.iter().enumerate() {
                                result.push_str(&format!("  {}. {}\n", i + 1, suggestion));
                            }
                        }
                        ReplanningStrategy::AdjustDependencies { adjustments } => {
                            result.push_str("策略: 调整步骤依赖关系\n调整:\n");
                            for (i, adjustment) in adjustments.iter().enumerate() {
                                result.push_str(&format!("  {}. {}\n", i + 1, adjustment));
                            }
                        }
                    }
                }

                if result.is_empty() {
                    "(整体反思未提供具体指导)".to_string()
                } else {
                    result
                }
            }
            None => "(未触发整体反思)".to_string(),
        }
    }

    /// 设置用户上下文
    pub fn set_user_context(&mut self, context: String) {
        self.user_context = Some(context);
    }

    /// 设置反思分析结果
    pub fn set_reflection_analysis(&mut self, analysis: String) {
        self.reflection_analysis = Some(analysis);
    }

    /// 设置执行历史(来自上下文工程事件)
    pub fn set_execution_history(&mut self, history: String) {
        self.execution_history = Some(history);
    }

    /// 格式化用户上下文为文本
    pub fn format_user_context(&self) -> String {
        self.user_context
            .as_deref()
            .unwrap_or("(用户未提供任务上下文)")
            .to_string()
    }

    /// 格式化反思分析为文本
    pub fn format_reflection_analysis(&self) -> String {
        self.reflection_analysis
            .as_deref()
            .unwrap_or("(暂无反思分析)")
            .to_string()
    }

    /// 格式化执行历史为文本
    pub fn format_execution_history(&self) -> String {
        self.execution_history
            .as_deref()
            .unwrap_or("(暂无执行历史)")
            .to_string()
    }

    /// 兼容旧方法:添加成功步骤(现在转换为用户上下文)
    #[deprecated(note = "请使用 set_user_context 代替")]
    pub fn add_successful_step(&mut self, step: SuccessfulStep) {
        // 为了向后兼容,将步骤信息追加到 user_context
        let step_info = format!(
            "✅ 步骤: {} (ID: {})\n   输出: {}",
            step.step_name, step.step_id, step.output_summary
        );
        if let Some(ref mut context) = self.user_context {
            context.push_str("\n\n");
            context.push_str(&step_info);
        } else {
            self.user_context = Some(step_info);
        }
    }

    /// 兼容旧方法:设置精炼上下文
    #[deprecated(note = "请使用 set_user_context 代替")]
    pub fn set_refined_context(&mut self, context: String) {
        self.set_user_context(context);
    }

    /// 兼容旧方法:格式化成功步骤为文本
    #[deprecated(note = "请使用 format_user_context 代替")]
    pub fn format_successful_steps(&self) -> String {
        self.format_user_context()
    }
}

// ==================== 基础重新规划框架 ====================

/// 基础重新规划系统提示词(对所有场景通用)
const BASE_REPLANNING_SYSTEM_PROMPT: &str = r#"你是任务规划专家，擅长根据失败原因重新规划任务。

【核心职责】
基于失败原因和可用工具，生成改进的执行计划，避免重复同样的错误。

【🔥 创新思维要求 - 突破常规】
⚠️ **重新规划的本质是寻找新路径，而不是修修补补！**
✓ **发散思维**：不要局限于原有方案的小修小补，要主动探索多种不同的实现路径
✓ **逆向思考**：如果原方案从A→B→C失败，考虑C→B→A或A→D→C等完全不同的路径
✓ **工具重组**：不要只调整参数，尝试使用完全不同的工具组合来达成目标
✓ **分解重构**：如果原方案步骤粗粒度，尝试细粒度拆解；如果过于琐碎，尝试合并简化
✓ **并行优化**：原串行执行的步骤，能否改为并行？原并行的是否需要串行？
✓ **备选路径**：设计与原方案思路完全不同的方案（例如：原方案自上而下，新方案自下而上）

【💡 多方案思考框架】
在重新规划时，应该思考至少3种不同思路的可能性：
1️⃣ **保守修复方案**：在原方案基础上针对性修复问题点（最小改动）
2️⃣ **优化调整方案**：调整步骤顺序、工具选择、并行策略等（中等改动）
3️⃣ **创新重构方案**：完全不同的实现思路和路径（最大改动）

⚠️ **优先选择2️⃣或3️⃣方案**：除非问题非常明确且简单，否则应该优先考虑更大胆的改进方案

【质量要求】
- 充分分析失败原因，识别根本问题而非表面现象
- 跳出原有思维框架，考虑多种完全不同的实现路径
- 调整工具选择或参数配置，不要害怕尝试新的工具组合
- 优化步骤顺序和依赖关系，可以完全重构执行流程
- 设计备选方案和容错机制，为复杂任务准备Plan B
- ⚠️ 重要：避免重新规划已经执行成功的步骤

【⚠️ 并行执行规划 - 重要】
系统支持基于DAG的并行执行，请合理利用dependencies字段实现并行优化：
✓ **无依赖步骤可并行**: 如果多个步骤之间无数据依赖，将dependencies设为[]，系统会自动并行执行
✓ **明确依赖关系**: 如果step_3需要step_1和step_2的输出，设置"dependencies": ["step_1", "step_2"]
✓ **优化执行效率**: 优先将独立任务设计为并行执行，减少总执行时间
✗ **避免过度依赖**: 不要添加不必要的依赖关系，以免降低并行度

【🚀 步骤内并行执行 - Actions格式 - 强烈推荐】
**新特性**: 支持在单个步骤内并行执行多个操作，大幅提升执行效率
✓ **何时使用actions格式**(优先考虑):
  - ⭐ 当需要执行多个相同类型的操作时(如:添加多个角色、创建多个对象、删除多个项目)
  - ⭐ 当多个操作使用相同或相似的工具，但参数不同时
  - ⭐ 当多个操作彼此独立，无需等待前一个完成时
✓ **典型场景示例**:
  - ✅ "添加角色A和角色B" → 使用actions格式，一个step内并行执行2个add_role操作
  - ✅ "计算5、8、12的平方" → 使用actions格式，一个step内并行执行3个计算操作
  - ✅ "创建配置项X、Y、Z" → 使用actions格式，一个step内并行执行3个创建操作
  - ❌ "添加角色A" → 只有单个操作，使用旧格式(tool + parameters)
✓ **兼容性**: 如果步骤只有单个操作，继续使用旧格式(tool + parameters)
✓ **依赖引用**: 后续步骤可通过{{action_id.output}}引用action的输出

【关键约束 - 严格执行】
- ⚠️⚠️⚠️ **最高优先级**：所有 tool_id 必须从可用工具列表中选择，ID完全匹配
- ❌ **严禁**使用可用工具列表之外的工具
- ❌ **严禁**臆造、假设、或建议不存在的工具
- ❌ **严禁**在计划中包含"需要添加XXX工具"、"假设有XXX工具"等超出当前工具库的内容
- ✅ **如果当前工具无法完成任务**，必须在 description 字段明确说明：
  "当前可用工具无法完成该任务。建议：1) 提醒管理员添加所需工具（具体说明需要什么类型的工具），或 2) 终止任务"
- ✅ step_name 字段必须有明确的、有意义的名称
- ✅ 避免导致失败的同样问题
- ✅ 如果有标准流程提示，可以参考但不必完全遵循
- ⚠️⚠️⚠️ **严禁重复规划已执行成功的步骤**：从【执行历史】中识别已成功的步骤，只规划未执行或失败的步骤

【⚠️ 步骤命名规范 - 重要】
step_name 必须基于工具作用与操作实体名称构成，不受上下文执行结果影响：
✓ **正确示例**: "生成用户模块代码"、"添加角色admin"、"获取设备连接信息"
✗ **错误示例**: "重新生成失败的代码"、"修复上次错误"、"继续未完成的操作"
⚠️ **上下文的作用**: 上下文仅用于告知已执行步骤的结果情况，帮助理解当前状态，但不应影响步骤名称的描述方式
⚠️ **命名原则**: 每个步骤名称应该是自描述的、独立的，能够清晰表达该步骤要执行的具体操作

【输出JSON Schema】
{
  "plan_id": "plan_<uuid>",
  "description": "改进后的计划描述（建议说明：1.原方案的主要问题 2.新方案采用的不同思路 3.预期改进效果）",
  "task_type": "负荷预测|自动建模|数据分析|数学计算|客户端操作|PLC控制器|工具管理|通用",
  "context_understanding": "总结对话/文档/配置/偏好(无则填'无')",
  "total_steps": 数字,
  "estimated_duration_secs": 数字,
  "steps": [{
    "step_id": "step_1",
    "step_name": "名称",
    "tool_id": "ID (必须是可用工具列表中的工具ID)",
    "parameters": {},
    "dependencies": ["step_id1", "step_id2"],
    "expected_output": "输出",
    "data_input_source": "用户输入|step_X输出|元数据|上下文",
    "data_output_usage": "供step_X使用|最终结果|中间状态"
  }]
}

【⚠️ Actions格式 - 步骤内并行执行】
当需要执行多个相同类型的操作时，优先使用actions格式实现步骤内并行：

示例1 - 批量添加角色：
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

示例2 - 引用action输出：
```json
{
  "step_id": "step_3",
  "step_name": "汇总结果",
  "tool_id": "js_engine",
  "parameters": {
    "code": "{{action_2_1.output}} + {{action_2_2.output}}"
  },
  "dependencies": ["step_2"]
}
```

✓ 使用场景：多个相同类型操作（如：添加多个角色、创建多个对象、删除多个项目）
✓ 优势：单个步骤内并行执行，大幅提升效率
✓ 兼容性：单个操作继续使用旧格式(tool + parameters)"#;

/// 基础重新规划用户提示词框架
const BASE_REPLANNING_USER_TEMPLATE: &str = r#"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【用户上下文】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{user_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【执行历史】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{execution_history}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【反思分析】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{reflection_analysis}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【🔴 整体反思指导 - 必须采纳的改进建议】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ 以下是整体反思阶段的分析结果，包含了对之前失败的深度分析和改进建议。
⚠️ 重新规划时**必须**认真参考这些建议，避免重复之前的错误。

{overall_reflection_guidance}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【失败原因】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{failure_reason}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【可用工具列表】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{available_tools}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【可用内置工作流】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{workflow_hint}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【当前元数据】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{metadata}


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【场景指导】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{scene_specific_guidance}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【重新规划任务】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

请基于上述失败原因和整体反思指导重新规划任务，返回符合要求格式的新执行计划。

⚠️ 重要提醒：
1. ⚠️⚠️⚠️ **最重要**：所有 tool_id 必须从上述【可用工具列表】中精确选择，严禁使用不存在的工具
2. ❌ **严禁**在计划中臆造工具、假设工具存在、或建议添加新工具（如 user_input_request、input_collector、js_engine 等）
3. ❌ **严禁**使用已被筛选掉的工具（即使它们在其他场景下可能存在）
4. ✅ **验证方法**：每选择一个工具，务必在上述【可用工具列表】中找到对应的 ID
5. ✅ **如果可用工具无法完成任务**：在 description 中说明"当前工具库无法完成任务，建议提醒管理员添加[具体类型]工具或终止任务"
6. 确保每个步骤的 step_name 都有明确的、有意义的名称
7. 避免导致失败的同样问题
8. 如果有标准流程提示，可以参考但不必完全遵循
9. 合理利用并行执行优化性能
10. ⚠️⚠️⚠️ **严禁重复规划已执行成功的步骤** - 这是关键约束：
   - 仔细查看【执行历史】，识别哪些步骤已经执行成功（状态为"已完成"或"成功"）
   - **只规划未执行、正在执行或失败的步骤**
   - 已成功步骤的输出可以作为后续步骤的输入（通过依赖关系引用其 step_id）
   - **如果任务已经全部完成，只需规划补充或优化步骤**
11. **如果需要复用已成功步骤的输出，请在新计划步骤的 dependencies 中引用该步骤的 step_id，而不是重新执行**
12. **🔴 必须采纳整体反思指导中的改进建议，特别是重新规划策略建议**

🌟 **发散思维要求 - 重要！**
在开始规划前，请先思考：
❓ **原方案为什么失败？** - 不只是表面错误，挖掘深层原因
❓ **有没有完全不同的路径？** - 不要陷入"修复原方案"的惯性思维
❓ **能否用不同的工具组合？** - 探索你可能没有尝试过的工具搭配
❓ **步骤顺序能否颠倒或重组？** - 换个角度看问题，也许从终点往起点规划更合理
❓ **能否通过并行优化提升效率？** - 识别可以同时进行的独立任务
❓ **是否需要增加中间验证步骤？** - 防止错误累积到最后才发现

💡 **鼓励创新方案**：
- ✅ 大胆尝试与原方案思路完全不同的解决路径
- ✅ 如果原方案是串行的，考虑并行化
- ✅ 如果原方案步骤过多，考虑合并简化
- ✅ 如果原方案步骤过少，考虑细化拆解增加鲁棒性
- ✅ 充分利用整体反思中的"替代方法/改进建议"
- ✅ 不要害怕推翻原有方案，创新往往来自大胆重构

请开始重新规划（记住：要发散思维，探索不同思路！）："#;

// ==================== 公开常量导出(用于向后兼容) ====================
// 这些常量用于保持与旧代码的兼容性
// 新代码应该使用 ReplanningPromptBuilder 来构建提示词

/// 重新规划系统提示词(公开常量，用于向后兼容)
pub const REPLANNING_SYSTEM_PROMPT: &str = BASE_REPLANNING_SYSTEM_PROMPT;

/// 重新规划用户模板(公开常量，用于向后兼容)
pub const REPLANNING_USER_TEMPLATE: &str = BASE_REPLANNING_USER_TEMPLATE;

// ==================== 场景特定重新规划指导 ====================
// 注意：场景指导内容已迁移到 scene_guidance.rs 统一管理

// ==================== 重新规划提示词构建器 ====================

/// 重新规划提示词构建器
pub struct ReplanningPromptBuilder {
    /// 统一的场景管理器
    scene_manager: SceneManager,
}

impl ReplanningPromptBuilder {
    /// 创建新的重新规划提示词构建器
    pub fn new() -> Self {
        Self {
            scene_manager: SceneManager::new(),
        }
    }

    /// 构建重新规划提示词
    ///
    /// # 参数
    /// - `failure_reason`: 失败原因描述
    /// - `available_tools`: 可用工具列表文本
    /// - `metadata`: 元数据
    /// - `workflow_hint`: 工作流提示(可选)
    /// - `context`: 重新规划上下文信息
    ///
    /// # 返回
    /// (system_prompt, user_prompt)
    pub fn build_replanning_prompt(
        &self,
        failure_reason: &str,
        available_tools: &str,
        metadata: &HashMap<String, String>,
        workflow_hint: Option<&str>,
        context: &ReplanningContext,
    ) -> (String, String) {
        // 1. 根据 task_type 选择场景特定的重新规划指导
        let scene_guidance = self
            .scene_manager
            .get_replanning_guidance(context.task_type.as_deref());

        // 2. 格式化用户上下文
        let user_context_text = context.format_user_context();

        // 3. 格式化执行历史
        let execution_history_text = context.format_execution_history();

        // 4. 格式化反思分析
        let reflection_analysis_text = context.format_reflection_analysis();

        // 4.5 格式化整体反思指导
        let overall_reflection_guidance_text = context.format_overall_reflection_guidance();

        // 5. 格式化元数据
        let metadata_str = if metadata.is_empty() {
            "无".to_string()
        } else {
            metadata
                .iter()
                .map(|(k, v)| format!("  - {}: {}", k, v))
                .collect::<Vec<_>>()
                .join("\n")
        };

        // 6. 格式化工作流程提示
        let workflow_str = if let Some(hint) = workflow_hint {
            format!("\n\n【匹配的标准任务流程】\n{}\n", hint)
        } else {
            String::new()
        };

        // 7. 组装用户提示词
        let user_prompt = BASE_REPLANNING_USER_TEMPLATE
            .replace("{user_context}", &user_context_text)
            .replace("{execution_history}", &execution_history_text)
            .replace("{reflection_analysis}", &reflection_analysis_text)
            .replace("{overall_reflection_guidance}", &overall_reflection_guidance_text)
            .replace("{failure_reason}", failure_reason)
            .replace("{available_tools}", available_tools)
            .replace("{metadata}", &metadata_str)
            .replace("{workflow_hint}", &workflow_str)
            .replace("{scene_specific_guidance}", scene_guidance);

        // 8. 返回系统提示词和用户提示词
        (BASE_REPLANNING_SYSTEM_PROMPT.to_string(), user_prompt)
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

impl Default for ReplanningPromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replanning_context_creation() {
        let context = ReplanningContext::new(
            Some("自动建模".to_string()),
            "步骤3执行失败".to_string()
        );
        assert_eq!(context.task_type, Some("自动建模".to_string()));
        assert_eq!(context.failure_reason, "步骤3执行失败");
        assert!(context.user_context.is_none());
    }

    #[test]
    fn test_set_user_context() {
        let mut context = ReplanningContext::new(None, "测试失败".to_string());
        context.set_user_context("用户提供的上下文信息".to_string());
        assert!(context.user_context.is_some());
        assert_eq!(context.user_context.unwrap(), "用户提供的上下文信息");
    }

    #[test]
    fn test_format_user_context_empty() {
        let context = ReplanningContext::new(None, "测试".to_string());
        let formatted = context.format_user_context();
        assert_eq!(formatted, "（用户未提供任务上下文）");
    }

    #[test]
    fn test_format_user_context_with_data() {
        let mut context = ReplanningContext::new(None, "测试".to_string());
        context.set_user_context("这是用户上下文信息".to_string());
        let formatted = context.format_user_context();
        assert!(formatted.contains("这是用户上下文信息"));
    }

    #[test]
    #[allow(deprecated)]
    fn test_add_successful_step_backward_compat() {
        let mut context = ReplanningContext::new(None, "测试失败".to_string());
        context.add_successful_step(SuccessfulStep {
            step_name: "测试步骤".to_string(),
            step_id: "step_1".to_string(),
            output_summary: "成功输出".to_string(),
            timestamp: None,
        });
        assert!(context.user_context.is_some());
    }

    #[test]
    fn test_builder_creation() {
        let builder = ReplanningPromptBuilder::new();
        let supported = builder.supported_task_types();
        assert!(supported.contains(&"自然语言建模".to_string()));
        assert!(supported.contains(&"工具管理".to_string()));
        assert!(supported.contains(&"客户端管理".to_string()));
        assert!(supported.contains(&"PLC控制器".to_string()));
    }

    #[test]
    fn test_build_prompt_with_tool_management() {
        let builder = ReplanningPromptBuilder::new();
        let context = ReplanningContext::new(
            Some("工具管理".to_string()),
            "工具创建失败".to_string()
        );

        let metadata = HashMap::new();
        let (system, user) = builder.build_replanning_prompt(
            "工具创建失败",
            "工具列表",
            &metadata,
            None,
            &context,
        );

        assert!(system.contains("任务规划专家"));
        assert!(user.contains("工具创建失败"));
    }

    #[test]
    fn test_build_prompt_with_default_guidance() {
        let builder = ReplanningPromptBuilder::new();
        let context = ReplanningContext::new(
            Some("未知场景".to_string()),
            "测试失败".to_string()
        );

        let metadata = HashMap::new();
        let (_, user) = builder.build_replanning_prompt(
            "测试失败",
            "工具列表",
            &metadata,
            None,
            &context,
        );

        assert!(user.contains("测试失败"));
    }
}
