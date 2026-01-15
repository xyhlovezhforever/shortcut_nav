//! 继续规划提示词管理模块
//!
//! 提供中断后继续规划的提示词模板管理
//! 支持基于历史执行流程的智能继续规划

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::scene_guidance::SceneManager;
use super::UnifiedLlmClient;

/// 历史执行步骤信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalStep {
    /// 步骤名称
    pub step_name: String,
    /// 步骤ID
    pub step_id: String,
    /// 工具ID
    pub tool_id: String,
    /// 工具参数
    pub parameters: String,
    /// 执行结果
    pub output: String,
    /// 执行状态(成功/失败)
    pub status: String,
    /// 执行时间戳
    pub timestamp: Option<String>,
}

/// 继续规划上下文信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuePlanningContext {
    /// 任务类型(如:自动建模、代码生成、客户端操作、负荷预测等)
    pub task_type: Option<String>,
    /// 用户传递的完整上下文(包含历史执行信息，模型根据最后一个任务状态继续规划)
    pub user_context: String,
}

impl ContinuePlanningContext {
    /// 创建新的继续规划上下文
    pub fn new(
        task_type: Option<String>,
        user_context: String,
    ) -> Self {
        Self {
            task_type,
            user_context,
        }
    }
}

// ==================== 继续规划提示词模板 ====================

/// 继续规划系统提示词
const CONTINUE_PLANNING_SYSTEM_PROMPT: &str = r#"你是任务规划专家，擅长基于上下文继续规划未完成的任务。

【核心职责】
分析用户提供的上下文，识别最后一个任务的执行状态，继续规划当前任务未完成的步骤。

【质量要求】
- 仔细分析上下文中的任务执行状态
- 识别最后一个任务是成功还是失败
- 如果最后一个任务成功，继续规划下一个步骤
- 如果最后一个任务失败，根据失败原因调整策略后继续
- 避免重复执行已成功的步骤
- 保持任务逻辑的连贯性

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

【继续规划策略】
1. **分析上下文中的最后一个任务**:
   - 查看最后一个任务的执行状态(成功/失败)
   - 如果成功，继续规划后续步骤
   - 如果失败，分析失败原因并调整策略

2. **任务连续性**:
   - 新步骤的step_id应从上下文中最后的步骤之后继续编号
   - 新步骤的dependencies应正确引用已有步骤和新步骤
   - 保持与上下文执行逻辑的一致性

3. **上下文利用**:
   - 充分利用上下文中已有步骤的输出作为后续步骤的输入
   - 理解原始任务的整体目标
   - 可以引用历史步骤的输出结果(使用{{step_id.output}}格式)

【关键约束】
- 仅使用提供的工具，ID完全匹配
- step_name 字段必须有明确的、有意义的名称
- 所有 tool_id 必须从可用工具列表中选择
- 新步骤的step_id必须从上下文中最后步骤之后继续编号
- 正确设置dependencies

【输出JSON Schema】
{
  "plan_id": "plan_<uuid>",
  "description": "继续执行计划描述",
  "task_type": "负荷预测|自动建模|数据分析|数学计算|客户端操作|PLC控制器|工具管理|通用",
  "context_understanding": "总结上下文中最后一个任务的执行状态和继续规划的理由",
  "continuation_strategy": "说明如何基于当前状态继续执行的策略",
  "total_steps": 数字,
  "estimated_duration_secs": 数字,
  "steps": [{
    "step_id": "step_N (N从上下文最后步骤之后继续)",
    "step_name": "名称",
    "tool_id": "ID (必须是可用工具列表中的工具ID)",
    "parameters": {},
    "dependencies": ["step_id1", "step_id2"],
    "expected_output": "输出",
    "data_input_source": "step_X输出|用户输入|元数据|上下文",
    "data_output_usage": "供step_Z使用|最终结果|中间状态"
  }]
}

【⚠️ Actions格式 - 步骤内并行执行】
当需要执行多个相同类型的操作时，优先使用actions格式实现步骤内并行：

示例1 - 批量添加角色：
```json
{
  "step_id": "step_4",
  "step_name": "批量添加角色",
  "actions": [
    {
      "action_id": "action_4_1",
      "name": "添加角色huarun_test_1",
      "tool_id": "add_role",
      "parameters": {"role_name": "huarun_test_1"},
      "dependencies": [],
      "expected_output": "角色添加成功"
    },
    {
      "action_id": "action_4_2",
      "name": "添加角色test_2",
      "tool_id": "add_role",
      "parameters": {"role_name": "test_2"},
      "dependencies": [],
      "expected_output": "角色添加成功"
    }
  ],
  "dependencies": ["step_3"]
}
```

示例2 - 引用已有步骤输出：
```json
{
  "step_id": "step_5",
  "step_name": "汇总结果",
  "tool_id": "js_engine",
  "parameters": {
    "code": "{{step_2.output}} + {{step_3.output}}"
  },
  "dependencies": ["step_2", "step_3"]
}
```

✓ 使用场景：多个相同类型操作(如:添加多个角色、创建多个对象、删除多个项目)
✓ 优势：单个步骤内并行执行，大幅提升效率
✓ 兼容性：单个操作继续使用旧格式(tool + parameters)"#;

/// 继续规划用户提示词模板
const CONTINUE_PLANNING_USER_TEMPLATE: &str = r#"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【用户上下文】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{user_context}

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
⚠️ 【场景专属规划指导 - 最高优先级】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{scene_guidance}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【继续规划任务】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

请分析上述用户上下文，识别最后一个任务的执行状态，继续规划未完成的步骤。

⚠️ 重要提醒：
1. **分析上下文中最后一个任务的执行状态**
2. **不要重复执行已成功的步骤**
3. **新步骤的step_id从上下文最后步骤之后继续编号**
4. **可以引用已有步骤的输出** (使用{{step_id.output}}格式)
5. **保持任务逻辑的连贯性和一致性**
6. 如果上方有场景专属指导，必须严格遵守
7. 合理利用并行执行优化性能
8. 多个相同类型操作优先使用actions格式

请开始继续规划："#;

// ==================== 继续规划提示词构建器 ====================

/// 继续规划提示词构建器
pub struct ContinuePlanningPromptBuilder {
    /// 统一的场景管理器
    scene_manager: SceneManager,
}

impl ContinuePlanningPromptBuilder {
    /// 创建新的继续规划提示词构建器
    pub fn new() -> Self {
        Self {
            scene_manager: SceneManager::new(),
        }
    }

    /// 构建继续规划提示词
    ///
    /// # 参数
    /// - `context`: 继续规划上下文信息
    /// - `available_tools`: 可用工具列表文本
    /// - `metadata`: 元数据
    /// - `workflow_hint`: 工作流提示(可选)
    ///
    /// # 返回
    /// (system_prompt, user_prompt)
    pub fn build_continue_planning_prompt(
        &self,
        context: &ContinuePlanningContext,
        available_tools: &str,
        metadata: &HashMap<String, String>,
        workflow_hint: Option<&str>,
    ) -> (String, String) {
        // 1. 根据 task_type 选择场景特定的规划指导
        let scene_guidance = self
            .scene_manager
            .get_planning_guidance(context.task_type.as_deref());

        // 2. 格式化元数据
        let metadata_str = if metadata.is_empty() {
            "无".to_string()
        } else {
            metadata
                .iter()
                .map(|(k, v)| format!("  - {}: {}", k, v))
                .collect::<Vec<_>>()
                .join("\n")
        };

        // 3. 格式化工作流程提示
        let workflow_str = if let Some(hint) = workflow_hint {
            format!("\n\n【匹配的标准任务流程】\n{}\n", hint)
        } else {
            "无".to_string()
        };

        // 4. 格式化场景指导
        let scene_guidance_str = if !scene_guidance.is_empty() && scene_guidance.trim() != "" {
            scene_guidance.to_string()
        } else {
            "无场景专属指导，使用通用规划策略。".to_string()
        };

        // 5. 组装用户提示词
        let user_prompt = CONTINUE_PLANNING_USER_TEMPLATE
            .replace("{user_context}", &context.user_context)
            .replace("{available_tools}", available_tools)
            .replace("{metadata}", &metadata_str)
            .replace("{workflow_hint}", &workflow_str)
            .replace("{scene_guidance}", &scene_guidance_str);

        // 6. 返回系统提示词和用户提示词
        (CONTINUE_PLANNING_SYSTEM_PROMPT.to_string(), user_prompt)
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

impl Default for ContinuePlanningPromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== LLM规划策略判断提示词 ====================

/// LLM规划策略判断系统提示词
const PLANNING_STRATEGY_DETECTION_SYSTEM_PROMPT: &str = r#"你是任务规划策略专家，根据用户的任务描述和当前上下文，判断应该使用哪种规划策略。

【策略类型】
1. **继续规划 (continue_planning)**: 基于现有执行历史继续完成剩余步骤
   - 适用场景：任务未完成但没有失败，只是中断了，用户想继续推进

2. **重新规划 (replanning)**: 因失败或需要调整而重新制定计划
   - 适用场景：上次执行失败，需要调整策略或重新设计方案

【判断标准】

✅ **继续规划**的情况：
- 用户明确要求继续："继续"、"接着做"、"完成剩余步骤"、"继续执行"
- 英文表达："continue"、"resume"、"keep going"、"proceed"
- 隐含继续意图："完成剩余的..."、"继续上次的..."、"接下来..."
- 任务处于中断状态，但之前的步骤是成功的
- 用户想在现有基础上继续推进，而不是推倒重来

✅ **重新规划**的情况：
- 用户明确要求重新规划："重新规划"、"重新开始"、"换个方案"、"重做"
- 英文表达："replan"、"start over"、"try different approach"
- 上次执行失败，需要调整策略
- 发现之前的方案有问题，需要重新设计
- 用户对当前进度不满意，要求重新制定计划

⚠️ **特殊情况**：
- 如果任务描述既不明确是继续也不明确是重新规划，默认判断为**普通规划**（全新任务）
- 如果用户只是简单的操作请求（如"查询数据"、"添加用户"），判断为**普通规划**

⚠️ **场景专属指导**：
如果下方提供了场景专属指导，应该将场景特征纳入判断依据。不同场景可能有特殊的策略判断规则，请优先遵循场景指导中的策略建议。

【输出格式】
必须返回JSON格式：
{
  "strategy": "continue_planning" | "replanning" | "normal_planning",
  "confidence": 0-100之间的数字,
  "reasoning": "判断理由（简短说明为什么选择这个策略）"
}

【示例】
输入："继续执行上次的任务"
输出：{"strategy": "continue_planning", "confidence": 95, "reasoning": "明确包含'继续执行'关键词，用户想在现有基础上继续"}

输入："上次失败了，重新规划一下"
输出：{"strategy": "replanning", "confidence": 90, "reasoning": "明确提到'失败'和'重新规划'，需要调整策略"}

输入："完成剩余的权限配置"
输出：{"strategy": "continue_planning", "confidence": 80, "reasoning": "包含'剩余'一词，暗示要继续之前未完成的配置"}

输入："创建一个新的用户账号"
输出：{"strategy": "normal_planning", "confidence": 95, "reasoning": "这是一个全新的独立操作请求，不涉及继续或重新规划"}

输入："换个方法重新生成工具"
输出：{"strategy": "replanning", "confidence": 85, "reasoning": "包含'换个方法'和'重新'，说明对之前的方案不满意，需要重新规划"}
"#;

/// LLM规划策略判断用户提示词模板
const PLANNING_STRATEGY_DETECTION_USER_TEMPLATE: &str = r#"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【任务描述】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{task_description}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 【场景专属规划策略指导 - 最高优先级】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{scene_guidance}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【判断任务】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

请分析上述任务描述，判断应该使用哪种规划策略（继续规划/重新规划/普通规划）。

如果上方提供了场景专属指导，必须优先遵守场景指导中的策略建议。

请返回JSON格式的判断结果。"#;

// ==================== LLM意图识别提示词（旧版，保留兼容）====================

/// LLM意图识别系统提示词
const CONTINUE_INTENT_DETECTION_SYSTEM_PROMPT: &str = r#"你是意图识别专家，擅长判断用户的任务描述是否包含"继续执行"的意图。

【核心职责】
分析用户的任务描述，判断用户是否想要继续之前中断的任务或执行流程。

【判断标准】
明确包含继续执行意图的情况：
- 明确提到"继续"、"接着"、"继续执行"、"继续做"等词汇
- 明确提到"resume"、"continue"、"keep going"等英文词汇
- 上下文暗示要继续之前的工作（如："完成剩余配置"、"继续上次的任务"）

不包含继续执行意图的情况：
- 启动新任务（如："开始新的项目"、"创建新任务"）
- 独立的操作请求（如："查询数据"、"添加用户"）
- 没有明确或隐含的"继续"语义

【输出格式】
必须返回JSON格式：
{
  "is_continue_intent": true/false,
  "confidence": 0-100之间的数字,
  "reasoning": "简短的判断理由"
}

【示例】
输入："继续执行上次的任务"
输出：{"is_continue_intent": true, "confidence": 95, "reasoning": "明确包含'继续执行'关键词"}

输入："创建新的用户账号"
输出：{"is_continue_intent": false, "confidence": 90, "reasoning": "这是一个独立的新操作请求，不涉及继续之前的任务"}

输入："完成剩余的权限配置"
输出：{"is_continue_intent": true, "confidence": 75, "reasoning": "包含'剩余'一词，暗示要继续之前未完成的配置"}
"#;

/// LLM意图识别用户提示词模板
const CONTINUE_INTENT_DETECTION_USER_TEMPLATE: &str = r#"请分析以下任务描述，判断是否包含"继续执行"的意图：

任务描述：{task_description}

请返回JSON格式的判断结果。"#;

// ==================== LLM规划策略判断数据结构 ====================

/// 规划策略类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlanningStrategy {
    /// 继续规划
    ContinuePlanning,
    /// 重新规划
    Replanning,
    /// 普通规划（全新任务）
    NormalPlanning,
}

/// LLM规划策略判断结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanningStrategyDetectionResult {
    /// 规划策略类型
    pub strategy: PlanningStrategy,
    /// 置信度（0-100）
    pub confidence: f32,
    /// 判断理由
    pub reasoning: String,
}

// ==================== LLM意图识别数据结构（旧版，保留兼容）====================

/// LLM意图识别结果
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ContinueIntentDetectionResult {
    /// 是否为继续执行意图
    is_continue_intent: bool,
    /// 置信度（0-100）
    confidence: f32,
    /// 判断理由
    reasoning: String,
}

// ==================== 规划策略判断工具函数 ====================

/// 使用LLM判断应该使用哪种规划策略（继续规划/重新规划/普通规划）
///
/// # 参数
/// - `task_description`: 任务描述
/// - `task_type`: 任务类型（可选，用于获取场景专属指导）
/// - `llm_client`: LLM客户端
/// - `confidence_threshold`: 置信度阈值（0-100）
///
/// # 返回
/// - Ok(PlanningStrategyDetectionResult): 规划策略判断结果
/// - Err: LLM调用失败
///
/// # 示例
/// ```rust
/// let result = detect_planning_strategy_with_llm(
///     "继续执行上次的任务",
///     Some("工具管理"),
///     &llm_client,
///     75.0
/// ).await?;
///
/// match result.strategy {
///     PlanningStrategy::ContinuePlanning => { /* 继续规划 */ },
///     PlanningStrategy::Replanning => { /* 重新规划 */ },
///     PlanningStrategy::NormalPlanning => { /* 普通规划 */ },
/// }
/// ```
pub async fn detect_planning_strategy_with_llm(
    task_description: &str,
    task_type: Option<&str>,
    llm_client: &UnifiedLlmClient,
    confidence_threshold: f32,
) -> Result<PlanningStrategyDetectionResult, Box<dyn std::error::Error + Send + Sync>> {
    use tracing::{info, warn, error};

    info!(
        "🚀 开始LLM规划策略判断 | 任务描述: '{}' | 任务类型: {:?} | 置信度阈值: {:.1}",
        task_description, task_type, confidence_threshold
    );

    // 创建场景管理器并获取场景指导
    let scene_manager = SceneManager::new();
    let scene_guidance = scene_manager.get_planning_guidance(task_type);

    // 格式化场景指导
    let scene_guidance_str = if !scene_guidance.is_empty() && scene_guidance.trim() != "" {
        scene_guidance.to_string()
    } else {
        "无场景专属指导，使用通用策略判断规则。".to_string()
    };

    // 构建提示词
    let user_prompt = PLANNING_STRATEGY_DETECTION_USER_TEMPLATE
        .replace("{task_description}", task_description)
        .replace("{scene_guidance}", &scene_guidance_str);

    info!("📤 正在调用LLM进行策略判断...");

    // 调用LLM
    let response = llm_client
        .call(PLANNING_STRATEGY_DETECTION_SYSTEM_PROMPT, &user_prompt, None)
        .await
        .map_err(|e| {
            error!("❌ LLM调用失败: {:?}", e);
            Box::new(e) as Box<dyn std::error::Error + Send + Sync>
        })?;

    info!("📥 LLM返回响应: {}", response);

    // 📝 记录到 test_log.txt
    let _ = crate::utils::DebugLogger::log_llm_interaction(
        "规划策略判断",
        PLANNING_STRATEGY_DETECTION_SYSTEM_PROMPT,
        &user_prompt,
        &response,
    );

    // 解析JSON响应
    let result: PlanningStrategyDetectionResult = serde_json::from_str(&response)
        .map_err(|e| {
            error!("❌ 解析JSON失败: {}, 响应内容: {}", e, response);
            warn!("⚠️ 解析LLM规划策略判断结果失败: {}, 响应: {}", e, response);
            Box::new(e) as Box<dyn std::error::Error + Send + Sync>
        })?;

    info!(
        "🎯 LLM规划策略判断结果 | 策略: {:?} | 置信度: {:.1}% | 理由: '{}'",
        result.strategy,
        result.confidence,
        result.reasoning
    );

    // 检查置信度是否达到阈值
    if result.confidence < confidence_threshold {
        warn!(
            "⚠️ 置信度 {:.1}% 低于阈值 {:.1}%，策略可能不准确",
            result.confidence,
            confidence_threshold
        );
    } else {
        info!(
            "✅ 置信度 {:.1}% 达到阈值 {:.1}%，策略判断可信",
            result.confidence,
            confidence_threshold
        );
    }

    Ok(result)
}

// ==================== 意图识别工具函数（旧版，保留兼容）====================

/// 使用LLM检测任务描述是否包含"继续执行"的意图
///
/// # 参数
/// - `task_description`: 任务描述
/// - `llm_client`: LLM客户端
/// - `confidence_threshold`: 置信度阈值（0-100）
///
/// # 返回
/// - Ok(true): 包含继续执行意图且置信度>=阈值
/// - Ok(false): 不包含继续执行意图或置信度<阈值
/// - Err: LLM调用失败
pub async fn detect_continue_intent_with_llm(
    task_description: &str,
    llm_client: &UnifiedLlmClient,
    confidence_threshold: f32,
) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
    // 构建提示词
    let user_prompt = CONTINUE_INTENT_DETECTION_USER_TEMPLATE
        .replace("{task_description}", task_description);

    // 调用LLM
    let response = llm_client
        .call(CONTINUE_INTENT_DETECTION_SYSTEM_PROMPT, &user_prompt, None)
        .await
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

    // 解析JSON响应
    let result: ContinueIntentDetectionResult = serde_json::from_str(&response)
        .map_err(|e| {
            tracing::warn!("⚠️ 解析LLM意图识别结果失败: {}, 响应: {}", e, response);
            Box::new(e) as Box<dyn std::error::Error + Send + Sync>
        })?;

    tracing::info!(
        "🔍 LLM意图识别结果: 是否继续={}, 置信度={}, 理由={}",
        result.is_continue_intent,
        result.confidence,
        result.reasoning
    );

    // 判断是否达到阈值
    let is_continue = result.is_continue_intent && result.confidence >= confidence_threshold;

    if result.is_continue_intent && result.confidence < confidence_threshold {
        tracing::info!(
            "⚠️ LLM识别为继续意图，但置信度({})低于阈值({}),  判定为非继续意图",
            result.confidence,
            confidence_threshold
        );
    }

    Ok(is_continue)
}

/// 检测任务描述是否包含"继续执行"的意图（关键词匹配版本）
///
/// # 参数
/// - `task_description`: 任务描述
///
/// # 返回
/// - true: 包含继续执行意图
/// - false: 不包含继续执行意图
///
/// # 识别策略
/// 1. 关键词匹配: "继续"、"接着"、"继续执行"、"continue"、"resume"等
/// 2. 意图标记: 任务描述格式为 "意图: 继续执行, 用户输入: ..." 或 "意图: continue, 用户输入: ..."
pub fn detect_continue_intent(task_description: &str) -> bool {
    let desc_lower = task_description.to_lowercase();

    // 策略1: 检查意图标记格式
    if desc_lower.contains("意图:") || desc_lower.contains("intent:") {
        if desc_lower.contains("继续")
            || desc_lower.contains("接着")
            || desc_lower.contains("continue")
            || desc_lower.contains("resume") {
            return true;
        }
    }

    // 策略2: 检查常见的继续执行关键词(独立词或词组)
    let continue_keywords = [
        "继续执行",
        "继续做",
        "继续完成",
        "接着执行",
        "接着做",
        "接着完成",
        "继续",
        "接着",
        "continue",
        "resume",
        "keep going",
        "carry on",
    ];

    for keyword in &continue_keywords {
        if desc_lower.contains(keyword) {
            return true;
        }
    }

    false
}

/// 从任务描述中提取用户的补充说明
///
/// # 参数
/// - `task_description`: 任务描述
///
/// # 返回
/// - Some(String): 用户补充说明
/// - None: 无补充说明
///
/// # 提取策略
/// 如果任务描述格式为 "意图: 继续执行, 用户输入: XXX"，提取"XXX"部分
/// 否则返回整个任务描述
pub fn extract_continuation_note(task_description: &str) -> Option<String> {
    // 尝试匹配 "用户输入:" 或 "user input:" 后面的内容
    if let Some(pos) = task_description.find("用户输入:") {
        let note = task_description[pos + "用户输入:".len()..].trim();
        if !note.is_empty() {
            return Some(note.to_string());
        }
    }

    if let Some(pos) = task_description.to_lowercase().find("user input:") {
        let note = task_description[pos + "user input:".len()..].trim();
        if !note.is_empty() {
            return Some(note.to_string());
        }
    }

    // 如果没有特殊格式，但检测到继续意图，返回整个描述
    if detect_continue_intent(task_description) {
        return Some(task_description.to_string());
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_continue_intent_with_keywords() {
        assert!(detect_continue_intent("继续执行上次的任务"));
        assert!(detect_continue_intent("接着做"));
        assert!(detect_continue_intent("请继续完成"));
        assert!(detect_continue_intent("continue the task"));
        assert!(detect_continue_intent("resume previous work"));
    }

    #[test]
    fn test_detect_continue_intent_with_intent_format() {
        assert!(detect_continue_intent("意图: 继续执行, 用户输入: 完成剩余步骤"));
        assert!(detect_continue_intent("intent: continue, user input: finish remaining steps"));
    }

    #[test]
    fn test_detect_continue_intent_negative() {
        assert!(!detect_continue_intent("创建新的任务"));
        assert!(!detect_continue_intent("开始一个全新的工作"));
        assert!(!detect_continue_intent("start a new task"));
    }

    #[test]
    fn test_extract_continuation_note() {
        let result = extract_continuation_note("意图: 继续执行, 用户输入: 完成数据分析");
        assert_eq!(result, Some("完成数据分析".to_string()));

        let result = extract_continuation_note("intent: continue, user input: analyze data");
        assert_eq!(result, Some("analyze data".to_string()));

        let result = extract_continuation_note("继续执行");
        assert_eq!(result, Some("继续执行".to_string()));

        let result = extract_continuation_note("创建新任务");
        assert_eq!(result, None);
    }

    #[test]
    fn test_continue_planning_context_creation() {
        let context = ContinuePlanningContext::new(
            Some("自动建模".to_string()),
            "用户上下文内容".to_string(),
        );
        assert_eq!(context.task_type, Some("自动建模".to_string()));
        assert_eq!(context.user_context, "用户上下文内容");
    }

    #[test]
    fn test_builder_creation() {
        let builder = ContinuePlanningPromptBuilder::new();
        let supported = builder.supported_task_types();
        assert!(supported.contains(&"自然语言建模".to_string()));
    }

    #[test]
    fn test_build_continue_planning_prompt() {
        let builder = ContinuePlanningPromptBuilder::new();
        let context = ContinuePlanningContext::new(
            Some("工具管理".to_string()),
            "创建计算器工具的上下文，包含历史执行步骤".to_string(),
        );

        let metadata = HashMap::new();
        let (system, user) = builder.build_continue_planning_prompt(
            &context,
            "工具列表",
            &metadata,
            None,
        );

        assert!(system.contains("任务规划专家"));
        assert!(system.contains("继续规划"));
        assert!(user.contains("创建计算器工具的上下文"));
        assert!(user.contains("工具列表"));
    }
}
