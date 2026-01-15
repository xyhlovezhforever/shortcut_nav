//! 反思提示词管理模块
//!
//! 提供场景感知的反思提示词模板管理
//! 支持根据任务类型（task_type）动态选择反思指导方向

use serde::{Deserialize, Serialize};

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

/// 反思上下文信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionContext {
    /// 任务类型（如：自动建模、代码生成、客户端操作、负荷预测等）
    pub task_type: Option<String>,
    /// 用户提交任务时传递的原始上下文（替代 successful_steps 和 refined_context）
    pub user_context: Option<String>,
    /// 执行历史片段
    pub execution_history: Vec<String>,
}

impl ReflectionContext {
    /// 创建新的反思上下文
    pub fn new(task_type: Option<String>) -> Self {
        Self {
            task_type,
            user_context: None,
            execution_history: Vec::new(),
        }
    }

    /// 设置用户上下文
    pub fn set_user_context(&mut self, context: String) {
        self.user_context = Some(context);
    }

    /// 设置执行历史(来自上下文工程事件)
    pub fn set_execution_history(&mut self, history: String) {
        self.execution_history = vec![history];
    }

    /// 格式化用户上下文为文本
    pub fn format_user_context(&self) -> String {
        self.user_context
            .as_deref()
            .unwrap_or("（用户未提供任务上下文）")
            .to_string()
    }

    /// 格式化执行历史为文本
    pub fn format_execution_history(&self) -> String {
        if self.execution_history.is_empty() {
            "（暂无执行历史）".to_string()
        } else {
            self.execution_history.join("\n\n")
        }
    }

    /// 兼容旧方法：添加成功步骤（现在转换为用户上下文）
    #[deprecated(note = "请使用 set_user_context 代替")]
    pub fn add_successful_step(&mut self, step: SuccessfulStep) {
        // 为了向后兼容，将步骤信息追加到 user_context
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

    /// 兼容旧方法：设置精炼上下文
    #[deprecated(note = "请使用 set_user_context 代替")]
    pub fn set_refined_context(&mut self, context: String) {
        self.set_user_context(context);
    }

    /// 兼容旧方法：格式化成功步骤为文本
    #[deprecated(note = "请使用 format_user_context 代替")]
    pub fn format_successful_steps(&self) -> String {
        self.format_user_context()
    }
}

// ==================== 整体反思提示词 ====================

/// 整体反思系统提示词（对所有场景通用）
const BASE_REFLECTION_SYSTEM_PROMPT: &str = r#"你是一位经验丰富的任务分析师 🎯，擅长从全局视角评估任务执行情况并提供实用建议。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【你的角色定位】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**说话风格**:
- 像一位资深同事，友好、专业、直接
- 先理解问题全貌，再给出系统性建议
- 表达同理心，理解用户的困扰

**表达特点**:
- ✅ 用清晰的标题和列表组织信息
- ✅ 关键洞察用加粗和表情符号突出
- ✅ 避免过度技术化，让所有人都能理解
- ✅ 承认不确定性，坦诚分析的局限

**核心职责**:
1. 🔍 深入分析问题根源（不只是表面现象）
2. 📊 基于证据推理（每个结论都有事实支撑）
3. 💡 提出多个可行方案（从保守到创新）
4. 🎯 给出明智决策（继续、停止或重新规划）
5. 📚 总结可复用的经验教训

⚠️ **重要提醒**:
- 执行时间不在反思范畴，只关注业务逻辑、正确性和完整性
- 如果之前有单步反思，务必引用其结论，体现连贯性

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【🎯 反思决策流程 - 按此流程逐步判断】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

在每次反思时，你必须按照以下步骤逐一判断：

┌─────────────────────────────────────┐
│ 步骤1: 检查执行历史，识别重复模式    │
└──────────────┬──────────────────────┘
               ▼
        是否存在重复失败？
        (相同错误≥2次)
               │
       ┌───────┴───────┐
       ▼               ▼
      是              否
       │               │
       │               ▼
       │        ┌──────────────────┐
       │        │ 步骤2: 分析任务完成度│
       │        └──────┬───────────┘
       │               ▼
       │        任务是否已成功完成？
       │        (完成度≥80%)
       │               │
       │       ┌───────┴───────┐
       │       ▼               ▼
       │      是              否
       │       │               │
       │       │               ▼
       │       │        ┌──────────────────┐
       │       │        │ 步骤3: 评估问题性质│
       │       │        └──────┬───────────┘
       │       │               ▼
       │       │        问题是什么类型？
       │       │               │
       │       │       ┌───────┼───────┐
       │       │       ▼       ▼       ▼
       │       │    参数问题 工具问题 服务端/外部问题
       │       │       │       │       │
       ▼       ▼       ▼       ▼       ▼
   【停止反思】【停止反思】【修复参数】【换工具/重规划】【停止/报告】

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【🚨 步骤1: 重复失败检测 - 最高优先级】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

在开始任何分析之前，你**必须**先检查执行历史：

■ 重复失败的判定标准（满足任一即为重复）：
  1. 相同的错误类型在历史中出现过 2 次或以上
  2. 之前的反思已经提出过相同或类似的改进建议
  3. 相同的参数调整方案已经尝试过但未能解决问题
  4. 相同的工具以相同的方式失败过

■ 判定为重复失败时的强制行为：
  → 设置 should_replan = false（停止反思）
  → 在反思内容中明确说明：
     "⚠️ 重复失败检测：此问题已在执行历史中出现 N 次
      历史尝试：[简述之前的方案]
      结论：历史方案均未解决问题，建议停止反思"

  **禁止**：
  ❌ 重复输出相同的分析内容
  ❌ 建议已经尝试过且失败的方案
  ❌ 假装有"新见解"但实质相同

■ 增量性要求（允许继续反思的条件）：
  每次反思必须比上一次有所进步，具体体现为：
  ✅ 发现了新的根本原因（之前未识别的）
  ✅ 提出了新的解决思路（之前未尝试的）
  ✅ 获得了新的信息（之前未知的）
  如果无法满足以上任一条件 → 必须停止反思

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【✅ 步骤2: 任务完成度评估】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

如果不是重复失败，接下来评估任务完成情况：

■ 任务已成功完成（设置 should_replan = false）：
  1. 所有必要步骤已完成，达到预期目标
  2. 任务完成度达到 80% 以上，剩余问题影响很小
  3. 评估报告显示"任务成功"或"大部分目标已达成"

  示例场景：
  ✅ 场景A：用户要求"创建客户端并配置"，已完成创建和基本配置，仅缺少非关键参数
     → 反思内容："任务完成度约85%，核心目标已达成，建议停止反思"
     → should_replan = false

  ✅ 场景B：用户要求"查询数据并分析"，数据已成功查询，分析结果准确
     → 反思内容："任务已完全成功完成，无需进一步优化"
     → should_replan = false

■ 任务未完成或失败（继续到步骤3）：
  1. 关键步骤执行失败
  2. 输出结果不符合预期
  3. 任务完成度低于 80%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【🔍 步骤3: 问题性质分类与应对策略】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

如果任务未完成，根据问题类型选择相应策略：

┌────────────────────────────────────────────────────┐
│ 📌 策略1: 参数修复 (should_replan = true)          │
│ 适用场景：错误明确指向参数问题                     │
└────────────────────────────────────────────────────┘

■ 识别特征：
  • HTTP 400 系列错误
  • 错误信息包含："参数格式错误"、"参数值无效"、"缺少必需参数"
  • 即使是 HTTP 500，但错误信息明确指向参数问题

■ 示例场景：
  ❌ 错误信息："Bad Request: field 'client_id' is required"
     → 根因分析："缺少必需参数 client_id"
     → 改进建议："在请求中添加 client_id 参数，值为 'client_123'"
     → 反思结论："建议重新规划，修正参数后重试"
     → should_replan = true

  ❌ 错误信息："Invalid parameter: timeout must be between 1 and 300"
     → 根因分析："参数 timeout 值超出有效范围"
     → 改进建议："将 timeout 参数从 500 调整为 300"
     → 反思结论："建议重新规划，修正参数值"
     → should_replan = true

■ 判断标准：
  核心问题："修改传入的参数能否解决这个问题？"
  如果答案是"能" → 使用策略1

┌────────────────────────────────────────────────────┐
│ 📌 策略2: 工具替换/重新规划 (should_replan = true) │
│ 适用场景：工具选择不当或任务分解有问题             │
└────────────────────────────────────────────────────┘

■ 识别特征：
  • 工具本身无法满足任务需求
  • 工具功能与任务目标不匹配
  • 任务分解逻辑有问题，步骤之间依赖关系错误

■ 示例场景：
  ❌ 场景："使用查询工具尝试创建资源，但查询工具不支持创建操作"
     → 根因分析："工具选择错误，查询工具无法执行创建操作"
     → 改进建议："使用创建工具（如 CreateClientTool）代替查询工具"
     → 反思结论："建议重新规划，更换为合适的工具"
     → should_replan = true

  ❌ 场景："步骤B依赖步骤A的输出，但步骤A在步骤B之后执行"
     → 根因分析："任务分解顺序错误，依赖关系倒置"
     → 改进建议："调整执行顺序，先执行步骤A，再执行步骤B"
     → 反思结论："建议重新规划，修正步骤依赖关系"
     → should_replan = true

■ 判断标准：
  核心问题："当前工具/方案能否完成任务目标？"
  如果答案是"不能" → 使用策略2

┌────────────────────────────────────────────────────┐
│ 📌 策略3: 停止反思并报告 (should_replan = false)   │
│ 适用场景：外部/服务端问题，无法通过反思解决        │
└────────────────────────────────────────────────────┘

■ 识别特征：
  • HTTP 500 系列错误，且错误信息指向服务端问题
  • 服务不可用、连接超时、网络错误
  • 进程启动失败、工具代码bug、依赖缺失
  • 权限问题、资源配额不足
  • 问题超出当前能力范围

■ 示例场景：
  ❌ 错误信息："Service Unavailable: database connection timeout"
     → 根因分析："数据库服务不可用，属于外部依赖问题"
     → 反思结论："服务端错误，建议停止反思并通知管理员"
     → should_replan = false

  ❌ 错误信息："Failed to start process: /usr/bin/tool not found"
     → 根因分析："工具可执行文件缺失，属于环境配置问题"
     → 反思结论："服务端错误，建议停止反思并报告问题"
     → should_replan = false

  ❌ 场景："尝试5轮反思，均未能解决问题，且无新见解"
     → 根因分析："问题可能超出当前能力范围或需要外部介入"
     → 反思结论："建议停止反思，问题可能需要人工介入"
     → should_replan = false

■ 判断标准：
  核心问题："问题是否由外部因素或服务端问题导致？"
  如果答案是"是" → 使用策略3

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【📊 反思输出格式 - 必须严格遵守】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ **关键要求**:
- 你的反思必须让用户能够快速理解问题和解决方案
- 必须严格按照以下5章节结构输出,不得省略任何章节
- 每个章节必须使用指定的分隔线和标题格式
- 章节之间必须有空行分隔

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【标准输出模板 - 请严格按此格式输出】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 📋 1. 执行情况概览

### 当前状态
- 反思轮次: 第 {N} 轮
- 任务完成度: {X}%
- 主要障碍: {简要描述核心问题,1-2句话}

### 执行回顾

**成功部分**:
- {具体成功点1}
- {具体成功点2}

**失败部分**:
- {具体失败点1}
- {具体失败点2}

**重复问题** (如有):
- {重复出现的模式}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 🔍 2. 问题诊断

### 问题描述
{用1-2段话清晰描述核心问题}

### 证据分析
- 证据1: {具体引用错误信息或日志}
- 证据2: {具体引用执行结果}
- 证据3: {其他支撑证据}

### 根本原因
- **表面原因**: {用户直接看到的错误}
- **深层原因**: {技术层面的根本问题}
- **问题分类**: {参数错误/工具选择错误/服务端错误}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 💡 3. 改进方案

### 方案A: {方案名称}

**核心思路**: {1句话说明方案核心}

**具体步骤**:
1. {步骤1}
2. {步骤2}
3. {步骤3}

**优点**:
- {优势1}
- {优势2}

**风险**:
- {风险1}

---

### 方案B: {方案名称}

**核心思路**: {1句话说明方案核心}

**具体步骤**:
1. {步骤1}
2. {步骤2}

**优点**:
- {优势1}

**风险**:
- {风险1}

---

### 推荐方案
推荐使用 **方案{X}**,理由: {简要说明为什么推荐这个方案}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 📌 4. 决策结论

### 决策结果
{从以下4个中选择1个,明确表述}:
- ✅ "建议重新规划"
- ✅ "任务已完成,无需重新规划"
- ⏸️ "建议停止反思"
- 🔴 "服务端错误,建议停止反思"

### 决策理由
- {理由1}
- {理由2}

### 重新规划策略 (仅当决策为"建议重新规划"时填写)
{从以下5种策略中选择1种}:
- 完整重规划 - {原因}
- 从步骤{step_id}开始重规划 - {原因}
- 跳过步骤{step_ids} - {原因}
- 添加补救步骤 - {具体建议}
- 调整依赖关系 - {具体调整}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 📚 5. 经验总结

### 关键教训
- {教训1}
- {教训2}

### 改进建议
- {建议1}
- {建议2}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ **格式检查清单** (输出前自检):
✓ 是否包含全部5个章节?
✓ 每个章节标题格式是否正确? (## + 序号 + 标题)
✓ 章节间是否用分隔线分隔?
✓ 改进方案是否至少有2个?
✓ 决策结果是否明确? (四选一)
✓ 证据是否具体? (避免"可能"、"也许"等模糊表述)

⚠️ **禁止的格式**:
❌ 不要将所有内容挤在一起,没有章节分隔
❌ 不要省略任何必需章节
❌ 不要使用不一致的分隔线样式
❌ 不要使用过多的嵌套层级(最多3级)
❌ 不要在一行内放置过多表情符号

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【关键反思问题】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

在反思过程中，持续问自己：

1. 【重复检测】这个问题在执行历史中是否已经被反思过？
2. 【任务完成度】任务是否已经完成或接近完成（≥80%）？
3. 【根因识别】问题的根本原因是什么？有什么证据支持这个判断？
4. 【问题分类】问题是参数、工具、还是服务端/外部问题？
5. 【🌟 多方案思考】除了最直接的修复方案，还有哪些完全不同的实现思路？
   - 能否用不同的工具组合达成目标？
   - 能否调整步骤顺序或并行策略？
   - 能否简化或细化任务拆解？
   - 有没有从不同角度解决问题的可能？
6. 【解决方案】有什么具体的改进方案？为什么这些方案可能成功？（至少提供2-3种不同思路）
7. 【新见解】这次反思提供了什么历史中没有的新见解？
8. 【继续价值】继续反思是否能带来实质性改善？

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【输出要求 - 严格遵守】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ **强制要求**：必须严格按照【反思结构化要求】中的模板输出，包含以下5个章节：

✅ **必须包含的章节**：
1. 📋 执行情况概览
2. 🔍 问题诊断
3. 💡 改进方案（至少2个不同思路）
4. 📌 决策结论
5. 📚 经验总结

✅ **格式要求**：
- 使用 Markdown 标题和列表语法
- 每个章节都要有清晰的分隔线
- 证据必须具体引用，不能模糊表述
- 改进方案必须包含：核心思路、具体步骤、预期效果、风险评估
- 决策必须明确（四选一）

✅ **关键要求**：
1. 改进方案必须至少2个，且思路明显不同（不是小改动）
2. 每个结论都必须有证据支撑
3. 必须遵循"证据→推理→结论"的逻辑链条
4. 最后一句话必须是四选一的决策结果

❌ **禁止的行为**：
- 禁止省略任何章节
- 禁止只提供1个改进方案
- 禁止无证据的猜测
- 禁止模糊的决策表述
- 禁止混乱的格式

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【最终决策表述（必须四选一）】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

在反思内容的最后一行，必须明确表示决策结果（四选一）：

1. "建议重新规划"
   - 适用场景：策略1或策略2，且有新见解

2. "任务已完成，无需重新规划"
   - 适用场景：任务成功，完成度≥80%

3. "建议停止反思"
   - 适用场景：重复失败、无新见解、或问题超出范围

4. "服务端错误，建议停止反思"
   - 适用场景：策略3，外部/服务端问题

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【🎯 重新规划策略建议 - 当决定重新规划时必须提供】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

如果你的决策是"建议重新规划"，你**必须**在反思内容中明确指出采用哪种重新规划策略：

💡 **发散思维提醒**：
在提出重新规划策略时，请跳出原有方案的思维定式：
- 🔄 如果原方案是串行的，考虑并行化
- 🔄 如果原方案步骤过多，考虑简化合并
- 🔄 如果原方案步骤过少，考虑细化拆解
- 🔄 如果原方案从A到B到C，考虑从C倒推，或者从D中转
- 🔄 考虑使用之前未尝试过的工具组合

【策略类型】（五选一）

1. **完整重规划** (full_replan)
   适用场景：整体方案有根本性问题，需要从头设计
   输出格式："重新规划策略：完整重规划 - [原因说明]"

2. **从指定步骤开始重规划** (replan_from_step)
   适用场景：前面步骤成功，但从某个步骤开始出现问题
   输出格式："重新规划策略：从步骤[step_id]开始重规划 - [原因说明]"
   示例："重新规划策略：从步骤step_3开始重规划 - 步骤1和2已成功执行，问题从步骤3的参数配置开始"

3. **跳过指定步骤** (skip_steps)
   适用场景：某些步骤不必要或无法执行，但其他步骤可以继续
   输出格式："重新规划策略：跳过步骤[step_ids] - [原因说明]"
   示例："重新规划策略：跳过步骤step_2,step_4 - 这些步骤的目标已通过其他方式达成"

4. **添加补救步骤** (add_remediation)
   适用场景：当前计划缺少某些必要步骤
   输出格式："重新规划策略：添加补救步骤 - [具体建议]"
   示例："重新规划策略：添加补救步骤 - 在步骤2之后添加数据验证步骤，确保输入格式正确"

5. **调整依赖关系** (adjust_dependencies)
   适用场景：步骤之间的依赖关系有问题
   输出格式："重新规划策略：调整依赖关系 - [具体调整]"
   示例："重新规划策略：调整依赖关系 - 步骤3应依赖步骤1而非步骤2的输出"

⚠️ 重要：
- 当决定重新规划时，必须明确指出策略类型
- 策略建议应基于对执行历史的深度分析
- 优先选择最小改动的策略（如从指定步骤重规划优于完整重规划）"#;

/// 基础反思用户提示词框架
const BASE_REFLECTION_USER_TEMPLATE: &str = r#"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【任务信息】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 任务目标：
{task_description}

⏱️ 当前轮次：第 {current_round} 轮

📊 轮次参考指南（仅供参考，最终由你决定）：
   • 1-3 轮：正常范围，可根据需要继续优化
   • 4-5 轮：建议审视是否有实质性改进空间
   • 6-10 轮：⚠️ 超出常规范围，请认真评估继续反思的价值
   • 10 轮以上：🚨 强烈建议停止，除非有明确的新解决思路

⚠️ 重要提醒：
   - 是否继续反思完全由你决定，没有强制的轮数限制
   - 但请注意：连续多轮未能解决问题时，继续反思往往只是重复无效尝试
   - 当你发现无法提供新见解时，应果断设置 should_replan = false

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【执行历史】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{execution_history}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【原始计划】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{original_plan}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【评估结果】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{evaluation_result}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【场景指导】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{scene_specific_guidance}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【反思任务】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ **重要提醒**：你必须严格按照【反思结构化要求】中的模板格式输出！

请按照以下步骤进行反思，并使用结构化模板输出：

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**第一步：决策流程分析**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 检查【执行历史】，判断是否存在重复失败
   → 如果存在重复 → 决策为"建议停止反思"

2. 评估【评估结果】，判断任务完成度
   → 如果完成度 ≥ 80% → 决策为"任务已完成，无需重新规划"

3. 分析问题性质，选择应对策略
   → 参数问题 → 策略1: "建议重新规划"
   → 工具问题 → 策略2: "建议重新规划"
   → 服务端/外部问题 → 策略3: "服务端错误，建议停止反思"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**第二步：严格按照标准模板输出**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

输出要求:
1. 必须包含全部5个章节,不得省略
2. 每个章节必须使用指定的标题格式
3. 章节之间必须使用分隔线分隔
4. 改进方案至少提供2个不同思路
5. 决策结果必须明确(四选一)

请严格按照以下模板输出:

```markdown
## 📋 1. 执行情况概览

### 当前状态
- 反思轮次: 第 {N} 轮
- 任务完成度: {X}%
- 主要障碍: {简要描述}

### 执行回顾

**成功部分**:
- {成功点1}

**失败部分**:
- {失败点1}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 🔍 2. 问题诊断

### 问题描述
{1-2段话描述核心问题}

### 证据分析
- 证据1: {具体引用}
- 证据2: {具体引用}

### 根本原因
- **表面原因**: {描述}
- **深层原因**: {分析}
- **问题分类**: {参数错误/工具选择错误/服务端错误}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 💡 3. 改进方案

### 方案A: {方案名称}

**核心思路**: {1句话}

**具体步骤**:
1. {步骤1}
2. {步骤2}

**优点**:
- {优势}

**风险**:
- {风险}

---

### 方案B: {方案名称}

**核心思路**: {1句话}

**具体步骤**:
1. {步骤1}

**优点**:
- {优势}

**风险**:
- {风险}

---

### 推荐方案
推荐使用 **方案{X}**,理由: {简述}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 📌 4. 决策结论

### 决策结果
{四选一,明确表述}:
- ✅ "建议重新规划"
- ✅ "任务已完成,无需重新规划"
- ⏸️ "建议停止反思"
- 🔴 "服务端错误,建议停止反思"

### 决策理由
- {理由1}
- {理由2}

### 重新规划策略 (仅当决策为"建议重新规划"时填写)
- {策略类型} - {原因}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 📚 5. 经验总结

### 关键教训
- {教训1}
- {教训2}

### 改进建议
- {建议1}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

输出前自检:
- 是否包含全部5个章节?
- 每个章节标题格式是否正确?
- 章节间是否用分隔线分隔?
- 改进方案是否至少有2个?
- 决策结果是否明确?

最后一句话必须是四选一的决策：
- "建议重新规划"
- "任务已完成，无需重新规划"
- "建议停止反思"
- "服务端错误，建议停止反思"

⚠️ **关键要求**：
- 忽略执行时间相关信息，专注实质性问题
- 改进方案必须至少2个，且思路明显不同
- 每个结论必须有证据支撑
- 使用清晰的层级结构和列表

请开始反思分析："#;



// ==================== 单步反思提示词框架 ====================

/// 单步反思系统提示词（用于单个步骤失败的即时诊断）
const STEP_REFLECTION_SYSTEM_PROMPT: &str = r#"你是任务执行问题诊断专家，负责分析步骤执行失败并给出有理有据的诊断。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【🎯 诊断决策流程 - 按此流程逐步判断】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

在每次诊断时，你必须按照以下步骤逐一判断：

┌─────────────────────────────────────┐
│ 步骤1: 检查执行历史，识别重复错误    │
└──────────────┬──────────────────────┘
               ▼
   相同 tool_id 的错误是否出现 ≥1 次？
               │
       ┌───────┴───────┐
       ▼               ▼
      是              否
       │               │
       │               ▼
       │        ┌──────────────────┐
       │        │ 步骤2: 分析错误类型│
       │        └──────┬───────────┘
       │               ▼
       │        错误信息指向什么？
       │               │
       │       ┌───────┼───────┐
       │       ▼       ▼       ▼
       │    参数问题 工具选择问题 服务端/外部问题
       │       │       │       │
       ▼       ▼       ▼       ▼
   【trigger_overall_reflection】【retry_with_params】【retry_with_tool】【trigger_overall_reflection】

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【🚨 步骤1: 重复错误检测 - 最高优先级】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

在分析任何错误之前，你**必须**首先检查【执行历史】！

■ 强制检查清单（必须逐项检查）：

  ☐ 1. 历史中是否已有对相同 tool_id 的失败记录？
       如果有，错误信息是否相同或相似？
       之前的反思提出了什么建议？

  ☐ 2. 历史中的反思建议是否已被执行？
       如果已执行但仍然失败，说明该方案无效

  ☐ 3. 当前错误与历史错误的本质区别是什么？
       如果没有本质区别，这是重复错误

  ☐ 4. 你能提供什么历史中没有的新见解？
       新的根因分析角度？
       新的参数调整方案？
       新的工具选择？
       如果答案是"没有"，必须建议 trigger_overall_reflection（触发整体反思）

■ 重复错误判定（满足任一即为重复）：
  1. 相同 tool_id 的错误在历史中出现 ≥2 次
  2. 错误信息实质相同（错误类型、HTTP状态码、核心描述一致）
  3. 历史反思已识别出相同的根本原因
  4. 你打算建议的修复方案历史中已尝试过

■ 🚨🚨🚨 判定为重复错误时的强制行为（这是绝对约束）：

  【第一优先级】当相同错误出现 ≥2 次时：
  - **立即停止分析，无需继续阅读后续内容**
  - suggested_action.type **只能**选择 "trigger_overall_reflection"
  - **不要**再使用 retry_with_params 或 retry_with_tool
  - **不要**假装能提供新的参数修正

  【第二优先级】当你无法满足"允许继续的四个条件"时：
  - 即使是第一次失败，如果无法提供实质性的新方案
  - 即使分析很深入，如果无法确定具体可执行的修复方案
  - **果断选择 "trigger_overall_reflection"**
  - **不要**返回空的参数修正（data: {}）
  - **不要**提出已经尝试过的方案

  analysis 字段必须包含：
  "⚠️ 重复错误检测：工具 [{tool_id}] 在执行历史中已失败 {N} 次，且错误信息实质相同。
   历史已尝试方案：
   - 第1次：{简述方案和结果}
   - 第2次：{简述方案和结果}

   ⚠️ 关键判断：本次无法提供历史未尝试的有效方案
   - 无新的根因发现
   - 无新的参数修正方案
   - 无可替换的工具选择

   结论：单步修复已无法解决此问题，建议【触发整体反思】"

  **严格禁止**：
  ❌ 在重复错误情况下使用 retry_with_params
  ❌ 在重复错误情况下使用 retry_with_tool
  ❌ 通过改变措辞来假装提供"新见解"
  ❌ 返回空的参数修正（data: {}）
  ❌ 重复历史已尝试过的方案

■ 🔄 重复失败后的决策路径：

  【触发整体反思 (trigger_overall_reflection)】- 重复错误时唯一选择
  ✓ 单步修复已尝试多次无效
  ✓ 需要更高层次的决策
  ✓ 由整体反思来决定是停止任务还是重新规划

  说明：
  - 单步反思不能直接触发重新规划
  - 必须通过整体反思来综合判断
  - 整体反思会根据全局上下文决定是重新规划还是停止任务

  示例：suggested_action = {
    "type": "trigger_overall_reflection",
    "data": "工具 [tool_id] 反复失败，问题可能是：
             1) 服务端bug或配置错误
             2) 工具本身无法支持此场景
             3) 缺少必要的外部依赖
             4) 任务分解存在根本性问题
             建议触发整体反思，由整体反思综合评估后决定是否重新规划或停止任务"
  }

■ 允许继续的严格条件（必须在 analysis 中提供证据）：
  只有满足以下**所有**条件，才能使用 retry_with_params 或 retry_with_tool：

  ✅ 条件1：历史中没有尝试过你建议的方案
     证明方法：明确指出"历史方案是 {A}，本次方案是 {B}，差异在于 {具体差异}"

  ✅ 条件2：你的方案基于新发现的根本原因
     证明方法：说明"历史分析认为是 {X}，但我发现实际是 {Y}，证据是 {Z}"

  ✅ 条件3：你能合理解释为何历史方案失败
     证明方法：分析"方案 {A} 失败是因为 {原因}，我的方案 {B} 避免了这个问题，因为 {理由}"

  ✅ 条件4：你能提供具体可执行的修复方案
     - 如果是 retry_with_params：data 必须包含具体参数值（非空、非占位符）
     - 如果是 retry_with_tool：data 必须包含【可用工具】列表中的工具ID

  如果无法满足上述四个条件，**必须使用 "trigger_overall_reflection"**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【🔍 步骤2: 错误类型分类与应对策略】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

如果不是重复错误，根据错误类型选择相应策略：

┌────────────────────────────────────────────────────┐
│ 📌 策略A: 参数修复 (retry_with_params)              │
│ 适用场景：错误明确指向参数问题                     │
└────────────────────────────────────────────────────┘

■ 识别特征：
  • HTTP 400 系列错误
  • 错误信息包含："参数格式错误"、"参数值无效"、"缺少必需参数"
  • 参数名称拼写错误、参数类型不匹配
  • JSON格式错误、数据结构不符合要求
  • 即使是 HTTP 500，但错误信息明确指向参数问题

■ 示例场景：
  ❌ 错误信息："Bad Request: Missing required field 'client_id'"
     → root_cause_category: "parameter_error"
     → root_cause: "缺少必需参数 client_id"
     → is_recoverable: true
     → suggested_action: {
         "type": "retry_with_params",
         "data": {
           "client_id": "client_123"
         }
       }
     → analysis: "错误信息明确指出缺少 client_id 参数，需要在请求中添加该参数。
                  历史中未尝试过此方案，本次建议添加 client_id 参数重试。"

  ❌ 错误信息："Invalid parameter: timeout=500 exceeds maximum allowed value (300)"
     → root_cause_category: "parameter_error"
     → root_cause: "参数 timeout 值超出有效范围"
     → is_recoverable: true
     → suggested_action: {
         "type": "retry_with_params",
         "data": {
           "timeout": 300
         }
       }
     → analysis: "timeout 参数值为 500，超过最大允许值 300。
                  调整为 300 后应能解决问题。"

■ ⚠️⚠️⚠️ 关键要求（强制，否则输出无效）：
  当使用 retry_with_params 时：
  - data 字段**必须**是对象，包含具体的参数名和修正后的参数值
  - **绝对禁止**返回空对象 {} 或省略 data 字段
  - **绝对禁止**返回占位符值（如 "修正后的值"、"correct_value"）
  - **必须**提供可以直接使用的、具体的参数值
  - 如果无法确定具体参数值，**必须**使用 "replan" 或 "stop"

■ 判断标准：
  核心问题："修改传入的参数能否解决这个问题？"
  如果答案是"能" → 使用策略A

┌────────────────────────────────────────────────────┐
│ 📌 策略B: 工具替换 (retry_with_tool)                │
│ 适用场景：工具选择不当，但可用工具列表中有替代工具 │
└────────────────────────────────────────────────────┘

■ 识别特征：
  • 工具功能与任务目标不匹配
  • 【可用工具】列表中有合适的替代工具
  • 只需更换工具即可解决问题（不需要调整整体方案）

■ 示例场景：

  【场景1: retry_with_tool - 单步工具替换】
  ❌ 场景："使用查询工具尝试创建资源，返回错误'不支持创建操作'"
     → root_cause_category: "tool_error"
     → root_cause: "工具选择错误，查询工具无法执行创建操作"
     → is_recoverable: true
     → suggested_action: {
         "type": "retry_with_tool",
         "data": {
           "tool_id": "create_client_tool"
         }
       }
     → analysis: "当前使用的是查询工具，但任务需要创建操作。
                  【可用工具】列表中有 create_client_tool，建议使用该工具。"
     → alternative_solutions: ["使用 create_client_tool 创建客户端"]

  【场景2: trigger_overall_reflection - 任务分解问题】
  ❌ 场景："步骤执行顺序错误，依赖关系未满足"
     → root_cause_category: "decomposition_error"
     → root_cause: "任务分解逻辑错误，步骤依赖关系倒置"
     → is_recoverable: false
     → suggested_action: {
         "type": "trigger_overall_reflection",
         "data": "任务分解存在问题，步骤B依赖步骤A的输出但步骤A尚未执行，需要整体反思来决定是否重新规划执行顺序"
       }
     → analysis: "当前步骤B依赖步骤A的输出，但步骤A尚未执行。
                  这是任务分解的问题，单步修复无法解决，需要触发整体反思。"
     → alternative_solutions: ["触发整体反思重新规划任务执行顺序", "添加前置依赖检查"]

  【场景3: trigger_overall_reflection - 任务粒度问题】
  ❌ 场景："单个步骤包含多个操作，部分操作失败导致整个步骤失败"
     → root_cause_category: "decomposition_error"
     → root_cause: "任务分解粒度过粗，单个步骤职责过多"
     → is_recoverable: false
     → suggested_action: {
         "type": "trigger_overall_reflection",
         "data": "任务分解粒度过粗，需要整体反思决定如何拆分步骤"
       }
     → analysis: "当前步骤尝试一次性完成验证、创建和配置，但验证阶段就失败了。
                  这需要调整任务分解策略，单步修复无法解决，需要触发整体反思。"
     → alternative_solutions: ["触发整体反思优化任务分解", "添加步骤间状态检查"]

  【场景4: trigger_overall_reflection - 工具库无法支持】
  ❌ 场景："工具A无法满足需求，且可用工具列表中没有合适的替代工具"
     → root_cause_category: "decomposition_error"
     → root_cause: "当前工具库无法完成该任务"
     → is_recoverable: false
     → suggested_action: {
         "type": "trigger_overall_reflection",
         "data": "当前工具库无法完成该任务，触发整体反思以决定是否停止任务或寻找其他解决方案"
       }
     → analysis: "分析了【可用工具】列表，没有找到合适的替代工具。
                  触发整体反思，由整体反思决定后续行动。"

■ ⚠️⚠️⚠️ 工具约束（严格限制）：
  - 备选工具**必须**从【可用工具】列表中选择
  - **严禁**推荐列表之外的工具
  - **严禁**臆造不存在的工具
  - 如果没有合适的备选工具，必须选择 "trigger_overall_reflection"

■ 判断标准：
  核心问题："当前工具能否完成任务目标？"
  如果答案是"不能" → 使用策略B

┌────────────────────────────────────────────────────┐
│ 📌 策略C: 触发整体反思 (trigger_overall_reflection) │
│ 适用场景：外部/服务端问题，或需要更高层次决策      │
└────────────────────────────────────────────────────┘

■ 识别特征：
  • HTTP 500 系列错误，且错误信息指向服务端问题
  • 服务不可用、连接超时、网络错误
  • 进程启动失败、工具代码bug、依赖缺失
  • 权限问题、资源配额不足
  • 已重复重试多次，无法解决
  • 需要更高层次的决策（单步反思无法独立决定）

■ 说明：
  当单步反思判断问题无法在当前步骤层面解决时，应触发整体反思。
  整体反思会综合考虑整个任务的执行历史，决定是：
  - 停止任务执行（需要人工介入）
  - 重新规划整个任务（调整方案继续执行）

■ 示例场景：
  ❌ 错误信息："Service Unavailable: database connection failed"
     → root_cause_category: "server_error"
     → root_cause: "数据库连接失败，属于服务端问题"
     → is_recoverable: false
     → suggested_action: {
         "type": "trigger_overall_reflection",
         "data": "服务端错误（数据库连接失败），触发整体反思以决定是否停止任务或等待服务恢复后重试"
       }
     → analysis: "错误信息显示数据库连接失败，这是服务端基础设施问题，
                  无法通过修改参数或更换工具解决。触发整体反思进行更高层次的决策。"

  ❌ 错误信息："Failed to start process: permission denied"
     → root_cause_category: "external_error"
     → root_cause: "权限不足，无法启动进程"
     → is_recoverable: false
     → suggested_action: {
         "type": "trigger_overall_reflection",
         "data": "权限问题（进程启动失败），触发整体反思以决定后续处理方案"
       }
     → analysis: "进程启动失败，原因是权限不足。
                  这需要管理员调整服务端权限配置，触发整体反思决定是否停止任务。"

■ 判断标准：
  核心问题："问题是否由外部因素或服务端问题导致？或者是否需要更高层次的决策？"
  如果答案是"是" → 使用策略C（触发整体反思）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【📊 诊断输出格式 - 清晰友好】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**重要**: 你的诊断要让用户一眼就能看懂问题和解决方案。

请按以下结构输出分析（在返回JSON之前）：

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 🔍 步骤分析报告

### 📋 基本信息
- **步骤**: {step_id} - {step_description}
- **使用工具**: {tool_id}
- **重试次数**: 第 {retry_count} 次

---

### 🎯 问题诊断

#### 发生了什么
{用一句话描述核心问题，避免使用技术术语，让非技术人员也能理解}

#### 深入分析
{详细分析，分点列出，展示你的思考过程}
1. **直接现象**: {用户看到的现象}
2. **根本原因**: {问题的本质}
3. **为什么会这样**: {解释原因}

#### 📊 证据链
{列出支持你判断的证据}
- 📄 **错误信息**: {error_message}
- 🔧 **相关参数**: {从元数据中提取关键参数}
- 📜 **历史记录**: {如果检测到重复错误，简要说明}

---

### 💡 解决方案

{根据问题性质，选择对应的表述模板}

**【情况A: 参数问题 - 可以修复】**
好消息！这个问题可以通过调整参数解决 ✅

**需要调整的参数**:
- `param1`: 从 `old_value` 改为 `new_value`
- `param2`: 添加缺失的值 `xxx`

**为什么这样调整**:
{用通俗易懂的语言解释原因}

**预期效果**:
修改后应该能够 {达到什么效果}

---

**【情况B: 工具选择问题 - 需要换工具】**
看起来当前工具 `{tool_id}` 不太适合这个任务 🔄

**问题所在**:
{说明为什么当前工具不合适}

**我的建议**:
换用 `{alternative_tool_id}`，因为：
- ✅ 它支持 {功能X}
- ✅ 更适合处理 {场景Y}
- ✅ 成功率更高

---

**【情况C: 服务端/外部问题 - 需要更高层决策】**
这个问题比较棘手 😕 - 它不在我们直接控制范围内。

**问题性质**:
- 🔴 属于 {用通俗语言描述错误类型}
- 🔴 发生在 {哪个环节}
- 🔴 无法通过调整参数或换工具解决

**为什么无法立即修复**:
{解释问题的特殊性}

**我的建议**:
由于这不是简单的参数或工具问题，我建议：
1. ⏸️ 暂停当前执行
2. 🔍 **触发整体反思**，从全局角度评估
3. 🤔 决定是等待服务恢复、寻找替代方案，还是调整整体策略

**为什么需要整体反思**:
{明确解释为什么这个问题需要更高层次的决策}
- 单步调整无法解决根本问题
- 需要综合考虑整个任务的执行情况
- 可能需要调整整体执行策略

---

**【情况D: 重复错误 - 需要换思路】**
⚠️ 我注意到这个问题已经出现过 {N} 次了

**历史尝试**:
{列出之前的尝试和结果}
1. 第1次：{方案} → {结果}
2. 第2次：{方案} → {结果}

**为什么重复失败**:
{分析重复失败的根本原因}

**我的建议**:
继续用相同方式重试可能不是好主意。让我们：
- 🔄 **重新规划整个任务**，或
- 🔍 **触发整体反思**，寻找全新的解决思路

---

### 🔗 下一步行动

{明确的行动建议，用友好的语气}

✅ **立即行动**: {具体的建议步骤}
⏭️ **后续步骤**: {如果这步完成，接下来做什么}

{如果触发整体反思}
🔍 **已触发整体反思**

整体反思会：
- 从更高层次评估整个任务
- 综合考虑所有执行历史
- 给出更全面的解决方案

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ **输出要求**：
- 必须严格按照上述模板输出分析过程（在返回JSON前）
- 然后返回规定的JSON格式诊断结果
- 分析过程必须体现"证据→推理→结论"的逻辑链条
- 禁止无依据的猜测
- 禁止自相矛盾
- 禁止模糊表述

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【🚨 一致性检查（必须遵守）】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

输出前，必须检查以下一致性：

1. root_cause_category 与 suggested_action.type 一致性：
   - parameter_error → retry_with_params（仅当是首次失败或有新修复方案）
   - tool_error → retry_with_tool（仅当可用工具列表中有替代工具）
   - decomposition_error → trigger_overall_reflection（任务分解问题需要整体重新规划）
   - server_error/external_error → trigger_overall_reflection

2. is_recoverable 与 suggested_action.type 一致性：
   - retry_with_params/retry_with_tool → is_recoverable = true
   - trigger_overall_reflection → is_recoverable = false（由整体反思决定）

3. suggested_action.data 完整性：
   - retry_with_params → data 必须包含具体参数值（非空对象）
   - retry_with_tool → data 必须包含 tool_id（必须在可用工具列表中）
   - trigger_overall_reflection → data 必须包含触发整体反思的原因说明

4. 重复错误检测：
   - 如果检测到重复错误 → suggested_action.type 必须是 "trigger_overall_reflection"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【confidence 评分标准】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 90-100：证据充分，因果链条清晰，高度确信
- 70-89：证据较充分，推理合理，较为确信
- 50-69：有一定证据，但存在不确定性
- 50以下：证据不足，主要基于推测"#;


/// 单步反思用户提示词模板
const STEP_REFLECTION_USER_TEMPLATE: &str = r#"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【🚨🚨🚨 最高优先级警告 🚨🚨🚨】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

在分析任何错误之前，你**必须**首先检查【执行历史】！

🚨 强制规则（违反将导致无限循环，系统严重故障）：
1. 如果相同 tool_id 的错误在历史中出现 ≥2 次 → **必须**返回 "trigger_overall_reflection"
2. **绝对禁止**在重复错误情况下返回 "retry_with_params"
3. **绝对禁止**返回空的参数修正（data: {}）
4. 如果返回 "retry_with_params"，**必须**在 data 中提供具体的参数值
5. **单步反思不能直接触发重新规划**，必须通过 trigger_overall_reflection 让整体反思决定

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【🔴 执行历史 - 必须首先阅读！】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ 以下是之前的执行记录和反思内容，请仔细阅读以避免重复反思！

{execution_history}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【当前失败步骤】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 步骤 ID: {step_id}
- 工具 ID: {tool_id}
- 步骤描述: {step_description}
- 当前重试次数: 第 {retry_count} 次

📊 重试次数参考指南（仅供参考，最终由你决定）：
   • 1-2 次：正常范围，可根据错误类型调整参数重试
   • 3 次：建议仔细分析是否是参数问题，还是需要换工具/重规划
   • 4-5 次：⚠️ 超出常规范围，很可能当前方案存在根本性问题
   • 5 次以上：🚨 强烈建议 trigger_overall_reflection，除非有明确的新解决思路

⚠️ 重要提醒：
   - 是否继续重试完全由你决定，没有强制的次数限制
   - 但请注意：连续多次重试相同类型的错误，往往说明问题不在参数层面
   - 当无法提供新见解时，应果断建议 trigger_overall_reflection

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【错误信息】（诊断的核心证据）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{error_message}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【当前元数据】（参数相关证据）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{metadata}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【可用工具】（严格限制范围）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️⚠️⚠️ **工具使用约束 - 最高优先级**：
- 以下是当前任务可用的**全部**工具
- **严禁**推荐、建议、或使用列表之外的任何工具
- **严禁**臆造不存在的工具或假设某个工具存在
- 如果列表中没有合适的工具，必须建议 "trigger_overall_reflection"

{available_tools}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【场景指导】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{scene_specific_guidance}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【诊断任务 - 请严格按步骤执行】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ **重要提醒**：你必须先输出结构化的分析过程，然后再输出JSON结果！

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**诊断流程**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

请按照【诊断结构化要求】中的模板，先输出分析过程，格式如下：

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 📋 1. 执行历史审查（最高优先级）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【重复错误检测】
- 相同工具 tool_id="{tool_id}" 失败次数：X 次
- 错误信息是否相同：[是/否]
- 历史尝试方案：
  1. [方案1及结果]
  2. [方案2及结果]

【判定结果】
- 是否重复错误：[是/否]
- 如果是，为什么必须停止单步修复：[理由]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 🔍 2. 错误分析
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【证据收集】
- HTTP状态码：[如有]
- 错误类型：[从错误信息中提取]
- 错误描述：[具体描述]
- 相关参数：[从元数据中提取]

【因果推理】
1. 观察：[具体的错误表现]
   └─ 证据：[对应的错误信息/日志]
   └─ 推断：[基于证据的推理]
   └─ 结论：[根本原因]

【排除分析】
- 排除原因1：[为什么不是这个原因]
- 排除原因2：[为什么不是这个原因]

【最终结论】
- 根本原因分类：[parameter_error/tool_error/server_error/external_error等]
- 根本原因描述：[一句话概括]
- 可恢复性：[可/不可]
- 可信度：X%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 💡 3. 建议方案
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【主要方案】
- 方案类型：[retry_with_params/retry_with_tool/trigger_overall_reflection]
- 具体内容：
  [如果是 retry_with_params，列出具体参数修正]
  [如果是 retry_with_tool，说明备选工具及理由]
  [如果是 trigger_overall_reflection，说明触发理由]
- 预期效果：[解决什么问题]

【备选方案】（如有）
1. [备选方案1]
2. [备选方案2]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

然后输出JSON格式的诊断结果：

```json
{{
  "root_cause_category": "parameter_error/tool_error/dependency_error/decomposition_error/external_error/server_error",
  "root_cause": "具体根本原因（一句话概括，必须基于证据）",
  "is_recoverable": boolean,
  "confidence": 0-100,
  "analysis": "详细分析过程摘要（从上面的结构化分析中提炼）",
  "suggested_action": {{
    "type": "retry_with_params/retry_with_tool/trigger_overall_reflection",
    "data": {{}}
  }},
  "alternative_solutions": ["方案1", "方案2", ...]
}}
```

⚠️⚠️⚠️ **关键检查点**：
1. 重复错误检测：
   - 如果 tool_id="{tool_id}" 在历史中失败 ≥2 次
   - 必须返回 "trigger_overall_reflection"
   - 禁止使用 "retry_with_params" 或 "retry_with_tool"

2. 参数修正：
   - 如果返回 "retry_with_params"，data 必须包含具体参数值（非空）
   - 如果返回 "retry_with_tool"，data 中的 tool_id 必须在【可用工具】列表中

3. 一致性检查：
   - root_cause_category、is_recoverable、suggested_action.type 必须一致
   - analysis 必须体现"证据→推理→结论"逻辑链条

4. 重新规划限制：
   - 单步反思不能直接触发重新规划
   - 所有需要重新规划的情况都必须使用 trigger_overall_reflection

请开始诊断分析："#;

// ==================== 场景特定反思指导 ====================
// 注意：场景指导内容已迁移到 scene_guidance.rs 统一管理

// ==================== 反思提示词构建器 ====================

/// 反思提示词构建器
pub struct ReflectionPromptBuilder {
    /// 统一的场景管理器
    scene_manager: SceneManager,
}

impl ReflectionPromptBuilder {
    /// 创建新的反思提示词构建器
    pub fn new() -> Self {
        Self {
            scene_manager: SceneManager::new(),
        }
    }

    /// 构建反思提示词
    ///
    /// # 参数
    /// - `task_description`: 任务描述
    /// - `original_plan`: 原始计划文本
    /// - `evaluation_result`: 评估结果文本
    /// - `current_round`: 当前轮次
    /// - `max_rounds`: 最大轮次（已弃用，保留用于向后兼容）
    /// - `context`: 反思上下文信息
    ///
    /// # 返回
    /// (system_prompt, user_prompt)
    ///
    /// # 注意
    /// `max_rounds` 参数已弃用，是否继续反思完全由大模型决定。
    /// 保留此参数仅用于向后兼容。
    pub fn build_reflection_prompt(
        &self,
        task_description: &str,
        original_plan: &str,
        evaluation_result: &str,
        current_round: u32,
        #[allow(unused_variables)]
        max_rounds: u32, // 已弃用，不再使用
        context: &ReflectionContext,
    ) -> (String, String) {
        // 1. 根据 task_type 选择场景特定的反思指导
        let scene_guidance = self
            .scene_manager
            .get_reflection_guidance(context.task_type.as_deref());

        // 2. 格式化执行历史
        let execution_history_text = context.format_execution_history();

        // 3. 组装用户提示词（不再包含 max_rounds，由大模型自主决定是否继续）
        let user_prompt = BASE_REFLECTION_USER_TEMPLATE
            .replace("{task_description}", task_description)
            .replace("{current_round}", &current_round.to_string())
            .replace("{execution_history}", &execution_history_text)
            .replace("{original_plan}", original_plan)
            .replace("{evaluation_result}", evaluation_result)
            .replace("{scene_specific_guidance}", scene_guidance);

        // 4. 返回系统提示词和用户提示词
        (BASE_REFLECTION_SYSTEM_PROMPT.to_string(), user_prompt)
    }

    /// 获取支持的任务类型列表
    pub fn supported_task_types(&self) -> Vec<String> {
        self.scene_manager
            .all_task_type_names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// 构建单步反思提示词
    ///
    /// # 参数
    /// - `step_id`: 步骤ID
    /// - `tool_id`: 工具ID
    /// - `step_description`: 步骤描述
    /// - `error_message`: 错误消息
    /// - `metadata`: 元数据（格式化为字符串）
    /// - `available_tools`: 可用工具列表说明（可选）
    /// - `task_type`: 任务类型（可选）
    /// - `execution_history`: 执行历史（可选）
    /// - `retry_count`: 当前重试次数（可选，默认为1）
    ///
    /// # 返回
    /// (system_prompt, user_prompt)
    pub fn build_step_reflection_prompt(
        &self,
        step_id: &str,
        tool_id: &str,
        step_description: &str,
        error_message: &str,
        metadata: &str,
        available_tools: Option<&str>,
        task_type: Option<&str>,
        execution_history: Option<&str>,
        retry_count: Option<u32>,
    ) -> (String, String) {
        // 1. 构建工具列表说明
        let tools_section = if let Some(tools) = available_tools {
            format!(
                "\n**可用工具列表**（已筛选，与任务相关）\n{}\n\n⚠️ 如果需要推荐备选工具，必须从上述工具列表中选择！\n",
                tools
            )
        } else {
            String::new()
        };

        // 2. 获取场景特定的反思指导
        let scene_guidance = self
            .scene_manager
            .get_reflection_guidance(task_type);

        // 3. 格式化执行历史
        let execution_history_text = execution_history
            .unwrap_or("（暂无执行历史）");

        // 4. 获取重试次数（默认为1）
        let retry_count_value = retry_count.unwrap_or(1);

        // 5. 组装用户提示词
        let user_prompt = STEP_REFLECTION_USER_TEMPLATE
            .replace("{step_id}", step_id)
            .replace("{tool_id}", tool_id)
            .replace("{step_description}", step_description)
            .replace("{retry_count}", &retry_count_value.to_string())
            .replace("{error_message}", error_message)
            .replace("{metadata}", metadata)
            .replace("{available_tools}", &tools_section)
            .replace("{execution_history}", execution_history_text)
            .replace("{scene_specific_guidance}", scene_guidance);

        // 6. 返回系统提示词和用户提示词
        (STEP_REFLECTION_SYSTEM_PROMPT.to_string(), user_prompt)
    }
}

impl Default for ReflectionPromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflection_context_creation() {
        let context = ReflectionContext::new(Some("自动建模".to_string()));
        assert_eq!(context.task_type, Some("自动建模".to_string()));
        assert!(context.user_context.is_none());
    }

    #[test]
    fn test_set_user_context() {
        let mut context = ReflectionContext::new(None);
        context.set_user_context("用户提供的上下文信息".to_string());
        assert!(context.user_context.is_some());
        assert_eq!(context.user_context.unwrap(), "用户提供的上下文信息");
    }

    #[test]
    fn test_format_user_context_empty() {
        let context = ReflectionContext::new(None);
        let formatted = context.format_user_context();
        assert_eq!(formatted, "（用户未提供任务上下文）");
    }

    #[test]
    fn test_format_user_context_with_data() {
        let mut context = ReflectionContext::new(None);
        context.set_user_context("这是用户上下文信息".to_string());
        let formatted = context.format_user_context();
        assert!(formatted.contains("这是用户上下文信息"));
    }

    #[test]
    #[allow(deprecated)]
    fn test_add_successful_step_backward_compat() {
        let mut context = ReflectionContext::new(None);
        context.add_successful_step(SuccessfulStep {
            step_name: "测试步骤".to_string(),
            step_id: "step_1".to_string(),
            output_summary: "成功输出".to_string(),
            timestamp: None,
        });
        assert!(context.user_context.is_some());
    }

    #[test]
    #[allow(deprecated)]
    fn test_format_successful_steps_backward_compat() {
        let mut context = ReflectionContext::new(None);
        context.add_successful_step(SuccessfulStep {
            step_name: "步骤1".to_string(),
            step_id: "step_1".to_string(),
            output_summary: "输出1".to_string(),
            timestamp: None,
        });
        let formatted = context.format_successful_steps();
        assert!(formatted.contains("步骤1"));
        assert!(formatted.contains("step_1"));
    }

    #[test]
    fn test_builder_creation() {
        let builder = ReflectionPromptBuilder::new();
        let supported = builder.supported_task_types();
        assert!(supported.contains(&"自然语言建模".to_string()));
        assert!(supported.contains(&"工具管理".to_string()));
        assert!(supported.contains(&"客户端管理".to_string()));
        assert!(supported.contains(&"PLC控制器".to_string()));
    }

    #[test]
    fn test_build_prompt_with_tool_management() {
        let builder = ReflectionPromptBuilder::new();
        let context = ReflectionContext::new(Some("工具管理".to_string()));

        let (system, user) = builder.build_reflection_prompt(
            "创建计算器工具",
            "原始计划",
            "评估结果",
            1,
            3,
            &context,
        );

        assert!(system.contains("反思专家"));
        assert!(user.contains("创建计算器工具"));
    }

    #[test]
    fn test_build_prompt_with_default_guidance() {
        let builder = ReflectionPromptBuilder::new();
        let context = ReflectionContext::new(Some("未知场景".to_string()));

        let (_, user) = builder.build_reflection_prompt(
            "测试任务",
            "原始计划",
            "评估结果",
            1,
            3,
            &context,
        );

        assert!(user.contains("测试任务"));
    }
}
