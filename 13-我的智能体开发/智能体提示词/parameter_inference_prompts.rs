//! 动态参数推断提示词模块
//!
//! 用于在步骤执行时根据上下文动态推断工具参数

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 动态参数推断的系统提示词
pub const PARAMETER_INFERENCE_SYSTEM_PROMPT: &str = r#"你是一个专业的工具参数推断专家。你的任务是根据执行上下文为当前步骤的工具调用生成准确的参数。

## 核心原则

1. **精确性**：参数值必须完全匹配工具的输入参数定义
2. **上下文感知**：充分利用已执行步骤的输出和元数据
3. **类型正确**：确保参数类型与工具定义一致（字符串、数字、布尔值、数组、对象）
4. **最小必要**：只生成必需的参数，可选参数仅在有明确依据时才生成

## 参数来源优先级

1. **前置步骤输出**：优先从已完成步骤的输出中提取参数
2. **任务元数据**：从任务初始化时传入的 metadata 中获取
3. **工具默认值**：当工具定义了默认值且无其他来源时使用
4. **合理推断**：仅在有足够上下文支撑时进行推断

## 输出格式

你必须输出一个有效的 JSON 对象，格式如下：
```json
{
  "parameters": {
    "参数名1": "参数值1",
    "参数名2": 123,
    "参数名3": ["item1", "item2"]
  },
  "reasoning": "简要说明参数推断的依据"
}
```

## 注意事项

- 如果某个必需参数无法确定，在 reasoning 中说明原因，但仍需提供最佳猜测
- 参数值应该是原始类型，不要使用占位符如 `{{xxx}}`
- 对于复杂对象或数组，确保 JSON 格式正确
"#;

/// 动态参数推断的用户提示词模板
pub const PARAMETER_INFERENCE_USER_TEMPLATE: &str = r#"请为以下步骤推断工具调用参数。

## 当前步骤信息

- **步骤ID**: {step_id}
- **步骤名称**: {step_name}
- **步骤描述**: {step_description}
- **调用工具**: {tool_id}

## 工具信息

### 工具描述
{tool_description}

### 输入参数定义
```json
{tool_input_params}
```

### 输出参数定义
```json
{tool_output_params}
```

## 执行上下文

### 任务元数据 (Metadata)
```json
{metadata}
```

### 已完成步骤的执行历史
{execution_history}

## 规划阶段的参数参考（可能包含占位符）
```json
{planned_parameters}
```

---

请根据以上信息，为工具 `{tool_id}` 生成准确的调用参数。输出 JSON 格式的结果。
"#;

/// 参数推断结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterInferenceResult {
    /// 推断出的参数
    pub parameters: HashMap<String, serde_json::Value>,
    /// 推断依据说明
    pub reasoning: String,
}

/// 参数推断上下文
#[derive(Debug, Clone)]
pub struct ParameterInferenceContext {
    /// 步骤ID
    pub step_id: String,
    /// 步骤名称
    pub step_name: String,
    /// 步骤描述
    pub step_description: String,
    /// 工具ID
    pub tool_id: String,
    /// 工具描述
    pub tool_description: String,
    /// 工具输入参数定义（JSON字符串）
    pub tool_input_params: String,
    /// 工具输出参数定义（JSON字符串）
    pub tool_output_params: String,
    /// 任务元数据
    pub metadata: HashMap<String, String>,
    /// 已完成步骤的执行历史
    pub execution_history: Vec<StepExecutionRecord>,
    /// 规划阶段的参数（可能包含占位符）
    pub planned_parameters: String,
}

/// 步骤执行记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepExecutionRecord {
    /// 步骤ID
    pub step_id: String,
    /// 步骤名称
    pub step_name: String,
    /// 工具ID
    pub tool_id: String,
    /// 是否成功
    pub is_success: bool,
    /// 执行输出（JSON格式）
    pub output: String,
    /// 使用的参数
    pub parameters: HashMap<String, String>,
}

/// 参数推断提示词构建器
pub struct ParameterInferencePromptBuilder;

impl ParameterInferencePromptBuilder {
    /// 构建用户提示词
    pub fn build_user_prompt(context: &ParameterInferenceContext) -> String {
        // 格式化元数据为 JSON
        let metadata_json = serde_json::to_string_pretty(&context.metadata)
            .unwrap_or_else(|_| "{}".to_string());

        // 格式化执行历史
        let execution_history = Self::format_execution_history(&context.execution_history);

        PARAMETER_INFERENCE_USER_TEMPLATE
            .replace("{step_id}", &context.step_id)
            .replace("{step_name}", &context.step_name)
            .replace("{step_description}", &context.step_description)
            .replace("{tool_id}", &context.tool_id)
            .replace("{tool_description}", &context.tool_description)
            .replace("{tool_input_params}", &context.tool_input_params)
            .replace("{tool_output_params}", &context.tool_output_params)
            .replace("{metadata}", &metadata_json)
            .replace("{execution_history}", &execution_history)
            .replace("{planned_parameters}", &context.planned_parameters)
    }

    /// 格式化执行历史
    fn format_execution_history(history: &[StepExecutionRecord]) -> String {
        if history.is_empty() {
            return "（暂无已完成的步骤）".to_string();
        }

        let mut result = String::new();
        for (idx, record) in history.iter().enumerate() {
            result.push_str(&format!(
                "\n### 步骤 {} - {} ({})\n",
                idx + 1,
                record.step_name,
                record.step_id
            ));
            result.push_str(&format!("- **工具**: {}\n", record.tool_id));
            result.push_str(&format!(
                "- **状态**: {}\n",
                if record.is_success { "✅ 成功" } else { "❌ 失败" }
            ));

            // 格式化使用的参数
            if !record.parameters.is_empty() {
                result.push_str("- **输入参数**:\n```json\n");
                if let Ok(params_json) = serde_json::to_string_pretty(&record.parameters) {
                    result.push_str(&params_json);
                }
                result.push_str("\n```\n");
            }

            // 格式化输出（截断过长的输出）
            result.push_str("- **输出**:\n```json\n");
            let output_display = if record.output.len() > 1000 {
                format!("{}... (输出已截断，共{}字节)", &record.output[..1000], record.output.len())
            } else {
                record.output.clone()
            };
            result.push_str(&output_display);
            result.push_str("\n```\n");
        }

        result
    }

    /// 获取系统提示词
    pub fn get_system_prompt() -> &'static str {
        PARAMETER_INFERENCE_SYSTEM_PROMPT
    }

    /// 解析LLM响应，提取参数推断结果
    pub fn parse_response(response: &str) -> Result<ParameterInferenceResult, String> {
        // 尝试从响应中提取 JSON
        let json_str = Self::extract_json_from_response(response)?;

        // 解析 JSON
        serde_json::from_str::<ParameterInferenceResult>(&json_str)
            .map_err(|e| format!("解析参数推断结果失败: {}", e))
    }

    /// 从响应中提取 JSON 内容
    fn extract_json_from_response(response: &str) -> Result<String, String> {
        // 尝试直接解析
        if response.trim().starts_with('{') {
            return Ok(response.trim().to_string());
        }

        // 尝试从 markdown 代码块中提取
        if let Some(start) = response.find("```json") {
            let start = start + 7;
            if let Some(end) = response[start..].find("```") {
                return Ok(response[start..start + end].trim().to_string());
            }
        }

        // 尝试从普通代码块中提取
        if let Some(start) = response.find("```") {
            let start = start + 3;
            // 跳过可能的语言标识符
            let start = response[start..].find('\n').map(|n| start + n + 1).unwrap_or(start);
            if let Some(end) = response[start..].find("```") {
                return Ok(response[start..start + end].trim().to_string());
            }
        }

        // 尝试找到 JSON 对象
        if let Some(start) = response.find('{') {
            // 找到匹配的闭合括号
            let mut depth = 0;
            let mut end = start;
            for (i, c) in response[start..].char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = start + i + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if end > start {
                return Ok(response[start..end].to_string());
            }
        }

        Err("无法从响应中提取JSON".to_string())
    }

    /// 将推断结果转换为 HashMap<String, String> 格式（兼容现有接口）
    pub fn result_to_string_map(result: &ParameterInferenceResult) -> HashMap<String, String> {
        let mut map = HashMap::new();
        for (key, value) in &result.parameters {
            let str_value = match value {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Null => String::new(),
                other => other.to_string(),
            };
            map.insert(key.clone(), str_value);
        }
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_response_direct_json() {
        let response = r#"{"parameters": {"name": "test"}, "reasoning": "test reason"}"#;
        let result = ParameterInferencePromptBuilder::parse_response(response);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_response_with_markdown() {
        let response = r#"
Here is the result:

```json
{"parameters": {"name": "test"}, "reasoning": "test reason"}
```
"#;
        let result = ParameterInferencePromptBuilder::parse_response(response);
        assert!(result.is_ok());
    }

    #[test]
    fn test_format_execution_history() {
        let history = vec![StepExecutionRecord {
            step_id: "step_1".to_string(),
            step_name: "获取数据".to_string(),
            tool_id: "get_data".to_string(),
            is_success: true,
            output: r#"{"data": "test"}"#.to_string(),
            parameters: HashMap::from([("id".to_string(), "123".to_string())]),
        }];

        let formatted = ParameterInferencePromptBuilder::format_execution_history(&history);
        assert!(formatted.contains("获取数据"));
        assert!(formatted.contains("get_data"));
        assert!(formatted.contains("✅ 成功"));
    }
}
