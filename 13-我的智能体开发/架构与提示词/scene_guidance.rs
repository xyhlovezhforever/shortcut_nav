//! 场景指导统一管理模块
//!
//! 提供跨阶段（Planning/Reflection/Replanning）的场景指导统一管理
//! 确保所有场景定义、别名、指导内容在一个地方维护

use std::collections::HashMap;

// ==================== 场景类型定义 ====================

/// 场景类型枚举（标准化所有场景）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SceneType {
    /// 自然语言建模场景
    NaturalLanguageModeling,
    /// 工具管理场景
    ToolManagement,
    /// PLC控制器场景
    PlcController,
    /// 客户端管理场景
    ClientManagement,
    /// 通用场景（默认）
    General,
}

impl SceneType {
    /// 获取场景的主名称（中文）
    pub fn primary_name(&self) -> &'static str {
        match self {
            Self::NaturalLanguageModeling => "自然语言建模",
            Self::ToolManagement => "工具管理",
            Self::PlcController => "PLC控制器",
            Self::ClientManagement => "客户端管理",
            Self::General => "通用",
        }
    }

    /// 获取场景的所有别名（用于匹配）
    pub fn aliases(&self) -> Vec<&'static str> {
        match self {
            Self::NaturalLanguageModeling => vec![
                "自然语言建模",
                "自动建模",
                "建模",
                "natural_language_modeling",
                "auto_modeling",
                "modeling",
            ],
            Self::ToolManagement => vec![
                "工具管理",
                "代码生成",
                "工具",
                "tool_management",
                "tool",
                "code_generation",
            ],
            Self::PlcController => vec![
                "PLC控制器",
                "plc控制器",
                "PLC",
                "plc",
                "plc_controller",
                "controller",
            ],
            Self::ClientManagement => vec![
                "客户端管理",
                "客户端操作",
                "客户端",
                "client_management",
                "client_operation",
                "client",
            ],
            Self::General => vec!["通用", "general", "default"],
        }
    }

    /// 获取场景的关键词（用于模糊匹配）
    pub fn keywords(&self) -> Vec<&'static str> {
        match self {
            Self::NaturalLanguageModeling => vec!["建模", "模型", "model", "仿真", "拓扑"],
            Self::ToolManagement => vec!["工具", "tool", "创建", "注册", "生成"],
            Self::PlcController => vec!["plc", "控制器", "controller", "设备"],
            Self::ClientManagement => vec!["客户端", "client", "角色", "role", "用户"],
            Self::General => vec![],
        }
    }
}

// ==================== 场景指导内容 ====================

/// 场景指导内容（包含所有阶段的提示词）
#[derive(Debug, Clone)]
pub struct SceneGuidance {
    /// 工具筛选阶段的场景指导
    pub selection_guidance: &'static str,
    /// 规划阶段的场景指导
    pub planning_guidance: &'static str,
    /// 执行评估阶段的场景指导
    pub evaluation_guidance: &'static str,
    /// 反思阶段的场景指导
    pub reflection_guidance: &'static str,
    /// 重新规划阶段的场景指导
    pub replanning_guidance: &'static str,
    /// 用户消息生成的场景指导
    pub message_guidance: &'static str,
}

// ==================== 各场景的指导内容定义 ====================

/// 自然语言建模场景的指导内容
const NATURAL_LANGUAGE_MODELING_GUIDANCE: SceneGuidance = SceneGuidance {
    selection_guidance: r#""#,

    planning_guidance: r#""#,

    evaluation_guidance: r#""#,

    reflection_guidance: r#""#,

    replanning_guidance: r#""#,

    message_guidance: r#""#,
};

/// 工具管理场景的指导内容
const TOOL_MANAGEMENT_GUIDANCE: SceneGuidance = SceneGuidance {
    selection_guidance: r#""#,

    planning_guidance: r#"
⚠️ **重要规划限制**

1. 如果用户任务是"创建xxx程序"、"开发xxx工具"、"生成xxx代码"等代码生成类任务：
- 默认只规划生成**方案文档**（设计方案、技术选型、架构设计等）
- **不要**直接规划生成具体代码以及后续的步骤
- 原因：方案生成后需要用户确认和选择具体方案，之后才能进行代码实现
只有当用户明确指定要实现某个具体方案时，才规划代码生成步骤。

2. 如果用户任务是"上线xxx工具"、"部署xxx工具"、"发布xxx工具"等工具上线类任务：

⚠️ ⚠️ ⚠️ **绝对强制规则：必须包含查询步骤！** ⚠️ ⚠️ ⚠️

**强制执行流程（不可省略任何步骤）：**
**第1步（强制必需）：查询工具代码状态**
- 目的：检查该工具是否已生成可执行代码
- 要求：从可用工具中选择能够查询工具列表或单个工具状态的工具
  （提示：查找工具名称或描述中包含"工具列表"、"工具状态"、"查询"、"list"、"status"等关键词的工具）
- ⚠️ 重要：方案文档 ≠ 可执行代码，必须明确查询代码实现状态
- ⚠️ 即使你从上下文判断代码已存在/不存在，也必须规划此查询步骤！
- ⚠️ 禁止：不要创造不存在的工具名称，必须从实际提供的工具列表中选择

**第2步（视第1步结果而定）：生成工具代码**
- 目的：如果代码不存在，则生成可执行代码
- 要求：从可用工具中选择代码生成类工具
- 依赖：第1步（查询步骤）

**第3步（强制必需）：测试工具**
- 目的：验证工具代码可正常运行
- 要求：从可用工具中选择测试类工具
- 依赖：第2步（生成步骤）

**第4步（强制必需）：注册并上线工具**
- 目的：将工具注册到系统并启动
- 要求：从可用工具中选择注册/上线类工具
- 依赖：第3步（测试步骤）

**🚨 绝对禁止的错误行为：**
❌ 跳过第1步查询步骤，直接规划生成/测试/上线
❌ 凭上下文信息自己判断代码是否存在
❌ 看到"方案已生成"就认为代码已生成



3.
    "#,

    evaluation_guidance: r#""#,

    reflection_guidance: r#"
1. 如果错误信息包含"找不到 tool_id 对应的工具"或"请确保目录名称与 tool_id 完全一致"，这是等待用户选择方案的提示，不是错误。返回 needs_replanning=false, suggested_action="wait_for_user"。

2. 判断是否为测试工具报错（工具名或步骤描述包含：test、run_tests、execute_tests、check、verify、测试、检查、验证）。如果不是，按常规反思流程处理。

3. 如果是测试工具报错，判断是否为代码问题（满足任一条件即为代码问题）：
   - 代码问题：SyntaxError、CompileError、TypeError、type mismatch、NameError、ReferenceError、undefined、not defined、ImportError、ModuleNotFoundError、AttributeError、assertion failed、test failed、包含文件路径和行号的运行时异常
   - 非代码问题：dependency missing、package not found、环境变量、permission denied、access forbidden、file not found、connection refused、timeout、config error

4. 如果是代码问题：
   - 返回 needs_replanning=true
   - suggested_action 为 retry_with_tool，从可选工具列表中选择代码更新工具
   - analysis 必须包含：错误类型、错误位置（文件和行号）、错误原因、修复方向

5. 如果是非代码问题，按常规反思流程处理，suggested_action 针对具体问题类型。
"#,

    replanning_guidance: r#"⚠️ **工具管理任务的重新规划指导**

## 适用场景
测试工具报错，且反思结果判定为代码问题需要修复时。

## 重新规划流程

### 步骤1：选择代码更新工具
从可选工具列表中选择代码更新工具：
- **要求**：必须从实际可用的工具列表中选择
- **要求**：工具名称必须与可选工具列表完全匹配
- **禁止**：使用不存在的工具

### 步骤2：构建新计划的步骤列表

新计划必须包含以下步骤（按顺序）：

#### 第1步：代码修复步骤
```json
{
  "step_id": 1,
  "description": "修复 [错误类型]：[简短描述]",
  "tool": "[步骤1选中的代码更新工具ID]",
  "params": {
    "file_path": "[需要修改的文件路径]",
    "error_type": "[错误类型，如 ImportError]",
    "error_message": "[原始错误信息摘要]",
    "fix_description": "[具体修复说明，来自反思分析的修复方向]"
    // 其他参数根据工具定义补充
  },
  "dependencies": []
}
```

#### 第2步：测试验证步骤
```json
{
  "step_id": 2,
  "description": "重新执行测试验证代码修复",
  "tool": "[原测试工具ID]",
  "params": {
    // 完全复制原测试步骤的参数
  },
  "dependencies": [1]
}
```

#### 第3步及以后：原计划的后续步骤（如有）
将原计划中未完成的步骤依次添加：
- 步骤ID从3开始递增
- 更新依赖关系：至少依赖步骤2（确保测试通过后才执行）
- 保持原步骤的工具和参数不变

### 步骤3：输出完整的新计划

**计划结构**：
```json
{
  "plan_id": "[新计划ID]",
  "goal": "[与原计划相同的目标]",
  "steps": [
    // 步骤1：代码修复
    // 步骤2：测试验证
    // 步骤3+：原计划后续步骤
  ]
}
```

## 完整示例

假设原计划有3个步骤：
1. 生成工具代码（已完成）
2. 测试工具（失败，触发重新规划）
3. 注册工具（未执行）

新计划应该是：

```json
{
  "plan_id": "plan_20240115_002",
  "goal": "创建并注册数据处理工具",
  "steps": [
    {
      "step_id": 1,
      "description": "修复 ImportError：补充缺失的函数导入",
      "tool": "update_tool_code",
      "params": {
        "file_path": "tools/data_processor/main.py",
        "error_type": "ImportError",
        "error_message": "cannot import name 'validate_input' from 'utils'",
        "fix_description": "在 utils 模块中实现 validate_input 函数，或修正导入语句使用正确的函数名"
      },
      "dependencies": []
    },
    {
      "step_id": 2,
      "description": "重新执行测试验证代码修复",
      "tool": "test_tool",
      "params": {
        "tool_id": "data_processor",
        "test_cases": ["basic_validation"]
      },
      "dependencies": [1]
    },
    {
      "step_id": 3,
      "description": "注册工具到系统",
      "tool": "register_tool",
      "params": {
        "tool_id": "data_processor"
      },
      "dependencies": [2]
    }
  ]
}
```

## 检查清单

在输出新计划前，确认以下各项：
- ✅ 第1步是代码修复，使用从可选工具中选择的更新工具
- ✅ 第2步是测试验证，使用原测试工具和参数
- ✅ 依赖关系正确：步骤2依赖步骤1，后续步骤依赖步骤2
- ✅ 代码修复步骤包含所有必需参数（file_path、error_type、error_message、fix_description）
- ✅ 工具ID存在于可选工具列表中
- ✅ 原计划的未完成步骤已添加到新计划

## 禁止事项

- ❌ 跳过代码修复，直接重试测试
- ❌ 使用不存在的工具名称
- ❌ 在测试步骤之前执行原计划的后续步骤
- ❌ 代码修复步骤缺少必需参数
- ❌ 遗漏原计划中未完成的步骤
"#,

    message_guidance: r#"⚠️ **工具管理任务的用户消息生成指导**

## 消息要点

### 工具设计方案生成任务
- 清晰说明工具设计方案已生成
- 提醒用户需要确认方案后才能实施
- 避免使用过于技术化的术语
- 使用简洁易懂的语言

### 其他工具管理任务
- 说明任务执行结果
- 必要时提供后续操作建议
"#,
};

/// PLC控制器场景的指导内容
const PLC_CONTROLLER_GUIDANCE: SceneGuidance = SceneGuidance {
    selection_guidance: r#""#,

    planning_guidance: r#""#,

    evaluation_guidance: r#""#,

    reflection_guidance: r#""#,

    replanning_guidance: r#""#,

    message_guidance: r#""#,
};

/// 客户端管理场景的指导内容
const CLIENT_MANAGEMENT_GUIDANCE: SceneGuidance = SceneGuidance {
    selection_guidance: r#""#,

    planning_guidance: r#""#,

    evaluation_guidance: r#""#,

    reflection_guidance: r#""#,

    replanning_guidance: r#""#,

    message_guidance: r#""#,
};

/// 通用场景的指导内容（默认）
const GENERAL_GUIDANCE: SceneGuidance = SceneGuidance {
    selection_guidance: r#""#,

    planning_guidance: r#""#,

    evaluation_guidance: r#""#,

    reflection_guidance: r#""#,

    replanning_guidance: r#""#,

    message_guidance: r#""#,
};

// ==================== 场景管理器 ====================

/// 场景管理器（单例，提供统一的查询接口）
pub struct SceneManager {
    /// 场景类型到别名的映射（用于快速查找）
    alias_map: HashMap<String, SceneType>,
}

impl SceneManager {
    /// 创建新的场景管理器
    pub fn new() -> Self {
        let mut alias_map = HashMap::new();

        // 注册所有场景的别名
        for scene_type in [
            SceneType::NaturalLanguageModeling,
            SceneType::ToolManagement,
            SceneType::PlcController,
            SceneType::ClientManagement,
            SceneType::General,
        ] {
            for alias in scene_type.aliases() {
                alias_map.insert(alias.to_lowercase(), scene_type);
            }
        }

        Self { alias_map }
    }

    /// 根据任务类型字符串匹配场景类型
    ///
    /// # 匹配策略
    /// 1. 精确匹配别名（不区分大小写）
    /// 2. 模糊匹配关键词
    /// 3. 默认返回通用场景
    pub fn match_scene(&self, task_type: Option<&str>) -> SceneType {
        let task_type_lower = task_type
            .map(|s| s.to_lowercase())
            .unwrap_or_else(|| "通用".to_string());

        // 1. 精确匹配别名
        if let Some(&scene_type) = self.alias_map.get(&task_type_lower) {
            return scene_type;
        }

        // 2. 模糊匹配关键词
        for scene_type in [
            SceneType::NaturalLanguageModeling,
            SceneType::ToolManagement,
            SceneType::PlcController,
            SceneType::ClientManagement,
        ] {
            for keyword in scene_type.keywords() {
                if task_type_lower.contains(&keyword.to_lowercase())
                    || keyword.to_lowercase().contains(&task_type_lower)
                {
                    return scene_type;
                }
            }
        }

        // 3. 默认返回通用场景
        SceneType::General
    }

    /// 获取场景的指导内容
    fn get_scene_guidance(&self, scene_type: SceneType) -> &SceneGuidance {
        match scene_type {
            SceneType::NaturalLanguageModeling => &NATURAL_LANGUAGE_MODELING_GUIDANCE,
            SceneType::ToolManagement => &TOOL_MANAGEMENT_GUIDANCE,
            SceneType::PlcController => &PLC_CONTROLLER_GUIDANCE,
            SceneType::ClientManagement => &CLIENT_MANAGEMENT_GUIDANCE,
            SceneType::General => &GENERAL_GUIDANCE,
        }
    }

    /// 获取工具筛选阶段的场景指导
    ///
    /// # 参数
    /// - `task_type`: 任务类型字符串（可选）
    ///
    /// # 返回
    /// 对应场景的筛选阶段指导字符串
    pub fn get_selection_guidance(&self, task_type: Option<&str>) -> &'static str {
        let scene_type = self.match_scene(task_type);
        self.get_scene_guidance(scene_type).selection_guidance
    }

    /// 获取规划阶段的场景指导
    ///
    /// # 参数
    /// - `task_type`: 任务类型字符串（可选）
    ///
    /// # 返回
    /// 对应场景的规划阶段指导字符串
    pub fn get_planning_guidance(&self, task_type: Option<&str>) -> &'static str {
        let scene_type = self.match_scene(task_type);
        self.get_scene_guidance(scene_type).planning_guidance
    }

    /// 获取执行评估阶段的场景指导
    ///
    /// # 参数
    /// - `task_type`: 任务类型字符串（可选）
    ///
    /// # 返回
    /// 对应场景的评估阶段指导字符串
    pub fn get_evaluation_guidance(&self, task_type: Option<&str>) -> &'static str {
        let scene_type = self.match_scene(task_type);
        self.get_scene_guidance(scene_type).evaluation_guidance
    }

    /// 获取反思阶段的场景指导
    ///
    /// # 参数
    /// - `task_type`: 任务类型字符串（可选）
    ///
    /// # 返回
    /// 对应场景的反思阶段指导字符串
    pub fn get_reflection_guidance(&self, task_type: Option<&str>) -> &'static str {
        let scene_type = self.match_scene(task_type);
        self.get_scene_guidance(scene_type).reflection_guidance
    }

    /// 获取重新规划阶段的场景指导
    ///
    /// # 参数
    /// - `task_type`: 任务类型字符串（可选）
    ///
    /// # 返回
    /// 对应场景的重新规划阶段指导字符串
    pub fn get_replanning_guidance(&self, task_type: Option<&str>) -> &'static str {
        let scene_type = self.match_scene(task_type);
        self.get_scene_guidance(scene_type).replanning_guidance
    }

    /// 获取用户消息生成的场景指导
    ///
    /// # 参数
    /// - `task_type`: 任务类型字符串（可选）
    ///
    /// # 返回
    /// 对应场景的消息生成指导字符串
    pub fn get_message_guidance(&self, task_type: Option<&str>) -> &'static str {
        let scene_type = self.match_scene(task_type);
        self.get_scene_guidance(scene_type).message_guidance
    }

    /// 获取所有支持的场景类型
    pub fn all_scene_types(&self) -> Vec<SceneType> {
        vec![
            SceneType::NaturalLanguageModeling,
            SceneType::ToolManagement,
            SceneType::PlcController,
            SceneType::ClientManagement,
            SceneType::General,
        ]
    }

    /// 获取所有支持的任务类型名称（主名称）
    pub fn all_task_type_names(&self) -> Vec<&'static str> {
        self.all_scene_types()
            .iter()
            .map(|st| st.primary_name())
            .collect()
    }
}

impl Default for SceneManager {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== 单元测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_type_primary_name() {
        assert_eq!(SceneType::NaturalLanguageModeling.primary_name(), "自然语言建模");
        assert_eq!(SceneType::ToolManagement.primary_name(), "工具管理");
        assert_eq!(SceneType::PlcController.primary_name(), "PLC控制器");
        assert_eq!(SceneType::ClientManagement.primary_name(), "客户端管理");
        assert_eq!(SceneType::General.primary_name(), "通用");
    }

    #[test]
    fn test_scene_type_aliases() {
        let aliases = SceneType::ToolManagement.aliases();
        assert!(aliases.contains(&"工具管理"));
        assert!(aliases.contains(&"tool_management"));
        assert!(aliases.contains(&"代码生成"));
    }

    #[test]
    fn test_scene_manager_match_exact() {
        let manager = SceneManager::new();

        // 精确匹配 - 中文
        assert_eq!(
            manager.match_scene(Some("工具管理")),
            SceneType::ToolManagement
        );
        assert_eq!(
            manager.match_scene(Some("自然语言建模")),
            SceneType::NaturalLanguageModeling
        );
        assert_eq!(
            manager.match_scene(Some("PLC控制器")),
            SceneType::PlcController
        );
        assert_eq!(
            manager.match_scene(Some("客户端管理")),
            SceneType::ClientManagement
        );

        // 精确匹配 - 英文
        assert_eq!(
            manager.match_scene(Some("tool_management")),
            SceneType::ToolManagement
        );
        assert_eq!(
            manager.match_scene(Some("plc_controller")),
            SceneType::PlcController
        );
    }

    #[test]
    fn test_scene_manager_match_fuzzy() {
        let manager = SceneManager::new();

        // 模糊匹配 - 关键词
        assert_eq!(
            manager.match_scene(Some("plc设备控制")),
            SceneType::PlcController
        );
        assert_eq!(
            manager.match_scene(Some("创建工具任务")),
            SceneType::ToolManagement
        );
    }

    #[test]
    fn test_scene_manager_match_default() {
        let manager = SceneManager::new();

        // 默认场景
        assert_eq!(manager.match_scene(Some("未知场景")), SceneType::General);
        assert_eq!(manager.match_scene(None), SceneType::General);
    }

    #[test]
    fn test_scene_manager_get_planning_guidance() {
        let manager = SceneManager::new();

        let guidance = manager.get_planning_guidance(Some("工具管理"));
        assert!(guidance.contains("创建工具的特殊规则"));
        assert!(guidance.contains("只规划一个步骤"));
    }

    #[test]
    fn test_scene_manager_get_reflection_guidance() {
        let manager = SceneManager::new();

        let guidance = manager.get_reflection_guidance(Some("工具管理"));
        assert!(guidance.contains("特殊场景处理"));
        assert!(guidance.contains("无需重新规划"));
    }

    #[test]
    fn test_scene_manager_all_scene_types() {
        let manager = SceneManager::new();
        let all_types = manager.all_scene_types();

        assert_eq!(all_types.len(), 5);
        assert!(all_types.contains(&SceneType::NaturalLanguageModeling));
        assert!(all_types.contains(&SceneType::ToolManagement));
        assert!(all_types.contains(&SceneType::PlcController));
        assert!(all_types.contains(&SceneType::ClientManagement));
        assert!(all_types.contains(&SceneType::General));
    }

    #[test]
    fn test_scene_manager_all_task_type_names() {
        let manager = SceneManager::new();
        let names = manager.all_task_type_names();

        assert_eq!(names.len(), 5);
        assert!(names.contains(&"自然语言建模"));
        assert!(names.contains(&"工具管理"));
        assert!(names.contains(&"PLC控制器"));
        assert!(names.contains(&"客户端管理"));
        assert!(names.contains(&"通用"));
    }

    #[test]
    fn test_alias_case_insensitive() {
        let manager = SceneManager::new();

        // 大小写不敏感
        assert_eq!(
            manager.match_scene(Some("TOOL_MANAGEMENT")),
            SceneType::ToolManagement
        );
        assert_eq!(
            manager.match_scene(Some("plc控制器")),
            SceneType::PlcController
        );
    }
}
