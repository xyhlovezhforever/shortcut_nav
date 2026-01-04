# MCP(模型上下文协议)技术详解与应用实战

## 一、MCP的本质:为AI打造的标准化连接协议

### 1.1 传统集成的碎片化困境

想象你是一家公司的技术负责人,需要让AI助手访问公司的各种数据源:

**挑战1:为每个数据源定制集成**
- 连接PostgreSQL数据库:需要学习数据库驱动API,编写SQL查询逻辑
- 连接Notion笔记:需要学习Notion API,处理OAuth认证,理解页面结构
- 连接Google Drive:又是一套全新的API,不同的认证方式
- 连接内部ERP系统:可能根本没有API,需要直接操作数据库或解析HTML

**挑战2:AI无法理解这些集成**

即使你写好了所有集成代码,AI也不知道:
- 哪些数据源是可用的?
- 每个数据源能提供什么信息?
- 如何调用这些数据源?
- 数据的格式是什么?

你需要再编写大量"胶水代码",将这些数据源包装成AI能理解的格式。

**挑战3:维护成本爆炸**

- Notion API升级,你的集成代码需要更新
- 数据库表结构变更,查询逻辑需要调整
- 新增数据源,整个流程重来一遍
- 团队成员离职,没人能维护这些定制代码

### 1.2 MCP:统一的连接标准

MCP(Model Context Protocol,模型上下文协议)就像为AI世界制定的"USB标准"。

**USB标准的类比:**

在USB出现之前:
- 鼠标、键盘、打印机、扫描仪都有各自的接口
- 每种设备需要专用的驱动程序
- 操作系统需要针对每种设备单独适配

USB标准出现后:
- 所有设备使用统一的物理接口
- 操作系统只需支持USB协议
- 设备厂商只需遵循USB规范

**MCP的作用完全类似:**

传统方式:
```
AI应用 --[定制代码A]--> 数据库
AI应用 --[定制代码B]--> Notion
AI应用 --[定制代码C]--> Google Drive
AI应用 --[定制代码D]--> Slack
```

MCP方式:
```
AI应用 --[MCP客户端]--> MCP协议 <--[MCP服务器A]-- 数据库
                                   <--[MCP服务器B]-- Notion
                                   <--[MCP服务器C]-- Google Drive
                                   <--[MCP服务器D]-- Slack
```

**核心价值:**

1. **AI应用开发者**:只需集成一次MCP客户端,就能连接所有MCP服务器
2. **数据源提供者**:只需实现一个MCP服务器,就能被所有支持MCP的AI应用访问
3. **用户**:可以灵活组合不同的AI应用和数据源,无需等待官方集成

### 1.3 MCP与其他协议的根本区别

#### 为什么不直接用REST API?

REST API是为**人类开发者**设计的,而MCP是为**AI模型**设计的。

**示例:查询天气**

REST API的设计:
```
GET /api/weather?city=beijing
Response: {"temperature": 25, "condition": "sunny"}
```

这对开发者很清晰,但AI面临的问题:
- 如何知道有个叫/api/weather的接口?
- city参数应该是"beijing"还是"北京"还是"Beijing, China"?
- temperature是摄氏度还是华氏度?
- 返回的JSON结构是什么?

**MCP的设计:**

服务器主动告诉AI:
```json
{
  "tools": [{
    "name": "get_weather",
    "description": "获取指定城市的天气信息",
    "inputSchema": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "城市名称,支持中英文,如'北京'或'Beijing'"
        }
      },
      "required": ["city"]
    }
  }]
}
```

AI看到这个描述后,就知道:
- 有一个叫get_weather的工具
- 需要传入city参数
- city可以是中文或英文

#### 为什么不用Function Calling?

Function Calling(如OpenAI的函数调用)是每个AI服务商各自的实现,没有统一标准。

**碎片化的问题:**

- OpenAI的函数定义格式与Anthropic不同
- Google Gemini又是另一套格式
- 本地模型(如Llama)可能完全不支持

**MCP的优势:**

- 统一的工具定义格式,所有AI模型都能理解
- 统一的调用协议,无需针对每个模型单独适配
- 社区可以共享MCP服务器,而不是为每个AI服务商重复开发

#### MCP的三大核心抽象

**1. Resources(资源):AI可以读取的数据**

就像文件系统中的文件,资源有URI标识,可以被读取:

```
resource://notion/page/abc123
resource://database/users/table
resource://gdrive/document/xyz789
```

**特点:**
- 只读:AI可以读取内容,但不直接修改
- 可订阅:资源更新时,AI可以收到通知
- 结构化:资源有明确的格式(文本、JSON、图片等)

**2. Tools(工具):AI可以执行的操作**

就像命令行中的命令,工具接收参数,执行操作,返回结果:

```
create_document(title, content)
send_email(to, subject, body)
run_sql_query(sql)
```

**特点:**
- 可修改状态:工具可以创建、更新、删除数据
- 有副作用:执行工具会改变外部系统
- 需要权限:工具调用需要用户授权

**3. Prompts(提示词模板):预定义的任务模板**

就像快捷指令,用户可以一键触发复杂的工作流:

```
analyze_sales_report → 读取销售数据 → 生成分析报告 → 发送给团队
weekly_summary → 汇总本周任务 → 总结进展 → 生成日报
```

**特点:**
- 封装复杂逻辑:一个提示词可以组合多个资源和工具
- 参数化:用户可以自定义部分参数
- 可复用:团队可以共享提示词模板

---

## 二、MCP架构设计:客户端与服务器的协作

### 2.1 双向通信:打破传统的请求-响应模式

#### 传统API的单向性

在REST API中,通信是单向的:
- 客户端发起请求
- 服务器被动响应
- 服务器无法主动联系客户端

**问题场景:**

假设AI正在分析一个长时间运行的数据处理任务:
- AI调用tool: analyze_data()
- 服务器开始处理,需要5分钟
- 客户端只能傻等,或者不断轮询"任务完成了吗?"

#### MCP的双向通信

MCP允许服务器主动向客户端发送消息:

**进度通知:**
```
服务器 → 客户端: "数据加载完成,已处理10%"
服务器 → 客户端: "分析进行中,已处理50%"
服务器 → 客户端: "分析完成,生成报告中"
服务器 → 客户端: "任务完成"
```

**资源更新通知:**
```
服务器 → 客户端: "数据库表users已更新,可能需要刷新"
```

**反向工具调用(Sampling):**

最强大的功能:服务器可以请求AI帮忙!

**场景:**用户让AI生成一篇文章并发布到博客

1. AI调用工具: publish_article(content)
2. 服务器发现content格式不对,主动询问AI:
   ```
   服务器 → AI: "请将文章转换为Markdown格式"
   ```
3. AI自动转换格式,再次提交
4. 服务器发现没有标题,再次询问:
   ```
   服务器 → AI: "请为文章生成一个吸引人的标题"
   ```
5. AI生成标题,最终发布成功

**这种协作在传统API中根本无法实现!**

### 2.2 传输层的灵活性

MCP协议本身不绑定特定的传输方式,可以通过多种通道通信。

#### stdio(标准输入输出):最简单的方式

**适用场景:**
- 本地工具集成(如Claude Desktop)
- IDE插件(如VSCode扩展)
- 命令行工具

**原理:**

AI应用启动MCP服务器进程:
```bash
node server.js
```

然后通过stdin发送请求,从stdout读取响应:
```
AI应用 --stdin--> MCP服务器进程 --stdout--> AI应用
```

**优点:**
- 简单:无需配置网络,无需担心端口冲突
- 安全:进程隔离,无需暴露网络端口
- 跨平台:所有操作系统都支持标准IO

**缺点:**
- 只能本地使用,无法远程访问
- 进程生命周期绑定,AI应用关闭时服务器也关闭

#### HTTP+SSE(服务器发送事件):适合远程服务

**适用场景:**
- 云端MCP服务(如数据库服务商提供的MCP接口)
- 需要多个客户端连接的服务
- 需要负载均衡的场景

**原理:**

- 客户端通过HTTP POST发送请求
- 服务器通过SSE推送通知和响应

```
客户端 --HTTP POST--> 服务器
客户端 <--SSE事件流-- 服务器
```

**优点:**
- 远程访问:可以部署在云端,多个客户端共享
- 标准协议:基于HTTP,易于部署在现有基础设施
- 防火墙友好:使用标准80/443端口

**缺点:**
- 需要配置HTTPS、认证、授权
- 网络延迟可能影响性能

#### WebSocket:全双工通信

虽然MCP规范中提到,但目前社区主要使用stdio和HTTP+SSE。

### 2.3 安全模型:细粒度的权限控制

#### 为什么AI访问外部数据是敏感的?

想象你的AI助手能访问:
- 公司财务数据库
- 客户个人信息
- 内部代码仓库

如果没有权限控制:
- AI可能泄露敏感数据
- 恶意提示词可能删除数据
- 第三方MCP服务器可能窃取信息

#### MCP的安全机制

**1. 资源级权限:**

用户可以精确控制AI能访问哪些资源:
```json
{
  "allowed_resources": [
    "database://public/*",
    "notion://team-docs/*"
  ],
  "denied_resources": [
    "database://sensitive/*",
    "notion://personal/*"
  ]
}
```

**2. 工具级权限:**

用户可以控制AI能调用哪些工具:
```json
{
  "allowed_tools": [
    "read_file",
    "search_documents"
  ],
  "denied_tools": [
    "delete_file",
    "execute_sql_write"
  ]
}
```

**3. 参数验证:**

服务器必须验证AI传入的参数:
```python
def delete_file(path: str):
    # 禁止删除系统文件
    if path.startswith("/system/"):
        raise PermissionError("Cannot delete system files")

    # 禁止使用路径遍历
    if ".." in path:
        raise ValueError("Invalid path")

    # 检查文件是否在允许的目录内
    if not path.startswith("/user/documents/"):
        raise PermissionError("Path not allowed")
```

**4. 用户确认:**

对于高风险操作,MCP客户端应该要求用户确认:
```
AI: 我将删除文件 /important-document.pdf
[用户确认] [取消]
```

Claude Desktop就实现了这种机制:AI调用工具前会显示弹窗,让用户确认。

#### 实际案例:数据库访问的分层控制

**场景:**公司有一个MCP数据库服务器

**安全策略:**

1. **网络层:**只允许公司内网访问
2. **认证层:**MCP服务器需要API密钥
3. **授权层:**不同角色有不同权限
   - 分析师:只能读取public schema
   - 开发者:可以读写dev schema
   - 管理员:可以访问所有schema
4. **工具层:**
   - 提供read_only_query工具:只能执行SELECT
   - 提供write_query工具:可以INSERT/UPDATE,但需要用户确认
   - 不提供execute_arbitrary_sql:防止SQL注入
5. **审计层:**记录所有查询,定期审查

---

## 三、资源(Resources):AI的数据源

### 3.1 资源的本质:只读的、可发现的数据

#### 资源vs文件系统

资源很像文件系统,但有关键区别:

**相似之处:**
- 都有唯一标识(文件路径 vs 资源URI)
- 都可以读取内容
- 都有层次结构(目录树 vs 资源树)

**关键区别:**
- **动态生成**:资源可以是实时生成的,而不是预先存储的
- **AI友好**:资源描述包含AI需要的元数据
- **跨系统**:资源可以来自任何数据源,不局限于本地文件

#### 资源的结构

**资源URI:**

```
database://company/users/table
notion://workspace/page/abc123
slack://channel/general/messages
file:///documents/report.pdf
```

URI明确标识了资源的来源和位置。

**资源描述:**

服务器告诉AI有哪些资源可用:
```json
{
  "resources": [
    {
      "uri": "database://company/users/table",
      "name": "用户表",
      "description": "包含所有注册用户的信息:ID、姓名、邮箱、注册时间",
      "mimeType": "application/json"
    },
    {
      "uri": "notion://workspace/page/abc123",
      "name": "产品路线图",
      "description": "2024年Q1-Q4的产品规划文档",
      "mimeType": "text/markdown"
    }
  ]
}
```

AI看到这些描述,就知道:
- 有哪些数据可用
- 每个数据源包含什么信息
- 数据的格式是什么

**资源内容:**

AI请求读取某个资源:
```json
{
  "method": "resources/read",
  "params": {
    "uri": "database://company/users/table"
  }
}
```

服务器返回资源内容:
```json
{
  "contents": [
    {
      "uri": "database://company/users/table",
      "mimeType": "application/json",
      "text": "[{\"id\":1,\"name\":\"Alice\",\"email\":\"alice@example.com\"}]"
    }
  ]
}
```

### 3.2 资源模板:动态资源的强大之处

#### 静态资源 vs 动态资源

**静态资源:**URI固定,内容可能变化

```
database://company/users/table
→ 始终指向users表,但表内容会更新
```

**动态资源:**URI本身包含参数

```
database://company/{table_name}/table
→ {table_name}是参数,可以是users、orders、products等
```

#### 资源模板的应用

**场景1:数据库表的通用访问**

不需要为每个表创建单独的资源:
```json
{
  "resourceTemplates": [
    {
      "uriTemplate": "database://company/{table}/records",
      "name": "数据库表 {table}",
      "description": "访问company数据库中的{table}表数据",
      "mimeType": "application/json"
    }
  ]
}
```

AI可以动态访问:
- database://company/users/records
- database://company/orders/records
- database://company/products/records

**场景2:分页数据**

```json
{
  "uriTemplate": "api://products/list?page={page}&size={size}",
  "name": "产品列表(第{page}页)",
  "description": "获取产品列表,每页{size}条"
}
```

AI可以逐页读取:
- api://products/list?page=1&size=50
- api://products/list?page=2&size=50

**场景3:时间范围数据**

```json
{
  "uriTemplate": "logs://server/access/{date}",
  "name": "{date}的访问日志",
  "description": "获取指定日期的服务器访问日志"
}
```

AI可以查询不同日期:
- logs://server/access/2024-01-15
- logs://server/access/2024-01-16

### 3.3 资源订阅:实时感知数据变化

#### 为什么需要订阅?

传统方式,AI需要不断轮询:

```
AI: 读取资源A
服务器: 返回内容(版本1)
(等待1秒)
AI: 读取资源A
服务器: 返回内容(版本1,未变化)
(等待1秒)
AI: 读取资源A
服务器: 返回内容(版本2,已更新!)
```

这种方式:
- 浪费带宽:90%的请求返回未变化的数据
- 延迟高:最多1秒才能感知变化
- 服务器压力大:大量无效请求

#### MCP的资源订阅

AI订阅资源:
```json
{
  "method": "resources/subscribe",
  "params": {
    "uri": "database://company/users/table"
  }
}
```

服务器在资源更新时主动通知:
```json
{
  "method": "notifications/resources/updated",
  "params": {
    "uri": "database://company/users/table"
  }
}
```

AI收到通知后,再去读取最新内容。

#### 实际应用场景

**协作文档编辑:**

用户在Notion中编辑文档,AI实时感知变化,提供建议:

```
用户修改了"产品设计"章节
→ 服务器通知AI资源已更新
→ AI读取最新内容
→ AI: "我注意到你添加了新功能,需要我更新用户手册吗?"
```

**数据监控:**

AI监控数据库表,发现异常时告警:

```
订单表新增了一条大额订单
→ 服务器通知AI
→ AI读取订单详情
→ AI: "检测到10万元订单,建议人工审核"
```

**多用户协作:**

团队成员都在使用AI助手,共享同一个任务列表:

```
成员A标记任务为完成
→ 服务器通知所有订阅的AI
→ 成员B的AI自动更新显示
→ 成员C的AI生成进度报告
```

---

## 四、工具(Tools):AI的执行能力

### 4.1 工具的本质:带副作用的操作

#### 资源vs工具的根本区别

**资源(Resources):**
- 类比:读书
- 特点:只读,幂等(读多少次结果一样)
- 风险:低,最多泄露信息
- 示例:查询用户信息、读取文档内容

**工具(Tools):**
- 类比:做手术
- 特点:可写,有副作用(执行一次改变一次)
- 风险:高,可能造成不可逆破坏
- 示例:删除文件、发送邮件、执行SQL UPDATE

**为什么要区分?**

安全考虑:
- 资源可以放心让AI读取
- 工具必须谨慎授权,最好需要用户确认

性能考虑:
- 资源可以缓存、批量读取
- 工具不能缓存,每次都要实际执行

### 4.2 工具的定义:让AI理解如何使用

#### 工具Schema的重要性

AI不是人,它不知道:
- create_file需要什么参数?
- path应该是绝对路径还是相对路径?
- content是纯文本还是可以包含二进制?

**一个完善的工具定义:**

```json
{
  "name": "create_file",
  "description": "在指定路径创建文件。如果文件已存在,将被覆盖。路径必须在用户的documents目录内。",
  "inputSchema": {
    "type": "object",
    "properties": {
      "path": {
        "type": "string",
        "description": "文件的绝对路径,必须以/users/documents/开头,不能包含..路径遍历",
        "pattern": "^/users/documents/.+$"
      },
      "content": {
        "type": "string",
        "description": "文件内容,纯文本,UTF-8编码"
      },
      "overwrite": {
        "type": "boolean",
        "description": "如果文件已存在,是否覆盖?默认false,如果文件存在且此项为false,将返回错误",
        "default": false
      }
    },
    "required": ["path", "content"]
  }
}
```

**每个字段的作用:**

- **name**:AI调用时使用的标识符
- **description**:AI理解工具用途的关键,写得越详细,AI用得越准确
- **inputSchema**:JSON Schema格式,精确定义参数类型、约束、默认值
- **pattern**:正则表达式,防止AI传入非法路径
- **required**:明确哪些参数必须提供

#### Description的艺术:引导AI正确使用工具

**差的description:**

```
"description": "创建文件"
```

AI可能:
- 不知道path要绝对路径还是相对路径
- 不知道overwrite是什么意思
- 不知道有路径限制

**好的description:**

```
"description": "在用户的documents目录中创建新文件。重要提示:path必须是绝对路径,以/users/documents/开头。如果文件已存在,默认会报错,除非设置overwrite=true。content只支持纯文本,不支持二进制文件。"
```

AI读到这个,就知道:
- 路径的格式要求
- 如何处理文件已存在的情况
- 内容的限制

**更进一步:提供使用示例**

```
"description": "在用户的documents目录中创建新文件。\n\n示例:\n- 创建新文件: {\"path\": \"/users/documents/note.txt\", \"content\": \"hello\"}\n- 覆盖已有文件: {\"path\": \"/users/documents/note.txt\", \"content\": \"new content\", \"overwrite\": true}\n\n注意:path必须以/users/documents/开头,不允许路径遍历(..)。"
```

AI看到示例,理解会更准确。

### 4.3 工具的实现:从定义到执行

#### 工具调用的生命周期

**1. AI决定调用工具:**

基于用户需求和工具描述,AI选择合适的工具:

```
用户: 帮我创建一个todo.txt文件,内容是"买牛奶"
AI思考: 需要创建文件,应该使用create_file工具
```

**2. AI生成工具调用请求:**

```json
{
  "method": "tools/call",
  "params": {
    "name": "create_file",
    "arguments": {
      "path": "/users/documents/todo.txt",
      "content": "买牛奶",
      "overwrite": false
    }
  }
}
```

**3. MCP服务器验证请求:**

```python
def validate_create_file(arguments):
    path = arguments["path"]

    # 检查路径是否合法
    if not path.startswith("/users/documents/"):
        raise ValueError("路径必须在/users/documents/目录内")

    # 检查路径遍历
    if ".." in path:
        raise ValueError("路径不能包含..")

    # 检查文件是否已存在
    if os.path.exists(path) and not arguments.get("overwrite", False):
        raise FileExistsError(f"文件{path}已存在,设置overwrite=true以覆盖")
```

**4. 服务器执行工具:**

```python
def execute_create_file(arguments):
    path = arguments["path"]
    content = arguments["content"]

    # 确保目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 写入文件
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return {
        "content": [{
            "type": "text",
            "text": f"文件已创建: {path}"
        }]
    }
```

**5. 返回执行结果:**

```json
{
  "result": {
    "content": [{
      "type": "text",
      "text": "文件已创建: /users/documents/todo.txt"
    }]
  }
}
```

**6. AI解读结果,回复用户:**

```
AI: 我已经创建了todo.txt文件,内容是"买牛奶"。
```

#### 错误处理:当工具执行失败

**常见失败场景:**

1. **参数验证失败:**
```python
raise ValueError("路径必须在/users/documents/目录内")
→ 返回错误,AI理解后可能重试或告诉用户
```

2. **权限不足:**
```python
raise PermissionError("没有权限访问此目录")
→ AI告诉用户需要授权
```

3. **资源不存在:**
```python
raise FileNotFoundError("父目录不存在")
→ AI可能先调用create_directory工具创建目录
```

4. **外部服务错误:**
```python
raise RuntimeError("网络连接失败,无法访问API")
→ AI告诉用户稍后重试
```

**错误消息的格式:**

```json
{
  "error": {
    "code": -32000,
    "message": "文件已存在: /users/documents/todo.txt",
    "data": {
      "path": "/users/documents/todo.txt",
      "suggestion": "设置overwrite=true以覆盖文件"
    }
  }
}
```

AI读到suggestion,可能会:
```
AI: 文件已存在,我可以覆盖它吗?
用户: 可以
AI: (重新调用工具,设置overwrite=true)
```

### 4.4 复杂工具:多步骤的操作编排

#### 原子工具 vs 复合工具

**原子工具:**单一职责,一次调用完成一个明确的操作

```
create_file(path, content)
delete_file(path)
send_email(to, subject, body)
```

**复合工具:**封装多个步骤的复杂操作

```
deploy_website(source_dir, target_url)
→ 内部执行:
  1. 压缩文件
  2. 上传到服务器
  3. 解压
  4. 重启服务
  5. 验证部署成功
```

**何时使用复合工具?**

**适合的场景:**
- 步骤高度固定,没有分支逻辑
- 中间步骤对用户不重要
- 需要保证原子性(要么全成功,要么全失败)

**示例:批量导入用户**

```python
def import_users(csv_content):
    # 解析CSV
    users = parse_csv(csv_content)

    # 验证所有用户数据
    for user in users:
        validate_user(user)

    # 批量插入数据库(事务)
    with transaction():
        for user in users:
            create_user(user)

    return f"成功导入{len(users)}个用户"
```

如果用原子工具,AI需要:
1. 调用parse_csv
2. 对每个用户调用validate_user
3. 对每个用户调用create_user

100个用户就是200次工具调用,效率低,而且中间步骤可能出错导致数据不一致。

**不适合的场景:**
- 步骤之间需要AI判断(如:生成代码→运行测试→如果失败则修复→重新测试)
- 中间结果需要用户确认
- 步骤可能失败,需要不同的恢复策略

**示例:代码审查工作流**

不要做成:
```
review_and_fix_code(file_path)
→ 读代码→发现问题→自动修复→提交
```

应该分成:
```
1. analyze_code(file_path) → 返回问题列表
2. AI展示给用户,用户选择要修复的问题
3. fix_code_issue(file_path, issue_id) → 修复单个问题
4. commit_changes(message) → 提交修改
```

这样用户有控制权,可以审查每一步。

---

## 五、提示词(Prompts):工作流的模板化

### 5.1 提示词的本质:封装重复性任务

#### 为什么需要提示词?

**场景:**每周五,产品经理需要生成周报

**不用提示词的方式:**

每周五,产品经理都要手动输入:
```
请帮我:
1. 读取Notion中本周的所有任务(workspace://tasks)
2. 统计已完成/进行中/未开始的任务数
3. 读取每个任务的描述和负责人
4. 生成一份周报,包括:
   - 本周进展总结
   - 各成员完成情况
   - 下周计划
5. 将周报发送到Slack的#weekly-updates频道
```

**问题:**
- 每次都要重新输入,容易遗漏步骤
- 描述不准确,AI可能理解偏差
- 无法在团队中共享

#### 提示词的解决方案

**定义提示词模板:**

```json
{
  "name": "generate_weekly_report",
  "description": "生成产品团队周报并发送到Slack",
  "arguments": [
    {
      "name": "week_number",
      "description": "第几周,如'2024-W03'",
      "required": true
    }
  ],
  "prompt": "请执行以下步骤生成{week_number}的周报:\n\n1. 读取Notion资源: workspace://tasks?week={week_number}\n2. 统计任务状态:\n   - 已完成:status=done\n   - 进行中:status=in_progress\n   - 未开始:status=todo\n3. 对每个任务,提取:\n   - 任务名称\n   - 负责人\n   - 完成时间(如果已完成)\n4. 生成周报,格式:\n   ```\n   ## {week_number}周报\n   \n   ### 本周数据\n   - 总任务数: X\n   - 已完成: Y (Z%)\n   - 进行中: ...\n   \n   ### 任务详情\n   (按负责人分组列出)\n   \n   ### 下周计划\n   (列出进行中和未开始的任务)\n   ```\n5. 使用send_slack_message工具发送到#weekly-updates"
}
```

**使用提示词:**

产品经理只需:
```
执行提示词: generate_weekly_report
参数: week_number = "2024-W03"
```

AI自动执行所有步骤,生成并发送周报。

### 5.2 提示词的参数化:灵活性与一致性的平衡

#### 静态提示词的局限

假设提示词写死了:
```
"读取Notion资源: workspace://tasks?week=2024-W03"
```

问题:
- 只能查询2024年第3周,下周就不能用了
- 不同团队可能有不同的Notion workspace

#### 动态参数

提示词支持参数替换:

```json
{
  "arguments": [
    {
      "name": "week_number",
      "description": "查询的周数,格式:YYYY-Wnn,如2024-W03",
      "required": true
    },
    {
      "name": "workspace",
      "description": "Notion工作空间ID",
      "required": false,
      "default": "default-workspace"
    },
    {
      "name": "slack_channel",
      "description": "发送周报的Slack频道",
      "required": false,
      "default": "#weekly-updates"
    }
  ],
  "prompt": "读取Notion资源: {workspace}://tasks?week={week_number}\n...\n发送到{slack_channel}"
}
```

现在提示词变得灵活:
- 默认使用default-workspace和#weekly-updates
- 用户可以自定义查询不同团队的数据
- 可以发送到不同的Slack频道

#### 参数验证

防止用户输入非法参数:

```json
{
  "name": "week_number",
  "description": "查询的周数",
  "required": true,
  "schema": {
    "type": "string",
    "pattern": "^\\d{4}-W\\d{2}$"
  }
}
```

如果用户输入"第3周"而不是"2024-W03",MCP客户端会提示格式错误。

### 5.3 提示词的组合:构建复杂工作流

#### 单一提示词 vs 提示词链

**简单任务:**单一提示词足够

```
summarize_document
→ 读取文档,生成摘要
```

**复杂任务:**需要多个提示词协作

**场景:从需求到上线的完整流程**

1. **需求分析提示词:**
```
analyze_requirement
→ 读取需求文档
→ 提取核心功能点
→ 识别技术难点
→ 生成技术方案
```

2. **代码生成提示词:**
```
generate_code
→ 基于技术方案生成代码框架
→ 实现核心逻辑
→ 添加单元测试
```

3. **代码审查提示词:**
```
review_code
→ 静态分析代码
→ 检查代码规范
→ 发现潜在bug
→ 生成审查报告
```

4. **部署提示词:**
```
deploy_to_staging
→ 运行所有测试
→ 构建Docker镜像
→ 部署到staging环境
→ 运行smoke test
→ 通知团队
```

#### 条件分支:根据结果选择下一步

提示词可以包含条件逻辑(由AI执行):

```
如果代码审查发现严重问题:
  → 执行fix_code_issues提示词
  → 重新运行review_code
否则:
  → 执行deploy_to_staging
```

实际实现中,这是AI的责任:
- 提示词定义了步骤和判断标准
- AI根据每一步的结果,决定下一步行动
- MCP只负责提供工具和资源

#### 提示词库:团队的知识沉淀

**组织层面的提示词:**

```
marketing/
  - generate_social_media_post
  - analyze_campaign_performance
  - create_email_newsletter

engineering/
  - onboard_new_developer
  - debug_production_issue
  - generate_api_documentation

sales/
  - qualify_lead
  - generate_proposal
  - analyze_competitor
```

**好处:**
- 新员工快速上手(直接使用提示词,无需从头学习)
- 质量一致性(避免不同人做同样任务时质量参差不齐)
- 持续改进(提示词效果不好可以集中优化,所有人受益)

---

## 六、采样(Sampling):服务器请求AI的能力

### 6.1 反向调用:打破传统的控制流

#### 传统工具调用的单向性

一般流程:
```
用户 → AI → 工具 → 执行 → 返回结果 → AI → 用户
```

AI是主动方,工具是被动执行方。

**问题场景:**

工具需要AI帮忙处理中间步骤。

**示例:翻译工具**

用户:"把这份英文合同翻译成中文,并提取关键条款"

理想流程:
```
1. AI调用translate工具,传入英文合同
2. translate工具发现合同很长,分段翻译
3. 每段翻译完,工具想让AI检查翻译质量
4. AI检查,发现专业术语翻译不准,建议修改
5. 工具采纳建议,重新翻译
6. 所有段落翻译完毕,工具想让AI提取关键条款
7. AI分析中文合同,提取条款
8. 工具返回最终结果
```

在传统API中,步骤3、4、6、7无法实现,因为工具无法"回调"AI。

#### MCP的Sampling机制

服务器可以请求AI帮忙!

**Sampling请求:**

```json
{
  "method": "sampling/createMessage",
  "params": {
    "messages": [
      {
        "role": "user",
        "content": {
          "type": "text",
          "text": "请检查以下翻译是否准确:\n原文: The party of the first part\n译文: 第一方\n\n如果不准确,请提供改进建议。"
        }
      }
    ],
    "maxTokens": 200
  }
}
```

**AI响应:**

```json
{
  "result": {
    "role": "assistant",
    "content": {
      "type": "text",
      "text": "翻译不够准确。在法律文书中,'The party of the first part'应译为'甲方',这是标准的合同术语。建议修改为'甲方'。"
    }
  }
}
```

服务器收到建议,调整翻译,继续执行。

### 6.2 Sampling的应用场景

#### 场景1:内容生成

**工具:generate_blog_post**

```
用户: 写一篇关于MCP的博客
AI: 调用generate_blog_post("MCP协议介绍")

服务器内部:
1. 生成文章大纲
2. 请求AI: "这个大纲是否合理?" (Sampling)
3. AI: "建议增加实际案例章节"
4. 服务器调整大纲
5. 生成每个章节的内容
6. 请求AI: "润色这段文字,使其更易读" (Sampling)
7. 生成最终文章
8. 返回给AI

AI: 收到文章,展示给用户
```

**价值:**
- 工具利用AI的语言能力,生成高质量内容
- AI不需要理解每一个中间步骤,只需回答具体问题

#### 场景2:决策辅助

**工具:optimize_sql_query**

```
用户: 优化这个SQL查询
AI: 调用optimize_sql_query(sql)

服务器内部:
1. 分析SQL,发现3种优化方案:
   A. 添加索引
   B. 重写为JOIN
   C. 使用物化视图
2. 请求AI: "以下3种方案,哪种更适合这个场景?" (Sampling)
   提供上下文:表大小、查询频率、数据更新频率
3. AI: "根据表大小(100万行)和查询频率(每秒10次),建议方案A"
4. 服务器应用方案A
5. 返回优化后的SQL

AI: 向用户解释优化逻辑
```

**价值:**
- 结合工具的专业知识(SQL优化技巧)和AI的推理能力(选择最佳方案)

#### 场景3:格式转换

**工具:convert_to_markdown**

```
用户: 把这个Word文档转成Markdown
AI: 调用convert_to_markdown(docx_file)

服务器内部:
1. 解析Word文档,提取文字和格式
2. 遇到复杂表格,请求AI: "请用Markdown表格表示这个数据" (Sampling)
   提供:表格数据的JSON
3. AI: 生成Markdown表格
4. 遇到图片,请求AI: "为这张图片生成alt文本" (Sampling)
5. AI: 分析图片,生成描述
6. 组装完整的Markdown
7. 返回结果
```

**价值:**
- 工具处理格式解析(机械部分)
- AI处理语义理解(如图片描述、表格总结)

### 6.3 Sampling的限制与注意事项

#### 限制1:不能无限递归

服务器请求AI → AI调用工具 → 工具请求AI → AI调用工具 → ...

这会导致无限循环。

**MCP的限制:**
- Sampling的AI响应不能再调用工具
- Sampling只能是"问答",不能触发新的工作流

#### 限制2:性能考虑

每次Sampling都是一次AI推理:
- 延迟:几百毫秒到几秒
- 成本:消耗tokens

**不要滥用:**

❌ 错误:
```python
for row in database_rows:
    # 对每一行都请求AI处理
    ai_result = sampling(f"总结这条数据: {row}")
```

如果有10000行,就是10000次AI调用,耗时且昂贵。

✅ 正确:
```python
# 批量处理,只请求一次AI
rows_text = "\n".join([str(row) for row in database_rows])
ai_result = sampling(f"总结以下{len(database_rows)}条数据:\n{rows_text}")
```

#### 限制3:上下文管理

Sampling请求中,AI看不到之前的对话历史。

服务器需要提供完整上下文:

❌ 错误:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "这个方案可行吗?"
    }
  ]
}
```

AI: "什么方案?我不知道你在说什么。"

✅ 正确:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "我有一个SQL查询需要优化:\nSELECT * FROM users WHERE age > 30\n\n我考虑了3种优化方案:\nA. 在age列添加索引\nB. 使用分区表\nC. 缓存查询结果\n\n根据以下条件,哪个方案最佳?\n- 表有100万行\n- 查询每秒10次\n- age分布均匀"
    }
  ]
}
```

AI有了完整信息,才能给出合理建议。

---

## 七、实际应用场景:MCP如何解决真实问题

### 7.1 个人知识管理:连接分散的信息孤岛

#### 问题场景

知识工作者的信息分散在:
- Notion:会议笔记、项目文档
- Obsidian:个人知识库、读书笔记
- Gmail:邮件、往来记录
- Google Drive:文件、表格
- Slack:团队沟通
- GitHub:代码、issue讨论

**用户需求:**

"帮我整理上周关于X项目的所有信息,包括会议讨论、邮件往来、代码变更,生成一份完整的项目总结"

**传统方式的困难:**
- 需要手动打开每个应用
- 在每个应用中搜索"X项目"
- 复制粘贴相关内容
- 人工整理、去重、归纳

#### MCP解决方案

**部署MCP服务器:**

1. **mcp-server-notion**:访问Notion笔记
2. **mcp-server-gmail**:访问邮件
3. **mcp-server-gdrive**:访问文件
4. **mcp-server-slack**:访问Slack消息
5. **mcp-server-github**:访问代码仓库

**AI工作流:**

用户:"整理上周X项目的信息"

```
AI自动执行:

1. 搜索Notion资源:
   notion://search?query=X项目&date_range=last_week
   → 找到3篇会议笔记

2. 搜索Gmail:
   gmail://search?query=X项目&after=7_days_ago
   → 找到15封相关邮件

3. 搜索Slack:
   slack://search?query=X项目&in=#project-channel
   → 找到42条讨论

4. 查询GitHub:
   github://repo/commits?path=project-x&since=last_week
   → 找到8次代码提交

5. 综合所有信息,生成总结:
   - 会议决议:...
   - 邮件沟通要点:...
   - Slack讨论的关键问题:...
   - 代码变更:...
   - 下周行动项:...
```

**用户体验:**
- 一句话需求,几秒钟得到完整总结
- 无需手动在多个应用间切换
- 信息自动去重、归类

### 7.2 开发工作流:IDE与AI的深度集成

#### 问题场景

开发者每天面对:
- 阅读代码库,理解业务逻辑
- 查询数据库,确认数据结构
- 编写代码,解决bug
- 编写测试,确保质量
- 提交代码,创建PR
- 查看CI/CD结果

**传统方式:**
- 在IDE写代码
- 切换到数据库客户端查询
- 切换到浏览器查看文档
- 切换到终端运行测试
- 切换到GitHub创建PR

#### MCP解决方案

**VSCode + MCP服务器:**

1. **mcp-server-filesystem**:访问项目文件
2. **mcp-server-database**:查询开发数据库
3. **mcp-server-git**:git操作
4. **mcp-server-terminal**:运行命令
5. **mcp-server-github**:管理PR

**实际场景:修复bug**

用户:"修复用户注册失败的bug"

```
AI工作流:

1. 分析日志:
   读取resource://logs/error.log
   → 发现:INSERT INTO users失败,违反唯一约束

2. 查询数据库:
   调用tool:run_sql_query("SELECT * FROM users WHERE email='test@example.com'")
   → 发现:email字段有重复数据

3. 分析代码:
   读取resource://src/auth/register.js
   → 发现:注册前没有检查email是否已存在

4. 修复代码:
   调用tool:edit_file(
     path="src/auth/register.js",
     changes=[
       {line: 10, insert: "const existing = await User.findByEmail(email);"},
       {line: 11, insert: "if (existing) throw new Error('Email already exists');"}
     ]
   )

5. 编写测试:
   调用tool:create_file(
     path="tests/auth/register.test.js",
     content="test('should reject duplicate email', async () => { ... })"
   )

6. 运行测试:
   调用tool:run_command("npm test")
   → 所有测试通过

7. 提交代码:
   调用tool:git_commit("Fix: check duplicate email before registration")
   调用tool:git_push()

8. 创建PR:
   调用tool:create_pull_request(
     title="Fix user registration duplicate email bug",
     body="修复了用户注册时未检查邮箱重复的问题..."
   )
```

**AI向用户汇报:**
"我已经修复了注册bug,原因是注册前未检查邮箱是否已存在。我添加了检查逻辑和测试,所有测试通过。代码已提交,PR已创建:#1234"

**用户体验:**
- 从发现bug到创建PR,全程在对话框完成
- 无需在多个工具间切换
- AI自动执行最佳实践(写测试、运行测试、写清晰的commit message)

### 7.3 数据分析:从数据到洞察的自动化

#### 问题场景

数据分析师需要:
- 连接多个数据源(数据库、API、Excel)
- 清洗、转换数据
- 探索性分析,发现模式
- 生成图表
- 撰写报告

**传统方式:**
- 用SQL查询数据
- 导出到CSV
- 用Python/R脚本清洗
- 用matplotlib/ggplot画图
- 用Word/Markdown写报告
- 大量手动工作,容易出错

#### MCP解决方案

**Jupyter + MCP服务器:**

1. **mcp-server-postgres**:查询生产数据库
2. **mcp-server-analytics-api**:调用分析平台API
3. **mcp-server-excel**:读写Excel文件
4. **mcp-server-visualization**:生成图表

**实际场景:分析用户留存**

用户:"分析上月新用户的30天留存率,按渠道对比"

```
AI工作流:

1. 查询新用户数据:
   tool:run_sql("""
     SELECT user_id, signup_date, channel
     FROM users
     WHERE signup_date >= '2024-01-01' AND signup_date < '2024-02-01'
   """)
   → 获得10000个新用户

2. 查询活跃数据:
   tool:run_sql("""
     SELECT user_id, DATE(activity_date) as date
     FROM user_activities
     WHERE activity_date >= '2024-01-01' AND activity_date < '2024-03-01'
   """)
   → 获得活跃记录

3. 计算留存率:
   (AI在内存中处理数据)
   - 按渠道分组
   - 计算每个用户的留存天数
   - 计算30天留存率

4. 生成图表:
   tool:create_chart({
     type: "line",
     data: {
       labels: ["Day 1", "Day 7", "Day 14", "Day 30"],
       datasets: [
         {label: "搜索引擎", data: [100, 65, 45, 32]},
         {label: "社交媒体", data: [100, 58, 38, 25]},
         {label: "直接访问", data: [100, 70, 52, 40]}
       ]
     },
     title: "30天留存率对比(按渠道)"
   })
   → 生成PNG图片

5. 发现洞察:
   (AI分析数据)
   - 直接访问渠道留存最好(40%)
   - 社交媒体留存最差(25%)
   - Day 7是关键流失点

6. 生成报告:
   tool:create_document({
     type: "markdown",
     content: "# 1月新用户留存分析\n\n## 核心发现\n- 直接访问用户留存率最高...\n\n## 建议\n1. 加大直接访问渠道投放\n2. 优化社交媒体用户的onboarding流程..."
   })
```

**AI向用户汇报:**
"分析完成!1月新用户30天留存率为32%。直接访问渠道表现最好(40%),社交媒体最差(25%)。建议优化社交媒体用户的首周体验。详细报告已生成,包含趋势图和改进建议。"

**用户体验:**
- 自然语言描述需求,无需编写SQL和Python代码
- 自动生成专业图表和报告
- AI主动发现数据洞察,提供可执行建议

### 7.4 客户支持:智能客服的知识库

#### 问题场景

客服团队需要:
- 访问产品文档
- 查询客户历史记录
- 搜索常见问题库
- 查询订单状态
- 创建工单

**传统方式:**
- 客服在5-6个系统间切换
- 新客服需要几个月培训才能熟练
- 人工查询慢,客户等待时间长

#### MCP解决方案

**客服工作台 + MCP服务器:**

1. **mcp-server-docs**:产品文档
2. **mcp-server-crm**:客户关系管理
3. **mcp-server-orders**:订单系统
4. **mcp-server-faq**:FAQ知识库
5. **mcp-server-ticketing**:工单系统

**实际场景:处理客户咨询**

客户:"我的订单#12345还没收到,怎么回事?"

```
AI工作流:

1. 查询订单:
   tool:get_order_details(order_id="12345")
   → 状态:已发货,物流单号:SF1234567890

2. 查询物流:
   tool:track_shipment(tracking_number="SF1234567890")
   → 当前位置:XX市分拨中心,预计明天送达

3. 查询客户历史:
   resource://crm/customer/by_order/12345
   → 发现:此客户是VIP,过去有过2次延迟配送投诉

4. 搜索相关FAQ:
   resource://faq/search?query=配送延迟
   → 找到标准应对话术

5. 生成回复:
   (AI综合信息)
   "您好!您的订单已于3天前发货,目前在XX市分拨中心,预计明天送达。考虑到您是我们的VIP客户,我已为您申请加急配送,并赠送您一张50元优惠券作为补偿。感谢您的耐心!"

6. 创建工单:
   tool:create_ticket({
     type: "配送延迟",
     priority: "high", // 因为是VIP
     order_id: "12345",
     compensation: "50元优惠券"
   })

7. 更新CRM:
   tool:add_customer_note({
     customer_id: "...",
     note: "订单12345配送延迟,已补偿50元券"
   })
```

**客服体验:**
- 所有信息一个界面呈现
- AI自动识别VIP,提供差异化服务
- AI生成专业回复,客服审核后发送

**客户体验:**
- 快速得到准确答复(几秒vs几分钟)
- 主动补偿,提升满意度

---

## 八、开发MCP服务器:从零开始

### 8.1 选择技术栈:Python vs TypeScript vs其他

#### Python:快速原型与数据处理

**优势:**
- 生态丰富:pandas、numpy、数据库驱动
- 适合数据密集型服务器(分析、ML、科学计算)
- 学习曲线平缓

**示例场景:**
- mcp-server-pandas:数据分析
- mcp-server-ml:机器学习模型推理
- mcp-server-scientific:科学计算工具

#### TypeScript:类型安全与Node生态

**优势:**
- 类型系统:编译时捕获错误
- Node生态:大量npm包
- 适合Web服务集成(REST API、GraphQL)

**示例场景:**
- mcp-server-github:GitHub API封装
- mcp-server-notion:Notion API
- mcp-server-slack:Slack集成

#### 其他语言

- **Rust**:性能关键型服务(大文件处理、实时数据)
- **Go**:云原生服务(Kubernetes operator、分布式系统)
- **Java**:企业系统集成(SAP、Oracle)

**选择建议:**

1. **数据源是什么?**
   - 数据库/文件 → Python
   - Web API → TypeScript
   - 企业系统 → Java

2. **性能要求?**
   - 一般 → Python/TypeScript
   - 高性能 → Rust/Go

3. **团队技能?**
   - 用团队最熟悉的语言,快速迭代

### 8.2 设计资源结构:如何组织数据

#### 扁平结构 vs 层次结构

**扁平结构:**所有资源在一个层级

```
database://users
database://orders
database://products
```

**优点:**简单,易于列举
**缺点:**资源多时难以管理

**层次结构:**按类别组织

```
database://company/
  users/
    table
    view/active_users
  orders/
    table
    view/recent_orders
```

**优点:**清晰,易于浏览
**缺点:**需要实现层次遍历

#### 动态资源的命名

使用{参数}表示动态部分:

```
database://company/{table}/records
api://products/{category}/list
logs://{service}/{date}
```

AI看到这些模板,知道可以替换参数访问不同资源。

#### 资源元数据的重要性

**差的资源定义:**

```json
{
  "uri": "db://users",
  "name": "users"
}
```

AI不知道:
- 这是什么类型的数据?
- 包含什么字段?
- 数据量有多大?

**好的资源定义:**

```json
{
  "uri": "database://company/users/table",
  "name": "用户表",
  "description": "包含所有注册用户的信息。字段:id(主键)、name(姓名)、email(邮箱)、created_at(注册时间)、role(角色:admin/user)。当前约10万条记录。",
  "mimeType": "application/json",
  "metadata": {
    "rowCount": 100000,
    "schema": {
      "id": "integer",
      "name": "string",
      "email": "string",
      "created_at": "timestamp",
      "role": "enum(admin,user)"
    }
  }
}
```

AI看到这个,就能:
- 理解数据结构
- 知道数据量级(影响查询策略)
- 了解字段含义

### 8.3 设计工具接口:让AI正确使用

#### 工具粒度的权衡

**原子工具:**一个工具做一件事

```
create_user(name, email)
update_user(id, name, email)
delete_user(id)
```

**优点:**
- 清晰,易于理解
- 灵活组合

**缺点:**
- 复杂任务需要多次调用
- 增加延迟和成本

**复合工具:**一个工具做多件事

```
manage_user(action, id, name, email)
→ action可以是"create"/"update"/"delete"
```

**优点:**
- 减少工具数量

**缺点:**
- 参数复杂,AI容易用错
- 难以细粒度权限控制

**建议:**
- 常用的原子操作:单独的工具
- 复杂的多步骤流程:复合工具
- 让AI选择:同时提供两种

#### 参数设计的最佳实践

**1. 明确必填vs可选:**

```json
{
  "required": ["email"],
  "properties": {
    "email": {...},
    "name": {"default": "Anonymous"}
  }
}
```

**2. 提供默认值:**

```json
{
  "page_size": {
    "type": "integer",
    "default": 50,
    "description": "每页记录数,默认50"
  }
}
```

**3. 限制取值范围:**

```json
{
  "priority": {
    "type": "string",
    "enum": ["low", "medium", "high"],
    "description": "任务优先级,只能是low/medium/high之一"
  }
}
```

**4. 验证格式:**

```json
{
  "email": {
    "type": "string",
    "format": "email",
    "description": "用户邮箱,必须是有效的邮箱格式"
  }
}
```

#### 返回值设计

**结构化返回:**

```json
{
  "content": [{
    "type": "text",
    "text": "用户创建成功"
  }],
  "metadata": {
    "user_id": 12345,
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

AI可以:
- 展示text给用户
- 使用metadata中的user_id进行后续操作

**丰富的内容类型:**

```json
{
  "content": [
    {
      "type": "text",
      "text": "查询结果:"
    },
    {
      "type": "resource",
      "resource": {
        "uri": "data://query_result_123",
        "mimeType": "application/json",
        "text": "[{...}, {...}]"
      }
    },
    {
      "type": "image",
      "data": "base64...",
      "mimeType": "image/png"
    }
  ]
}
```

支持文本、资源引用、图片,AI可以丰富地展示结果。

### 8.4 错误处理与重试机制

#### 明确的错误分类

不要所有错误都返回"操作失败",而是明确类型:

**参数错误:**
```json
{
  "error": {
    "code": -32602,
    "message": "Invalid params",
    "data": {
      "field": "email",
      "issue": "Invalid email format"
    }
  }
}
```

AI可以:
- 修正email格式
- 重新调用工具

**权限错误:**
```json
{
  "error": {
    "code": -32001,
    "message": "Permission denied",
    "data": {
      "required_permission": "admin",
      "current_permission": "user"
    }
  }
}
```

AI可以:
- 告诉用户需要管理员权限
- 或请求用户授权

**资源不存在:**
```json
{
  "error": {
    "code": -32002,
    "message": "Resource not found",
    "data": {
      "uri": "database://users/12345"
    }
  }
}
```

AI可以:
- 提示用户ID可能错误
- 或先创建资源再继续

**外部服务错误:**
```json
{
  "error": {
    "code": -32003,
    "message": "External service error",
    "data": {
      "service": "GitHub API",
      "error": "Rate limit exceeded",
      "retry_after": 60
    }
  }
}
```

AI可以:
- 等待60秒后重试
- 或告诉用户稍后再试

#### 重试策略

**哪些错误应该重试?**

✅ 应该重试:
- 网络超时
- 临时服务不可用(503)
- 速率限制(但要等待retry_after)

❌ 不应该重试:
- 参数错误(400)
- 权限错误(403)
- 资源不存在(404)

**指数退避:**

```python
def call_tool_with_retry(tool_name, args, max_retries=3):
    for attempt in range(max_retries):
        try:
            return call_tool(tool_name, args)
        except TemporaryError as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt  # 1s, 2s, 4s
            time.sleep(wait_time)
```

---

## 九、引发思考的问题

1. **标准化 vs 灵活性:MCP会限制创新吗?**
   - 统一协议带来互操作性,但会不会限制每个应用的独特功能?
   - 当所有AI应用都支持MCP,它们会变得趋同而失去差异化吗?
   - 如何在遵循标准和保持创新之间平衡?

2. **隐私与安全:AI访问所有数据源的风险**
   - 当AI可以访问邮件、文档、数据库,如何防止敏感信息泄露?
   - 用户真的理解授予AI的权限范围吗?还是像安装APP时盲目点"允许"?
   - MCP服务器提供者如何保证不窃取用户数据?

3. **依赖与控制权:过度依赖AI的风险**
   - 当所有任务都由AI自动化,人类会失去技能吗?
   - 如果MCP服务器出错,AI执行了错误操作(如删除重要数据),谁负责?
   - 用户应该对AI的每个操作都审查吗?那样还有自动化的意义吗?

4. **生态系统的集中化:巨头垄断的担忧**
   - 如果大公司控制了主流MCP服务器,是否会形成新的数据垄断?
   - 小开发者的MCP服务器如何与巨头竞争?
   - 开源MCP服务器能否持续维护,不被商业服务挤压?

5. **性能与成本:AI调用的经济性**
   - 每次AI调用工具都消耗tokens,成本如何控制?
   - 复杂任务可能调用几十次工具,用户愿意为此付费吗?
   - 如何优化MCP通信,减少不必要的往返?

6. **多模态融合:MCP能处理图片、视频、音频吗?**
   - 当前MCP主要处理文本和结构化数据,如何扩展到多模态?
   - 传输大文件(如视频)时,性能如何保证?
   - AI如何理解和操作非文本资源?

7. **版本兼容性:MCP协议升级时的挑战**
   - 当MCP协议升级,旧的服务器和客户端如何兼容?
   - 如果客户端支持MCP 2.0,但服务器只支持1.0,如何降级处理?
   - 社区如何协调协议演进,避免碎片化?

8. **实时性需求:MCP适合实时应用吗?**
   - 金融交易、游戏、实时监控等场景,MCP的延迟是否可接受?
   - 如何在MCP上实现流式传输(如实时日志、视频流)?
   - WebSocket传输能否成为MCP的主流选择?

9. **跨语言AI:MCP如何处理多语言场景?**
   - 资源描述用英文,AI能准确理解中文用户的需求吗?
   - 工具返回的内容是中文,但JSON字段名是英文,AI会混淆吗?
   - 如何设计多语言友好的MCP服务器?

10. **AI能力边界:什么任务不应该自动化?**
    - 法律文书审核、医疗诊断决策,AI+MCP能完全自动化吗?
    - 哪些任务必须保留人类的最终判断?
    - 如何在MCP中标记"此操作需要人工确认"?
