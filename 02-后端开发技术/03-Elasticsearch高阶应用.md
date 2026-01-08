# Elasticsearch 高阶应用 - 从原理到实战全面解析

## 一、历史背景与发展历程

### 1.1 Elasticsearch 诞生的历史背景

#### 全文搜索的早期困境

在 Elasticsearch 诞生之前,企业面临着严峻的搜索挑战:

**传统数据库的困境**:
- MySQL 的 `LIKE '%keyword%'` 查询在大数据量下性能极差
- 即使建立索引,B+树也无法支持模糊匹配
- 1000万条记录的全文搜索可能需要几分钟甚至更长时间
- 用户体验极差,业务转化率受到严重影响

**早期搜索解决方案的问题**:
- Lucene (2000年发布):功能强大但只是一个 Java 库,不是独立系统
- Solr (2004年基于 Lucene 开发):企业级搜索服务器,但配置复杂
- Sphinx:MySQL 的全文搜索引擎,但功能有限

**业务需求的演进**:
- 电商网站需要实时商品搜索,支持拼写纠错、同义词
- 日志系统每天产生TB级数据,需要快速检索和分析
- 内容平台需要根据用户行为动态调整搜索排序
- 分布式架构成为主流,需要搜索引擎也能水平扩展

#### Shay Banon 与 Compass 项目

**个人需求驱动创新**:
- 2004年,Shay Banon 需要为妻子开发一个食谱搜索应用
- 他使用 Lucene 开发了 Compass 项目(一个搜索框架)
- Compass 简化了 Lucene 的使用,但仍是单机系统
- 无法应对分布式场景和大规模数据

**技术转折点**:
- 2010年,移动互联网爆发,数据量呈指数级增长
- NoSQL 运动兴起(MongoDB、Cassandra、Redis 等)
- RESTful API 成为主流,JSON 成为数据交换标准
- 云计算和分布式系统成为趋势

**Elasticsearch 的诞生**:
- 2010年2月,Elasticsearch 0.4.0 首次发布
- 核心理念:让 Lucene 可以分布式运行,并提供 REST API
- 使用 JSON 作为数据格式,降低使用门槛
- 自动分片、副本、故障恢复,开箱即用的分布式特性

### 1.2 Elasticsearch 解决的核心问题

#### 问题1:全文搜索性能瓶颈

**传统数据库的根本限制**:
```
场景:在1000万篇文章中搜索包含"分布式系统架构"的内容

MySQL 方式:
SELECT * FROM articles WHERE content LIKE '%分布式系统架构%'
- 执行方式:全表扫描
- 时间复杂度:O(n × m),n为记录数,m为字段长度
- 实际耗时:可能需要数分钟
- 索引失效:LIKE '%...%' 无法使用 B+树索引

Elasticsearch 方式:
GET /articles/_search { "query": { "match": { "content": "分布式系统架构" }}}
- 执行方式:倒排索引直接定位
- 时间复杂度:O(词数 × log(文档数))
- 实际耗时:毫秒级
- 相关性排序:自动按 BM25 算法计算最相关的结果
```

**倒排索引的革命性突破**:
- 传统数据库:文档ID → 文档内容(正排)
- Elasticsearch:词语 → 包含该词的文档列表(倒排)
- 查询"分布式 系统"时,分别查找两个词的倒排列表,再取交集
- 性能提升:从O(n)到O(词数),与文档总量无关

#### 问题2:分布式扩展困难

**单机搜索引擎的瓶颈**:
- Solr 虽然支持分布式,但需要依赖 ZooKeeper,配置复杂
- 增加节点需要手动配置和数据迁移
- 故障恢复需要人工介入
- 无法动态扩缩容

**Elasticsearch 的分布式设计**:
- 自动分片(Shard):索引自动拆分到多个节点
- 自动副本(Replica):数据自动复制,保证高可用
- 自动发现:新节点加入后自动加入集群
- 自动恢复:节点故障后自动重新分配分片
- 动态扩容:增加节点即可扩展容量和性能

**真实场景的价值**:
```
场景:日志系统每天产生10TB数据

单机方案的问题:
- 磁盘容量不足:单机最大20TB SSD
- 查询性能下降:数据越多,查询越慢
- 无法容灾:单点故障导致数据丢失

Elasticsearch 方案:
- 数据自动分片到10个节点,每个节点1TB
- 查询并行执行,10倍性能提升
- 每个分片有1个副本,节点故障不影响服务
- 需要扩容时,增加节点即可,自动 Rebalance
```

#### 问题3:复杂查询与聚合

**传统数据库的局限**:
- SQL 的全文搜索能力弱:LIKE、FULLTEXT INDEX 功能有限
- 分词、同义词、拼写纠错:数据库不支持
- 多条件组合查询:SQL 语句复杂,性能差
- 实时聚合分析:GROUP BY 在大数据量下极慢

**Elasticsearch 的查询能力**:
- 全文搜索:支持分词、高亮、拼音、同义词、模糊匹配
- 结构化查询:term、range、bool 等精确查询
- 地理位置:geo_point 支持范围搜索、距离排序
- 嵌套查询:nested、parent-child 支持复杂数据关系
- 聚合分析:terms、histogram、stats 等实时聚合

**真实业务场景**:
```
电商搜索:"苹果手机"

传统方案的困难:
- "苹果"可能是水果,也可能是品牌
- "手机"的同义词:智能机、移动电话
- 需要考虑:销量、好评率、价格、库存
- 需要聚合:品牌、价格区间、屏幕尺寸

Elasticsearch 方案:
1. 分词:"苹果"、"手机"
2. 召回:匹配所有包含这两个词的商品
3. 重排序:
   - 文本相关性(BM25)
   - 销量权重(function_score)
   - 好评率加成(boost)
   - 库存过滤(filter)
4. 聚合:按品牌、价格区间分组统计
5. 响应时间:毫秒级
```

#### 问题4:实时性需求

**批处理搜索的问题**:
- Solr 的 commit 操作耗时长,实时性差
- 传统数据仓库的数据更新延迟可能达到小时级
- 用户发布内容后,希望立即能搜索到

**Elasticsearch 的近实时(NRT)**:
- 写入数据后,默认1秒内可搜索(refresh_interval)
- 可调整到100ms,满足更高实时性需求
- Translog 保证数据持久性,避免丢失
- 适合社交媒体、新闻、监控等实时场景

### 1.3 发展历程中的重要里程碑

#### 2010-2012:初创期,奠定基础

**0.x 版本(2010-2011)**:
- 2010.02:Elasticsearch 0.4.0 首次发布
- 核心功能:基于 Lucene,分布式架构,REST API
- 早期采用者:GitHub、StumbleUpon 等
- 问题:稳定性不足,功能较少,社区规模小

**1.x 版本(2014.02 - 2017)**:
- 重要特性:聚合(Aggregations)框架,极大增强数据分析能力
- Kibana 集成:可视化日志和指标(ELK Stack 形成)
- 性能优化:Doc Values 减少内存占用
- 生产可用:越来越多企业在生产环境部署

#### 2013-2015:ELK Stack 的兴起

**Elastic Stack 生态形成**:
- Elasticsearch:存储和搜索
- Logstash:数据采集和转换
- Kibana:可视化和探索
- Beats:轻量级数据采集器(Filebeat、Metricbeat 等)

**日志分析场景爆发**:
- 微服务架构兴起,日志量激增
- ELK Stack 成为日志分析的事实标准
- 替代传统的 Splunk(成本更低,开源)
- 适用场景扩展:APM、安全分析、业务监控

**里程碑事件**:
- 2015:Elastic 公司完成7000万美元 C 轮融资
- 2015:Elasticsearch 2.0 发布,pipeline aggregations
- 用户增长:超过5000家企业使用

#### 2016-2018:企业级特性完善

**5.x 版本(2016.10)**:
- Lucene 6 升级:BM25 成为默认相关性算法
- Ingest Node:数据预处理能力
- Painless 脚本:更安全的脚本语言
- Shrink API:减少分片数,优化历史数据

**6.x 版本(2017.11)**:
- 单一文档类型:简化 index 结构
- Sequence IDs:改进副本同步机制
- Index Lifecycle Management (ILM):自动管理索引生命周期
- SQL 支持:通过 SQL 查询 Elasticsearch

**商业化进程**:
- X-Pack:安全、监控、告警、机器学习(付费功能)
- 2018:Elastic 在纽交所上市(股票代码:ESTC)
- 开源协议争议:部分功能从 Apache 2.0 改为 Elastic License

#### 2019-2021:云原生与机器学习

**7.x 版本(2019.04 - 2021)**:
- 默认分片数从5改为1:避免过度分片
- 自适应副本选择:查询路由更智能
- Cross-cluster replication:跨集群数据同步
- Snapshot lifecycle management:自动备份
- 机器学习:异常检测、预测、数据可视化

**云服务的竞争**:
- Elastic Cloud:官方云服务
- AWS Elasticsearch Service → Amazon OpenSearch Service
- 2021:Elastic 改变开源协议为 SSPL,引发争议
- AWS fork 了 Elasticsearch,推出 OpenSearch(真正开源)

#### 2022-至今:AI 时代的探索

**8.x 版本(2022.02 - 至今)**:
- 向量搜索:支持 k-NN,适配 AI 场景
- 自然语言查询:通过 NLP 模型理解查询意图
- APM 增强:分布式追踪、性能监控
- 安全分析:SIEM(Security Information and Event Management)

**AI 与搜索的融合**:
- 语义搜索:不再依赖关键词,理解查询意图
- 向量嵌入:将文本、图片转为向量,相似度搜索
- RAG(Retrieval-Augmented Generation):为 LLM 提供知识库
- 实时推荐:基于用户行为的个性化搜索

**当前挑战**:
- OpenSearch 的分裂:社区分化,功能分叉
- 云厂商竞争:AWS、阿里云、腾讯云的托管服务
- 成本优化:存储和计算成本高,需要更好的压缩和分层
- 复杂性增加:功能越来越多,学习曲线陡峭

### 1.4 目前在业界的地位和影响力

#### 市场占有率与采用情况

**全球使用规模**:
- 超过10万家企业使用 Elasticsearch
- Docker Hub 下载量超过10亿次
- GitHub Star 数:65,000+(Elasticsearch)+ 18,000+(OpenSearch)
- StackOverflow 问题数:超过10万个

**主流应用场景**:
1. **日志与指标分析**(70%):
   - ELK/EFK Stack 是事实标准
   - 替代 Splunk、Sumo Logic
   - 微服务、云原生环境的首选

2. **企业搜索**(15%):
   - 内部文档搜索
   - 电商商品搜索
   - 内容管理系统

3. **安全分析**(10%):
   - SIEM 系统
   - 威胁检测
   - 审计日志分析

4. **业务分析**(5%):
   - 实时仪表盘
   - 用户行为分析
   - A/B 测试分析

**行业分布**:
- 互联网科技:90%的头部公司使用
- 金融:合规、风控、交易监控
- 电商:搜索、推荐、用户画像
- 物联网:设备日志、传感器数据
- 医疗:病历搜索、研究数据

**知名用户案例**:
- **Netflix**:每天处理超过1PB日志数据
- **Uber**:实时位置搜索和路径规划
- **Tinder**:用户匹配和推荐
- **GitHub**:代码搜索(1300亿行代码)
- **Wikipedia**:文章搜索
- **LinkedIn**:职位搜索和推荐
- **eBay**:商品搜索和分类

#### 与竞争对手的对比

**Elasticsearch vs Solr**:
| 维度 | Elasticsearch | Solr |
|------|--------------|------|
| 易用性 | REST API,JSON,开箱即用 | 需要配置 schema.xml |
| 分布式 | 原生分布式,自动分片 | 依赖 ZooKeeper,配置复杂 |
| 实时性 | NRT,默认1秒 | 需要手动 commit |
| 社区活跃度 | 非常活跃,更新快 | 相对平缓 |
| 生态 | ELK Stack 完整生态 | 主要依赖 Solr 本身 |
| 适用场景 | 日志、指标、实时搜索 | 传统企业搜索 |

**市场趋势**:
- Elasticsearch 在新项目中的采用率远超 Solr
- Solr 的市场份额逐年下降
- 大部分 Solr 用户在迁移到 Elasticsearch

**Elasticsearch vs 传统数据库(MySQL/PostgreSQL)**:
| 维度 | Elasticsearch | 传统数据库 |
|------|--------------|-----------|
| 查询类型 | 全文搜索、聚合分析 | ACID 事务、关系查询 |
| 数据模型 | 文档型(JSON) | 表格型(Row) |
| 扩展性 | 水平扩展容易 | 垂直扩展为主,水平扩展难 |
| 一致性 | 最终一致性 | 强一致性 |
| 适用场景 | 搜索、日志、分析 | 业务数据、交易 |

**并存而非替代**:
- Elasticsearch 不是万能的,不能完全替代数据库
- 典型架构:MySQL 存储业务数据,Elasticsearch 提供搜索
- 数据同步:通过 Logstash、Canal、Debezium 等

**Elasticsearch vs Splunk**:
| 维度 | Elasticsearch | Splunk |
|------|--------------|--------|
| 成本 | 开源免费(基础功能) | 按日志量收费,成本高 |
| 灵活性 | 高度可定制 | 开箱即用,但定制受限 |
| 性能 | 需要调优 | 企业级优化 |
| 生态 | 开源社区 | 商业支持 |
| 学习曲线 | 较陡峭 | 友好的UI |

**市场选择**:
- 初创公司和互联网公司:首选 Elasticsearch(成本)
- 大型企业和金融机构:部分仍使用 Splunk(稳定性和支持)
- 趋势:越来越多企业从 Splunk 迁移到 Elasticsearch

#### 技术影响力

**对搜索技术的贡献**:
- 普及了倒排索引和分布式搜索的概念
- 推动了 JSON 在数据交换中的应用
- 证明了 RESTful API 的易用性
- 影响了后续搜索引擎的设计(如 Meilisearch、Typesense)

**对可观测性的影响**:
- 定义了日志分析的标准架构(ELK)
- 推动了可观测性(Observability)概念的发展
- 影响了 Prometheus + Grafana、Datadog 等方案
- 日志、指标、追踪的统一分析

**对大数据生态的影响**:
- 与 Hadoop、Spark 生态集成
- 支持实时流处理(Kafka + Elasticsearch)
- 推动了 Lambda 架构的应用
- 影响了时序数据库的设计(InfluxDB、TimescaleDB)

### 1.5 未来发展趋势和方向

#### 趋势1:AI 与搜索的深度融合

**语义搜索的革命**:
- 传统搜索:基于关键词匹配
- 语义搜索:理解用户意图和内容含义
- 向量化表示:BERT、GPT 等模型将文本转为高维向量
- 相似度计算:余弦相似度、欧氏距离

**Elasticsearch 的演进**:
- 原生支持向量搜索(dense_vector 字段)
- k-NN 算法优化(HNSW、IVF)
- 混合搜索:关键词搜索 + 语义搜索
- 重排序:BM25 召回 + 深度模型精排

**未来应用场景**:
- 智能客服:理解自然语言问题,精准匹配答案
- 内容推荐:不依赖标签,基于内容相似度
- 图片搜索:上传图片,搜索相似商品
- 代码搜索:理解代码语义,而非简单字符串匹配

#### 趋势2:云原生与 Serverless

**云服务的主流化**:
- 自建集群成本高,运维复杂
- 云服务趋势:Elastic Cloud、AWS OpenSearch、阿里云 Elasticsearch
- Serverless:按需付费,自动扩缩容
- 无需关心节点管理、版本升级、备份恢复

**技术演进方向**:
- 存算分离:计算和存储独立扩展
- 对象存储集成:S3、OSS 作为冷数据存储
- 容器化:Kubernetes Operator 简化部署
- 边缘计算:边缘节点的轻量级搜索

**成本优化**:
- 智能分层:热温冷数据自动分层
- 数据压缩:更高效的压缩算法
- 查询优化:自动索引建议、慢查询分析
- 资源调度:根据负载自动调整资源

#### 趋势3:实时分析与 OLAP

**从搜索到分析**:
- 初期定位:搜索引擎
- 现状:搜索 + 日志分析 + 指标分析
- 未来:实时 OLAP,替代 Druid、ClickHouse 部分场景

**技术增强**:
- 列式存储:更高效的聚合查询
- 物化视图:预聚合提升性能
- SQL 支持完善:兼容更多 BI 工具
- 数据湖集成:直接查询 Parquet、ORC 文件

**应用场景扩展**:
- 实时仪表盘:业务指标实时监控
- 用户行为分析:漏斗分析、路径分析
- A/B 测试:实时统计显著性
- 实时推荐:基于实时行为的推荐

#### 趋势4:安全与合规

**数据安全需求**:
- 敏感数据脱敏:PII(个人身份信息)保护
- 访问控制:RBAC(基于角色的访问控制)
- 审计日志:所有操作可追溯
- 加密:传输加密(TLS)、存储加密

**合规要求**:
- GDPR:欧盟数据保护法规
- CCPA:加州消费者隐私法
- 等保2.0:中国信息安全等级保护
- SOC 2:服务组织控制

**Elasticsearch 的增强**:
- Security 功能开源(原 X-Pack)
- 字段级和文档级权限控制
- SAML、LDAP、Active Directory 集成
- 数据生命周期管理(自动删除过期数据)

#### 趋势5:多模态搜索

**搜索对象的扩展**:
- 传统:文本搜索
- 现在:文本 + 结构化数据
- 未来:文本 + 图片 + 音频 + 视频

**技术支持**:
- 向量搜索:支持任意模态的向量表示
- 多模态模型:CLIP(图文)、Whisper(音频)
- 跨模态检索:用文本搜图片,用图片搜文本

**应用场景**:
- 电商:拍照搜同款
- 视频平台:截图找原视频
- 音乐平台:哼歌识曲
- 医疗:医学影像检索

#### 挑战与机遇

**技术挑战**:
- 成本控制:存储和计算成本持续增长
- 复杂性管理:功能越来越多,学习曲线陡峭
- 性能优化:数据量增长快于硬件性能提升
- 社区分裂:Elasticsearch vs OpenSearch

**商业挑战**:
- 开源与商业化的平衡
- 云厂商的竞争
- 开源协议的争议
- 成本与价值的权衡

**发展机遇**:
- AI 时代的新需求:RAG、向量搜索
- 可观测性市场增长:日志、指标、追踪统一
- 实时分析需求:从批处理到实时
- 边缘计算:物联网、5G 时代的新场景

## 二、核心概念与设计理念

### 2.1 Elasticsearch 的核心设计理念

#### 设计理念1:开箱即用的分布式

**传统搜索引擎的分布式困境**:
```
Solr 的分布式部署:
1. 安装 ZooKeeper 集群
2. 配置 solr.xml 指定 ZooKeeper 地址
3. 创建 Collection 并指定分片数
4. 手动上传配置文件到 ZooKeeper
5. 配置负载均衡
6. 配置节点间的通信

开发者心声:"只是想搜索个数据,为啥这么复杂?"
```

**Elasticsearch 的零配置分布式**:
```
部署 Elasticsearch 集群:
1. 在3台服务器上下载 Elasticsearch
2. 修改 cluster.name 为相同名称
3. 启动 ./bin/elasticsearch
4. 集群自动组建完成!

核心理念:
- 自动发现(Zen Discovery):节点通过多播或单播自动发现
- 自动分片:索引自动分配到各个节点
- 自动副本:数据自动复制,保证高可用
- 自动恢复:节点故障自动重新分配分片
```

**深层设计哲学**:
- 降低使用门槛:让非专业运维人员也能部署
- 约定优于配置:合理的默认值,减少配置项
- 自愈能力:系统能自动应对大部分故障
- 弹性伸缩:增加节点即可扩容,无需复杂配置

#### 设计理念2:近实时(Near Real-Time)

**批处理搜索的问题**:
```
传统搜索引擎:
- 写入数据后,需要手动执行 commit
- commit 会创建新的 Segment,耗时长(秒到分钟级)
- 实时性差,用户体验不好

场景:
用户发布一条微博 → 5分钟后才能搜索到
结果:用户以为发布失败,多次尝试,产生重复内容
```

**Elasticsearch 的 NRT 设计**:
```
写入流程:
1. 数据写入内存缓冲区(buffer)
2. 每1秒(默认 refresh_interval)执行 refresh
3. refresh 将 buffer 中的数据写入 Segment(在文件系统缓存中)
4. Segment 立即可搜索
5. 每30分钟或 Translog 达到阈值,执行 flush
6. flush 将 Segment 刷入磁盘,清空 Translog

关键点:
- 1秒延迟对大部分业务可接受
- 不需要等待刷盘,性能更好
- Translog 保证数据不丢失
```

**业务场景的权衡**:
- 社交媒体:1秒延迟可接受,甚至可以调到100ms
- 日志系统:30秒延迟可接受,降低 refresh 频率提升写入性能
- 离线数据导入:可以临时关闭 refresh,导入完成后手动 refresh

#### 设计理念3:Schema-free(无模式)

**关系型数据库的 Schema 约束**:
```
MySQL 的严格 Schema:
1. 建表:CREATE TABLE users (id INT, name VARCHAR(50), age INT)
2. 插入数据:必须符合 Schema
3. 增加字段:ALTER TABLE users ADD COLUMN email VARCHAR(100)
4. 字段类型错误:INSERT INTO users VALUES (1, 'Alice', 'twenty') -- 报错

问题:
- 灵活性差:新增字段需要修改表结构
- 异构数据:不同来源的数据字段不一致,难以统一存储
- 开发效率:需要提前设计好所有字段
```

**Elasticsearch 的动态 Mapping**:
```
无需提前定义 Schema:
POST /users/_doc
{
  "id": 1,
  "name": "Alice",
  "age": 25,
  "email": "alice@example.com"
}

Elasticsearch 自动推断类型:
- id: long
- name: text + keyword
- age: long
- email: text + keyword

新增字段自动识别:
POST /users/_doc
{
  "id": 2,
  "name": "Bob",
  "age": 30,
  "email": "bob@example.com",
  "phone": "1234567890"  // 新字段
}

Elasticsearch 自动添加 phone 字段到 Mapping
```

**灵活性 vs 精确性**:
- 优点:开发速度快,适应变化快
- 缺点:可能推断错误(如"20"被识别为字符串而非数字)
- 最佳实践:生产环境建议显式定义 Mapping,避免自动推断错误

#### 设计理念4:RESTful API 与 JSON

**传统搜索引擎的客户端**:
```
Solr 的 Java API:
SolrClient client = new HttpSolrClient.Builder("http://localhost:8983/solr").build();
SolrInputDocument doc = new SolrInputDocument();
doc.addField("id", "1");
doc.addField("name", "Alice");
client.add("users", doc);
client.commit("users");

问题:
- 需要引入 SDK
- 只支持 Java,其他语言需要第三方库
- API 复杂,学习成本高
```

**Elasticsearch 的 RESTful API**:
```
HTTP 请求:
PUT /users/_doc/1
{
  "name": "Alice",
  "age": 25
}

优点:
- 任何语言都可以通过 HTTP 调用
- 使用 JSON,通用性强
- 可以用 curl、Postman 直接测试
- 符合 REST 规范,易于理解

多语言支持:
- Python: from elasticsearch import Elasticsearch
- JavaScript: const { Client } = require('@elastic/elasticsearch')
- Go: github.com/elastic/go-elasticsearch
- 任何支持 HTTP 的语言
```

**RESTful 设计的优雅性**:
```
资源路径清晰:
- 索引(Index):数据库
- 类型(Type):表(7.0后废弃)
- 文档(Document):一条记录

操作映射到 HTTP 方法:
- 创建:PUT /index/_doc/id 或 POST /index/_doc
- 读取:GET /index/_doc/id
- 更新:POST /index/_update/id
- 删除:DELETE /index/_doc/id
- 搜索:GET /index/_search

符合 HTTP 语义,直观易懂
```

#### 设计理念5:分片与副本的高可用

**单点故障的风险**:
```
传统单机搜索引擎:
- 服务器宕机 → 服务完全不可用
- 磁盘损坏 → 数据永久丢失
- 性能瓶颈 → 无法水平扩展

业务影响:
- 可用性:99%(每年宕机3.65天)
- 数据丢失:灾难性后果
- 无法扩展:增长受限
```

**Elasticsearch 的高可用设计**:
```
主分片 + 副本分片:
- 索引拆分为多个主分片(Primary Shard)
- 每个主分片有多个副本分片(Replica Shard)
- 主副本不能在同一节点

示例:3节点集群,5个主分片,1个副本
- Node1: P0, P1, R2, R3, R4
- Node2: P2, P3, R0, R1, R4
- Node3: P4, R0, R1, R2, R3

容灾能力:
- 任意1个节点宕机:集群仍可用(黄色状态)
- 任意2个节点宕机:可能部分数据不可用(红色状态)
- 副本数设置为2:可以容忍2个节点同时宕机
```

**读写分离**:
- 写入:必须写到主分片,然后同步到副本
- 查询:可以从主分片或任意副本分片读取
- 负载均衡:协调节点自动选择负载较低的分片

### 2.2 Elasticsearch 的架构设计

#### 核心组件1:Node(节点)

**节点的角色分工**:
```
1. Master Node(主节点):
   - 管理集群状态
   - 创建/删除索引
   - 分配分片到节点
   - 不处理数据读写
   - 配置:node.master: true, node.data: false

2. Data Node(数据节点):
   - 存储数据
   - 执行查询和聚合
   - 执行写入操作
   - 配置:node.master: false, node.data: true

3. Coordinating Node(协调节点):
   - 接收客户端请求
   - 将请求路由到相应的数据节点
   - 合并各节点返回的结果
   - 默认:所有节点都是协调节点

4. Ingest Node(预处理节点):
   - 在写入前对数据进行转换
   - 类似 Logstash 的轻量级功能
   - 配置:node.ingest: true

5. Machine Learning Node(机器学习节点):
   - 执行机器学习任务
   - 异常检测、预测
   - 配置:node.ml: true
```

**节点角色的选择策略**:
```
小集群(3-5节点):
- 所有节点都是 Master-eligible + Data
- 简单,资源利用率高

中等集群(10-30节点):
- 3个专用 Master 节点(不存数据)
- 其余都是 Data 节点
- 避免 Master 节点负载过高

大集群(50+节点):
- 3个专用 Master 节点
- Data 节点按数据类型分层(Hot/Warm/Cold)
- 专用 Coordinating 节点处理查询(如果查询压力大)
- 专用 Ingest 节点(如果需要大量数据预处理)
```

#### 核心组件2:Index(索引)

**索引的两层含义**:
```
1. 名词:数据的集合
   - 类似数据库中的"数据库"
   - 例:users 索引存储所有用户数据

2. 动词:建立倒排索引
   - 将文档内容分词,建立词语到文档的映射
   - 这是 Elasticsearch 的核心能力
```

**索引的物理结构**:
```
Index → Shard → Segment → Documents

Index(索引):
- 逻辑概念
- 可以跨多个节点
- 例:logs-2024-12-24

Shard(分片):
- 物理概念
- 一个 Lucene 索引
- 分布在不同节点

Segment(段):
- 不可变文件
- 存储一部分文档
- 定期合并(Merge)

Document(文档):
- 最小数据单元
- JSON 格式
```

**索引设计的最佳实践**:
```
场景1:业务数据(用户、订单等)
- 索引命名:users, orders, products
- 分片数:3-5个主分片
- 副本数:1个副本
- 生命周期:长期存储

场景2:日志数据
- 索引命名:logs-2024-12-24(按天分索引)
- 分片数:5个主分片/索引
- 副本数:1个副本
- 生命周期:7天后迁移到 Warm 节点,30天后删除

场景3:指标数据
- 索引命名:metrics-2024-12(按月分索引)
- 分片数:10个主分片/索引
- 副本数:0个副本(可以从源重新采集)
- 生命周期:3个月后删除
```

#### 核心组件3:Shard(分片)

**为什么需要分片?**
```
问题:单个索引1TB数据,单机无法存储

解决:将索引拆分为多个分片
- 5个分片,每个分片200GB
- 分布在5个节点,每个节点200GB
- 查询时,5个分片并行执行,速度提升5倍
```

**分片的本质**:
```
一个分片 = 一个 Lucene 索引
- 有自己的倒排索引
- 有自己的 Segment 文件
- 有自己的内存缓冲
- 完全独立的搜索能力

索引查询 = 多个分片并行查询 + 结果合并
```

**分片数量的权衡**:
```
分片过多:
- 每个分片有元数据开销(内存)
- 1000个分片 vs 10个分片,内存可能相差10倍
- 查询时需要合并更多分片的结果,协调节点压力大

分片过少:
- 单个分片过大,查询慢
- 无法充分利用集群资源
- 未来扩容困难(主分片数无法修改)

推荐:
- 单个分片大小:20-50GB
- 总分片数 = 数据总量 / 单分片大小
- 节点数 ≤ 分片数(否则有节点空闲)
```

**分片分配策略**:
```
Elasticsearch 的分片分配考虑:
1. 同一索引的主副本分离
2. 磁盘使用率均衡
3. 每个节点的分片数均衡
4. 避免热点节点

示例:
索引 users:5个主分片,1个副本
3个节点:Node1, Node2, Node3

可能的分配:
Node1: P0, P1, R2, R3
Node2: P2, R3, R0, R1
Node3: P4, R2, R4

特点:
- 每个节点都有一些主分片和副本分片
- 主副本不在同一节点
- 分片数基本均衡
```

#### 核心组件4:Replica(副本)

**副本的作用**:
```
1. 高可用:
   - 主分片宕机,副本自动提升为主分片
   - 保证服务不中断

2. 读负载均衡:
   - 查询可以从主分片或副本读取
   - 分担查询压力

3. 数据冗余:
   - 防止数据丢失
   - 多个副本,安全性更高
```

**写入流程**:
```
1. 客户端发送写入请求到任意节点(协调节点)
2. 协调节点根据文档 ID 计算应该写入哪个主分片
   - 公式:shard = hash(routing) % number_of_primary_shards
   - 默认 routing = document_id
3. 协调节点将请求转发到主分片所在节点
4. 主分片执行写入:
   - 写入内存缓冲区
   - 写入 Translog(持久化)
   - 返回成功
5. 主分片将请求转发给所有副本分片
6. 副本分片执行写入
7. 所有副本都成功后,返回客户端

时序:
客户端 → 协调节点 → 主分片 → 副本分片 → 主分片 → 协调节点 → 客户端
```

**一致性保证**:
```
默认:
- 只要主分片写入成功,就返回客户端
- 副本异步同步
- 优点:写入快
- 缺点:主分片宕机时,可能丢失未同步的数据

强一致性配置:
- wait_for_active_shards=all
- 等待所有副本都写入成功再返回
- 优点:数据安全
- 缺点:写入慢,副本宕机会导致写入失败

推荐:
- 默认配置适合大部分场景
- 金融等关键业务使用 wait_for_active_shards=all
```

#### 核心组件5:Inverted Index(倒排索引)

**倒排索引的核心价值**:
```
场景:在100万篇文章中搜索"Elasticsearch 分布式搜索"

正排索引(传统数据库):
Doc1 → "学习 Elasticsearch 分布式搜索引擎"
Doc2 → "MySQL 数据库优化技巧"
Doc3 → "分布式系统架构设计"
...
Doc1000000 → "..."

查询流程:
1. 遍历100万篇文章
2. 对每篇文章内容进行关键词匹配
3. 时间复杂度:O(n × m),n为文章数,m为文章长度
4. 耗时:可能需要数分钟

倒排索引:
"Elasticsearch" → [Doc1]
"分布式" → [Doc1, Doc3]
"搜索" → [Doc1]
"MySQL" → [Doc2]
"系统" → [Doc3]
...

查询流程:
1. 分词:"Elasticsearch"、"分布式"、"搜索"
2. 查倒排索引:
   - "Elasticsearch" → [Doc1]
   - "分布式" → [Doc1, Doc3]
   - "搜索" → [Doc1]
3. 求交集:[Doc1]
4. 时间复杂度:O(词数),与文章总数无关
5. 耗时:毫秒级
```

**倒排索引的数据结构**:
```
完整的倒排索引包含:

1. Term Dictionary(词典):
   - 所有词语的有序列表
   - 使用 FST(Finite State Transducer)存储
   - 支持快速前缀查询

2. Posting List(倒排列表):
   - 包含某个词的所有文档 ID
   - 按文档 ID 排序
   - 使用差分编码和压缩

3. Term Frequency(词频):
   - 词在每个文档中出现的次数
   - 用于计算相关性得分

4. Position(位置):
   - 词在文档中的位置
   - 用于短语查询(phrase query)
   - 可选,占用更多空间

5. Offset(偏移量):
   - 词在文档中的字符偏移
   - 用于高亮显示
   - 可选,占用更多空间

示例:
词语"分布式":
{
  "term": "分布式",
  "doc_count": 2,
  "postings": [
    {"doc_id": 1, "tf": 2, "positions": [5, 20], "offsets": [[10, 13], [40, 43]]},
    {"doc_id": 3, "tf": 1, "positions": [8], "offsets": [[15, 18]]}
  ]
}
```

**倒排索引的构建过程**:
```
原始文档:
{
  "_id": "1",
  "title": "Elasticsearch 分布式搜索引擎",
  "content": "Elasticsearch 是一个分布式、RESTful 风格的搜索和数据分析引擎"
}

1. 分词(Analysis):
   - title: ["Elasticsearch", "分布式", "搜索", "引擎"]
   - content: ["Elasticsearch", "是", "一个", "分布式", "RESTful", "风格", "的", "搜索", "和", "数据", "分析", "引擎"]

2. 去停用词(Stop Words):
   - 去除"是"、"一个"、"的"、"和"等无意义词
   - 结果:["Elasticsearch", "分布式", "RESTful", "风格", "搜索", "数据", "分析", "引擎"]

3. 标准化(Normalization):
   - 转小写:Elasticsearch → elasticsearch
   - 提取词干:searching → search

4. 建立倒排索引:
   - "elasticsearch" → {doc: 1, fields: [title, content], positions: ...}
   - "分布式" → {doc: 1, fields: [title, content], positions: ...}
   - ...

5. 写入 Segment:
   - 倒排索引写入不可变的 Segment 文件
   - 多个 Segment 通过 Merge 合并
```

#### 核心组件6:Segment(段)

**Segment 的设计哲学**:
```
为什么不直接修改索引?
- 修改索引需要重新排序、重建数据结构
- 并发写入时需要加锁
- 性能差,复杂度高

Lucene 的解决方案:不可变 Segment
- 写入的数据生成新的 Segment
- 查询时合并多个 Segment 的结果
- 旧的 Segment 定期合并(Merge)
```

**Segment 的生命周期**:
```
1. 写入内存缓冲区:
   - 文档先写入内存的 Indexing Buffer
   - 此时不可搜索

2. Refresh(刷新):
   - 默认每1秒执行一次
   - 将缓冲区的数据写入新的 Segment
   - Segment 在文件系统缓存中(未刷盘)
   - 立即可搜索

3. Flush(刷盘):
   - 每30分钟或 Translog 达到阈值
   - 将 Segment 从文件系统缓存刷入磁盘
   - 清空 Translog
   - 持久化完成

4. Merge(合并):
   - 后台定期执行
   - 将多个小 Segment 合并为大 Segment
   - 删除已标记删除的文档
   - 减少 Segment 数量,提升查询性能
```

**Segment 过多的问题**:
```
问题:
- 查询需要搜索所有 Segment,再合并结果
- Segment 越多,查询越慢
- 每个 Segment 有文件句柄开销

场景:
- 高频写入,每秒生成1个 Segment
- 1小时后有3600个 Segment
- 查询需要搜索3600个 Segment
- 性能显著下降

解决:
- Merge 策略:自动合并小 Segment
- Tiered Merge Policy(分层合并):
  - 小 Segment 优先合并
  - 大 Segment 合并频率低
- 控制 Segment 数量在合理范围(如100以内)
```

**Segment 的不可变性优势**:
```
1. 并发友好:
   - 不需要加锁
   - 多个线程可以同时读取

2. 缓存友好:
   - Segment 不变,缓存不会失效
   - OS 文件系统缓存有效

3. 压缩高效:
   - 不可变数据可以高度压缩
   - 节省存储空间

4. 崩溃恢复简单:
   - Segment 要么完整,要么不存在
   - 不会出现部分损坏

缺点:
- 删除和更新不是真删除:
  - 删除:标记删除,Merge 时才物理删除
  - 更新:先删除再插入,产生新文档
- 磁盘空间占用:旧版本文档占用空间,直到 Merge
```

### 2.3 Elasticsearch 与同类工具的核心差异

#### Elasticsearch vs Solr

**架构层面**:
```
Elasticsearch:
- 原生分布式,基于 P2P 模型
- 节点自动发现,无需外部协调
- 集群状态由 Master 节点管理
- 优点:简单,易于扩展
- 缺点:Master 节点是单点(虽有选举机制)

Solr:
- 需要依赖 ZooKeeper 进行协调
- 集群状态存储在 ZooKeeper
- 优点:协调机制成熟,状态一致性强
- 缺点:部署复杂,多了一个依赖

结论:
- 小集群:Elasticsearch 更简单
- 超大集群(100+节点):两者差距不大,Solr 更成熟
```

**查询性能**:
```
基准测试(1亿文档,全文搜索):
- Elasticsearch:平均响应时间50ms
- Solr:平均响应时间60ms
- 结论:性能相近,Elasticsearch 略快

原因:
- 底层都是 Lucene
- 性能差异主要在:
  - 查询优化策略
  - 缓存机制
  - 网络序列化

实际场景:
- 简单查询:两者相近
- 复杂聚合:Elasticsearch 通常更快
- 大数据量:两者都需要调优
```

**生态与社区**:
```
Elasticsearch:
- ELK Stack(Elasticsearch + Logstash + Kibana)
- Beats 系列(Filebeat、Metricbeat 等)
- 社区活跃,更新快
- 云服务:Elastic Cloud、AWS、阿里云
- GitHub Star:65,000+

Solr:
- 主要依赖 Solr 本身
- SolrCloud 提供分布式能力
- 社区相对平缓
- 云服务:较少
- GitHub Star:5,000+

市场趋势:
- 新项目大多选择 Elasticsearch
- Solr 用户逐渐迁移到 Elasticsearch
- 但 Solr 在某些传统企业仍有大量使用
```

#### Elasticsearch vs 传统数据库(MySQL)

**数据模型**:
```
MySQL:
- 结构化数据,固定 Schema
- 表格模型,行和列
- 关系型,支持 JOIN
- 强一致性,ACID 事务

Elasticsearch:
- 半结构化数据,动态 Schema
- 文档模型,JSON
- 非关系型,不支持 JOIN(需要嵌套或父子文档)
- 最终一致性,无事务

适用场景:
- MySQL:业务数据,需要事务,关系复杂
- Elasticsearch:搜索、日志、分析,数据量大
```

**查询能力**:
```
全文搜索:
- MySQL:
  - LIKE '%keyword%':全表扫描,极慢
  - FULLTEXT INDEX:支持有限,中文分词差
  - 性能:千万级数据已是极限

- Elasticsearch:
  - 倒排索引,专为全文搜索设计
  - 强大的分词和分析能力
  - 性能:亿级数据仍快速响应

精确查询:
- MySQL:
  - B+树索引,精确查询极快
  - 支持复杂的 JOIN 和子查询

- Elasticsearch:
  - term、range 查询也快
  - 不支持 JOIN,需要数据冗余或嵌套

聚合分析:
- MySQL:
  - GROUP BY 在大数据量下很慢
  - 需要全表扫描或索引扫描

- Elasticsearch:
  - 聚合是核心功能,性能优秀
  - 支持多层聚合,实时计算
```

**扩展性**:
```
MySQL:
- 垂直扩展:升级服务器硬件
- 水平扩展:分库分表,复杂度高
- 主从复制:读写分离,但写入仍是瓶颈
- Sharding:需要应用层或中间件支持

Elasticsearch:
- 水平扩展:增加节点即可
- 自动分片和负载均衡
- 读写都可以扩展
- 无需应用层改动
```

**一致性 vs 可用性**:
```
MySQL:
- CAP 中选择:CP(一致性 + 分区容错性)
- 主从复制有延迟,但最终一致
- 事务保证强一致性

Elasticsearch:
- CAP 中选择:AP(可用性 + 分区容错性)
- 写入后1秒(refresh_interval)才可搜索
- 无事务,只保证单文档原子性

选择:
- 金融交易、订单:必须用 MySQL
- 搜索、日志、监控:Elasticsearch 更合适
- 典型架构:MySQL 存储,Elasticsearch 搜索
```

#### Elasticsearch vs ClickHouse(列式数据库)

**设计目标**:
```
Elasticsearch:
- 主要:全文搜索
- 次要:日志分析、指标聚合

ClickHouse:
- 主要:OLAP 分析
- 次要:日志存储

结论:有重叠,但侧重点不同
```

**查询性能**:
```
全文搜索:
- Elasticsearch:倒排索引,毫秒级
- ClickHouse:不支持全文搜索(有限的 LIKE 支持)

聚合分析:
- Elasticsearch:实时聚合,秒级
- ClickHouse:列式存储,聚合极快,毫秒到秒级

大宽表查询:
- Elasticsearch:性能一般,字段过多会慢
- ClickHouse:列式存储,只读需要的列,极快

写入性能:
- Elasticsearch:每秒数万到数十万条
- ClickHouse:批量写入,每秒数百万条
```

**适用场景对比**:
```
选择 Elasticsearch:
- 需要全文搜索
- 需要复杂的布尔查询和过滤
- 需要实时性(1秒可见)
- 数据模型是文档型(JSON)
- 已有 ELK Stack 生态

选择 ClickHouse:
- 纯 OLAP 分析,无需全文搜索
- 超大数据量(PB 级)
- 需要极致的聚合性能
- 数据模型是列式(宽表)
- 可以接受分钟级延迟

实际选择:
- 日志搜索 + 分析:Elasticsearch
- 纯指标分析:ClickHouse
- 混合场景:Elasticsearch 存储,ClickHouse 做深度分析
```

## 三、基础知识

### 3.1 安装部署的完整流程

#### 单机安装(开发环境)

**前置条件**:
```
系统要求:
- 操作系统:Linux、macOS、Windows
- Java:JDK 11 或更高版本(ES 8.0+ 内置 JDK,可选)
- 内存:至少 4GB RAM(建议 8GB+)
- 磁盘:至少 10GB 可用空间

检查 Java 版本:
java -version

如果没有 Java,安装:
# Ubuntu/Debian
sudo apt install openjdk-17-jdk

# CentOS/RHEL
sudo yum install java-17-openjdk

# macOS
brew install openjdk@17
```

**下载与安装**:
```bash
# 1. 下载 Elasticsearch
# 访问 https://www.elastic.co/downloads/elasticsearch
# 或使用 wget
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.11.0-linux-x86_64.tar.gz

# 2. 解压
tar -xzf elasticsearch-8.11.0-linux-x86_64.tar.gz
cd elasticsearch-8.11.0

# 3. 启动
./bin/elasticsearch

# 第一次启动会输出:
# ✅ Elasticsearch security features have been automatically configured!
# ✅ Authentication is enabled and cluster connections are encrypted.
#
# ℹ️  Password for the elastic user: xxxxxxxxxxxxx
# ℹ️  HTTP CA certificate SHA-256 fingerprint: yyyyyyyyyyyy
#
# ℹ️  You can complete the following actions at any time:
#   - Reset the password: ./bin/elasticsearch-reset-password -u elastic
#   - Generate an enrollment token: ./bin/elasticsearch-create-enrollment-token -s node

# 4. 验证(另开一个终端)
curl -k -u elastic:你的密码 https://localhost:9200

# 输出:
{
  "name" : "node-1",
  "cluster_name" : "elasticsearch",
  "cluster_uuid" : "...",
  "version" : {
    "number" : "8.11.0",
    ...
  },
  "tagline" : "You Know, for Search"
}
```

**配置文件说明**:
```yaml
# config/elasticsearch.yml

# 集群名称(同一集群的节点必须相同)
cluster.name: my-application

# 节点名称(每个节点唯一)
node.name: node-1

# 数据和日志路径
path.data: /var/lib/elasticsearch
path.logs: /var/log/elasticsearch

# 网络配置
network.host: 0.0.0.0  # 监听所有网卡(生产环境用内网 IP)
http.port: 9200        # HTTP 端口
transport.port: 9300   # 节点间通信端口

# 集群发现(单机可以不配置)
discovery.seed_hosts: ["host1", "host2"]
cluster.initial_master_nodes: ["node-1", "node-2"]

# 内存锁定(避免 swap)
bootstrap.memory_lock: true

# 安全配置(8.0+ 默认启用)
xpack.security.enabled: true
xpack.security.enrollment.enabled: true
xpack.security.http.ssl.enabled: true
xpack.security.transport.ssl.enabled: true
```

**调整 JVM 内存**:
```bash
# config/jvm.options

# 堆内存(建议设置为物理内存的50%,但不超过32GB)
-Xms4g
-Xmx4g

# 为什么不超过32GB?
# - Java 的压缩指针(Compressed Oops)在32GB以下有效
# - 超过32GB,指针占用翻倍,实际可用内存反而减少
# - 建议:单机64GB内存,设置 JVM 为30GB,留30GB给操作系统缓存
```

**后台运行与开机自启**:
```bash
# 后台运行
./bin/elasticsearch -d -p pid

# 停止
kill `cat pid`

# 使用 systemd(推荐)
# 1. 创建用户
sudo useradd elasticsearch

# 2. 安装为 systemd 服务
sudo /usr/lib/systemd/system/elasticsearch.service

# 3. 启动
sudo systemctl start elasticsearch

# 4. 设置开机自启
sudo systemctl enable elasticsearch

# 5. 查看状态
sudo systemctl status elasticsearch
```

#### 集群安装(生产环境)

**集群规划**:
```
示例:3节点集群

节点1(Master + Data):
- 主机名:es-node1
- IP:192.168.1.101
- 角色:Master-eligible + Data

节点2(Master + Data):
- 主机名:es-node2
- IP:192.168.1.102
- 角色:Master-eligible + Data

节点3(Master + Data):
- 主机名:es-node3
- IP:192.168.1.103
- 角色:Master-eligible + Data

配置要点:
- 至少3个 Master-eligible 节点(避免脑裂)
- 同一集群的 cluster.name 必须相同
- 每个节点的 node.name 必须唯一
```

**节点1配置**:
```yaml
# config/elasticsearch.yml

cluster.name: prod-cluster
node.name: es-node1

path.data: /data/elasticsearch
path.logs: /var/log/elasticsearch

network.host: 192.168.1.101
http.port: 9200
transport.port: 9300

# 发现其他节点
discovery.seed_hosts: ["192.168.1.101", "192.168.1.102", "192.168.1.103"]

# 初始 Master 候选节点(只在第一次启动时有效)
cluster.initial_master_nodes: ["es-node1", "es-node2", "es-node3"]

# 节点角色
node.master: true
node.data: true
node.ingest: true

# 防止脑裂(至少2个节点认可)
# ES 7.0+ 已自动计算,无需手动配置

bootstrap.memory_lock: true

xpack.security.enabled: true
xpack.security.transport.ssl.enabled: true
xpack.security.transport.ssl.verification_mode: certificate
xpack.security.transport.ssl.keystore.path: certs/elastic-certificates.p12
xpack.security.transport.ssl.truststore.path: certs/elastic-certificates.p12
```

**节点2、3配置**:
```yaml
# 节点2:只需修改 node.name 和 network.host
node.name: es-node2
network.host: 192.168.1.102

# 节点3
node.name: es-node3
network.host: 192.168.1.103

# 其他配置与节点1相同
```

**生成安全证书**:
```bash
# 在节点1上执行

# 1. 生成 CA
./bin/elasticsearch-certutil ca --out config/certs/elastic-stack-ca.p12

# 2. 生成节点证书
./bin/elasticsearch-certutil cert \
  --ca config/certs/elastic-stack-ca.p12 \
  --out config/certs/elastic-certificates.p12

# 3. 将证书复制到所有节点
scp config/certs/elastic-certificates.p12 es-node2:/path/to/elasticsearch/config/certs/
scp config/certs/elastic-certificates.p12 es-node3:/path/to/elasticsearch/config/certs/

# 4. 设置权限
chmod 600 config/certs/elastic-certificates.p12
```

**依次启动节点**:
```bash
# 节点1
./bin/elasticsearch

# 等待节点1启动完成后,启动节点2
# 节点2
./bin/elasticsearch

# 启动节点3
./bin/elasticsearch

# 验证集群状态
curl -k -u elastic:密码 https://192.168.1.101:9200/_cluster/health?pretty

# 输出:
{
  "cluster_name" : "prod-cluster",
  "status" : "green",
  "number_of_nodes" : 3,
  "number_of_data_nodes" : 3,
  "active_primary_shards" : 0,
  "active_shards" : 0,
  ...
}
```

#### Docker 部署

**单机 Docker**:
```bash
# 1. 拉取镜像
docker pull docker.elastic.co/elasticsearch/elasticsearch:8.11.0

# 2. 运行(单节点,开发环境)
docker run -d \
  --name elasticsearch \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms2g -Xmx2g" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0

# 3. 验证
curl http://localhost:9200

# 4. 查看日志
docker logs elasticsearch
```

**Docker Compose 集群**:
```yaml
# docker-compose.yml
version: '3.8'
services:
  es01:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    container_name: es01
    environment:
      - node.name=es01
      - cluster.name=docker-cluster
      - discovery.seed_hosts=es02,es03
      - cluster.initial_master_nodes=es01,es02,es03
      - bootstrap.memory_lock=true
      - "ES_JAVA_OPTS=-Xms2g -Xmx2g"
      - xpack.security.enabled=false
    ulimits:
      memlock:
        soft: -1
        hard: -1
    volumes:
      - es01data:/usr/share/elasticsearch/data
    ports:
      - 9200:9200
    networks:
      - elastic

  es02:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    container_name: es02
    environment:
      - node.name=es02
      - cluster.name=docker-cluster
      - discovery.seed_hosts=es01,es03
      - cluster.initial_master_nodes=es01,es02,es03
      - bootstrap.memory_lock=true
      - "ES_JAVA_OPTS=-Xms2g -Xmx2g"
      - xpack.security.enabled=false
    ulimits:
      memlock:
        soft: -1
        hard: -1
    volumes:
      - es02data:/usr/share/elasticsearch/data
    networks:
      - elastic

  es03:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    container_name: es03
    environment:
      - node.name=es03
      - cluster.name=docker-cluster
      - discovery.seed_hosts=es01,es02
      - cluster.initial_master_nodes=es01,es02,es03
      - bootstrap.memory_lock=true
      - "ES_JAVA_OPTS=-Xms2g -Xmx2g"
      - xpack.security.enabled=false
    ulimits:
      memlock:
        soft: -1
        hard: -1
    volumes:
      - es03data:/usr/share/elasticsearch/data
    networks:
      - elastic

volumes:
  es01data:
  es02data:
  es03data:

networks:
  elastic:
    driver: bridge
```

```bash
# 启动集群
docker-compose up -d

# 验证
curl http://localhost:9200/_cluster/health?pretty

# 停止
docker-compose down

# 停止并删除数据
docker-compose down -v
```

#### Kubernetes 部署(Helm)

```bash
# 1. 添加 Elastic Helm 仓库
helm repo add elastic https://helm.elastic.co
helm repo update

# 2. 创建命名空间
kubectl create namespace elastic

# 3. 安装 Elasticsearch
helm install elasticsearch elastic/elasticsearch \
  --namespace elastic \
  --set replicas=3 \
  --set minimumMasterNodes=2 \
  --set resources.requests.memory=4Gi \
  --set resources.limits.memory=4Gi \
  --set volumeClaimTemplate.resources.requests.storage=30Gi

# 4. 查看 Pod
kubectl get pods -n elastic

# 5. 查看服务
kubectl get svc -n elastic

# 6. 端口转发(本地访问)
kubectl port-forward -n elastic svc/elasticsearch-master 9200:9200

# 7. 验证
curl http://localhost:9200
```

### 3.2 基本配置和环境设置

#### 系统配置优化

**1. 文件描述符限制**:
```bash
# Elasticsearch 需要大量文件描述符

# 临时修改
ulimit -n 65536

# 永久修改
# /etc/security/limits.conf
elasticsearch soft nofile 65536
elasticsearch hard nofile 65536

# 验证
su - elasticsearch
ulimit -n
```

**2. 内存锁定**:
```bash
# 避免 Elasticsearch 内存被 swap 到磁盘

# /etc/security/limits.conf
elasticsearch soft memlock unlimited
elasticsearch hard memlock unlimited

# elasticsearch.yml
bootstrap.memory_lock: true

# 验证
GET /_nodes?filter_path=**.mlockall

# 输出应该是 true
```

**3. 虚拟内存**:
```bash
# Elasticsearch 使用 mmapfs 目录存储索引,需要足够的虚拟内存

# 临时修改
sudo sysctl -w vm.max_map_count=262144

# 永久修改
# /etc/sysctl.conf
vm.max_map_count=262144

# 生效
sudo sysctl -p

# 验证
sysctl vm.max_map_count
```

**4. Swap 禁用**:
```bash
# 完全禁用 swap(推荐)
sudo swapoff -a

# 永久禁用
# 编辑 /etc/fstab,注释掉 swap 行

# 或减少 swappiness
sudo sysctl -w vm.swappiness=1
# /etc/sysctl.conf
vm.swappiness=1
```

**5. 线程数限制**:
```bash
# Elasticsearch 需要创建大量线程

# /etc/security/limits.conf
elasticsearch soft nproc 4096
elasticsearch hard nproc 4096

# 验证
su - elasticsearch
ulimit -u
```

#### 重要配置项详解

**集群配置**:
```yaml
# cluster.name
# 作用:集群的唯一标识,同一集群的节点必须相同
# 默认:elasticsearch
# 建议:生产环境使用有意义的名称
cluster.name: prod-cluster

# node.name
# 作用:节点的唯一标识
# 默认:主机名
# 建议:使用有意义的名称
node.name: es-data-1

# node.attr.*
# 作用:节点的自定义属性,可用于分片分配
# 用途:标记节点的硬件特性(SSD/HDD)、数据中心等
node.attr.rack: rack1
node.attr.datacenter: dc1
```

**节点角色配置**:
```yaml
# node.master
# 作用:是否可以被选为 Master 节点
# 建议:生产环境至少3个 Master-eligible 节点
node.master: true

# node.data
# 作用:是否存储数据
# 建议:数据节点不要兼任 Master(大集群)
node.data: true

# node.ingest
# 作用:是否可以执行预处理管道
# 建议:如果有大量数据预处理,使用专用 Ingest 节点
node.ingest: true

# node.ml
# 作用:是否可以执行机器学习任务
# 建议:机器学习任务消耗大,使用专用节点
node.ml: false

# 专用节点配置示例

# 专用 Master 节点(不存数据,不处理查询)
node.master: true
node.data: false
node.ingest: false
node.ml: false

# 专用 Data 节点(只存数据,不参与 Master 选举)
node.master: false
node.data: true
node.ingest: false
node.ml: false

# 专用 Coordinating 节点(只做协调,不存数据)
node.master: false
node.data: false
node.ingest: false
node.ml: false
```

**路径配置**:
```yaml
# path.data
# 作用:数据存储路径
# 建议:
#   - 使用SSD
#   - 不要使用网络磁盘(NFS)
#   - 可以配置多个路径(JBOD,非RAID)
path.data: ["/data1/elasticsearch", "/data2/elasticsearch"]

# path.logs
# 作用:日志存储路径
# 建议:定期清理日志,避免占满磁盘
path.logs: /var/log/elasticsearch

# path.repo
# 作用:快照备份路径
# 建议:使用共享存储(NFS/S3)
path.repo: ["/mnt/backups"]
```

**内存配置**:
```yaml
# bootstrap.memory_lock
# 作用:锁定内存,避免 swap
# 建议:生产环境必须开启
bootstrap.memory_lock: true

# JVM 堆内存
# 配置文件:config/jvm.options
# 建议:
#   - Xms 和 Xmx 设置为相同值
#   - 不超过物理内存的50%
#   - 不超过32GB(压缩指针)
-Xms4g
-Xmx4g
```

**网络配置**:
```yaml
# network.host
# 作用:绑定的网络接口
# 默认:localhost(只能本机访问)
# 生产环境:内网 IP
network.host: 192.168.1.101

# http.port
# 作用:HTTP API 端口
# 默认:9200
http.port: 9200

# transport.port
# 作用:节点间通信端口
# 默认:9300
transport.port: 9300

# http.max_content_length
# 作用:HTTP 请求的最大大小
# 默认:100mb
# 建议:如果有大批量请求,可以增加
http.max_content_length: 200mb
```

**集群发现配置**:
```yaml
# discovery.seed_hosts
# 作用:集群中其他节点的地址列表
# 用途:新节点加入时,联系这些节点加入集群
discovery.seed_hosts:
  - 192.168.1.101:9300
  - 192.168.1.102:9300
  - 192.168.1.103:9300
  - es-node1  # 也可以使用主机名

# cluster.initial_master_nodes
# 作用:初次启动时的 Master 候选节点列表
# 注意:只在集群首次启动时有效,之后会被忽略
cluster.initial_master_nodes:
  - es-node1
  - es-node2
  - es-node3

# discovery.type
# 作用:发现类型
# 选项:multi-node(默认),single-node
# 用途:single-node 用于单节点开发环境
discovery.type: multi-node
```

#### 安全配置

**启用安全功能**:
```yaml
# xpack.security.enabled
# 作用:启用安全功能(认证、授权)
# ES 8.0+ 默认开启
xpack.security.enabled: true

# xpack.security.transport.ssl.enabled
# 作用:启用节点间通信加密
# 建议:生产环境必须开启
xpack.security.transport.ssl.enabled: true
xpack.security.transport.ssl.verification_mode: certificate
xpack.security.transport.ssl.keystore.path: certs/elastic-certificates.p12
xpack.security.transport.ssl.truststore.path: certs/elastic-certificates.p12

# xpack.security.http.ssl.enabled
# 作用:启用 HTTPS
# 建议:生产环境开启
xpack.security.http.ssl.enabled: true
xpack.security.http.ssl.keystore.path: certs/http.p12
```

**创建用户与角色**:
```bash
# 创建用户
PUT /_security/user/myuser
{
  "password" : "mypassword",
  "roles" : [ "myrole" ],
  "full_name" : "My User"
}

# 创建角色
PUT /_security/role/myrole
{
  "indices": [
    {
      "names": [ "myindex*" ],
      "privileges": [ "read", "write" ]
    }
  ]
}

# 修改密码
POST /_security/user/myuser/_password
{
  "password" : "newpassword"
}
```

### 3.3 核心功能的基础使用

#### 索引管理

**创建索引**:
```bash
# 创建索引(使用默认设置)
PUT /myindex

# 创建索引并指定分片和副本数
PUT /myindex
{
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1
  }
}

# 创建索引并定义 Mapping
PUT /users
{
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1
  },
  "mappings": {
    "properties": {
      "name": {
        "type": "text"
      },
      "age": {
        "type": "integer"
      },
      "email": {
        "type": "keyword"
      },
      "created_at": {
        "type": "date"
      }
    }
  }
}
```

**查看索引**:
```bash
# 查看所有索引
GET /_cat/indices?v

# 查看特定索引
GET /users

# 查看索引设置
GET /users/_settings

# 查看索引 Mapping
GET /users/_mapping
```

**修改索引设置**:
```bash
# 修改副本数(动态设置)
PUT /users/_settings
{
  "number_of_replicas": 2
}

# 修改 refresh_interval
PUT /users/_settings
{
  "refresh_interval": "30s"
}

# 注意:主分片数(number_of_shards)不能修改
# 如果需要修改,只能使用 Reindex API
```

**删除索引**:
```bash
# 删除单个索引
DELETE /myindex

# 删除多个索引
DELETE /index1,index2,index3

# 删除匹配的索引(危险!)
DELETE /logs-2024-*

# 防止误删,禁用通配符删除
PUT /_cluster/settings
{
  "persistent": {
    "action.destructive_requires_name": true
  }
}
```

#### 文档操作(CRUD)

**创建文档**:
```bash
# 指定文档 ID
PUT /users/_doc/1
{
  "name": "Alice",
  "age": 25,
  "email": "alice@example.com",
  "created_at": "2024-12-24T10:00:00Z"
}

# 自动生成文档 ID
POST /users/_doc
{
  "name": "Bob",
  "age": 30,
  "email": "bob@example.com"
}

# 只在文档不存在时创建(_create)
PUT /users/_create/1
{
  "name": "Alice",
  "age": 25
}
# 如果 ID=1 已存在,返回 409 Conflict
```

**读取文档**:
```bash
# 根据 ID 读取
GET /users/_doc/1

# 只获取 _source
GET /users/_source/1

# 获取特定字段
GET /users/_doc/1?_source=name,email

# 批量获取(mget)
GET /_mget
{
  "docs": [
    { "_index": "users", "_id": "1" },
    { "_index": "users", "_id": "2" }
  ]
}
```

**更新文档**:
```bash
# 全量更新(整个文档替换)
PUT /users/_doc/1
{
  "name": "Alice Updated",
  "age": 26,
  "email": "alice@example.com",
  "created_at": "2024-12-24T10:00:00Z"
}

# 部分更新(只更新指定字段)
POST /users/_update/1
{
  "doc": {
    "age": 26
  }
}

# 使用脚本更新
POST /users/_update/1
{
  "script": {
    "source": "ctx._source.age += params.count",
    "params": {
      "count": 1
    }
  }
}

# upsert(不存在则创建)
POST /users/_update/1
{
  "doc": {
    "age": 26
  },
  "doc_as_upsert": true
}
```

**删除文档**:
```bash
# 根据 ID 删除
DELETE /users/_doc/1

# 根据查询条件删除(delete by query)
POST /users/_delete_by_query
{
  "query": {
    "range": {
      "age": {
        "lt": 18
      }
    }
  }
}
```

**批量操作(Bulk)**:
```bash
# Bulk API:批量执行多个操作
POST /_bulk
{ "index": { "_index": "users", "_id": "1" }}
{ "name": "Alice", "age": 25 }
{ "create": { "_index": "users", "_id": "2" }}
{ "name": "Bob", "age": 30 }
{ "update": { "_index": "users", "_id": "1" }}
{ "doc": { "age": 26 }}
{ "delete": { "_index": "users", "_id": "3" }}

# 返回:
{
  "took": 30,
  "errors": false,
  "items": [
    { "index": { "_index": "users", "_id": "1", "result": "created", "status": 201 }},
    { "create": { "_index": "users", "_id": "2", "result": "created", "status": 201 }},
    { "update": { "_index": "users", "_id": "1", "result": "updated", "status": 200 }},
    { "delete": { "_index": "users", "_id": "3", "result": "not_found", "status": 404 }}
  ]
}

# 注意:
# - 每行都是一个完整的 JSON(不能换行美化)
# - 最后一行必须有换行符
# - 如果某个操作失败,不影响其他操作
```

#### 搜索查询

**基础查询**:
```bash
# 查询所有文档
GET /users/_search
{
  "query": {
    "match_all": {}
  }
}

# 全文搜索(match)
GET /users/_search
{
  "query": {
    "match": {
      "name": "Alice"
    }
  }
}

# 精确匹配(term)
GET /users/_search
{
  "query": {
    "term": {
      "email.keyword": "alice@example.com"
    }
  }
}

# 范围查询(range)
GET /users/_search
{
  "query": {
    "range": {
      "age": {
        "gte": 20,
        "lte": 30
      }
    }
  }
}

# 多条件组合(bool)
GET /users/_search
{
  "query": {
    "bool": {
      "must": [
        { "match": { "name": "Alice" }}
      ],
      "filter": [
        { "range": { "age": { "gte": 20 }}}
      ],
      "must_not": [
        { "term": { "status": "deleted" }}
      ]
    }
  }
}
```

**分页与排序**:
```bash
# 分页(from + size)
GET /users/_search
{
  "query": { "match_all": {} },
  "from": 0,
  "size": 10
}

# 排序
GET /users/_search
{
  "query": { "match_all": {} },
  "sort": [
    { "age": { "order": "desc" }},
    { "_score": { "order": "desc" }}
  ]
}

# 只返回特定字段
GET /users/_search
{
  "query": { "match_all": {} },
  "_source": ["name", "email"]
}
```

**聚合统计**:
```bash
# 按字段分组统计
GET /users/_search
{
  "size": 0,
  "aggs": {
    "age_groups": {
      "terms": {
        "field": "age"
      }
    }
  }
}

# 统计指标(平均值、最大值等)
GET /users/_search
{
  "size": 0,
  "aggs": {
    "avg_age": {
      "avg": {
        "field": "age"
      }
    },
    "max_age": {
      "max": {
        "field": "age"
      }
    }
  }
}

# 嵌套聚合
GET /users/_search
{
  "size": 0,
  "aggs": {
    "age_ranges": {
      "range": {
        "field": "age",
        "ranges": [
          { "to": 20 },
          { "from": 20, "to": 30 },
          { "from": 30 }
        ]
      },
      "aggs": {
        "avg_score": {
          "avg": {
            "field": "score"
          }
        }
      }
    }
  }
}
```

### 3.4 常用命令/API 详解

#### 集群API

```bash
# 集群健康状态
GET /_cluster/health

# 详细健康状态
GET /_cluster/health?level=indices

# 集群状态
GET /_cluster/state

# 集群统计
GET /_cluster/stats

# 节点信息
GET /_nodes

# 节点统计
GET /_nodes/stats

# 集群设置
GET /_cluster/settings

# 修改集群设置
PUT /_cluster/settings
{
  "persistent": {
    "indices.recovery.max_bytes_per_sec": "50mb"
  }
}
```

#### Cat API(表格输出)

```bash
# 查看所有索引
GET /_cat/indices?v

# 查看节点
GET /_cat/nodes?v

# 查看分片
GET /_cat/shards?v

# 查看未分配的分片
GET /_cat/shards?v&h=index,shard,prirep,state,unassigned.reason

# 查看线程池
GET /_cat/thread_pool?v

# 查看 Master 节点
GET /_cat/master?v
```

#### Index API

```bash
# 打开/关闭索引
POST /myindex/_close
POST /myindex/_open

# 刷新索引
POST /myindex/_refresh

# 刷盘
POST /myindex/_flush

# 强制合并(Force Merge)
POST /myindex/_forcemerge?max_num_segments=1

# 清空缓存
POST /myindex/_cache/clear

# 索引别名
POST /_aliases
{
  "actions": [
    { "add": { "index": "logs-2024-12-24", "alias": "logs-current" }},
    { "remove": { "index": "logs-2024-12-23", "alias": "logs-current" }}
  ]
}

# 查看别名
GET /_alias

# Reindex(重建索引)
POST /_reindex
{
  "source": {
    "index": "old_index"
  },
  "dest": {
    "index": "new_index"
  }
}
```

#### Snapshot API(备份恢复)

```bash
# 注册快照仓库
PUT /_snapshot/my_backup
{
  "type": "fs",
  "settings": {
    "location": "/mnt/backups/elasticsearch"
  }
}

# 创建快照
PUT /_snapshot/my_backup/snapshot_1
{
  "indices": "index1,index2",
  "ignore_unavailable": true,
  "include_global_state": false
}

# 查看快照
GET /_snapshot/my_backup/snapshot_1

# 恢复快照
POST /_snapshot/my_backup/snapshot_1/_restore
{
  "indices": "index1,index2"
}

# 删除快照
DELETE /_snapshot/my_backup/snapshot_1
```

### 3.5 新手必须掌握的基础概念

#### 概念1:Index vs Database

```
关系型数据库 → Elasticsearch
Database → Index
Table → (Type,已废弃)
Row → Document
Column → Field

示例:
MySQL: users 表在 mydb 数据库
Elasticsearch: users 索引

注意:
- ES 7.0+ 废弃了 Type 概念
- 一个索引只有一个 Type(_doc)
- 不同类型的数据应该放在不同索引
```

#### 概念2:Document 的结构

```json
{
  "_index": "users",        // 所属索引
  "_id": "1",               // 文档 ID
  "_version": 3,            // 版本号(每次更新+1)
  "_seq_no": 5,             // 序列号(用于并发控制)
  "_primary_term": 1,       // 主分片任期号
  "_source": {              // 原始文档内容
    "name": "Alice",
    "age": 25,
    "email": "alice@example.com"
  }
}

_source:
- 存储原始 JSON
- 查询时返回的内容
- 可以禁用以节省空间(但无法获取原始数据)
```

#### 概念3:Mapping 与数据类型

```
Mapping = Schema
- 定义文档结构
- 定义字段类型
- 定义分析器

常用数据类型:
1. text:全文搜索,会分词
2. keyword:精确匹配,不分词
3. long:长整型
4. integer:整型
5. short:短整型
6. byte:字节
7. double:双精度浮点数
8. float:浮点数
9. date:日期
10. boolean:布尔值
11. object:嵌套对象
12. nested:嵌套数组
13. geo_point:地理坐标
14. ip:IP地址

text vs keyword:
- text:
  - "Elasticsearch Tutorial" → ["elasticsearch", "tutorial"]
  - 用于全文搜索
  - 例:文章内容、商品描述

- keyword:
  - "Elasticsearch Tutorial" → "Elasticsearch Tutorial"
  - 用于精确匹配、排序、聚合
  - 例:标签、状态、邮箱、ID
```

#### 概念4:Analyzer(分析器)

```
分析器的作用:
- 将文本转换为词语(Token)
- 用于索引和搜索

分析器的组成:
1. Character Filter:字符过滤
   - 例:去除 HTML 标签

2. Tokenizer:分词
   - 例:按空格分词、按标点分词

3. Token Filter:词语过滤
   - 例:转小写、去停用词、提取词干

内置分析器:
1. standard:默认,按空格和标点分词
2. simple:按非字母分词,并转小写
3. whitespace:只按空格分词
4. stop:去除停用词
5. keyword:不分词,整个字符串作为一个词
6. pattern:正则表达式分词

中文分词器(需安装插件):
1. IK:
   - ik_max_word:最细粒度,召回率高
   - ik_smart:粗粒度,精确度高

2. jieba:结巴分词
3. THULAC:清华分词

测试分析器:
GET /_analyze
{
  "analyzer": "ik_max_word",
  "text": "我爱北京天安门"
}

返回:
{
  "tokens": [
    { "token": "我", "start_offset": 0, "end_offset": 1 },
    { "token": "爱", "start_offset": 1, "end_offset": 2 },
    { "token": "北京", "start_offset": 2, "end_offset": 4 },
    { "token": "北京天安门", "start_offset": 2, "end_offset": 7 },
    { "token": "天安门", "start_offset": 4, "end_offset": 7 },
    { "token": "天安", "start_offset": 4, "end_offset": 6 },
    { "token": "安门", "start_offset": 5, "end_offset": 7 }
  ]
}
```

#### 概念5:Query vs Filter

```
Query 上下文:
- 计算相关性得分(_score)
- 结果按得分排序
- 不能缓存
- 用于:"这个文档有多匹配?"

Filter 上下文:
- 只判断是否匹配(Yes/No)
- 不计算得分
- 可以缓存
- 用于:"这个文档是否匹配?"

示例:
GET /products/_search
{
  "query": {
    "bool": {
      "must": [
        { "match": { "title": "手机" }}   // Query:计算相关性
      ],
      "filter": [
        { "term": { "brand": "Apple" }},   // Filter:精确匹配,可缓存
        { "range": { "price": { "gte": 3000, "lte": 8000 }}}  // Filter
      ]
    }
  }
}

最佳实践:
- 全文搜索用 Query(match、multi_match)
- 精确匹配用 Filter(term、range、exists)
- Filter 会被缓存,性能更好
```

#### 概念6:相关性得分(_score)

```
_score 的计算(BM25 算法):
- 词频(TF):词在文档中出现的次数
- 逆文档频率(IDF):词在多少文档中出现
- 文档长度归一化:长文档不会占优势

影响得分的因素:
1. 词频:出现次数越多,得分越高
2. 稀有度:越稀有的词,得分越高
3. 字段长度:短字段中的匹配得分更高
4. 查询词覆盖度:匹配的查询词越多,得分越高

查看得分计算过程:
GET /users/_search
{
  "query": {
    "match": { "name": "Alice" }
  },
  "explain": true
}

返回:
{
  "_explanation": {
    "value": 0.2876821,
    "description": "weight(name:alice in 0) ...",
    "details": [...]
  }
}
```

### 3.6 最简单的"Hello World"级别示例

#### 示例1:完整的索引创建与搜索流程

```bash
# 1. 创建索引
PUT /blogs
{
  "mappings": {
    "properties": {
      "title": { "type": "text", "analyzer": "standard" },
      "content": { "type": "text", "analyzer": "standard" },
      "author": { "type": "keyword" },
      "publish_date": { "type": "date" },
      "tags": { "type": "keyword" }
    }
  }
}

# 2. 插入文档
POST /blogs/_doc/1
{
  "title": "Elasticsearch 入门教程",
  "content": "Elasticsearch 是一个分布式搜索和分析引擎,基于 Lucene 构建...",
  "author": "Alice",
  "publish_date": "2024-12-24",
  "tags": ["elasticsearch", "tutorial", "search"]
}

POST /blogs/_doc/2
{
  "title": "MySQL 性能优化",
  "content": "MySQL 是最流行的关系型数据库,性能优化包括索引优化、查询优化...",
  "author": "Bob",
  "publish_date": "2024-12-23",
  "tags": ["mysql", "database", "optimization"]
}

POST /blogs/_doc/3
{
  "title": "Elasticsearch 高级特性",
  "content": "Elasticsearch 的高级特性包括聚合、脚本、机器学习等...",
  "author": "Alice",
  "publish_date": "2024-12-22",
  "tags": ["elasticsearch", "advanced"]
}

# 3. 搜索
# 3.1 搜索标题包含"Elasticsearch"的文章
GET /blogs/_search
{
  "query": {
    "match": {
      "title": "Elasticsearch"
    }
  }
}
# 返回:文档1和3

# 3.2 搜索作者为"Alice"的文章
GET /blogs/_search
{
  "query": {
    "term": {
      "author": "Alice"
    }
  }
}
# 返回:文档1和3

# 3.3 搜索最近3天的文章
GET /blogs/_search
{
  "query": {
    "range": {
      "publish_date": {
        "gte": "now-3d/d"
      }
    }
  }
}
# 返回:所有文档

# 3.4 组合搜索:标题包含"Elasticsearch"且作者为"Alice"
GET /blogs/_search
{
  "query": {
    "bool": {
      "must": [
        { "match": { "title": "Elasticsearch" }}
      ],
      "filter": [
        { "term": { "author": "Alice" }}
      ]
    }
  }
}
# 返回:文档1和3

# 3.5 聚合:按作者统计文章数
GET /blogs/_search
{
  "size": 0,
  "aggs": {
    "authors": {
      "terms": {
        "field": "author"
      }
    }
  }
}
# 返回:Alice: 2篇,Bob: 1篇

# 3.6 高亮显示
GET /blogs/_search
{
  "query": {
    "match": { "title": "Elasticsearch" }
  },
  "highlight": {
    "fields": {
      "title": {}
    }
  }
}
# 返回:
{
  "hits": {
    "hits": [
      {
        "_source": { "title": "Elasticsearch 入门教程", ...},
        "highlight": {
          "title": ["<em>Elasticsearch</em> 入门教程"]
        }
      }
    ]
  }
}
```

#### 示例2:电商商品搜索

```bash
# 1. 创建商品索引
PUT /products
{
  "mappings": {
    "properties": {
      "name": { "type": "text", "analyzer": "ik_max_word" },
      "description": { "type": "text", "analyzer": "ik_max_word" },
      "brand": { "type": "keyword" },
      "category": { "type": "keyword" },
      "price": { "type": "double" },
      "sales": { "type": "long" },
      "rating": { "type": "float" },
      "stock": { "type": "integer" },
      "created_at": { "type": "date" }
    }
  }
}

# 2. 插入商品数据
POST /_bulk
{"index":{"_index":"products","_id":"1"}}
{"name":"Apple iPhone 15 Pro Max","description":"苹果最新旗舰手机","brand":"Apple","category":"手机","price":9999,"sales":1000,"rating":4.8,"stock":50,"created_at":"2024-12-01"}
{"index":{"_index":"products","_id":"2"}}
{"name":"华为 Mate 60 Pro","description":"华为5G旗舰手机","brand":"Huawei","category":"手机","price":7999,"sales":800,"rating":4.7,"stock":30,"created_at":"2024-11-20"}
{"index":{"_index":"products","_id":"3"}}
{"name":"小米14 Ultra","description":"小米影像旗舰","brand":"Xiaomi","category":"手机","price":5999,"sales":1200,"rating":4.6,"stock":100,"created_at":"2024-12-10"}

# 3. 搜索"苹果手机"
GET /products/_search
{
  "query": {
    "bool": {
      "must": [
        {
          "multi_match": {
            "query": "苹果手机",
            "fields": ["name^2", "description"]  // name 权重2倍
          }
        }
      ],
      "filter": [
        { "term": { "category": "手机" }},
        { "range": { "stock": { "gt": 0 }}}  // 有库存
      ]
    }
  },
  "sort": [
    { "_score": { "order": "desc" }},
    { "sales": { "order": "desc" }}
  ]
}

# 4. 聚合:价格区间分布
GET /products/_search
{
  "size": 0,
  "query": { "term": { "category": "手机" }},
  "aggs": {
    "price_ranges": {
      "range": {
        "field": "price",
        "ranges": [
          { "to": 3000, "key": "3000以下" },
          { "from": 3000, "to": 6000, "key": "3000-6000" },
          { "from": 6000, "to": 9000, "key": "6000-9000" },
          { "from": 9000, "key": "9000以上" }
        ]
      }
    },
    "brands": {
      "terms": {
        "field": "brand",
        "size": 10
      }
    }
  }
}
```

通过以上基础知识的学习,你已经掌握了 Elasticsearch 的核心概念和基本操作,可以开始构建简单的搜索应用了!接下来我们将深入学习更高级的功能和优化技巧。

## 四、核心功能详解

(由于篇幅限制,此处是文档的前三部分。完整文档将继续包含第四到第十三部分的详细内容,总计约2800-3000行,涵盖核心功能、应用场景、进阶技巧、高阶知识、生产环境实践、最佳实践、学习路线、学习资源、常见问题和实战练习等所有章节。)

---

*此文档将继续完善后续章节...*
