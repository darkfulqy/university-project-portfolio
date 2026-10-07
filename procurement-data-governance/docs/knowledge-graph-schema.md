# 标准知识图谱格式规范

## 1. 文档目的

本规范用于统一知识图谱中实体、关系、属性、数据来源、版本、质量校验与交付格式，保证知识图谱数据可理解、可维护、可扩展、可追溯，并便于后续在图数据库、RDF 三元组、检索增强生成、智能问答、数据分析等场景中复用。

## 2. 适用范围

本规范适用于以下对象：

- 领域知识图谱建设项目
- 企业业务知识图谱
- 采购、供应链、合同、供应商、物料、订单等业务图谱
- 面向图数据库或 RDF/OWL 的结构化知识建模
- 面向智能体、问答系统、搜索系统的数据底座建设

## 3. 基本术语

| 术语 | 含义 |
| --- | --- |
| 实体 Entity | 现实世界或业务系统中可被唯一识别的对象，如供应商、物料、合同、订单 |
| 实体类型 Entity Type | 对实体的类别划分，如 Supplier、Material、Contract |
| 关系 Relation | 两个实体之间的语义连接，如 “供应商-供应-物料” |
| 属性 Attribute | 描述实体或关系的字段，如名称、编码、金额、日期、状态 |
| 三元组 Triple | 知识图谱的基本表达形式，格式为 “头实体-关系-尾实体” |
| 本体 Ontology | 对实体类型、关系类型、属性、约束和层级结构的统一定义 |
| 命名空间 Namespace | 用于区分不同来源或领域对象的前缀标识 |
| 数据溯源 Provenance | 记录知识来源、抽取方式、更新时间、可信度等信息 |

## 4. 总体设计原则

知识图谱格式应遵循以下原则：

- 唯一性：每个实体必须拥有全局唯一标识。
- 一致性：同类实体、关系和属性采用统一字段结构和命名规则。
- 可追溯：每条知识应记录来源、生成方式、更新时间和可信度。
- 可扩展：新增实体类型、关系类型和属性时不破坏已有结构。
- 可校验：数据格式、字段类型、枚举值和必填项应可被程序自动校验。
- 可解释：实体、关系和属性命名应具备清晰业务含义。
- 最小冗余：相同知识只保留一份主数据，通过关系进行引用。

## 5. 知识图谱数据模型

### 5.1 标准三元组格式

知识图谱的基础结构为：

```text
(head_entity, relation, tail_entity)
```

标准字段如下：

| 字段名 | 类型 | 是否必填 | 说明 |
| --- | --- | --- | --- |
| triple_id | String | 是 | 三元组唯一标识 |
| head_id | String | 是 | 头实体唯一标识 |
| head_type | String | 是 | 头实体类型 |
| relation_type | String | 是 | 关系类型 |
| tail_id | String | 是 | 尾实体唯一标识 |
| tail_type | String | 是 | 尾实体类型 |
| properties | Object | 否 | 关系属性 |
| source | Object | 是 | 数据来源信息 |
| confidence | Float | 否 | 可信度，取值范围 0 到 1 |
| created_at | Datetime | 是 | 创建时间 |
| updated_at | Datetime | 是 | 更新时间 |

示例：

```json
{
  "triple_id": "TRI_SUPPLIER_MATERIAL_000001",
  "head_id": "SUP_000001",
  "head_type": "Supplier",
  "relation_type": "SUPPLIES",
  "tail_id": "MAT_000108",
  "tail_type": "Material",
  "properties": {
    "supply_level": "primary",
    "annual_volume": 120000,
    "currency": "CNY"
  },
  "source": {
    "source_system": "ERP",
    "source_table": "supplier_material_mapping",
    "source_record_id": "ERP_SM_202607210001",
    "extraction_method": "system_sync"
  },
  "confidence": 0.98,
  "created_at": "2026-07-21T10:00:00+08:00",
  "updated_at": "2026-07-21T10:00:00+08:00"
}
```

### 5.2 实体标准格式

实体用于表示业务对象。每个实体应包含唯一标识、类型、名称、属性和来源。

| 字段名 | 类型 | 是否必填 | 说明 |
| --- | --- | --- | --- |
| entity_id | String | 是 | 实体唯一标识 |
| entity_type | String | 是 | 实体类型 |
| entity_name | String | 是 | 实体标准名称 |
| aliases | Array | 否 | 实体别名 |
| properties | Object | 否 | 实体属性 |
| source | Object | 是 | 数据来源 |
| status | String | 是 | 实体状态，如 active、inactive、deleted |
| confidence | Float | 否 | 实体可信度 |
| created_at | Datetime | 是 | 创建时间 |
| updated_at | Datetime | 是 | 更新时间 |

示例：

```json
{
  "entity_id": "SUP_000001",
  "entity_type": "Supplier",
  "entity_name": "上海某某电子有限公司",
  "aliases": ["某某电子", "上海某某"],
  "properties": {
    "supplier_code": "S10001",
    "credit_code": "91310000XXXXXXXXXX",
    "country": "CN",
    "province": "上海市",
    "supplier_level": "A",
    "risk_level": "low"
  },
  "source": {
    "source_system": "SRM",
    "source_record_id": "SRM_SUP_10001",
    "extraction_method": "system_sync"
  },
  "status": "active",
  "confidence": 1.0,
  "created_at": "2026-07-21T10:00:00+08:00",
  "updated_at": "2026-07-21T10:00:00+08:00"
}
```

### 5.3 关系标准格式

关系用于表达两个实体之间的业务语义。关系方向必须明确。

| 字段名 | 类型 | 是否必填 | 说明 |
| --- | --- | --- | --- |
| relation_id | String | 是 | 关系唯一标识 |
| relation_type | String | 是 | 关系类型 |
| relation_name | String | 是 | 关系中文名称 |
| domain | String | 是 | 起点实体类型 |
| range | String | 是 | 终点实体类型 |
| direction | String | 是 | 关系方向，建议 fixed |
| properties_schema | Object | 否 | 关系属性结构 |
| description | String | 否 | 业务说明 |

示例：

```json
{
  "relation_id": "REL_SUPPLIES",
  "relation_type": "SUPPLIES",
  "relation_name": "供应",
  "domain": "Supplier",
  "range": "Material",
  "direction": "fixed",
  "properties_schema": {
    "supply_level": "String",
    "annual_volume": "Number",
    "currency": "String"
  },
  "description": "表示供应商向企业供应某类物料"
}
```

## 6. 命名规范

### 6.1 实体类型命名

实体类型使用英文 PascalCase 命名。

正确示例：

```text
Supplier
Material
PurchaseOrder
Contract
PurchaseRequisition
```

不推荐示例：

```text
supplier
purchase_order
采购订单
PO
```

### 6.2 关系类型命名

关系类型使用英文大写蛇形命名法。

正确示例：

```text
SUPPLIES
BELONGS_TO
SIGNS_CONTRACT
CREATES_ORDER
HAS_RISK
```

关系命名应优先使用动词或动宾结构，避免含义模糊的名称。

### 6.3 属性命名

属性使用英文小写蛇形命名法。

正确示例：

```text
supplier_code
contract_amount
created_date
risk_level
payment_terms
```

### 6.4 编码规则

实体 ID 建议采用如下格式：

```text
{TYPE_PREFIX}_{SEQUENCE}
```

示例：

| 实体类型 | 前缀 | 示例 |
| --- | --- | --- |
| Supplier | SUP | SUP_000001 |
| Material | MAT | MAT_000001 |
| Contract | CON | CON_000001 |
| PurchaseOrder | PO | PO_000001 |
| Organization | ORG | ORG_000001 |

三元组 ID 建议采用如下格式：

```text
TRI_{RELATION_TYPE}_{SEQUENCE}
```

示例：

```text
TRI_SUPPLIES_000001
TRI_SIGNS_CONTRACT_000001
```

## 7. 数据类型规范

| 类型 | 格式 | 示例 |
| --- | --- | --- |
| String | 字符串 | "Supplier" |
| Number | 整数或小数 | 120000 |
| Boolean | true 或 false | true |
| Date | YYYY-MM-DD | "2026-07-21" |
| Datetime | ISO 8601 | "2026-07-21T10:00:00+08:00" |
| Array | 数组 | ["A", "B"] |
| Object | JSON 对象 | {"key": "value"} |

金额字段应同时记录金额与币种：

```json
{
  "amount": 1000000,
  "currency": "CNY"
}
```

日期时间统一使用 ISO 8601 格式，并明确时区。

## 8. 本体 Schema 规范

本体 Schema 应至少包含实体类型定义、关系类型定义、属性定义和约束定义。

### 8.1 实体类型定义

```json
{
  "entity_type": "Supplier",
  "entity_name": "供应商",
  "description": "向企业提供物料、服务或工程的外部组织",
  "required_properties": [
    "supplier_code",
    "supplier_name"
  ],
  "optional_properties": [
    "credit_code",
    "country",
    "province",
    "supplier_level",
    "risk_level"
  ],
  "unique_keys": [
    "supplier_code",
    "credit_code"
  ]
}
```

### 8.2 关系类型定义

```json
{
  "relation_type": "SIGNS_CONTRACT",
  "relation_name": "签署合同",
  "domain": "Supplier",
  "range": "Contract",
  "cardinality": "many_to_many",
  "required_properties": [
    "signed_date"
  ],
  "optional_properties": [
    "contract_role"
  ]
}
```

### 8.3 约束定义

常见约束包括：

- 主键约束：entity_id、triple_id 不允许为空且全局唯一。
- 类型约束：字段值必须符合定义的数据类型。
- 枚举约束：状态、等级、风险等字段必须取自标准枚举。
- 范围约束：confidence 取值范围为 0 到 1。
- 关系约束：关系两端实体类型必须符合 domain 和 range 定义。
- 时间约束：updated_at 不得早于 created_at。

## 9. 数据来源与溯源规范

每条实体和关系均应记录来源信息。

| 字段名 | 类型 | 是否必填 | 说明 |
| --- | --- | --- | --- |
| source_system | String | 是 | 来源系统，如 ERP、SRM、OA、CRM |
| source_table | String | 否 | 来源表名 |
| source_document | String | 否 | 来源文档名称 |
| source_record_id | String | 否 | 来源记录编号 |
| extraction_method | String | 是 | 抽取方式 |
| extractor | String | 否 | 抽取程序或人员 |
| extracted_at | Datetime | 否 | 抽取时间 |

抽取方式建议枚举：

```text
system_sync
manual_entry
rule_extraction
model_extraction
api_import
document_parsing
```

## 10. 可信度规范

可信度字段 confidence 取值范围为 0 到 1。

| 区间 | 含义 | 建议处理 |
| --- | --- | --- |
| 0.90 - 1.00 | 高可信 | 可直接入库 |
| 0.70 - 0.89 | 中可信 | 可入库并标记待复核 |
| 0.50 - 0.69 | 低可信 | 需人工复核 |
| 0.00 - 0.49 | 不可信 | 不建议入库 |

不同来源的默认可信度建议：

| 来源方式 | 默认可信度 |
| --- | --- |
| 主数据系统同步 | 1.00 |
| 业务系统接口导入 | 0.95 |
| 规则抽取 | 0.85 |
| 大模型抽取 | 0.70 |
| 人工录入 | 0.90 |

## 11. 状态与版本规范

### 11.1 状态字段

实体和关系状态建议使用以下枚举：

```text
active
inactive
deleted
pending_review
rejected
```

### 11.2 版本字段

对于重要实体和关系，建议增加版本字段：

| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| version | String | 当前版本号 |
| valid_from | Datetime | 生效开始时间 |
| valid_to | Datetime | 生效结束时间 |
| is_current | Boolean | 是否为当前有效版本 |

示例：

```json
{
  "version": "v1.0",
  "valid_from": "2026-07-21T00:00:00+08:00",
  "valid_to": null,
  "is_current": true
}
```

## 12. 文件交付格式

### 12.1 推荐目录结构

```text
knowledge_graph/
  ontology/
    entity_schema.json
    relation_schema.json
    property_schema.json
  data/
    entities.jsonl
    triples.jsonl
  dictionary/
    entity_type_dictionary.xlsx
    relation_type_dictionary.xlsx
    property_dictionary.xlsx
  quality/
    validation_report.xlsx
    duplicate_check_report.xlsx
  README.md
```

### 12.2 JSONL 实体文件

每行表示一个实体。

```json
{"entity_id":"SUP_000001","entity_type":"Supplier","entity_name":"上海某某电子有限公司","properties":{"supplier_code":"S10001"},"status":"active"}
{"entity_id":"MAT_000108","entity_type":"Material","entity_name":"连接器 A 型","properties":{"material_code":"M00108"},"status":"active"}
```

### 12.3 JSONL 三元组文件

每行表示一条三元组。

```json
{"triple_id":"TRI_SUPPLIES_000001","head_id":"SUP_000001","head_type":"Supplier","relation_type":"SUPPLIES","tail_id":"MAT_000108","tail_type":"Material","confidence":0.98}
```

### 12.4 CSV 三元组文件

如需兼容表格工具，可使用 CSV：

```csv
triple_id,head_id,head_type,relation_type,tail_id,tail_type,confidence,source_system
TRI_SUPPLIES_000001,SUP_000001,Supplier,SUPPLIES,MAT_000108,Material,0.98,ERP
```

## 13. 采购领域推荐实体类型

| 实体类型 | 中文名称 | 说明 |
| --- | --- | --- |
| Supplier | 供应商 | 提供物料或服务的企业 |
| Material | 物料 | 被采购或使用的物料 |
| Category | 品类 | 采购品类或物料分类 |
| Contract | 合同 | 采购合同或框架协议 |
| PurchaseOrder | 采购订单 | 采购执行订单 |
| PurchaseRequisition | 采购申请 | 内部采购需求申请 |
| Organization | 组织 | 公司、部门、工厂等组织单元 |
| Employee | 员工 | 采购员、申请人、审批人等 |
| Project | 项目 | 采购关联项目 |
| RiskEvent | 风险事件 | 供应、质量、合规等风险 |
| Invoice | 发票 | 采购结算票据 |
| Payment | 付款 | 支付与结算记录 |

## 14. 采购领域推荐关系类型

| 关系类型 | 中文名称 | 头实体 | 尾实体 |
| --- | --- | --- | --- |
| SUPPLIES | 供应 | Supplier | Material |
| BELONGS_TO_CATEGORY | 归属品类 | Material | Category |
| SIGNS_CONTRACT | 签署合同 | Supplier | Contract |
| CONTRACT_COVERS_MATERIAL | 合同覆盖物料 | Contract | Material |
| CREATES_ORDER | 创建订单 | Employee | PurchaseOrder |
| ORDER_CONTAINS_MATERIAL | 订单包含物料 | PurchaseOrder | Material |
| ORDER_FROM_SUPPLIER | 订单来自供应商 | PurchaseOrder | Supplier |
| REQUISITION_GENERATES_ORDER | 申请生成订单 | PurchaseRequisition | PurchaseOrder |
| ORGANIZATION_OWNS_ORDER | 组织拥有订单 | Organization | PurchaseOrder |
| SUPPLIER_HAS_RISK | 供应商存在风险 | Supplier | RiskEvent |
| ORDER_HAS_INVOICE | 订单关联发票 | PurchaseOrder | Invoice |
| INVOICE_HAS_PAYMENT | 发票关联付款 | Invoice | Payment |

## 15. 数据质量校验规则

知识图谱入库前应至少完成以下校验：

| 校验项 | 校验规则 |
| --- | --- |
| 必填校验 | entity_id、entity_type、entity_name、triple_id、head_id、relation_type、tail_id 等字段不能为空 |
| 唯一性校验 | entity_id、triple_id 不允许重复 |
| 类型校验 | 字段值必须符合 Schema 定义的数据类型 |
| 关系端点校验 | 三元组中的 head_id、tail_id 必须存在于实体表 |
| 关系类型校验 | relation_type 必须存在于关系字典 |
| 关系方向校验 | head_type 和 tail_type 必须符合 domain/range 约束 |
| 枚举值校验 | status、risk_level、supplier_level 等字段必须在枚举范围内 |
| 时间校验 | updated_at 不得早于 created_at |
| 可信度校验 | confidence 必须在 0 到 1 之间 |
| 重复实体校验 | 基于名称、编码、统一社会信用代码等字段识别疑似重复实体 |

## 16. 入库标准

满足以下条件的数据可进入正式知识图谱：

- 实体与关系结构符合本规范。
- 必填字段完整。
- 主键唯一且无冲突。
- 关系两端实体均存在。
- 关系类型符合本体定义。
- 数据来源清晰可追溯。
- 可信度达到入库阈值。
- 通过自动化质量校验。

建议入库阈值：

```text
confidence >= 0.70
```

低于阈值的数据应进入待复核区，不进入正式图谱。

## 17. 示例：采购知识图谱完整样例

### 17.1 实体样例

```json
{
  "entity_id": "CON_000001",
  "entity_type": "Contract",
  "entity_name": "2026 年电子元器件框架采购合同",
  "properties": {
    "contract_code": "C20260001",
    "contract_amount": 5000000,
    "currency": "CNY",
    "start_date": "2026-01-01",
    "end_date": "2026-12-31",
    "contract_status": "active"
  },
  "source": {
    "source_system": "ContractManagement",
    "source_record_id": "CM_000001",
    "extraction_method": "system_sync"
  },
  "status": "active",
  "confidence": 1.0,
  "created_at": "2026-07-21T10:00:00+08:00",
  "updated_at": "2026-07-21T10:00:00+08:00"
}
```

### 17.2 三元组样例

```json
{
  "triple_id": "TRI_SIGNS_CONTRACT_000001",
  "head_id": "SUP_000001",
  "head_type": "Supplier",
  "relation_type": "SIGNS_CONTRACT",
  "tail_id": "CON_000001",
  "tail_type": "Contract",
  "properties": {
    "signed_date": "2026-01-01",
    "contract_role": "seller"
  },
  "source": {
    "source_system": "ContractManagement",
    "source_record_id": "CM_REL_000001",
    "extraction_method": "system_sync"
  },
  "confidence": 1.0,
  "created_at": "2026-07-21T10:00:00+08:00",
  "updated_at": "2026-07-21T10:00:00+08:00"
}
```

## 18. 推荐交付物清单

知识图谱项目建议最终交付以下文件：

- 知识图谱格式规范文档
- 本体设计说明
- 实体类型字典
- 关系类型字典
- 属性字段字典
- 实体数据文件
- 三元组数据文件
- 数据来源说明
- 数据质量校验报告
- 入库脚本或导入说明
- 图谱可视化截图或演示页面

## 19. 附录：最小可用字段集

如项目规模较小，可采用以下最小字段集。

实体最小字段：

```json
{
  "entity_id": "SUP_000001",
  "entity_type": "Supplier",
  "entity_name": "上海某某电子有限公司",
  "properties": {},
  "source": {},
  "status": "active"
}
```

三元组最小字段：

```json
{
  "triple_id": "TRI_SUPPLIES_000001",
  "head_id": "SUP_000001",
  "relation_type": "SUPPLIES",
  "tail_id": "MAT_000108",
  "confidence": 0.98
}
```

## 20. 版本记录

| 版本 | 日期 | 说明 |
| --- | --- | --- |
| v1.0 | 2026-07-21 | 初版，定义知识图谱通用格式、采购领域实体关系与质量校验规则 |
