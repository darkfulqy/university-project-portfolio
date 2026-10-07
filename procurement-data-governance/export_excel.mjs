import fs from "node:fs/promises";
import path from "node:path";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const [input, output] = process.argv.slice(2);
if (!input || !output) throw new Error("Usage: node export_excel.mjs dataset.json output.xlsx");
const data = JSON.parse(await fs.readFile(input, "utf8"));
const wb = Workbook.create();
const directory = path.dirname(output);
await fs.mkdir(directory, { recursive: true });

const clean = value => typeof value === "string" ? value.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f]/g, "").replace(/\r\n?/g, "\n") : value ?? null;
const literal = value => typeof value === "string" && value.startsWith("=") ? "'" + value : value;
const display = value => value === null ? "null" : typeof value === "object" ? JSON.stringify(value) : String(value);
function chunks(text, size = 160) {
  const chars = Array.from(clean(text));
  return chars.length ? Array.from({ length: Math.ceil(chars.length / size) }, (_, i) => chars.slice(i * size, (i + 1) * size).join("")) : [""];
}
function* flatten(value, prefix = "") {
  if (value && typeof value === "object" && Object.keys(value).length) {
    for (const [key, child] of Object.entries(value)) yield* flatten(child, Array.isArray(value) ? `${prefix}[${key}]` : prefix ? `${prefix}.${key}` : key);
  } else yield [prefix, value];
}
function column(n) {
  let result = "";
  for (n++; n; n = Math.floor((n - 1) / 26)) result = String.fromCharCode(65 + (n - 1) % 26) + result;
  return result;
}
function typed(value, heading) {
  if (value === null || value === undefined || value === "") return null;
  if (data.date_columns.includes(heading) && /^20\d{2}-\d{2}-\d{2}/.test(value)) {
    // Wall-clock times on the source sites are all Beijing time. Avoid host TZ conversion.
    const date = new Date(value.replace(" ", "T") + (value.length === 10 ? "T00:00:00Z" : "Z"));
    if (!Number.isNaN(date.getTime())) return date;
  }
  return literal(clean(value));
}
const longFields = [];
const rawFields = [];
const bodies = [];
const items = [];
const attachments = [];
const summary = [];

for (const notice of data.notices) {
  const r = notice.record;
  summary.push(data.columns.map(key => {
    const value = r[key];
    if (typeof value === "string" && (value.length > 120 || value.includes("\n")) && !/URL|公告标题|项目名称/.test(key)) {
      chunks(value).forEach((part, i) => longFields.push([r.公告ID, key, i + 1, part, r.来源URL]));
      return value.slice(0, 60).replace(/\n/g, " ") + " [完整值见字段全文]";
    }
    return typed(value, key);
  }));
  chunks(notice.text).forEach((part, i) => bodies.push([r.公告ID, i + 1, part, r.来源URL]));
  for (const [key, value] of flatten(notice.raw)) {
    chunks(display(value)).forEach((part, i) => rawFields.push([r.公告ID, key, value === null ? "null" : Array.isArray(value) ? "array" : typeof value, i + 1, part]));
  }
  for (const [i, item] of notice.items.entries()) {
    items.push([r.公告ID, i + 1, item.deviceName ?? null, item.deviceBrand ?? null, item.deviceModel ?? null,
      item.orderNum ?? null, item.unit ?? null, item.detailBudget ?? null,
      item.deviceSpec?.length > 180 ? item.deviceSpec.slice(0, 90) + " [完整值见原始字段]" : item.deviceSpec ?? null,
      item.afterService?.length > 180 ? item.afterService.slice(0, 90) + " [完整值见原始字段]" : item.afterService ?? null, r.来源URL]);
  }
  for (const item of notice.attachments) attachments.push([r.公告ID, item.name, item.status, item.bytes ?? null, item.url, item.file ?? null, item.sha256 ?? null]);
}

const sheets = [];
function table(name, headings, rows, widths, tableName) {
  const sheet = wb.worksheets.add(name);
  sheet.showGridLines = false;
  const count = Math.max(1, rows.length + 1);
  const matrix = [headings, ...rows].map(row => row.map(value => value instanceof Date ? value : literal(clean(value))));
  sheet.getRangeByIndexes(0, 0, count, headings.length).values = matrix;
  const range = sheet.getRangeByIndexes(0, 0, count, headings.length);
  range.format.font = { name: "Arial", size: 10, color: "#222222" };
  range.format.verticalAlignment = "top";
  range.format.wrapText = true;
  for (let i = 0; i < headings.length; i++) {
    const col = sheet.getRangeByIndexes(0, i, count, 1);
    col.format.columnWidth = widths[i] ?? 24;
    if (data.date_columns.includes(headings[i])) col.setNumberFormat("yyyy-mm-dd hh:mm");
    if (data.number_columns.includes(headings[i])) col.setNumberFormat(headings[i].endsWith("_元") ? "#,##0.00" : "#,##0");
  }
  if (rows.length) {
    const t = sheet.tables.add(`A1:${column(headings.length - 1)}${count}`, true, tableName);
    t.style = "TableStyleLight1";
    t.showFilterButton = true;
  }
  const header = sheet.getRangeByIndexes(0, 0, 1, headings.length);
  header.format.fill = "#304C49";
  header.format.font = { name: "Arial", size: 10, bold: true, color: "#FFFFFF" };
  header.format.rowHeight = 32;
  header.format.verticalAlignment = "center";
  header.format.horizontalAlignment = "center";
  header.format.borders = { insideVertical: { style: "thin", color: "#FFFFFF" } };
  for (let row = 0; row < rows.length; row++) {
    const lines = rows[row].reduce((max, v, c) => {
      if (typeof v !== "string") return max;
      const estimated = v.split("\n").reduce((sum, s) => sum + Math.max(1, Math.ceil(Array.from(s).reduce((n, x) => n + (x.charCodeAt(0) > 255 ? 2 : 1), 0) / ((widths[c] ?? 24) - 2))), 0);
      return Math.max(max, estimated);
    }, 1);
    sheet.getRangeByIndexes(row + 1, 0, 1, headings.length).format.rowHeight = Math.min(400, Math.max(34, lines * 15 + 8));
  }
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(name === "公告汇总" ? 2 : 1);
  sheets.push(sheet);
  return sheet;
}

const widths = data.columns.map(key => {
  if (key === "公告标题" || key === "项目名称") return 56;
  if (key === "公告ID") return 24;
  if (/URL|原始文件/.test(key)) return 62;
  if (/需求|资格|政策|事宜|期限|备注|方式|地点|地址/.test(key)) return 44;
  if (data.date_columns.includes(key)) return 23;
  if (data.number_columns.includes(key)) return 18;
  if (/联系人|联系方式|联系电话|编号|来源网站|名称/.test(key)) return 32;
  return 23;
});
if (data.bulk) {
  const records = new Map(data.notices.map(n => [n.record.公告ID, n.record]));
  const documentRows = (data.documents ?? []).map(d => {
    const r = records.get(d.notice_id) ?? {};
    return [d.document_id, d.notice_id, r.项目名称 ?? r.公告标题, d.name, d.category, d.format, d.pages,
      r.项目编号, r.来源网站, typed(r.发布时间, "发布时间"), r.预算金额_元, r.采购人名称,
      d.file, d.url, d.sha256, d.parent_id || null, d.archive_member || null, d.project_code_check,
      d.evidence, d.inspection_error || null, d.bytes, r.来源URL];
  });
  table("标书文件索引", ["文件ID", "公告ID", "项目名称", "文件名", "文件类别", "格式", "PDF页数", "项目编号", "来源网站", "发布时间", "预算金额_元", "采购人名称", "本地文件", "原始下载URL", "SHA256", "父压缩包文件ID", "压缩包内路径", "项目编号核验", "分类依据", "检查备注", "文件字节数", "公告URL"],
    documentRows, [26, 24, 52, 58, 26, 12, 12, 32, 30, 23, 18, 38, 72, 72, 68, 28, 62, 28, 52, 45, 18, 72], "TenderDocuments");
}
const main = table("公告汇总", data.columns, summary, widths, "NoticesTable");
main.tabColor = "#304C49";
const statusCol = column(data.columns.indexOf("采集状态"));
if (summary.length) main.getRange(`${statusCol}2:${statusCol}${summary.length + 1}`).conditionalFormats.add("containsText", { text: "仅列表", format: { fill: "#FFF1CC", font: { color: "#8B5200" } } });
table("采购明细", ["公告ID", "明细序号", "标的名称", "品牌", "型号", "数量", "单位", "明细预算_元", "技术参数", "售后服务", "来源URL"], items, [24, 12, 40, 20, 30, 12, 12, 18, 65, 50, 65], "ItemsTable");
table("附件清单", ["公告ID", "附件名称", "下载状态", "文件字节数", "附件URL", "本地文件", "SHA256"], attachments, [24, 56, 32, 18, 75, 75, 70], "AttachmentsTable");
table("公告正文", ["公告ID", "段序号", "正文片段", "来源URL"], bodies, [24, 12, 100, 70], "BodiesTable");
table("字段全文", ["公告ID", "字段名称", "片段序号", "字段完整值片段", "来源URL"], longFields, [24, 24, 12, 100, 70], "LongFieldsTable");
table("原始字段", ["公告ID", "原始字段路径", "数据类型", "片段序号", "原始字段值片段"], rawFields, [24, 64, 14, 12, 100], "RawFieldsTable");
const statusHeaders = ["省份", "网站", "公告数", "已取得正文数", "状态", "说明", "入口URL", "采集时间", "站点代码"];
table("站点状态", statusHeaders, data.sites.map(s => statusHeaders.map(h => typed(s[h], h))), [22, 44, 12, 18, 22, 95, 70, 23, 24], "SitesTable");
const explanation = [
  ["空值", "源站未公开、字段不适用或规则未识别时留空；不将未知金额写成0。完整正文及接口字段可用于复核。"],
  ["任务省份与项目地区", "任务省份按李卿阳的网站分工填写；全国性平台的项目可能来自其他省份。地区代码单独保留。"],
  ["金额", "标准化金额以人民币元计，保留原文；存在多个分包金额或单位不明确时不强行换算。"],
  ["日期", "采用源站北京时间。只有日期时，Excel中的00:00不表示源站公布了准确时刻。"],
  ["长字段", "汇总表中较长或包含换行的字段显示摘要，完整内容按公告ID、字段名称、片段序号存于字段全文。"],
  ["正文", "公告正文按公告ID和段序号拼接可还原采集文本；正文不包含脚本和样式。"],
  ["原始字段", "包含全部接口字段、null、空数组、网页元信息、公告表格及分节内容。长值按片段序号直接拼接。Excel统一换行符，原始JSON保持不变。"],
  ["采购明细", "接口提供结构化商品时按商品逐行列出。网页表格保留在原始字段的html_tables路径，不推测未标明的列含义。"],
  ["附件", data.bulk ? "记录已发现的公开附件。原文件与压缩包成员按公告ID关联，文件类别区分完整标书、候选、公告和其他附件。" : "记录全部已发现的附件链接，每站限量下载验证。网站登录限制或下载失败会记录在下载状态。"],
  ["站点状态", "17个分工入口逐一记录。入口访问成功不等于取得公告，未适配或访问失败的站点不会生成虚构记录。"],
  ["数据来源方式", "本次在线采集和历史本地样本分别标识；历史样本保留原采集时间及原始来源。"],
  ["采集范围", data.bulk ? `不限年份。本卷为数据库导出快照，生成时间：${data.generated_at}。进度以数据库及progress.json为准。` : "每站默认2条，仅用于采集脚本验证，不表示全量或均匀抽样。"],
];
if (data.bulk) explanation.push(["标书识别", "完整标书为标题和目录结构的规则识别结果；原件是否缺页需人工复核。PDF检查文本取前8页，扫描件暂不做OCR，Word原格式保留。"],
  ["压缩包", "ZIP自动展开；RAR/7Z使用本机libarchive/bsdtar展开，失败或加密时保留原包并说明。文件索引的父文件ID关联来源压缩包。"]);
table("字段说明", ["主题", "说明"], explanation, [30, 110], "DictionaryTable");

wb.recalculate();
console.log((await wb.inspect({ kind: "table", range: "公告汇总!A1:M3", include: "values", tableMaxRows: 3, tableMaxCols: 13, maxChars: 1800 })).ndjson);
console.log((await wb.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!", options: { useRegex: true, maxResults: 10 }, maxChars: 800 })).ndjson);
const previews = path.join(directory, "previews");
await fs.mkdir(previews, { recursive: true });
for (const sheet of sheets) {
  const last = sheet.name === "标书文件索引" ? "G" : sheet.name === "公告汇总" ? "G" : sheet.name === "采购明细" ? "H" : sheet.name === "站点状态" ? "F" : sheet.name === "字段说明" ? "B" : sheet.name === "公告正文" ? "C" : sheet.name === "附件清单" ? "D" : sheet.name === "字段全文" ? "D" : "E";
  const preview = await wb.render({ sheetName: sheet.name, range: `A1:${last}${sheet.name === "字段说明" ? 6 : 4}`, scale: 1.5, format: "png" });
  await fs.writeFile(path.join(previews, sheet.name + ".png"), new Uint8Array(await preview.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(wb)).save(output);
console.log(`Excel: ${output}\n${data.notices.length} notices; ${rawFields.length} raw-field rows; ${attachments.length} attachments`);
