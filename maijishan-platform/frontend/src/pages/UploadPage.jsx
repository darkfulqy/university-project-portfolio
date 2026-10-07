import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { listDocuments, listRecentUploads, uploadFile } from "../api";
import { getToken } from "../api/client";

const ACCEPT_TEXT = "支持 PDF、CAJ、DOC、DOCX、TXT 与常见图片";
const RECENT_LIMIT = 6;
const RECENT_FETCH_SIZE = 120;

const fullTimeFormatter = new Intl.DateTimeFormat("zh-CN", {
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});

const dateBadgeFormatter = new Intl.DateTimeFormat("zh-CN", {
  month: "2-digit",
  day: "2-digit",
});

const hourBadgeFormatter = new Intl.DateTimeFormat("zh-CN", {
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});

function formatFileSize(size) {
  const num = Number(size || 0);
  if (!Number.isFinite(num) || num <= 0) return "--";
  if (num < 1024) return `${num} B`;
  if (num < 1024 * 1024) return `${(num / 1024).toFixed(1)} KB`;
  return `${(num / (1024 * 1024)).toFixed(1)} MB`;
}

function getTimestamp(value) {
  if (!value) return 0;
  const stamp = new Date(value).getTime();
  return Number.isFinite(stamp) ? stamp : 0;
}

function formatRecentTime(value) {
  if (!getTimestamp(value)) return "时间待同步";
  return fullTimeFormatter.format(new Date(value));
}

function formatRecentDateBadge(value) {
  if (!getTimestamp(value)) return "--.--";
  return dateBadgeFormatter.format(new Date(value)).replaceAll("/", ".");
}

function formatRecentHourBadge(value) {
  if (!getTimestamp(value)) return "--:--";
  return hourBadgeFormatter.format(new Date(value));
}

function normalizeFileType(value = "") {
  const token = String(value).trim().toLowerCase();
  if (!token) return "";
  const mapping = {
    pdf: "PDF",
    caj: "CAJ",
    doc: "DOC",
    docx: "DOCX",
    txt: "TXT",
    md: "MD",
    png: "PNG",
    jpg: "JPG",
    jpeg: "JPG",
    webp: "WEBP",
  };
  return mapping[token] || token.toUpperCase();
}

function inferFileTypeFromUrl(value = "") {
  const text = String(value || "").trim();
  const matched = text.match(/\.([a-z0-9]+)(?:$|[?#])/i);
  return matched ? normalizeFileType(matched[1]) : "";
}

function isLibraryDocument(item) {
  const numericId = Number(item?.id ?? item?.document_id);
  return Number.isFinite(numericId) || !!item?.title || !!item?.filename;
}

function resolveStatus(item) {
  const raw = String(item?.ingestion_status || item?.processing_status || "").toLowerCase();
  if (raw === "failed" || raw === "error") return { label: "处理失败", tone: "error" };
  if (raw === "pending" || raw === "processing" || raw === "queued") return { label: "解析中", tone: "pending" };
  if (raw === "done" || raw === "completed" || raw === "success") return { label: "已入库", tone: "ready" };
  if (isLibraryDocument(item)) return { label: "已入列", tone: "ready" };
  return { label: "已上传", tone: "queued" };
}

function normalizeRecentItem(item) {
  if (!item || !isLibraryDocument(item)) return null;
  const fileTypes = Array.isArray(item.available_file_types)
    ? item.available_file_types.map(normalizeFileType).filter(Boolean)
    : [];
  const status = resolveStatus(item);
  const createdAt = item.created_at || item.updated_at || null;
  const numericId = Number(item.id ?? item.document_id);
  const fallbackType =
    normalizeFileType(item.file_format) ||
    inferFileTypeFromUrl(item.download_url) ||
    inferFileTypeFromUrl(item.preview_url) ||
    inferFileTypeFromUrl(item.source_url) ||
    inferFileTypeFromUrl(Array.isArray(item.source_urls) ? item.source_urls[0] : "");
  const detailHref = Number.isFinite(numericId) ? `/detail/documents/${numericId}` : null;
  const downloadHref =
    item.download_url ||
    item.pdf_download_url ||
    item.caj_download_url ||
    item.preview_url ||
    null;
  const primaryFileType = fileTypes[0] || fallbackType;
  const sourceLabel = item.source_db || item.source_type || "站内入列";
  return {
    key: `recent:${item.document_id ?? item.id ?? item.filename ?? Math.random().toString(16).slice(2)}`,
    title: item.title || item.filename || "未命名文献",
    timeText: formatRecentTime(createdAt),
    dateBadge: formatRecentDateBadge(createdAt),
    hourBadge: formatRecentHourBadge(createdAt),
    fileTypeLabel: fileTypes.join(" / ") || primaryFileType || "文档",
    statusLabel: status.label,
    statusTone: status.tone,
    detailHref: item.detail_url || item.document_url || detailHref,
    downloadHref,
    downloadLabel: primaryFileType ? `下载 ${primaryFileType}` : "下载文件",
    createdAt,
    sourceLabel,
  };
}

function mergeRecentItems(...groups) {
  const map = new Map();
  groups.flat().forEach((item) => {
    if (!item) return;
    const existing = map.get(item.key);
    map.set(item.key, existing ? { ...existing, ...item } : item);
  });
  return [...map.values()]
    .sort((a, b) => getTimestamp(b.createdAt) - getTimestamp(a.createdAt))
    .slice(0, RECENT_LIMIT);
}

function createOptimisticItem(result, file) {
  if (!result?.document_id) return null;
  return normalizeRecentItem({
    document_id: result.document_id,
    title: result.document_title || String(file?.name || "").replace(/\.[^.]+$/, ""),
    file_format: String(file?.name || "").split(".").pop() || "",
    available_file_types: [String(file?.name || "").split(".").pop() || ""],
    detail_url: result.document_url,
    download_url: `/documents/${result.document_id}/file?download=1`,
    ingestion_status: result.ingestion_status || "pending",
    ingestion_error: result.ingestion_error || null,
    created_at: new Date().toISOString(),
  });
}

export default function UploadPage() {
  const hasToken = !!getToken();
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [recentItems, setRecentItems] = useState([]);
  const [optimisticItems, setOptimisticItems] = useState([]);
  const [recentLoading, setRecentLoading] = useState(true);
  const [recentError, setRecentError] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    let active = true;
    setRecentLoading(true);
    setRecentError("");
    const loadRecentItems = async () => {
      try {
        const res = await listRecentUploads({ limit: RECENT_LIMIT });
        return Array.isArray(res?.items) ? res.items.map(normalizeRecentItem).filter(Boolean) : [];
      } catch (uploadErr) {
        const res = await listDocuments({ page: 1, page_size: RECENT_FETCH_SIZE });
        if (Array.isArray(res?.items)) {
          return res.items.map(normalizeRecentItem).filter(Boolean);
        }
        throw uploadErr;
      }
    };

    loadRecentItems()
      .then((normalized) => {
        if (!active) return;
        setRecentItems(normalized);
      })
      .catch((err) => {
        if (!active) return;
        setRecentItems([]);
        setRecentError(err.message || "最新提交加载失败");
      })
      .finally(() => {
        if (active) setRecentLoading(false);
      });
    return () => {
      active = false;
    };
  }, [refreshKey]);

  const latestItems = useMemo(
    () => mergeRecentItems(optimisticItems, recentItems),
    [optimisticItems, recentItems],
  );

  const submit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("请选择要上传的文件");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await uploadFile(file);
      setResult(res);
      const optimistic = createOptimisticItem(res, file);
      if (optimistic) {
        setOptimisticItems((prev) => mergeRecentItems([optimistic], prev));
      }
      setRefreshKey((value) => value + 1);
    } catch (err) {
      setError(err.message || "上传失败");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="upload-wrap page-wide">
      <div className="upload-shell">
        <div className="upload-copy">
          <p className="section-kicker">文件上传</p>
          <h1>上传你的文献与研究附件</h1>
          <p className="upload-subtitle">
            上传后的文献会直接进入站内整理流程，并在下方的最新提交里保留一条清晰记录。
            适合快速补充个人收集的论文、扫描件和研究附件，再进入文献页继续查看。
          </p>
          <div className="upload-note-list">
            <span>登录后可用</span>
            <span>{ACCEPT_TEXT}</span>
            <span>上传完成后可直接进入文献详情</span>
            <span>下半部分会显示最近提交</span>
          </div>
        </div>

        <form className="upload-card" onSubmit={submit}>
          {!hasToken ? (
            <div className="upload-empty">
              <h2>需要先登录</h2>
              <p className="muted">当前未检测到登录状态，先登录后才能上传并进入文献解析流程。</p>
              <Link className="login-btn" to="/login">前往登录</Link>
            </div>
          ) : (
            <>
              <div className="upload-dropzone">
                <label className="upload-picker">
                  <span className="upload-picker-title">{file ? file.name : "选择文件"}</span>
                  <span className="upload-picker-hint">
                    {file ? `${file.type || "未知类型"} · ${formatFileSize(file.size)}` : ACCEPT_TEXT}
                  </span>
                  <input
                    type="file"
                    accept=".pdf,.caj,.doc,.docx,.txt,image/png,image/jpeg,image/webp"
                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                  />
                </label>
              </div>

              {error ? <p className="error-text">{error}</p> : null}

              <button type="submit" disabled={loading}>
                {loading ? "正在上传..." : "开始上传"}
              </button>

              {result ? (
                <div className="upload-result">
                  <div className="upload-result-head">
                    <strong>上传完成</strong>
                    <span>{result.filename}</span>
                  </div>
                  <div className="upload-result-grid">
                    <div className="meta-item"><span className="meta-label">文件大小：</span>{formatFileSize(result.size)}</div>
                    <div className="meta-item"><span className="meta-label">文件类型：</span>{result.mime_type}</div>
                    <div className="meta-item"><span className="meta-label">当前状态：</span>{resolveStatus(result).label}</div>
                  </div>
                  <div className="upload-result-actions">
                    {result.document_url ? <Link className="btn-small" to={result.document_url}>进入文献详情</Link> : null}
                    <a className="btn-small" href={result.url} target="_blank" rel="noreferrer">打开上传文件</a>
                  </div>
                  <p className="upload-result-tip">
                    {result.document_id
                      ? "这条文献已经进入下方“最新提交”，你可以继续进入详情页查看当前入列结果。"
                      : "当前文件已上传成功，但不会进入文献列表。"}
                  </p>
                </div>
              ) : null}
            </>
          )}
        </form>
      </div>

      <section className="upload-latest" aria-labelledby="upload-latest-title">
        <div className="upload-latest-head">
          <div>
            <p className="section-kicker">最新提交</p>
            <h2 id="upload-latest-title">最近入列的文献条目</h2>
            <p className="upload-latest-note">
              下方展示最近进入文献库的条目。新上传完成后会优先出现在这里，方便你直接继续进入文献详情或回到文献库检索。
            </p>
          </div>
          <div className="upload-latest-tools">
            <span className="upload-latest-count">
              {latestItems.length ? `展示最近 ${latestItems.length} 条` : "等待新的文献进入列表"}
            </span>
            <Link className="btn-small" to="/resources?type=documents&sort=latest">进入文献库</Link>
          </div>
        </div>

        {recentLoading ? <div className="upload-latest-refresh">正在同步最新提交...</div> : null}
        {recentError ? <div className="upload-latest-refresh upload-latest-refresh--error">{recentError}</div> : null}

        {!recentLoading && !latestItems.length ? (
          <div className="upload-latest-empty">
            <strong>还没有可展示的上传文献</strong>
            <p className="muted">上传 PDF 后，这里会显示最新提交，并给出详情与下载入口。</p>
          </div>
        ) : latestItems.length ? (
          <div className="upload-latest-list">
            {latestItems.map((item, index) => (
              <article className="upload-latest-item" key={item.key}>
                <div className="upload-latest-stamp">
                  <span className="upload-latest-index">{String(index + 1).padStart(2, "0")}</span>
                  <strong className="upload-latest-day">{item.dateBadge}</strong>
                  <span className="upload-latest-hour">{item.hourBadge}</span>
                </div>

                <div className="upload-latest-main">
                  <div className="upload-latest-row">
                    <h3>{item.title}</h3>
                    <span className={`upload-status upload-status--${item.statusTone}`}>{item.statusLabel}</span>
                  </div>
                  <div className="upload-latest-meta">
                    <span>{item.timeText}</span>
                    <span>{item.fileTypeLabel}</span>
                    <span>{item.sourceLabel}</span>
                  </div>
                </div>

                <div className="upload-latest-actions">
                  {item.detailHref ? (
                    <Link className="upload-latest-link" to={item.detailHref}>文献详情</Link>
                  ) : (
                    <span className="upload-latest-placeholder">详情待生成</span>
                  )}
                  {item.downloadHref ? (
                    <a className="upload-latest-link upload-latest-link--ghost" href={item.downloadHref} target="_blank" rel="noreferrer">
                      {item.downloadLabel}
                    </a>
                  ) : (
                    <span className="upload-latest-placeholder">暂无下载</span>
                  )}
                </div>
              </article>
            ))}
          </div>
        ) : null}
      </section>
    </section>
  );
}
