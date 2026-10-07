import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { getOverviewStats, getPostImages, getPostPreview, listDocuments, listPosts } from "../api";
import Card from "../components/Card";

function Pager({ page, total, pageSize, onChange }) {
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  return (
    <div className="pager">
      <button type="button" disabled={page <= 1} onClick={() => onChange(page - 1)}>上一页</button>
      <span>{page} / {totalPages}</span>
      <button type="button" disabled={page >= totalPages} onClick={() => onChange(page + 1)}>下一页</button>
    </div>
  );
}

function DocumentActions({ item }) {
  if (!item?.pdf_download_url && !item?.caj_download_url) return null;
  return (
    <div className="doc-actions-inline">
      {item.pdf_download_url ? <a className="btn-small" href={item.pdf_download_url} target="_blank" rel="noreferrer">PDF 下载</a> : null}
      {item.caj_download_url ? <a className="btn-small" href={item.caj_download_url} target="_blank" rel="noreferrer">CAJ 下载</a> : null}
    </div>
  );
}

function formatCompactNumber(value) {
  const num = Number(value || 0);
  if (!Number.isFinite(num)) return "--";
  return new Intl.NumberFormat("zh-CN").format(num);
}

export default function ResourcesPage() {
  const [params, setParams] = useSearchParams();
  const type = params.get("type") === "documents" ? "documents" : "posts";
  const page = Number(params.get("page") || 1);
  const keyword = params.get("keyword") || "";
  const sort = params.get("sort") || "latest";

  const [input, setInput] = useState(keyword);
  const [state, setState] = useState({ items: [], total: 0, page_size: 24 });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [overview, setOverview] = useState(null);
  const [postPreviewMap, setPostPreviewMap] = useState({});
  const [postImagesMap, setPostImagesMap] = useState({});

  useEffect(() => setInput(keyword), [keyword]);

  useEffect(() => {
    let active = true;
    getOverviewStats().then((res) => {
      if (active) setOverview(res);
    }).catch(() => {});
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    const fn = type === "documents" ? listDocuments : listPosts;
    fn({ keyword, page, page_size: 24 })
      .then((res) => {
        if (!active) return;
        setState(res);
      })
      .catch((e) => {
        if (!active) return;
        setError(e.message || "加载失败");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => { active = false; };
  }, [type, page, keyword]);

  useEffect(() => {
    if (type !== "posts" || !state.items.length) {
      setPostPreviewMap({});
      setPostImagesMap({});
      return;
    }
    let active = true;
    state.items.forEach((item) => {
      if (!item?.id) return;
      getPostPreview(item.id).then((res) => {
        if (active && res) setPostPreviewMap((prev) => ({ ...prev, [item.id]: res }));
      }).catch(() => {});
      getPostImages(item.id, { limit: 3 }).then((res) => {
        if (active) setPostImagesMap((prev) => ({ ...prev, [item.id]: res?.items || [] }));
      }).catch(() => {});
    });
    return () => { active = false; };
  }, [type, state.items]);

  const sortedItems = useMemo(() => {
    const items = [...(state.items || [])];
    if (type === "documents") {
      if (sort === "title") items.sort((a, b) => String(a.title || "").localeCompare(String(b.title || ""), "zh-CN"));
      else if (sort === "author") items.sort((a, b) => String(a.author || "").localeCompare(String(b.author || ""), "zh-CN"));
      else items.sort((a, b) => String(b.publication_date || "").localeCompare(String(a.publication_date || "")));
    } else {
      if (sort === "hot") items.sort((a, b) => (b.views_count || 0) - (a.views_count || 0));
      else if (sort === "likes") items.sort((a, b) => (b.likes_count || 0) - (a.likes_count || 0));
      else items.sort((a, b) => String(b.publish_date || "").localeCompare(String(a.publish_date || "")));
    }
    return items;
  }, [state.items, type, sort]);

  const title = useMemo(() => (type === "documents" ? "经籍文献" : "石窟图文"), [type]);
  const subtitle = useMemo(
    () => (type === "documents" ? "按作者、题名与年份筛选整理后的研究文献" : "按热度、时间与图像预览浏览石窟图文资料"),
    [type],
  );

  const applySearch = (next = {}) => {
    const p = new URLSearchParams(params);
    if (next.type) p.set("type", next.type);
    if (next.page) p.set("page", String(next.page));
    if (next.sort) p.set("sort", String(next.sort));
    if (typeof next.keyword === "string") {
      if (next.keyword) p.set("keyword", next.keyword);
      else p.delete("keyword");
    }
    if (!p.get("type")) p.set("type", type);
    if (!p.get("page")) p.set("page", "1");
    if (!p.get("sort")) p.set("sort", "latest");
    setParams(p);
  };

  const dbCards = [
    { label: "文献总量", value: overview?.documents ?? "--", note: "已整理入库的研究条目" },
    { label: "石窟图文", value: overview?.posts ?? "--", note: "可检索的图文内容" },
    { label: "图片资源", value: overview?.images ?? "--", note: "预览与图像附件数量" },
    { label: "用户数量", value: overview?.users ?? "--", note: "当前注册用户规模" },
  ];

  return (
    <section className="resources-wrap page-wide">
      <div className="resources-hero">
        <div className="resources-copy">
          <p className="section-kicker">资源检索</p>
          <h1>{title}</h1>
          <p className="resources-subtitle">{subtitle}</p>
        </div>

        <div className="db-overview-grid">
          {dbCards.map((card) => (
            <div className="db-overview-card" key={card.label}>
              <span className="db-overview-label">{card.label}</span>
              <span className="db-overview-value">{formatCompactNumber(card.value)}</span>
              <span className="db-overview-note">{card.note}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="resource-control-panel">
        <div className="resource-tools">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && applySearch({ keyword: input, page: 1 })}
            placeholder={type === "documents" ? "搜索论文、作者、机构..." : "搜索石窟、造像、帖子标题..."}
          />
          <button type="button" onClick={() => applySearch({ keyword: input, page: 1 })}>检索</button>
        </div>

        <div className="resource-control-row">
          <div className="tab-row">
            <button type="button" className={type === "posts" ? "active" : ""} onClick={() => applySearch({ type: "posts", page: 1 })}>石窟图文</button>
            <button type="button" className={type === "documents" ? "active" : ""} onClick={() => applySearch({ type: "documents", page: 1 })}>经籍文献</button>
          </div>

          <div className="resource-meta-tools">
            <select className="sort-select" value={sort} onChange={(e) => applySearch({ sort: e.target.value, page: 1 })}>
              {type === "documents" ? (
                <>
                  <option value="latest">按时间</option>
                  <option value="title">按标题</option>
                  <option value="author">按作者</option>
                </>
              ) : (
                <>
                  <option value="latest">按时间</option>
                  <option value="hot">按浏览量</option>
                  <option value="likes">按点赞量</option>
                </>
              )}
            </select>
            <span className="result-pill">共 {formatCompactNumber(state.total || 0)} 条</span>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="skeleton-grid">
          {Array.from({ length: 6 }).map((_, i) => <div key={i} className="skeleton-card" />)}
        </div>
      ) : null}
      {error ? <p className="error-text">{error}</p> : null}

      {!loading && !error ? (
        <>
          <div className="grid-list">
            {sortedItems.map((item) => {
              const isDoc = type === "documents";
              const summary = isDoc ? item.abstract : item.content;
              const who = isDoc ? item.author : item.author_name;
              const when = isDoc ? item.publication_date : item.publish_date;
              const preview = !isDoc ? postPreviewMap[item.id] : null;
              const images = !isDoc ? (postImagesMap[item.id] || []) : [];
              const thumb = !isDoc ? (images[0]?.image_url || item.image_url || preview?.image_url) : null;
              return (
                <Link className="card-link" key={`${type}-${item.id}`} to={`/detail/${type}/${item.id}`}>
                  <Card>
                    <h3>{item.title || "未命名"}</h3>
                    <p className="summary">{summary || "暂无摘要"}</p>
                    {type === "posts" && thumb ? <img className="post-thumb" src={thumb} alt="帖子图片" loading="lazy" /> : null}
                    {type === "posts" && (preview?.source_url || item.source_url) ? (
                      <p className="source">来源：<a href={preview?.source_url || item.source_url} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()}>{preview?.source_name || item.source_name || item.source_url}</a></p>
                    ) : null}
                    <div className="meta-row"><span>{who || "佚名"}</span><span>{when || "未知"}</span></div>
                    {isDoc ? <DocumentActions item={item} /> : null}
                  </Card>
                </Link>
              );
            })}
          </div>
          <Pager page={page} total={state.total || 0} pageSize={state.page_size || 12} onChange={(p) => applySearch({ page: p })} />
        </>
      ) : null}
    </section>
  );
}
