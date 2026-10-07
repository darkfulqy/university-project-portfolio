import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getDocument, getPost, getPostImages, getPostPreview } from "../api";
import Card from "../components/Card";

function DocFileActions({ doc }) {
  if (!doc) return null;
  return (
    <div className="doc-actions-inline">
      {doc.preview_url ? (
        <a className="btn-small" href={doc.preview_url} target="_blank" rel="noreferrer">在线预览</a>
      ) : null}
      {doc.pdf_download_url ? (
        <a className="btn-small" href={doc.pdf_download_url} target="_blank" rel="noreferrer">PDF 下载</a>
      ) : null}
      {doc.caj_download_url ? (
        <a className="btn-small" href={doc.caj_download_url} target="_blank" rel="noreferrer">CAJ 下载</a>
      ) : null}
    </div>
  );
}

function DocMeta({ doc }) {
  const rows = [
    ["作者", doc.author],
    ["来源库", doc.source_db],
    ["来源类型", doc.source_type],
    ["发表时间", doc.publication_date],
    ["DOI", doc.doi],
    ["ISSN", doc.issn],
    ["ISBN", doc.isbn],
    ["卷期", [doc.volume, doc.issue].filter(Boolean).join(" / ")],
    ["页码", doc.pages],
    ["机构", doc.author_unit],
    ["学校", doc.school],
    ["导师", doc.advisor],
  ].filter(([, val]) => !!val);

  return (
    <div className="meta-grid">
      {rows.map(([k, v]) => (
        <div className="meta-item" key={k}><span className="meta-label">{k}：</span>{v}</div>
      ))}
      {doc.source_url ? (
        <div className="meta-item"><span className="meta-label">来源链接：</span><a className="meta-link" href={doc.source_url} target="_blank" rel="noreferrer">{doc.source_url}</a></div>
      ) : null}
    </div>
  );
}

export default function DetailPage() {
  const { type = "posts", id } = useParams();
  const isDoc = type === "documents";

  const [item, setItem] = useState(null);
  const [postPreview, setPostPreview] = useState(null);
  const [postImages, setPostImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    const fn = isDoc ? getDocument : getPost;
    fn(id)
      .then((res) => {
        if (!active) return;
        setItem(res);
      })
      .catch((e) => {
        if (!active) return;
        setError(e.message || "加载失败");
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    if (!isDoc) {
      getPostPreview(id).then(setPostPreview).catch(() => setPostPreview(null));
      getPostImages(id, { limit: 6 }).then((x) => setPostImages(x?.items || [])).catch(() => setPostImages([]));
    }

    return () => {
      active = false;
    };
  }, [id, isDoc]);

  return (
    <section className="detail-wrap page-wide">
      <Link className="back-link" to={`/resources?type=${isDoc ? "documents" : "posts"}`}>← 返回检索</Link>
      {loading ? <p className="muted">加载中...</p> : null}
      {error ? <p className="error-text">{error}</p> : null}
      {!loading && !error && item ? (
        <Card className="detail-card">
          <h1>{item.title}</h1>
          <div className="meta-row top">
            <span>{isDoc ? item.author : item.author_name}</span>
            <span>{isDoc ? item.publication_date : item.publish_date}</span>
          </div>

          {isDoc ? <DocFileActions doc={item} /> : null}

          {!isDoc && postPreview?.source_url ? (
            <p className="source">来源：<a href={postPreview.source_url} target="_blank" rel="noreferrer">{postPreview.source_name || postPreview.source_url}</a></p>
          ) : null}

          {!isDoc && postImages.length ? (
            <div className="post-image-grid">
              {postImages.map((img) => (
                <img key={img.id} src={img.image_url} alt="帖子图片" />
              ))}
            </div>
          ) : null}

          <article className="detail-text">{isDoc ? (item.abstract || "暂无摘要") : (item.content || "暂无内容")}</article>

          {isDoc ? <DocMeta doc={item} /> : null}

          {isDoc && item.keywords?.length ? <p className="keywords">关键词：{item.keywords.join(" · ")}</p> : null}
        </Card>
      ) : null}
    </section>
  );
}
