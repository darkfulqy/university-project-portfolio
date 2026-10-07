import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getCommunityPost, likeCommunityPost } from "../api/community";

function formatFileSize(size) {
  const value = Number(size || 0);
  if (!Number.isFinite(value) || value <= 0) return "--";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function statusText(attachment) {
  if (attachment.kind !== "attachment" || !attachment.filename.toLowerCase().endsWith(".pdf")) {
    return "已上传";
  }
  if (attachment.processing_status === "done") return "已转文献";
  if (attachment.processing_status === "failed") return `处理失败: ${attachment.processing_error || "未知错误"}`;
  if (attachment.processing_status === "processing") return "正在解析 PDF";
  return "等待解析";
}

export default function CommunityDetailPage() {
  const { id } = useParams();
  const [post, setPost] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadPost = async () => {
      setLoading(true);
      setError("");
      try {
        const response = await getCommunityPost(id);
        setPost(response);
      } catch (err) {
        setError(err.message || "加载帖子失败");
      } finally {
        setLoading(false);
      }
    };
    loadPost();
  }, [id]);

  const handleLike = async () => {
    if (!post) return;
    try {
      const response = await likeCommunityPost(post.id);
      setPost((current) => (current ? { ...current, likes_count: response.likes_count } : current));
    } catch (err) {
      setError(err.message || "点赞失败");
    }
  };

  if (loading) {
    return <section className="detail-wrap page-wide"><div className="detail-card card">正在加载帖子...</div></section>;
  }

  if (error) {
    return <section className="detail-wrap page-wide"><div className="detail-card card error-text">{error}</div></section>;
  }

  if (!post) {
    return <section className="detail-wrap page-wide"><div className="detail-card card">帖子不存在</div></section>;
  }

  const images = (post.attachments || []).filter((item) => item.kind === "image");
  const files = (post.attachments || []).filter((item) => item.kind !== "image");

  return (
    <section className="detail-wrap page-wide">
      <Link to="/community" className="back-link">返回社区</Link>
      <article className="detail-card card">
        <div className="community-post-meta top">
          <span>浏览 {post.views_count}</span>
          <span>点赞 {post.likes_count}</span>
          <span>评论 {post.comments_count}</span>
        </div>
        <h1>{post.title}</h1>
        <p className="detail-text">{post.content}</p>
        <div className="doc-actions-inline">
          <button type="button" onClick={handleLike}>点赞帖子</button>
        </div>

        {images.length ? (
          <>
            <h2 className="community-section-title">帖子图片</h2>
            <div className="community-image-grid">
              {images.map((image) => (
                <a key={image.id} href={image.url} target="_blank" rel="noreferrer">
                  <img src={image.url} alt={image.filename} />
                </a>
              ))}
            </div>
          </>
        ) : null}

        {files.length ? (
          <>
            <h2 className="community-section-title">附件与文献解析</h2>
            <div className="community-file-grid">
              {files.map((attachment) => (
                <div key={attachment.id} className="community-file-card">
                  <div className="community-file-head">
                    <strong>{attachment.filename}</strong>
                    <span>{formatFileSize(attachment.size)}</span>
                  </div>
                  <p className="muted">{statusText(attachment)}</p>
                  <div className="doc-actions-inline">
                    <a className="btn-small" href={attachment.url} target="_blank" rel="noreferrer">打开附件</a>
                    {attachment.linked_document_url ? (
                      <Link className="btn-small" to={attachment.linked_document_url}>查看入库文献</Link>
                    ) : null}
                  </div>
                </div>
              ))}
            </div>
          </>
        ) : null}
      </article>
    </section>
  );
}
