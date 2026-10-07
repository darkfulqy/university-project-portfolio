import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { createCommunityPost, listCommunityPosts } from "../api/community";
import { getToken } from "../api/client";

function formatFileSize(size) {
  const value = Number(size || 0);
  if (!Number.isFinite(value) || value <= 0) return "--";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

export default function CommunityPage() {
  const hasToken = !!getToken();
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [submitError, setSubmitError] = useState("");
  const [form, setForm] = useState({ title: "", content: "", images: [], attachments: [] });

  const selectedFiles = useMemo(
    () => [...form.images.map((file) => ({ file, kind: "图片" })), ...form.attachments.map((file) => ({ file, kind: "附件" }))],
    [form.attachments, form.images],
  );

  const loadPosts = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await listCommunityPosts({ page: 1, page_size: 20 });
      setPosts(response.posts || []);
    } catch (err) {
      setError(err.message || "加载帖子失败");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPosts();
  }, []);

  const submitPost = async (event) => {
    event.preventDefault();
    if (!form.title.trim() || !form.content.trim()) {
      setSubmitError("请先填写标题和正文");
      return;
    }

    setSubmitting(true);
    setSubmitError("");
    try {
      const created = await createCommunityPost({
        title: form.title.trim(),
        content: form.content.trim(),
        images: form.images,
        attachments: form.attachments,
      });
      setPosts((current) => [created, ...current]);
      setForm({ title: "", content: "", images: [], attachments: [] });
    } catch (err) {
      setSubmitError(err.message || "发布帖子失败");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <section className="community-wrap page-wide">
      <div className="community-shell">
        <div className="community-column">
          <div className="community-hero">
            <p className="section-kicker">学术社区</p>
            <h1>发帖时直接附带图片与文献附件</h1>
            <p className="community-subtitle">
              支持在帖子里上传图片、PDF 和研究附件。检测到 PDF 后，后端会异步送去 MinerU 转 Markdown，再调用 Qwen
              提取作者、来源、摘要等文献信息并写入文献库。
            </p>
          </div>

          <div className="community-feed">
            <div className="community-feed-head">
              <h2>最新帖子</h2>
              <button type="button" onClick={loadPosts}>刷新列表</button>
            </div>

            {loading ? <div className="card">正在加载帖子...</div> : null}
            {error ? <p className="error-text">{error}</p> : null}

            <div className="community-posts">
              {posts.map((post) => (
                <article key={post.id} className="community-post-card">
                  <div className="community-post-meta">
                    <span>浏览 {post.views_count}</span>
                    <span>点赞 {post.likes_count}</span>
                    <span>附件 {post.attachments?.length || 0}</span>
                  </div>
                  <h3>
                    <Link to={`/community/${post.id}`}>{post.title}</Link>
                  </h3>
                  <p className="summary">{post.content}</p>
                  {post.attachments?.length ? (
                    <div className="community-attachment-list">
                      {post.attachments.slice(0, 4).map((attachment) => (
                        <span key={attachment.id} className="attachment-pill">
                          {attachment.kind === "image" ? "图片" : "附件"} · {attachment.filename}
                        </span>
                      ))}
                    </div>
                  ) : null}
                </article>
              ))}
            </div>
          </div>
        </div>

        <aside className="community-composer card">
          <div className="community-composer-head">
            <h2>发布帖子</h2>
            <p className="muted">正文、图片和 PDF 附件会一起提交。</p>
          </div>

          {!hasToken ? (
            <div className="community-login-state">
              <p className="muted">当前未登录，先登录后才能发布帖子。</p>
              <Link className="login-btn" to="/login">前往登录</Link>
            </div>
          ) : (
            <form className="community-form" onSubmit={submitPost}>
              <label>
                标题
                <input
                  type="text"
                  value={form.title}
                  onChange={(event) => setForm((current) => ({ ...current, title: event.target.value }))}
                  placeholder="输入帖子标题"
                />
              </label>

              <label>
                正文
                <textarea
                  value={form.content}
                  onChange={(event) => setForm((current) => ({ ...current, content: event.target.value }))}
                  placeholder="输入你的研究发现、问题或资料说明"
                  rows={8}
                />
              </label>

              <label>
                帖子图片
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  multiple
                  onChange={(event) =>
                    setForm((current) => ({ ...current, images: Array.from(event.target.files || []) }))
                  }
                />
              </label>

              <label>
                文献附件
                <input
                  type="file"
                  accept=".pdf,.doc,.docx,.txt,.zip,image/png,image/jpeg"
                  multiple
                  onChange={(event) =>
                    setForm((current) => ({ ...current, attachments: Array.from(event.target.files || []) }))
                  }
                />
              </label>

              {selectedFiles.length ? (
                <div className="community-selected-files">
                  {selectedFiles.map(({ file, kind }) => (
                    <div key={`${kind}-${file.name}-${file.size}`} className="selected-file-row">
                      <span>{kind}</span>
                      <strong>{file.name}</strong>
                      <span>{formatFileSize(file.size)}</span>
                    </div>
                  ))}
                </div>
              ) : null}

              {submitError ? <p className="error-text">{submitError}</p> : null}

              <button type="submit" disabled={submitting}>
                {submitting ? "正在发布..." : "发布帖子"}
              </button>
            </form>
          )}
        </aside>
      </div>
    </section>
  );
}
