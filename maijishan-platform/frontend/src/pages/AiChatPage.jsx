import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { aiChat } from "../api";

const SUGGESTED_PROMPTS = [
  "请概括麦积山第44窟的主要特点。",
  "整理麦积山彩塑保护研究中的几个常见主题。",
  "比较北朝时期麦积山与敦煌造像风格的差异。",
];

const RESEARCH_STEPS = [
  {
    title: "站内检索优先",
    body: "先检索文献和图文资料，再基于命中内容组织回答。",
  },
  {
    title: "回答聚焦结论",
    body: "主答案优先给结论，参考文献只做简要提示与跳转。",
  },
  {
    title: "延伸到原文",
    body: "需要深挖时，可直接跳转到站内检索页或文献详情继续查看。",
  },
];

function buildResourceHref(source) {
  const query = new URLSearchParams({
    type: "documents",
    keyword: source || "",
    page: "1",
  });
  return `/resources?${query.toString()}`;
}

function previewText(text, maxLength = 88) {
  const compact = String(text || "").replace(/\s+/g, " ").trim();
  if (!compact) return "暂无预览";
  if (compact.length <= maxLength) return compact;
  return `${compact.slice(0, maxLength)}...`;
}

function CitationImages({ images = [] }) {
  const visible = images.slice(0, 1);
  if (!visible.length) return null;
  return (
    <div className="chat-citation-image-grid compact">
      {visible.map((src) => (
        <a key={src} href={src} target="_blank" rel="noreferrer" className="chat-citation-image-link">
          <img
            src={src}
            alt="相关图像"
            className="chat-citation-image"
            loading="lazy"
            onError={(e) => {
              e.currentTarget.closest(".chat-citation-image-link")?.remove();
            }}
          />
        </a>
      ))}
    </div>
  );
}

function CitationCard({ citation, index }) {
  return (
    <article className="chat-citation-card compact" key={`${citation.source}-${index}`}>
      <div className="chat-citation-head">
        <strong title={citation.source}>{citation.source || "未命名文献"}</strong>
        <span>相关度 {Number(citation.score || 0).toFixed(2)}</span>
      </div>
      <p>{previewText(citation.text)}</p>
      <div className="chat-citation-actions">
        <Link className="chat-citation-link" to={buildResourceHref(citation.source)}>
          查看文献
        </Link>
        <span className="chat-citation-hint">跳转到站内检索页</span>
      </div>
      <CitationImages images={citation.images || []} />
    </article>
  );
}

function MessageBubble({ item }) {
  const isAssistant = item.role === "assistant";

  return (
    <article className={`chat-item ${isAssistant ? "assistant" : "user"}`}>
      <div className="chat-avatar" aria-hidden>{isAssistant ? "答" : "问"}</div>
      <div className="chat-message-stack">
        <div className="chat-item-meta">
          <strong>{isAssistant ? "灵境问答" : "你的提问"}</strong>
          <span>{isAssistant ? "优先展示结论，文献作为补充入口" : "已提交到检索链路"}</span>
        </div>
        <div className="chat-bubble">
          <p>{item.content}</p>
        </div>
        {isAssistant && item.citations?.length ? (
          <div className="chat-citation-wrap">
            <div className="chat-citation-summary">
              <strong>参考文献</strong>
              <span>{item.citations.length} 条，点击可跳转到站内检索页继续查看</span>
            </div>
            <div className="chat-citation-list compact">
              {item.citations.map((citation, index) => (
                <CitationCard citation={citation} index={index} key={`${citation.source}-${index}`} />
              ))}
            </div>
          </div>
        ) : null}
      </div>
    </article>
  );
}

export default function AiChatPage() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const logRef = useRef(null);

  useEffect(() => {
    if (!logRef.current) return;
    logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [messages, pending]);

  const totalCitations = useMemo(
    () => messages.reduce((count, item) => count + (item.citations?.length || 0), 0),
    [messages],
  );

  const submitQuestion = async (input) => {
    const content = (input ?? question).trim();
    if (!content || pending) return;

    setError("");
    setPending(true);
    setMessages((prev) => [...prev, { role: "user", content }]);
    setQuestion("");

    try {
      const res = await aiChat({ user_message: content });
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.assistant_message || "暂未返回可展示的回答。",
          citations: res.citations || [],
        },
      ]);
    } catch (e) {
      setError(e?.response?.data?.detail || e?.message || "问答服务暂时不可用，请稍后重试。");
    } finally {
      setPending(false);
    }
  };

  return (
    <section className="page-wide ai-chat-shell">
      <aside className="ai-chat-rail">
        <div className="ai-chat-hero card">
          <p className="section-kicker">知识问答</p>
          <h1>灵境问道</h1>
          <p className="ai-chat-summary">
            面向麦积山研究资料的检索问答入口。主答案优先给出结论，参考文献缩成索引卡片，便于继续跳转查看。
          </p>
          <div className="ai-chat-stat-grid">
            <div className="ai-chat-stat-card">
              <span>会话轮次</span>
              <strong>{messages.length}</strong>
            </div>
            <div className="ai-chat-stat-card">
              <span>参考条目</span>
              <strong>{totalCitations}</strong>
            </div>
          </div>
        </div>

        <div className="ai-chat-notes card">
          <div className="community-section-title">问答方式</div>
          <div className="ai-chat-step-list">
            {RESEARCH_STEPS.map((item) => (
              <article className="ai-chat-step" key={item.title}>
                <strong>{item.title}</strong>
                <p>{item.body}</p>
              </article>
            ))}
          </div>
        </div>

        <div className="ai-chat-notes card">
          <div className="community-section-title">可直接提问</div>
          <div className="ai-chat-prompt-list">
            {SUGGESTED_PROMPTS.map((prompt) => (
              <button
                type="button"
                key={prompt}
                className="ai-chat-prompt"
                onClick={() => submitQuestion(prompt)}
                disabled={pending}
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>
      </aside>

      <div className="chat-panel ai-chat-panel">
        <div className="chat-title ai-chat-panel-head">
          <div>
            <strong>学术问答席</strong>
            <span>回答为主，文献为辅，点击引用即可跳转继续检索</span>
          </div>
          <div className="ai-chat-status">
            <span className={`ai-chat-status-dot ${pending ? "pending" : "ready"}`} />
            {pending ? "正在检索与生成" : "可以开始提问"}
          </div>
        </div>

        <div className="chat-log" ref={logRef}>
          {messages.length === 0 ? (
            <div className="ai-chat-empty">
              <div className="ai-chat-empty-mark">问</div>
              <h2>从一个明确问题开始</h2>
              <p>例如询问某一洞窟的造像特征、某篇文献的核心观点，或某类保护研究的代表案例。</p>
            </div>
          ) : null}

          {messages.map((item, index) => (
            <MessageBubble key={`${item.role}-${index}-${item.content.slice(0, 12)}`} item={item} />
          ))}

          {pending ? (
            <article className="chat-item assistant">
              <div className="chat-avatar" aria-hidden>答</div>
              <div className="chat-message-stack">
                <div className="chat-item-meta">
                  <strong>灵境问答</strong>
                  <span>知识库正在整理相关资料</span>
                </div>
                <div className="chat-bubble chat-bubble-loading">
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            </article>
          ) : null}
        </div>

        <div className="chat-input-row ai-chat-input-row">
          <label className="ai-chat-composer">
            <span className="ai-chat-composer-label">提问内容</span>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="请输入你的问题，例如：请概括麦积山第44窟的主要特点。"
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  submitQuestion();
                }
              }}
            />
          </label>
          <button type="button" onClick={() => submitQuestion()} disabled={pending || !question.trim()}>
            {pending ? "生成中" : "发送"}
          </button>
        </div>
        {error ? <p className="error-text ai-chat-error">{error}</p> : null}
      </div>
    </section>
  );
}
