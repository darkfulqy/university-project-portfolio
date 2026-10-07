import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getHomeStats } from "../api";

function formatCompactNumber(value) {
  const num = Number(value || 0);
  if (!Number.isFinite(num)) return "--";
  return new Intl.NumberFormat("zh-CN").format(num);
}

function Bars({ title, items = [], keyName, scrollable = false, onItemClick = null }) {
  const max = Math.max(1, ...items.map((x) => x.count || 0));
  return (
    <div>
      <h3 className="stats-section-title">{title}</h3>
      {items.length === 0 ? <p className="muted">暂无数据</p> : null}
      <div className={scrollable ? "bar-scroll" : ""}>
        {items.map((item, idx) => (
          <div
            className={`bar-row ${onItemClick ? "clickable" : ""}`}
            key={`${keyName}-${idx}`}
            onClick={() => onItemClick?.(item[keyName])}
            role={onItemClick ? "button" : undefined}
            tabIndex={onItemClick ? 0 : undefined}
            onKeyDown={(e) => {
              if (!onItemClick) return;
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onItemClick(item[keyName]);
              }
            }}
          >
            <span className="bar-label">{item[keyName]}</span>
            <div className="bar-track"><div className="bar-fill" style={{ width: `${(item.count / max) * 100}%` }} /></div>
            <span className="bar-val">{item.count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Trend({ items = [] }) {
  const [hoverIndex, setHoverIndex] = useState(null);
  const chartPoints = useMemo(() => {
    if (!items.length) return [];
    const max = Math.max(1, ...items.map((x) => x.count || 0));
    return items.map((item, i) => {
      const x = (i / Math.max(1, items.length - 1)) * 100;
      const y = 100 - ((item.count || 0) / max) * 100;
      return { x, y, year: item.year, count: item.count || 0 };
    });
  }, [items]);
  const points = chartPoints.map((p) => `${p.x},${p.y}`).join(" ");

  return (
    <div>
      <h3 className="stats-section-title">年份-文献数量趋势</h3>
      {items.length === 0 ? <p className="muted">暂无数据</p> : null}
      {items.length > 0 ? (
        <>
          <div className="trend-wrap">
            <svg
              className="trend"
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
              onMouseMove={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const x = ((e.clientX - rect.left) / rect.width) * 100;
                let nearest = 0;
                let best = Infinity;
                chartPoints.forEach((p, idx) => {
                  const d = Math.abs(p.x - x);
                  if (d < best) {
                    best = d;
                    nearest = idx;
                  }
                });
                setHoverIndex(nearest);
              }}
              onMouseLeave={() => setHoverIndex(null)}
            >
              <polyline points={points} className="trend-line" />
              {hoverIndex !== null ? (
                <>
                  <line x1={chartPoints[hoverIndex].x} x2={chartPoints[hoverIndex].x} y1="0" y2="100" className="trend-hover-line" />
                  <circle cx={chartPoints[hoverIndex].x} cy={chartPoints[hoverIndex].y} r="1.8" className="trend-hover-dot" />
                </>
              ) : null}
            </svg>
            {hoverIndex !== null ? (
              <div
                className="trend-tooltip"
                style={{ left: `${chartPoints[hoverIndex].x}%`, top: `${Math.max(8, chartPoints[hoverIndex].y - 8)}%` }}
              >
                <div>{chartPoints[hoverIndex].year}年</div>
                <div>{chartPoints[hoverIndex].count} 篇</div>
              </div>
            ) : null}
          </div>
          <div className="trend-timeline">
            {(items.length <= 8 ? items : [items[0], ...items.filter((_, i) => i % Math.ceil(items.length / 6) === 0), items[items.length - 1]])
              .filter((v, i, arr) => i === 0 || v.year !== arr[i - 1].year)
              .map((it) => (
                <span key={`t-${it.year}`}>{it.year}</span>
              ))}
          </div>
        </>
      ) : null}
    </div>
  );
}

export default function HomePage() {
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [index, setIndex] = useState(0);

  useEffect(() => {
    let active = true;
    getHomeStats()
      .then((res) => {
        if (!active) return;
        setData(res);
      })
      .catch((e) => {
        if (!active) return;
        setError(e.message || "统计加载失败");
      });
    return () => {
      active = false;
    };
  }, []);

  const slides = useMemo(() => {
    if (!data) return [];
    const jumpToKeywordSearch = (keyword) => {
      if (!keyword) return;
      const query = new URLSearchParams({ type: "documents", keyword, page: "1" });
      navigate(`/resources?${query.toString()}`);
    };
    const jumpToAuthorSearch = (author) => {
      if (!author) return;
      const query = new URLSearchParams({ type: "documents", keyword: author, page: "1" });
      navigate(`/resources?${query.toString()}`);
    };
    return [
      {
        key: "keywords",
        title: "热门关键词",
        node: <Bars title="热门关键词（点击可检索）" items={data.keywords || []} keyName="keyword" scrollable onItemClick={jumpToKeywordSearch} />,
      },
      { key: "years", title: "年份趋势", node: <Trend items={data.year_series || []} /> },
      {
        key: "authors",
        title: "热门作者",
        node: <Bars title="热门作者（点击可检索）" items={data.top_authors || []} keyName="author" scrollable onItemClick={jumpToAuthorSearch} />,
      },
    ];
  }, [data, navigate]);

  const current = slides[index] || null;
  const statTiles = [
    { label: "文献总量", value: data?.total_documents ?? "--", note: "研究文献入库" },
    { label: "石窟图文", value: data?.total_posts ?? "--", note: "图文资料可检索" },
    { label: "图片资源", value: data?.total_images ?? "--", note: "图像与预览文件" },
    { label: "平台用户", value: data?.total_users ?? "--", note: "注册账户规模" },
  ];
  const bgImages = [
    "/shouye/0e7f585187f043e7d9ba0490705da373517ba7d022e67fe9c33c47f3afcfaeff.jpg",
    "/shouye/102.jpg",
    "/shouye/1a0025b5f3ac957c199b6bef1615c25c8ae21379c0c166f165e4739cd7f53982.jpg",
    "/shouye/247.jpg",
    "/shouye/462.png",
    "/shouye/739.png",
    "/shouye/771.jpg",
    "/shouye/becc35501972055d2e18b9bf4338e2ad8f4c6e0e9ca42268ed309499b640d8a2.jpg",
    "/shouye/f3d9b47f69382de0a1b5c4f9c48fb3e33e3fb3ac0eeb58a933510df6de61be0d.jpg",
    "/shouye/fdf0055fc10024b21bdc277a12544b75e28aa28d289629c9c29adba8acffbf3a.jpg",
  ];

  return (
    <section className="home-wrap home-wide home-collage">
      <div className="home-bg" aria-hidden>
        <div className="home-bg-row">
          {[...bgImages, ...bgImages].map((src, i) => <img key={`r1-${i}`} src={src} alt="" />)}
        </div>
        <div className="home-bg-row reverse">
          {[...bgImages, ...bgImages].map((src, i) => <img key={`r2-${i}`} src={src} alt="" />)}
        </div>
        <div className="home-bg-row slow">
          {[...bgImages, ...bgImages].map((src, i) => <img key={`r3-${i}`} src={src} alt="" />)}
        </div>
      </div>

      <div className="hero-card">
        <p className="section-kicker">首页总览</p>
        <h1>麦积山石窟数字信息平台</h1>
        <p>文献统计、趋势洞察与检索导览，收束到同一套数字浏览界面。</p>
        <div className="hero-metrics">
          {statTiles.map((item) => (
            <div className="hero-metric-card" key={item.label}>
              <span className="hero-metric-label">{item.label}</span>
              <strong className="hero-metric-value">{formatCompactNumber(item.value)}</strong>
              <span className="hero-metric-note">{item.note}</span>
            </div>
          ))}
        </div>
      </div>

      <aside className="stats-card">
        <div className="stats-header">
          <strong>{current?.title || "数据看板"}</strong>
          <span>{data?.updated_at ? new Date(data.updated_at).toLocaleString() : "加载中"}</span>
        </div>
        <div className="stats-body">
          {error ? <p className="error-text">{error}</p> : null}
          {!error && !current ? <p className="muted">正在加载...</p> : null}
          {!error && current ? current.node : null}
        </div>
        <div className="stats-nav">
          <button type="button" onClick={() => setIndex((x) => (x - 1 + slides.length) % slides.length)} disabled={!slides.length}>上一页</button>
          <div className="dots">
            {slides.map((s, i) => (
              <span key={s.key} className={`dot ${i === index ? "active" : ""}`} />
            ))}
          </div>
          <button type="button" onClick={() => setIndex((x) => (x + 1) % slides.length)} disabled={!slides.length}>下一页</button>
        </div>
      </aside>
    </section>
  );
}
