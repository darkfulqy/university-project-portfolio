import { NavLink, Outlet, Link } from "react-router-dom";
import { getToken } from "../api/client";

const navClass = ({ isActive }) => `nav-link ${isActive ? "active" : ""}`;

export default function Layout() {
  const hasToken = !!getToken();

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="container topbar-inner">
          <Link to="/" className="brand">
            <span className="brand-badge">麦</span>
            <div>
              <div className="brand-title">麦积山</div>
              <div className="brand-sub">数字信息平台</div>
            </div>
          </Link>
          <nav className="nav">
            <NavLink to="/" className={navClass}>首页</NavLink>
            <NavLink to="/resources?type=posts" className={navClass}>石窟遗珍</NavLink>
            <NavLink to="/resources?type=documents" className={navClass}>经籍文献</NavLink>
            <NavLink to="/community" className={navClass}>学术社区</NavLink>
            <NavLink to="/upload" className={navClass}>文件上传</NavLink>
            <NavLink to="/ai-chat" className={navClass}>灵境问道</NavLink>
            <NavLink to="/login" className="login-btn">{hasToken ? "我的" : "登入"}</NavLink>
          </nav>
        </div>
      </header>

      <main className="main-content">
        <Outlet />
      </main>

      <footer className="footer">
        <div className="container footer-inner">
          <span>麦积山数字信息平台 · 文化遗产数字化展示</span>
          <span>数据与内容服务由后端 API 提供</span>
        </div>
      </footer>
    </div>
  );
}
