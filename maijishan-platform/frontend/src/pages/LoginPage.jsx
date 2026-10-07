import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { login, register } from "../api";
import { setToken } from "../api/client";

const initialLoginForm = { username: "", password: "" };
const initialRegisterForm = { username: "", email: "", password: "", confirmPassword: "" };

export default function LoginPage() {
  const nav = useNavigate();
  const [mode, setMode] = useState("login");
  const [loginForm, setLoginForm] = useState(initialLoginForm);
  const [registerForm, setRegisterForm] = useState(initialRegisterForm);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const switchMode = (nextMode) => {
    setMode(nextMode);
    setError("");
    setSuccess("");
  };

  const submitLogin = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess("");
    setLoading(true);
    try {
      const res = await login(loginForm);
      setToken(res.access_token);
      nav("/");
    } catch (err) {
      setError(err.message || "登录失败");
    } finally {
      setLoading(false);
    }
  };

  const submitRegister = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess("");

    if (registerForm.password !== registerForm.confirmPassword) {
      setError("两次输入的密码不一致");
      return;
    }

    setLoading(true);
    try {
      await register({
        username: registerForm.username,
        email: registerForm.email,
        password: registerForm.password,
      });
      setRegisterForm(initialRegisterForm);
      setMode("login");
      setSuccess("注册成功，请使用新账号登录");
    } catch (err) {
      setError(err.message || "注册失败");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="login-wrap">
      <div className="login-card">
        <div className="login-head">
          <p className="section-kicker">用户系统</p>
          <h2>云上麦积</h2>
          <p className="muted">{mode === "login" ? "登录后可使用上传与个人权限相关功能" : "注册新账号后即可登录使用平台"}</p>
        </div>

        <div className="login-switch" role="tablist" aria-label="登录与注册切换">
          <button type="button" className={mode === "login" ? "active" : ""} onClick={() => switchMode("login")}>登录</button>
          <button type="button" className={mode === "register" ? "active" : ""} onClick={() => switchMode("register")}>注册</button>
        </div>

        {mode === "login" ? (
          <form className="login-form" onSubmit={submitLogin}>
            <label>
              用户名
              <input
                type="text"
                value={loginForm.username}
                onChange={(e) => setLoginForm({ ...loginForm, username: e.target.value })}
                required
              />
            </label>
            <label>
              密码
              <input
                type="password"
                value={loginForm.password}
                onChange={(e) => setLoginForm({ ...loginForm, password: e.target.value })}
                required
              />
            </label>
            {error ? <p className="error-text">{error}</p> : null}
            {success ? <p className="success-text">{success}</p> : null}
            <button type="submit" disabled={loading}>{loading ? "正在登录..." : "登录"}</button>
          </form>
        ) : (
          <form className="login-form" onSubmit={submitRegister}>
            <label>
              用户名
              <input
                type="text"
                value={registerForm.username}
                onChange={(e) => setRegisterForm({ ...registerForm, username: e.target.value })}
                required
              />
            </label>
            <label>
              邮箱
              <input
                type="email"
                value={registerForm.email}
                onChange={(e) => setRegisterForm({ ...registerForm, email: e.target.value })}
                required
              />
            </label>
            <label>
              密码
              <input
                type="password"
                value={registerForm.password}
                onChange={(e) => setRegisterForm({ ...registerForm, password: e.target.value })}
                required
              />
            </label>
            <label>
              确认密码
              <input
                type="password"
                value={registerForm.confirmPassword}
                onChange={(e) => setRegisterForm({ ...registerForm, confirmPassword: e.target.value })}
                required
              />
            </label>
            {error ? <p className="error-text">{error}</p> : null}
            {success ? <p className="success-text">{success}</p> : null}
            <button type="submit" disabled={loading}>{loading ? "正在注册..." : "创建账号"}</button>
          </form>
        )}
      </div>
    </section>
  );
}
