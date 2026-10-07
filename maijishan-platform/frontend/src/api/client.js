import axios from "axios";

const TOKEN_KEY = "access_token";

export const getToken = () => localStorage.getItem(TOKEN_KEY);
export const setToken = (token) => {
  if (token) localStorage.setItem(TOKEN_KEY, token);
};
export const clearToken = () => localStorage.removeItem(TOKEN_KEY);

const api = axios.create({
  baseURL: "",
  timeout: 30000,
});

api.interceptors.request.use((config) => {
  const token = getToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => {
    const payload = response.data || {};
    if (payload.error) {
      return Promise.reject(payload.error);
    }
    return payload.data;
  },
  (error) => {
    if (error?.response?.status === 401) {
      clearToken();
      window.location.href = "/login";
    }
    const message =
      error?.code === "ECONNABORTED"
        ? "请求超时，请稍后重试"
        : error?.response?.data?.error?.message || error?.message || "请求失败";
    return Promise.reject({ message });
  },
);

export default api;
