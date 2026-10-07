import api from "./client";

export const login = (payload) => api.post("/auth/login", payload);
export const register = (payload) => api.post("/auth/register", payload);
export const listPosts = (params) => api.get("/posts", { params });
export const getPost = (id) => api.get(`/posts/${id}`);
export const getPostPreview = (id) => api.get(`/posts/${id}/preview`);
export const getPostImages = (id, params) => api.get(`/posts/${id}/images`, { params });

export const listDocuments = (params) => api.get("/documents", { params });
export const getDocument = (id) => api.get(`/documents/${id}`);
export const getDocumentGraph = (params) => api.get("/documents/graph", { params });
export const getHomeStats = () => api.get("/stats/home");
export const getOverviewStats = () => api.get("/stats/overview");
export const aiChat = (payload) => api.post("/ai/chat", payload, { timeout: 180000 });
export const listRecentUploads = (params) => api.get("/uploads", { params });
export const uploadFile = (file) => {
  const formData = new FormData();
  formData.append("file", file);
  return api.post("/uploads", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
};

export {
  createCommunityPost,
  getCommunityPost,
  likeCommunityPost,
  listCommunityPosts,
} from "./community";
