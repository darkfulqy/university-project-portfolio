import api from "./client";

export const listCommunityPosts = (params) => api.get("/community/", { params });

export const getCommunityPost = (id) => api.get(`/community/${id}`);

export const likeCommunityPost = (id) => api.post(`/community/${id}/like`);

export const createCommunityPost = ({ title, content, images = [], attachments = [] }) => {
  const formData = new FormData();
  formData.append("title", title);
  formData.append("content", content);
  images.forEach((file) => formData.append("images", file));
  attachments.forEach((file) => formData.append("attachments", file));
  return api.post("/community/", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
};
