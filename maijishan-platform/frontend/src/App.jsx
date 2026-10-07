import { createBrowserRouter, RouterProvider } from "react-router-dom";
import Layout from "./components/Layout";
import HomePage from "./pages/HomePage";
import LoginPage from "./pages/LoginPage";
import ResourcesPage from "./pages/ResourcesPage";
import DetailPage from "./pages/DetailPage";
import AiChatPage from "./pages/AiChatPage";
import UploadPage from "./pages/UploadPage";
import CommunityPage from "./pages/CommunityPage";
import CommunityDetailPage from "./pages/CommunityDetailPage";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      { index: true, element: <HomePage /> },
      { path: "login", element: <LoginPage /> },
      { path: "upload", element: <UploadPage /> },
      { path: "community", element: <CommunityPage /> },
      { path: "community/:id", element: <CommunityDetailPage /> },
      { path: "resources", element: <ResourcesPage /> },
      { path: "detail/:type/:id", element: <DetailPage /> },
      { path: "ai-chat", element: <AiChatPage /> },
    ],
  },
]);

export default function App() {
  return <RouterProvider router={router} />;
}
