import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import HomePage from "./pages/HomePage";
import RepoPage from "./pages/RepoPage";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/repo/:owner/:name" element={<RepoPage />} />
      </Routes>
    </Layout>
  );
}
