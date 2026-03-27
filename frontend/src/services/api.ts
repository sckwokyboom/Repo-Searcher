import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:7860";

const api = axios.create({
  baseURL: `${API_BASE}/api`,
  timeout: 300000,
});

export const WS_BASE = API_BASE.replace("http", "ws");

export default api;
