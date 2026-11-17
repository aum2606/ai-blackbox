import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The dashboard talks only to the documented /v1 API, proxied so no CORS is opened.
export default defineConfig({
  plugins: [react()],
  server: { host: "127.0.0.1", proxy: { "/v1": "http://127.0.0.1:8000" } },
});
