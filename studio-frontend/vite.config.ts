import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",
  plugins: [react()],
  build: {
    outDir: "../studio/next",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/healthz": "http://127.0.0.1:8080",
      "/runs": "http://127.0.0.1:8080",
      "/sessions": "http://127.0.0.1:8080",
      "/engine": "http://127.0.0.1:8080",
      "/jlens": "http://127.0.0.1:8080",
      "/snapshots": "http://127.0.0.1:8080",
      "/steer": "http://127.0.0.1:8080",
      "/sampling": "http://127.0.0.1:8080",
      "/guard": "http://127.0.0.1:8080",
      "/models": "http://127.0.0.1:8080",
    },
  },
});
