import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",
  plugins: [react()],
  build: {
    outDir: "../studio/app",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/sessions": "http://127.0.0.1:8080",
      "/runs": "http://127.0.0.1:8080",
      "/engine": "http://127.0.0.1:8080",
      "/models": "http://127.0.0.1:8080",
      "/experiment-results": "http://127.0.0.1:8080",
    },
  },
});
