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
      "/runs": "http://127.0.0.1:8080",
      "/v1": "http://127.0.0.1:8080",
    },
  },
});
