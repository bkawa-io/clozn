import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",
  plugins: [react()],
  build: {
    outDir: "../studio/v3",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/sessions": "http://127.0.0.1:8080",
      "/runs": "http://127.0.0.1:8080",
    },
  },
});
