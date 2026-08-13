import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const gateway = "http://127.0.0.1:8080";

export default defineConfig({
  base: "./",
  plugins: [react()],
  build: {
    outDir: "../studio/v2",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/healthz": gateway,
      "/readyz": gateway,
      "/runtime": gateway,
      "/runs": gateway,
      "/sessions": gateway,
      "/diff": gateway,
    },
  },
});
