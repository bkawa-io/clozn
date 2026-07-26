import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./app/App";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/workspace.css";
import "./styles/observatory.css";
import "./styles/compare.css";
import "./styles/runs.css";
import "./styles/lens.css";
import "./styles/behavior.css";
import "./styles/model.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
