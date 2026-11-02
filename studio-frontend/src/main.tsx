import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource-variable/spline-sans";
import "@fontsource-variable/spline-sans-mono";
import { App } from "./app/App";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/shell.css";

document.documentElement.dataset.theme = "clozn-dark";

createRoot(document.getElementById("root")!).render(
  <StrictMode><App /></StrictMode>,
);
