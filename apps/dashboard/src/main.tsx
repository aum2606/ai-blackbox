import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { RunPage } from "./RunPage";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RunPage runId={new URLSearchParams(location.search).get("run")} />
  </StrictMode>,
);
