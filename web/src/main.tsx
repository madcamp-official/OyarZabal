import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import HumanBenchmarkAdmin from "./HumanBenchmarkAdmin";
import HumanBenchmarkPrototype from "./HumanBenchmarkPrototype";
import "./styles.css";

const pathname = window.location.pathname.replace(/\/+$/, "");
const Root = pathname === "/admin/human-benchmark"
  ? HumanBenchmarkAdmin
  : window.location.hostname === "www.pitchtest.madcamp-kaist.org" ||
      pathname === "/prototype/human-benchmark"
    ? HumanBenchmarkPrototype
    : App;

document.documentElement.classList.toggle(
  "human-benchmark-page",
  Root === HumanBenchmarkPrototype,
);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Root />
  </StrictMode>,
);
