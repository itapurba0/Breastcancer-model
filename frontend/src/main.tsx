import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";

window.addEventListener("error", (event) => {
  console.error("Breast Cancer Companion Global Error Caught:", event.error || event.message);
});

window.addEventListener("unhandledrejection", (event) => {
  console.error("Breast Cancer Companion Global Unhandled Promise Rejection:", event.reason);
});

try {
  const container = document.getElementById("root");
  if (!container) {
    console.error("Breast Cancer Companion Boot Error: Root container element '#root' not found in DOM!");
  } else {
    const root = createRoot(container);
    root.render(<App />);
  }
} catch (error) {
  console.error("Breast Cancer Companion Sync Render Boot Crash:", error);
}
