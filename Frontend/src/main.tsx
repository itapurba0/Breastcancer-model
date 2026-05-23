import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";

console.log("ClassifierAI Main: Bootstrapping application...");

window.addEventListener("error", (event) => {
  console.error("ClassifierAI Global Error Caught:", event.error || event.message);
});

window.addEventListener("unhandledrejection", (event) => {
  console.error("ClassifierAI Global Unhandled Promise Rejection:", event.reason);
});

try {
  const container = document.getElementById("root");
  if (!container) {
    console.error("ClassifierAI Boot Error: Root container element '#root' not found in DOM!");
  } else {
    const root = createRoot(container);
    console.log("ClassifierAI Main: React root created, mounting <App />...");
    root.render(<App />);
    console.log("ClassifierAI Main: Render call completed.");
  }
} catch (error) {
  console.error("ClassifierAI Sync Render Boot Crash:", error);
}
