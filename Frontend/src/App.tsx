import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import Classification from "./pages/Classification";
import Chatbot from "./pages/Chatbot";
import NotFound from "./pages/NotFound";
import GlobalLayout from "@/components/layout/GlobalLayout";

const queryClient = new QueryClient();

const App = () => {
  console.log("ClassifierAI: Rendering <App /> component tree...");
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <GlobalLayout>
            <Routes>
              <Route path="/" element={<Index />} />
              <Route path="/classification" element={<Classification />} />
              <Route path="/chatbot" element={<Chatbot />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </GlobalLayout>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
};

export default App;
