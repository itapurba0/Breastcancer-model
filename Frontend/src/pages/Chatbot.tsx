import Header from "@/components/layout/Header";
import ChatInterface from "@/components/chatbot/ChatInterface";
import { Info, ShieldCheck, Clock } from "lucide-react";

const Chatbot = () => {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      
      <main className="container mx-auto px-4 py-8 md:py-12">
        <div className="max-w-4xl mx-auto">
          {/* Page Header */}
          <div className="text-center mb-8">
            <h1 className="text-3xl md:text-4xl font-bold text-foreground mb-3">
              Medical Chat Assistant
            </h1>
            <p className="text-muted-foreground max-w-xl mx-auto">
              Ask questions about breast cancer, screening procedures, symptoms, and get reliable health information.
            </p>
          </div>

          {/* Info Cards */}
          <div className="grid sm:grid-cols-3 gap-4 mb-8">
            <div className="flex items-center gap-3 bg-card rounded-xl p-4 border border-border">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-accent">
                <Info className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-sm font-medium text-foreground">Informational Only</p>
                <p className="text-xs text-muted-foreground">Not medical advice</p>
              </div>
            </div>
            <div className="flex items-center gap-3 bg-card rounded-xl p-4 border border-border">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-accent">
                <ShieldCheck className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-sm font-medium text-foreground">Private & Secure</p>
                <p className="text-xs text-muted-foreground">Conversations encrypted</p>
              </div>
            </div>
            <div className="flex items-center gap-3 bg-card rounded-xl p-4 border border-border">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-accent">
                <Clock className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-sm font-medium text-foreground">24/7 Available</p>
                <p className="text-xs text-muted-foreground">Always here to help</p>
              </div>
            </div>
          </div>

          {/* Chat Interface */}
          <ChatInterface />
        </div>
      </main>
    </div>
  );
};

export default Chatbot;
