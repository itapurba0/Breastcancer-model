import Header from "@/components/layout/Header";
import ChatInterface from "@/components/chatbot/ChatInterface";
import HeroCanvas from "@/components/layout/HeroCanvas";

const Chatbot = () => {
  return (
    <div className="min-h-screen bg-transparent text-foreground relative selection:bg-secondary/40 selection:text-foreground">
      <HeroCanvas />
      <Header />
      
      <main id="main-content" className="relative z-10 min-h-screen px-4 py-12 md:py-16">
        <div className="max-w-4xl mx-auto space-y-8">
          <div className="text-center space-y-2">
            <h1 className="text-2xl md:text-3xl font-heading font-bold text-foreground tracking-tight">
              Medical Chat
            </h1>
            <p className="text-sm text-muted-foreground font-sans max-w-md mx-auto">
              Ask me about breast cancer diagnosis, treatment options, or screening guidelines.
            </p>
          </div>
          <ChatInterface />
        </div>
      </main>
    </div>
  );
};

export default Chatbot;
