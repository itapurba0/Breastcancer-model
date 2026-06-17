import { motion } from "framer-motion";
import Header from "@/components/layout/Header";
import ChatInterface from "@/components/chatbot/ChatInterface";
import HeroCanvas from "@/components/layout/HeroCanvas";
import { Info, ShieldCheck, Clock } from "lucide-react";

const pageVariants = {
  hidden: { opacity: 0, filter: "blur(10px)" },
  visible: {
    opacity: 1,
    filter: "blur(0px)",
    transition: {
      type: "tween",
      ease: "easeOut",
      duration: 0.55,
    },
  },
};

const Chatbot = () => {
  return (
    <div className="min-h-screen bg-transparent text-foreground relative selection:bg-secondary/40 selection:text-foreground">
      <HeroCanvas />
      <Header />
      
      <main className="container mx-auto px-4 sm:px-6 py-12 md:py-16 max-w-4xl relative z-10">
        <motion.div
          initial="hidden"
          animate="visible"
          variants={pageVariants}
          className="space-y-10"
        >
          <div className="text-center space-y-4">
            <h1 className="text-3xl md:text-5xl font-extrabold text-foreground tracking-[-0.03em] font-heading">
              Neural Assistant Terminal
            </h1>
            <p className="text-sm md:text-base text-muted-foreground max-w-xl mx-auto leading-relaxed font-sans">
              Interact with our support agent to clear oncological screening questions, parse model activation layers, and query literature citations.
            </p>
          </div>

          <div className="grid sm:grid-cols-3 gap-4">
            <div className="glass-panel glass-panel-hover rounded-[1.5rem] p-4 flex items-center gap-3.5 border border-brand/10 shadow-sm">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-muted border border-brand/10">
                <Info className="h-5 w-5 text-brand" />
              </div>
              <div className="space-y-0.5">
                <p className="text-xs font-bold text-foreground font-mono uppercase tracking-wide">INFORMATIONAL_ONLY</p>
                <p className="text-xs text-muted-foreground font-sans">No clinical diagnostic advice</p>
              </div>
            </div>

            <div className="glass-panel glass-panel-hover rounded-[1.5rem] p-4 flex items-center gap-3.5 border border-brand/10 shadow-sm">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-muted border border-brand/10">
                <ShieldCheck className="h-5 w-5 text-brand" />
              </div>
              <div className="space-y-0.5">
                <p className="text-xs font-bold text-foreground font-mono uppercase tracking-wide">SECURE_RAG_NODE</p>
                <p className="text-xs text-muted-foreground font-sans">256-bit encryption active</p>
              </div>
            </div>

            <div className="glass-panel glass-panel-hover rounded-[1.5rem] p-4 flex items-center gap-3.5 border border-brand/10 shadow-sm">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-muted border border-brand/10">
                <Clock className="h-5 w-5 text-brand" />
              </div>
              <div className="space-y-0.5">
                <p className="text-xs font-bold text-foreground font-mono uppercase tracking-wide">LATENCY_INDEX_LOW</p>
                <p className="text-xs text-muted-foreground font-sans">Literature sync online</p>
              </div>
            </div>
          </div>

          {/* Frosted Chat Interface Console */}
          <ChatInterface />
        </motion.div>
      </main>
    </div>
  );
};

export default Chatbot;
