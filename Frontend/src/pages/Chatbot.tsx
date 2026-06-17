import { motion } from "framer-motion";
import Header from "@/components/layout/Header";
import ChatInterface from "@/components/chatbot/ChatInterface";
import HeroCanvas from "@/components/layout/HeroCanvas";

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
              Ask me about breast cancer diagnosis, treatment options, or screening guidelines.
            </p>
          </div>

          {/* Frosted Chat Interface Console */}
          <ChatInterface />
        </motion.div>
      </main>
    </div>
  );
};

export default Chatbot;
