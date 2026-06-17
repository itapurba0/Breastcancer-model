import { Link } from "react-router-dom";
import { ArrowRight, Microscope, MessageCircle, ShieldAlert, Cpu, Heart, CheckCircle2 } from "lucide-react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import Header from "@/components/layout/Header";
import HeroCanvas from "@/components/layout/HeroCanvas";

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.12,
      delayChildren: 0.15,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 15, filter: "blur(10px)" },
  visible: {
    opacity: 1,
    y: 0,
    filter: "blur(0px)",
    transition: {
      type: "tween",
      ease: "easeOut",
      duration: 0.55,
    },
  },
};

const Index = () => {
  const nodes = [
    {
      icon: Microscope,
      title: "MAMMOGRAPHY_INF_SCAN",
      description: "Deep learning tensor classification models providing prompt visual tissue activation reports.",
    },
    {
      icon: MessageCircle,
      title: "NEURAL_MED_CHAT",
      description: "Clinical support LLM chatbot offering instant assistance and general medical screening advice.",
    },
    {
      icon: Cpu,
      title: "EXPLAINABLE_GRAD_CAM",
      description: "Tensors are mapped back into standard visual spaces, overlaying tissue weights and gradients.",
    },
    {
      icon: ShieldAlert,
      title: "HIPAA_DATA_SECURE",
      description: "All uploaded dicoms, ultrasonography, and medical metadata are fully encrypted in-flight.",
    },
  ];

  return (
    <div className="min-h-screen bg-transparent text-foreground relative selection:bg-secondary/40 selection:text-foreground">
      <HeroCanvas />
      <Header />

      <main className="container mx-auto px-4 sm:px-6 py-16 md:py-24 max-w-5xl relative z-10">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="space-y-12"
        >
          <motion.div
            variants={itemVariants}
            className="glass-panel rounded-[2.5rem] p-6 sm:p-8 md:p-14 relative overflow-hidden shadow-lg border border-brand/15"
          >
            <div className="absolute -top-40 -right-40 w-96 h-96 bg-secondary/10 rounded-full blur-3xl pointer-events-none" />
            <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-sage/30 rounded-full blur-3xl pointer-events-none" />

            <div className="max-w-3xl mx-auto text-center space-y-8">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-muted/80 border border-brand/25 text-xs sm:text-sm font-mono text-foreground tracking-wide font-bold">
                <Heart className="h-3.5 w-3.5 animate-pulse text-brand" />
                <span>[ AI_TRIAGE_CORE_v2.1 ]</span>
              </div>

              <h1 className="text-3xl sm:text-4xl md:text-7xl font-extrabold text-foreground leading-tight font-heading tracking-[-0.04em]">
                Deep Learning Triage for
                <span className="block mt-2 bg-gradient-to-r from-foreground via-brand to-brand/70 bg-clip-text text-transparent">
                  Breast Cancer Screening
                </span>
              </h1>

              <p className="text-sm md:text-base text-muted-foreground font-sans max-w-2xl mx-auto leading-relaxed font-semibold">
                ClassifierAI harnesses medical deep convolutional architectures to map visual tissues.
                Upload mammography images for visual gradient explanations and chat with our support model.
              </p>

              <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
                <Link to="/classification">
                  <Button variant="default" className="h-12 px-6 hover:scale-105 transition-all duration-300 font-sans font-bold tracking-wide rounded-full flex items-center gap-2 shadow-sm">
                    <Microscope className="h-4 w-4" />
                    Initialize Scan Routine
                    <ArrowRight className="h-3.5 w-3.5 ml-0.5" />
                  </Button>
                </Link>
                <Link to="/chatbot">
                  <Button variant="outline" className="h-12 px-6 bg-white/80 hover:scale-105 transition-all duration-300 font-sans font-bold tracking-wide rounded-full flex items-center gap-2 shadow-sm">
                    <MessageCircle className="h-4 w-4" />
                    Query Clinical LLM
                  </Button>
                </Link>
              </div>
            </div>
          </motion.div>

          <motion.div
            variants={itemVariants}
            className="grid md:grid-cols-2 gap-4 sm:gap-6"
          >
            {nodes.map((node, index) => {
              const Icon = node.icon;
              return (
                <motion.div
                  key={index}
                  whileHover={{
                    y: -4,
                    scale: 1.01,
                    boxShadow: "0 30px 60px hsl(var(--brand) / 0.12)"
                  }}
                  transition={{ type: "tween", ease: "easeOut", duration: 0.35 }}
                  className="glass-panel rounded-[2rem] p-5 sm:p-7 flex gap-5 shadow-sm border border-brand/15 transition-all duration-300 bg-white"
                >
                  <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-muted border border-brand/20">
                    <Icon className="h-5 w-5 text-brand" />
                  </div>
                  <div className="space-y-2">
                    <h3 className="text-xs font-bold tracking-wider font-mono text-foreground">
                      {node.title}
                    </h3>
                    <p className="text-xs sm:text-sm text-muted-foreground font-sans leading-relaxed font-semibold">
                      {node.description}
                    </p>
                  </div>
                </motion.div>
              );
            })}
          </motion.div>

          <motion.div
            variants={itemVariants}
            className="glass-panel rounded-2xl p-5 flex items-center justify-center gap-3 text-center border border-brand/15 bg-white/70 shadow-sm"
          >
            <CheckCircle2 className="h-4 w-4 text-brand" />
            <p className="text-xs sm:text-sm text-muted-foreground font-sans font-semibold">
              Developed as clinical support assistance. All screening reports must be confirmed by qualified medical oncology staff.
            </p>
          </motion.div>
        </motion.div>
      </main>

      <footer className="py-10 border-t border-brand/10 relative z-10 bg-muted/30">
        <div className="container mx-auto px-4 sm:px-6 max-w-5xl flex flex-col md:flex-row items-center justify-between gap-4">
          <span className="footer-small font-mono text-muted-foreground font-bold">
            &copy; 2026 CLASSIFIER_AI_LABS. All rights reserved.
          </span>
          <span className="footer-small font-mono text-muted-foreground flex items-center gap-2 font-bold">
            <span className="w-2.5 h-2.5 rounded-full bg-teal-500 animate-ping" />
            VIRTUAL_SCAN_NODE_ONLINE
          </span>
        </div>
      </footer>
    </div>
  );
};

export default Index;
