import { Link } from "react-router-dom";
import { ArrowRight, Microscope, MessageCircle, ShieldCheck, CheckCircle2 } from "lucide-react";
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

const features = [
  {
    icon: Microscope,
    title: "Image Analysis",
    description:
      "Upload ultrasound images for instant AI-powered screening with explainable Grad-CAM visualizations.",
  },
  {
    icon: MessageCircle,
    title: "Medical Chat",
    description:
      "Ask questions about breast health, screening guidelines, and risk factors through our retrieval-augmented clinical assistant.",
  },
  {
    icon: ShieldCheck,
    title: "Privacy & Security",
    description:
      "All uploads are processed in-memory and never stored. Your medical data remains fully under your control.",
  },
];

const Index = () => {
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
          {/* Hero */}
          <motion.div variants={itemVariants} className="max-w-3xl space-y-6">
            <p className="text-sm font-semibold tracking-wide text-primary/80 font-sans uppercase">
              AI-powered breast cancer screening
            </p>
            <h1 className="text-3xl sm:text-4xl md:text-7xl font-heading font-bold tracking-tight text-foreground leading-tight">
              Early detection saves lives
            </h1>
            <p className="text-base md:text-lg text-muted-foreground font-sans max-w-xl leading-relaxed">
              Classify breast tissue images in seconds with our deep learning model and get
              clear, explained results — designed to support, not replace, clinical judgment.
            </p>
            <div className="flex flex-col sm:flex-row items-start gap-4 pt-2">
              <Link to="/classification">
                <Button className="bg-primary text-primary-foreground h-12 px-6 font-sans font-bold tracking-wide rounded-full flex items-center gap-2 hover:scale-105 transition-all duration-300 shadow-sm">
                  <Microscope className="h-4 w-4" />
                  Start scan
                  <ArrowRight className="h-3.5 w-3.5 ml-0.5" />
                </Button>
              </Link>
              <Link to="/chatbot">
                <Button
                  variant="outline"
                  className="border border-primary/20 text-primary h-12 px-6 font-sans font-bold tracking-wide rounded-full flex items-center gap-2 hover:scale-105 transition-all duration-300"
                >
                  <MessageCircle className="h-4 w-4" />
                  Chat with assistant
                </Button>
              </Link>
            </div>
          </motion.div>

          {/* Feature cards */}
          <motion.div variants={itemVariants} className="space-y-4">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={index}
                  whileHover={{
                    y: -2,
                    transition: { type: "tween", ease: "easeOut", duration: 0.3 },
                  }}
                  className="glass-panel rounded-2xl p-5 sm:p-6 flex items-start gap-5 shadow-sm border border-primary/10"
                >
                  <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-sage/60 border border-primary/10">
                    <Icon className="h-5 w-5 text-primary" />
                  </div>
                  <div className="space-y-1.5">
                    <h3 className="font-heading font-bold tracking-tight text-foreground text-sm sm:text-base">
                      {feature.title}
                    </h3>
                    <p className="text-xs sm:text-sm text-muted-foreground font-sans leading-relaxed">
                      {feature.description}
                    </p>
                  </div>
                </motion.div>
              );
            })}
          </motion.div>

          {/* Disclaimer */}
          <motion.div
            variants={itemVariants}
            className="flex items-center justify-center gap-3 py-4"
          >
            <CheckCircle2 className="h-4 w-4 text-primary/50 shrink-0" />
            <p className="text-xs text-muted-foreground/70 font-sans text-center max-w-lg leading-relaxed">
              For educational and research purposes only. Always consult a qualified healthcare
              provider for diagnosis and treatment decisions.
            </p>
          </motion.div>
        </motion.div>
      </main>

      {/* Footer */}
      <footer className="py-10 border-t border-primary/10 relative z-10 bg-muted/30">
        <div className="container mx-auto px-4 sm:px-6 max-w-5xl flex flex-col md:flex-row items-center justify-between gap-4">
          <span className="footer-small font-mono text-muted-foreground">
            &copy; 2026 Breast Cancer Companion. All rights reserved.
          </span>
          <span className="footer-small font-mono text-muted-foreground flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-primary animate-pulse" />
            System online
          </span>
        </div>
      </footer>
    </div>
  );
};

export default Index;
