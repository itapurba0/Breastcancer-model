import { motion } from "framer-motion";
import { Upload, Cpu, CheckCircle } from "lucide-react";
import Header from "@/components/layout/Header";
import ImageUploader from "@/components/classification/ImageUploader";
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

const steps = [
  {
    icon: Upload,
    title: "Upload",
    description: "Drop a mammography or ultrasound image",
  },
  {
    icon: Cpu,
    title: "Analyze",
    description: "Neural layers process the tissue in real time",
  },
  {
    icon: CheckCircle,
    title: "Results",
    description: "Classification, heatmap, and triage assessment",
  },
];

const Classification = () => {
  return (
    <div className="min-h-screen bg-transparent text-foreground relative selection:bg-secondary/40 selection:text-foreground">

      <div className="print:hidden">
        <HeroCanvas />

        <Header />
      </div>

      <main className="container mx-auto px-4 sm:px-6 py-12 md:py-16 max-w-3xl relative z-10">
        <motion.div
          initial="hidden"
          animate="visible"
          variants={pageVariants}
          className="space-y-10"
        >
          {/* Page Header */}
          <div className="space-y-3">
            <p className="text-xs font-semibold tracking-widest text-primary/70 uppercase font-sans">
              AI-powered analysis
            </p>
            <h1 className="text-2xl sm:text-3xl font-heading font-bold tracking-tight text-foreground">
              Breast tissue classification
            </h1>
            <p className="text-sm sm:text-base text-muted-foreground font-sans leading-relaxed max-w-xl">
              Upload a mammography or ultrasound image for AI-powered screening with
              explainable heatmaps and clinical triage recommendations.
            </p>
          </div>

          <ImageUploader />

          {/* How It Works */}
          <div className="glass-panel rounded-2xl p-5 sm:p-6 border border-primary/10 bg-white/60">
            <p className="text-xs font-semibold tracking-widest text-muted-foreground uppercase font-sans mb-4">
              How it works
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 sm:gap-6">
              {steps.map((step, i) => {
                const Icon = step.icon;
                return (
                  <div key={i} className="flex items-start gap-3">
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary text-xs font-bold">
                      {i + 1}
                    </div>
                    <div className="space-y-0.5 pt-1">
                      <div className="flex items-center gap-1.5">
                        <Icon className="h-3.5 w-3.5 text-primary" />
                        <span className="text-sm font-semibold text-foreground">{step.title}</span>
                      </div>
                      <p className="text-xs text-muted-foreground leading-relaxed">{step.description}</p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </motion.div>
      </main>
    </div>
  );
};

export default Classification;
