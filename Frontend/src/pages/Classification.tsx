import { motion } from "framer-motion";
import Header from "@/components/layout/Header";
import ImageUploader from "@/components/classification/ImageUploader";
import HeroCanvas from "@/components/layout/HeroCanvas";
import { ShieldCheck, Sparkles, Network } from "lucide-react";

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

const Classification = () => {
  const benefits = [
    { icon: ShieldCheck, text: "HIPAA_ENCRYPTED_TRANSFER" },
    { icon: Sparkles, text: "AI_EXPLAINABLE_GRAD_CAM" },
    { icon: Network, text: "TRIAGE_CLINICAL_ADVICE" },
  ];

  function humanize(token: string) {
    return token
      .replace(/_/g, " ")
      .toLowerCase()
      .replace(/\b\w/g, (c) => c.toUpperCase());
  }

  return (
    <div className="min-h-screen bg-transparent text-[#333333] relative selection:bg-[#FFE082]/40 selection:text-[#333333]">
      {/* 3D breathing Neural Net background void */}
      <HeroCanvas />

      {/* Floating Header */}
      <Header />

      <main className="container mx-auto px-6 py-12 md:py-16 max-w-4xl relative z-10">
        <motion.div
          initial="hidden"
          animate="visible"
          variants={pageVariants}
          className="space-y-12"
        >
          {/* Frosted Heading Panel */}
          {/* <div className="glass-panel rounded-[2rem] p-8 md:p-10 text-center space-y-5 border border-[#78909C]/15 shadow-md relative overflow-hidden">
            <div className="absolute -top-24 -left-24 w-48 h-48 bg-[#FFE082]/5 rounded-full blur-2xl pointer-events-none" />

            <h1 className="text-3xl md:text-5xl font-extrabold text-[#455A64] tracking-[-0.03em] font-heading leading-tight">
              Visual Diagnostic Classification
            </h1>
            <p className="text-sm md:text-base text-[#616161] font-sans max-w-xl mx-auto leading-relaxed font-semibold">
              Feed digital mammography or ultrasonography images to the core convolutional layers.
              The system classifies tissue, generates visual gradient heatmaps, and computes risk parameters.
            </p>

            
            <div className="flex flex-wrap items-center justify-center gap-3 mt-6 pt-5 border-t border-slate-100">
              {benefits.map((benefit, index) => {
                const Icon = benefit.icon;
                return (
                  <div
                    key={index}
                    className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-slate-100 border border-[#78909C]/20 text-[10px] font-mono text-[#455A64] tracking-wide font-bold"
                  >
                    <Icon className="h-3.5 w-3.5 text-[#78909C]" />
                    <span>{humanize(benefit.text)}</span>
                  </div>
                );
              })}
            </div>
          </div>*/}

          {/* Refined Upload Terminal */}
          <ImageUploader />

          {/* Removed: Deep-learning Technical Specs and global disclaimer per UX request */}
        </motion.div>
      </main>
    </div>
  );
};

export default Classification;
