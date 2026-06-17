import { motion } from "framer-motion";
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
      ease: [0.25, 1, 0.5, 1],
      duration: 0.45,
    },
  },
};

const Classification = () => {
  return (
    <div className="min-h-screen bg-transparent text-foreground relative selection:bg-secondary/40 selection:text-foreground">
      <div className="print:hidden">
        <HeroCanvas />
        <Header />
      </div>

      <main className="container mx-auto px-4 sm:px-6 py-12 md:py-16 max-w-7xl relative z-10">
        <motion.div
          initial="hidden"
          animate="visible"
          variants={pageVariants}
        >
          {/* Page Header — left-aligned, not centered, hidden on print */}
          <div className="space-y-3 mb-10 print:hidden">
            <p className="text-xs font-semibold tracking-widest text-primary/70 uppercase font-sans">
              AI-powered analysis
            </p>
            <h1 className="text-2xl sm:text-3xl font-heading font-bold tracking-tight text-foreground text-balance">
              Breast tissue classification
            </h1>
            <p className="text-sm sm:text-base text-muted-foreground font-sans leading-relaxed max-w-xl">
              Upload a mammography or ultrasound image for AI-powered screening with
              explainable heatmaps and clinical triage recommendations.
            </p>
          </div>

          {/* Full-width Uploader */}
          <ImageUploader />
        </motion.div>
      </main>
    </div>
  );
};

export default Classification;
