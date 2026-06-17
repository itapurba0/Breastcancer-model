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
      ease: "easeOut",
      duration: 0.55,
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

      <main className="container mx-auto px-4 sm:px-6 py-12 md:py-16 max-w-4xl relative z-10">
        <motion.div
          initial="hidden"
          animate="visible"
          variants={pageVariants}
          className="space-y-12"
        >
          <ImageUploader />
        </motion.div>
      </main>
    </div>
  );
};

export default Classification;
