import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useLocation } from "react-router-dom";
import ParticleCanvas from "@/components/particles/ParticleCanvas";

const pageTransition = {
    initial: { opacity: 0, y: 8, scale: 0.995 },
    animate: { opacity: 1, y: 0, scale: 1, transition: { duration: 0.45, ease: "easeOut" } },
    exit: { opacity: 0, y: -8, scale: 0.995, transition: { duration: 0.28, ease: "easeOut" } },
};

const GlobalLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const location = useLocation();

    return (
        <div className="min-h-screen relative bg-white text-foreground">
            {location.pathname === "/" && <ParticleCanvas />}
            <div style={{ position: "relative", zIndex: 10 }}>
                <AnimatePresence mode="wait">
                    <motion.main
                        key={location.pathname || "app"}
                        initial="initial"
                        animate="animate"
                        exit="exit"
                        variants={pageTransition}
                        className="container mx-auto px-4 py-8"
                    >
                        {children}
                    </motion.main>
                </AnimatePresence>
            </div>
        </div>
    );
};

export default GlobalLayout;
