import { useCallback } from "react";
import {
  X, Image as ImageIcon, Loader2, Microscope,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface UploadPanelProps {
  selectedImage: string | null;
  selectedFile: File | null;
  isDragging: boolean;
  isAnalyzing: boolean;
  analysisStep: number;
  onDragOver: (e: React.DragEvent) => void;
  onDragLeave: (e: React.DragEvent) => void;
  onDrop: (e: React.DragEvent) => void;
  onFileSelect: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onClear: () => void;
  onAnalyze: () => void;
}

const UploadPanel = ({
  selectedImage,
  selectedFile,
  isDragging,
  isAnalyzing,
  analysisStep,
  onDragOver,
  onDragLeave,
  onDrop,
  onFileSelect,
  onClear,
  onAnalyze,
}: UploadPanelProps) => {
  return (
    <div className="space-y-4">
      <motion.div
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        whileHover={!selectedImage ? { y: -2 } : undefined}
        transition={{ type: "tween", ease: [0.25, 1, 0.5, 1], duration: 0.3 }}
        className={cn(
          "rounded-3xl p-8 overflow-hidden relative bg-white border print:hidden min-h-[280px]",
          isDragging ? "border-primary/40 bg-primary/5 soft-shadow-md" : "border-brand/15 hover:border-brand/30",
          selectedImage && "border-brand/25"
        )}
      >
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 bg-primary/5 rounded-full blur-3xl pointer-events-none" />

        {!selectedImage ? (
          <div className="flex flex-col items-center gap-6 py-8 sm:py-12 relative z-10">
            <div className="flex h-16 w-16 sm:h-20 sm:w-20 items-center justify-center rounded-full bg-primary/10 border-primary/20 soft-shadow-sm transition-colors duration-300 hover:bg-secondary/10">
              <Microscope className="h-7 w-7 sm:h-9 sm:w-9 text-primary stroke-[2px]" />
            </div>
            <div className="text-center space-y-2">
              <h2 className="text-lg sm:text-xl font-bold text-foreground font-heading tracking-tight">
                Upload mammography image
              </h2>
              <p className="text-sm sm:text-base text-muted-foreground font-sans max-w-sm font-semibold">
                Drag and drop or click to browse
              </p>
            </div>
            <input
              type="file"
              accept="image/*"
              onChange={onFileSelect}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer min-h-[44px]"
              aria-label="Upload mammography image"
            />
            <div className="flex items-center gap-2 px-4 py-1.5 rounded-full bg-muted border border-brand/15 text-xs text-muted-foreground">
              <ImageIcon className="h-3.5 w-3.5 text-primary" />
              <span>Supports: JPEG, PNG</span>
            </div>
          </div>
        ) : (
          <div className="space-y-6 relative z-10">
            <div className="relative rounded-3xl overflow-hidden bg-muted/70 aspect-video soft-shadow-md border border-brand/15 flex items-center justify-center">
              <img
                src={selectedImage}
                alt="Selected tissue scan"
                className="max-h-full max-w-full object-contain"
              />
              {isAnalyzing && (
                <div className="absolute inset-0 bg-highlight/[0.02] pointer-events-none overflow-hidden z-10">
                  <motion.div
                    initial={{ y: "-10%" }}
                    animate={{ y: "110%" }}
                    transition={{ duration: 1.8, ease: "easeInOut" }}
                    className="w-full h-1.5 bg-gradient-to-r from-transparent via-highlight to-transparent"
                  />
                  <div className="absolute inset-0 bg-gradient-to-b from-highlight/0 via-highlight/6 to-highlight/0 opacity-30" />
                </div>
              )}
              <button
                onClick={onClear}
                disabled={isAnalyzing}
                aria-label="Remove image"
                className="absolute top-4 right-4 p-3 rounded-3xl bg-white/95 text-foreground border border-brand/20 hover:bg-muted/95 transition-colors duration-200 z-20 disabled:opacity-50 soft-shadow-sm min-h-[44px] min-w-[44px] flex items-center justify-center cursor-pointer"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-t border-brand/10 pt-5">
              <p className="text-xs sm:text-sm text-muted-foreground font-mono truncate max-w-full sm:max-w-[280px] font-bold">
                {selectedFile?.name}
              </p>
              <Button
                onClick={onAnalyze}
                disabled={isAnalyzing}
                className="h-12 w-full sm:w-auto px-6 bg-secondary hover:bg-secondary/95 border border-secondary/40 text-foreground transition-colors duration-200 font-sans font-semibold rounded-3xl flex items-center gap-2 soft-shadow-sm"
              >
                {isAnalyzing ? (
                  <><Loader2 className="h-4 w-4 animate-spin" /> Analyzing…</>
                ) : (
                  <><Microscope className="h-4 w-4" /> Analyze Image</>
                )}
              </Button>
            </div>
          </div>
        )}
      </motion.div>

      {/* Progress Steps */}
      <AnimatePresence>
        {isAnalyzing && (
          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -15 }}
            className="rounded-2xl p-6 border border-primary/10 bg-white"
          >
            <div className="space-y-3">
              {["Receiving image", "Processing neural layers", "Generating heatmap", "Compiling results"].map((step, i) => (
                <div key={step} className="flex items-center gap-3">
                  <div className={cn(
                    "flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-bold",
                    analysisStep > i ? "bg-primary text-white" : analysisStep === i ? "bg-primary/10 text-primary animate-pulse" : "bg-muted text-muted-foreground"
                  )}>
                    {analysisStep > i ? "\u2713" : i + 1}
                  </div>
                  <span className={cn("text-sm font-sans", analysisStep >= i ? "text-foreground" : "text-muted-foreground")}>
                    {step}
                  </span>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {isAnalyzing && (
        <div className="animate-pulse bg-muted rounded-2xl h-64 w-full" />
      )}
    </div>
  );
};

export default UploadPanel;
