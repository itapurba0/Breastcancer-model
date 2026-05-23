import { useState, useCallback } from "react";
import { Upload, X, Image as ImageIcon, Loader2, AlertCircle, CheckCircle, Microscope, Sparkles, Activity, ShieldCheck, HeartPulse } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ClassificationResult {
  prediction: string;
  confidence: number;
  details: string;
  gradcam?: string;
  triage?: {
    tier: string;
    recommendation: string;
    rationale: string;
    confidence_score: number;
  };
}

const ImageUploader = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<ClassificationResult | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      processFile(file);
    }
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const processFile = (file: File) => {
    setSelectedFile(file);
    setResult(null);
    const reader = new FileReader();
    reader.onload = (e) => {
      setSelectedImage(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const clearImage = () => {
    setSelectedImage(null);
    setSelectedFile(null);
    setResult(null);
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setResult(null);
    try {
      const formData = new FormData();
      formData.append("file", selectedFile, selectedFile.name);

      const res = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        setResult({
          prediction: "Analysis Failed",
          confidence: 0,
          details: `HTTP error: ${res.status} ${text}`,
        });
        return;
      }

      const json = await res.json();

      const rawConfidence = Number(json.confidence ?? 0);
      const normalizedConfidence = Number.isFinite(rawConfidence)
        ? Math.round((rawConfidence <= 1 ? rawConfidence * 100 : rawConfidence))
        : 0;

      setResult({
        prediction: json.predicted ?? json.prediction ?? json.label ?? "Unknown",
        confidence: normalizedConfidence,
        details: json.details ?? JSON.stringify(json),
        gradcam: json.gradcam_image,
        triage: json.triage,
      });
    } catch (err) {
      setResult({
        prediction: "Analysis Failed",
        confidence: 0,
        details: String(err),
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto space-y-8">
      {/* 1. Frosted Uploader Panel */}
      <motion.div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        whileHover={{ y: -4, scale: 1.01 }}
        transition={{ type: "tween", ease: "easeOut", duration: 0.35 }}
        className={cn(
          "glass-panel rounded-3xl p-8 overflow-hidden relative transition-transform duration-300 bg-white",
          isDragging ? "border-highlight bg-muted/50 soft-shadow-md" : "border-brand/15 hover:border-brand/40",
          selectedImage && "border-brand/25"
        )}
      >
        {/* Subtle light corner glows */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 bg-[#FFE082]/5 rounded-full blur-3xl pointer-events-none" />

        {!selectedImage ? (
          <div className="flex flex-col items-center gap-6 py-12 relative z-10">
            <div className="flex h-20 w-20 items-center justify-center rounded-3xl bg-muted border border-brand/20 soft-shadow-sm group transition-transform duration-500 hover:scale-105 hover:bg-highlight/10">
              <Upload className="h-9 w-9 text-brand stroke-[2px]" />
            </div>
            <div className="text-center space-y-2">
              <h2 className="text-xl font-bold text-[#333333] font-heading tracking-tight">
                MAMMOGRAPHY_IMAGE_DROP_ZONE
              </h2>
              <p className="text-xs text-[#616161] font-sans max-w-sm font-semibold">
                Drag & drop digital DICOM/ultrasound vectors here, or browse local systems.
              </p>
            </div>
            <input
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <div className="flex items-center gap-2 px-4 py-1.5 rounded-full bg-muted border border-brand/15 text-[10px] text-muted-foreground font-mono font-bold">
              <ImageIcon className="h-3.5 w-3.5 text-brand" />
              <span>SUPPORTED_VECTORS: JPG, PNG, DCM</span>
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

              {/* Holographic Laser Grid Scanner Overlay */}
              {isAnalyzing && (
                <div className="absolute inset-0 bg-highlight/[0.02] pointer-events-none overflow-hidden z-10">
                  <motion.div
                    initial={{ y: "-10%" }}
                    animate={{ y: "110%" }}
                    transition={{
                      repeat: Infinity,
                      repeatType: "reverse",
                      duration: 1.8,
                      ease: "easeInOut",
                    }}
                    className="w-full h-1.5 bg-gradient-to-r from-transparent via-highlight to-transparent"
                  />
                  <div className="absolute inset-0 bg-gradient-to-b from-highlight/0 via-highlight/6 to-highlight/0 opacity-30" />
                </div>
              )}

              <button
                onClick={clearImage}
                disabled={isAnalyzing}
                className="absolute top-4 right-4 p-2.5 rounded-3xl bg-white/95 text-foreground border border-brand/20 hover:bg-muted/95 transition-transform duration-200 hover:scale-105 z-20 disabled:opacity-50 soft-shadow-sm"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="flex items-center justify-between border-t border-slate-100 pt-5">
              <p className="text-xs text-[#616161] font-mono truncate max-w-[280px] font-bold">
                FILE: {selectedFile?.name}
              </p>
              <Button
                onClick={analyzeImage}
                disabled={isAnalyzing}
                className="h-12 px-6 bg-highlight hover:bg-highlight/95 border border-highlight/40 text-foreground transition-transform duration-300 font-sans font-semibold rounded-3xl flex items-center gap-2 soft-shadow-sm"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Executing Deep Inference...
                  </>
                ) : (
                  <>
                    <Microscope className="h-4 w-4" />
                    Initialize Deep Inference
                  </>
                )}
              </Button>
            </div>
          </div>
        )}
      </motion.div>

      {/* 2. Interactive Analytical Scan Progress */}
      <AnimatePresence>
        {isAnalyzing && (
          <motion.div
            initial={{ opacity: 0, y: 15, filter: "blur(5px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            exit={{ opacity: 0, y: -15, filter: "blur(5px)" }}
            transition={{ duration: 0.45, ease: "easeOut" }}
            className="glass-panel rounded-3xl p-6 border border-brand/15 soft-shadow-md relative overflow-hidden bg-white"
          >
            {/* Pulsing light */}
            <div className="absolute inset-0 bg-highlight/[0.02] animate-pulse pointer-events-none" />

            <div className="flex items-center gap-4 relative z-10">
              <div className="relative flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-white border border-brand/15 soft-shadow-sm">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ repeat: Infinity, duration: 3, ease: "linear" }}
                  className="absolute inset-1 border-t-2 border-r-2 border-brand rounded-2xl"
                />
                <Activity className="h-5 w-5 text-brand" />
              </div>
              <div className="flex-1">
                <div className="flex justify-between items-center mb-1">
                  <p className="font-bold text-[#333333] font-heading text-sm">COGNITIVE_SCANNER_CALC</p>
                  <span className="text-[10px] font-bold text-[#455A64] bg-[#FFE082]/30 border border-[#FFE082]/60 px-3 py-0.5 rounded-full animate-pulse uppercase tracking-wider font-mono">Calculating</span>
                </div>
                <p className="text-xs text-[#616161] font-sans leading-relaxed font-semibold">
                  Mapping visual tissue saliencies & processing deep layers...
                </p>
              </div>
            </div>

            {/* Log logs */}
            <div className="mt-5 relative">
              <div className="h-2 bg-muted rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: "0%" }}
                  animate={{ width: "95%" }}
                  transition={{
                    duration: 5.5,
                    ease: "easeOut",
                  }}
                  className="h-full bg-gradient-to-r from-[#78909C] via-[#FFE082] to-[#78909C] rounded-full"
                />
              </div>
              <div className="flex justify-between text-[9px] text-[#616161] mt-2.5 font-mono tracking-wide font-bold">
                <span>[STACK_01] TENSOR_NORMALIZATION</span>
                <span>[STACK_02] GRAD_CAM_SALIENCY</span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 3. Frosted Clinical Result Panel */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 25, filter: "blur(10px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            exit={{ opacity: 0, y: -25, filter: "blur(10px)" }}
            transition={{ duration: 0.55, ease: "easeOut" }}
            className="glass-panel rounded-3xl p-8 md:p-10 border border-brand/20 soft-shadow-lg relative overflow-hidden space-y-8 bg-white"
          >
            {/* Background highlight */}
            <div className="absolute -top-32 -right-32 w-64 h-64 bg-[#FFE082]/5 rounded-full blur-3xl pointer-events-none" />

            {/* Top Section: Status Badge & Prediction Verdict */}
            <div className="flex flex-col md:flex-row items-start gap-6 border-b border-slate-100 pb-8">
              {(() => {
                const isFailed = result.prediction.toLowerCase().includes("fail");
                const isNormal = result.prediction.toLowerCase().includes("normal");
                const isBenign = result.prediction.toLowerCase().includes("benign");

                let accentColor = "text-red-700 border-red-200 bg-red-50/70";
                if (isNormal) {
                  accentColor = "text-[#37474F] border-[#78909C]/20 bg-[#78909C]/10";
                } else if (isBenign) {
                  accentColor = "text-[#004D40] border-[#E0F2F1]/80 bg-[#E0F2F1]/80";
                }

                return (
                  <>
                    <div className={cn(
                      "flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl border",
                      accentColor
                    )}>
                      {isNormal || isBenign ? (
                        <CheckCircle className="h-6 w-6 stroke-[2px]" />
                      ) : (
                        <AlertCircle className="h-6 w-6 stroke-[2px]" />
                      )}
                    </div>
                    <div className="flex-1 space-y-4">
                      <div className="flex flex-wrap items-center gap-3">
                        <span className="text-[10px] font-mono text-[#616161] uppercase tracking-widest font-bold">[ DIAGNOSTIC_PREDICTION ]</span>
                        <h3 className="text-2xl md:text-3xl font-black text-[#333333] font-heading tracking-tight w-full md:w-auto leading-none">
                          {result.prediction.toUpperCase()}
                        </h3>
                        {!isFailed && (
                          <span className="px-3.5 py-1 rounded-full text-xs font-mono font-bold tracking-tight border border-[#FFE082] bg-[#FFE082]/30 text-[#455A64]">
                            {result.confidence}% CONFIDENCE_SCORE
                          </span>
                        )}
                      </div>

                      <p className="text-xs md:text-sm text-[#616161] leading-relaxed font-sans font-semibold">
                        Explainable visual overlays (Grad-CAM) are calculated to map mathematical weights. The heat signals denote visual tissue coordinates that mathematically informed the neural output.
                      </p>
                    </div>
                  </>
                );
              })()}
            </div>

            {/* Middle Section: Central Solitary Grad-CAM Heatmap Visual Focus */}
            <div className="flex flex-col items-center justify-center gap-4 w-full">
              <div className="w-full max-w-xl glass-panel rounded-3xl p-6 border border-[#78909C]/15 shadow-sm bg-slate-50/50 flex flex-col gap-4">
                <span className="text-[10px] font-mono text-[#616161] uppercase tracking-widest flex items-center justify-center gap-2 font-bold">
                  <Sparkles className="h-4 w-4 text-[#78909C] animate-pulse" />
                  EXPLAINABLE_GRAD_CAM_MAP
                </span>
                <div className="overflow-hidden rounded-2xl bg-slate-100/80 flex items-center justify-center h-80 border border-[#78909C]/10 relative shadow-inner">
                  {result.gradcam ? (
                    <img
                      src={result.gradcam}
                      alt="Tissue activation heatmap mapping"
                      className="max-h-full max-w-full object-contain"
                    />
                  ) : (
                    <div className="px-4 text-center text-xs text-[#616161] font-sans font-semibold">
                      Grad-CAM explanation mapping is not available for this session index.
                    </div>
                  )}
                  <div className="absolute bottom-3 left-3 px-3 py-1 rounded-full bg-white/90 backdrop-blur-sm text-[9px] font-mono text-[#455A64] border border-[#78909C]/20 font-bold shadow-2xs">
                    HEATMAP_SALIENCY_VIEW
                  </div>
                </div>
              </div>
            </div>

            {/* Bottom Section: Risk Triage Index & Recommendations */}
            {result.triage && (
              <div className="p-6 md:p-8 rounded-[2rem] bg-slate-50/60 border border-slate-100 space-y-4">
                <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                  <span className="text-[10px] font-mono text-[#616161] uppercase tracking-widest font-bold">[ RISK_TRIAGE_INDEX ]</span>
                  <span className={cn(
                    "px-3 py-1 rounded-full text-[10px] font-mono font-bold uppercase tracking-wider",
                    result.triage.tier === "high concern"
                      ? "bg-red-50 text-red-700 border border-red-200"
                      : result.triage.tier === "moderate confidence"
                        ? "bg-amber-50 text-amber-800 border border-amber-200"
                        : "bg-slate-100 text-[#455A64] border border-[#78909C]/25"
                  )}>
                    {result.triage.tier}
                  </span>
                </div>
                <div className="space-y-2">
                  <h4 className="text-sm md:text-base font-bold text-[#333333] font-heading tracking-tight leading-snug">
                    Recommendation: {result.triage.recommendation}
                  </h4>
                  <p className="text-xs md:text-sm text-[#616161] leading-relaxed font-sans font-semibold">
                    {result.triage.rationale}
                  </p>
                </div>
              </div>
            )}

            {/* Footer Compliance Warnings */}
            <div className="pt-6 border-t border-slate-100 flex items-start gap-3">
              <ShieldCheck className="h-5 w-5 text-[#78909C] shrink-0 mt-0.5" />
              <p className="text-[10px] md:text-xs text-[#616161] font-sans leading-relaxed font-semibold">
                All inputs are anonymized prior to layer convolutions. Predictions generated are triage aids and must be validated through standard visual biopsies by oncologist teams.
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ImageUploader;
