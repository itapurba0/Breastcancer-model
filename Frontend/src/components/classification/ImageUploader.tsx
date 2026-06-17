import { useState, useCallback } from "react";
import {
  X, Image as ImageIcon, Loader2, AlertCircle, CheckCircle,
  Microscope, Activity, ShieldCheck, FileText, Printer, User, FileOutput,
  Eye
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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

interface PatientDetails {
  patientName: string;
  patientAge: string;
  patientId: string;
  clinicalNotes: string;
}

const ImageUploader = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStep, setAnalysisStep] = useState(-1);
  const [result, setResult] = useState<ClassificationResult | null>(null);

  // Report Generation States
  const [reportStep, setReportStep] = useState<"hidden" | "form" | "preview">("hidden");
  const [patientData, setPatientData] = useState<PatientDetails>({
    patientName: "",
    patientAge: "",
    patientId: "",
    clinicalNotes: "",
  });

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
    setReportStep("hidden");
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
    setReportStep("hidden");
  };

  const handlePatientDataChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setPatientData({ ...patientData, [e.target.name]: e.target.value });
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setResult(null);
    setReportStep("hidden");

    try {
      const steps = [0, 1, 2, 3];
      for (const step of steps) {
        await new Promise(r => setTimeout(r, 600));
        setAnalysisStep(step);
      }

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
      setAnalysisStep(-1);
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
          "glass-panel rounded-3xl p-8 overflow-hidden relative transition-transform duration-300 bg-white print:hidden min-h-[280px]",
          isDragging ? "border-primary/40 bg-primary/5 soft-shadow-md" : "border-brand/15 hover:border-brand/40",
          selectedImage && "border-brand/25"
        )}
      >
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 bg-primary/5 rounded-full blur-3xl pointer-events-none" />

        {!selectedImage ? (
          <div className="flex flex-col items-center gap-6 py-8 sm:py-12 relative z-10">
            <div className="flex h-16 w-16 sm:h-20 sm:w-20 items-center justify-center rounded-full bg-primary/10 border-primary/20 soft-shadow-sm group transition-transform duration-500 hover:scale-105 hover:bg-secondary/10">
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
              onChange={handleFileSelect}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer min-h-[44px]"
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
                    transition={{ repeat: Infinity, repeatType: "reverse", duration: 1.8, ease: "easeInOut" }}
                    className="w-full h-1.5 bg-gradient-to-r from-transparent via-highlight to-transparent"
                  />
                  <div className="absolute inset-0 bg-gradient-to-b from-highlight/0 via-highlight/6 to-highlight/0 opacity-30" />
                </div>
              )}
              <button
                onClick={clearImage}
                disabled={isAnalyzing}
                className="absolute top-4 right-4 p-3 rounded-3xl bg-white/95 text-foreground border border-brand/20 hover:bg-muted/95 transition-transform duration-200 hover:scale-105 z-20 disabled:opacity-50 soft-shadow-sm min-h-[44px] min-w-[44px] flex items-center justify-center"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-t border-brand/10 pt-5">
              <p className="text-xs sm:text-sm text-muted-foreground font-mono truncate max-w-full sm:max-w-[280px] font-bold">
                {selectedFile?.name}
              </p>
              <Button
                onClick={analyzeImage}
                disabled={isAnalyzing}
                className="h-12 w-full sm:w-auto px-6 bg-secondary hover:bg-secondary/95 border border-secondary/40 text-foreground transition-transform duration-300 font-sans font-semibold rounded-3xl flex items-center gap-2 soft-shadow-sm"
              >
                {isAnalyzing ? (
                  <><Loader2 className="h-4 w-4 animate-spin" /> Analyzing...</>
                ) : (
                  <><Microscope className="h-4 w-4" /> Analyze Image</>
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
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -15 }}
            className="glass-panel rounded-3xl p-6 border border-primary/10 bg-white"
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

      {/* Skeleton placeholder during analysis */}
      {isAnalyzing && !result && (
        <div className="animate-pulse bg-muted rounded-2xl h-64 w-full" />
      )}

      {/* 3. Main Result Panel & Report Generator */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 25, filter: "blur(10px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            exit={{ opacity: 0, y: -25, filter: "blur(10px)" }}
            transition={{ duration: 0.55, ease: "easeOut" }}
            className="glass-panel rounded-3xl p-8 md:p-10 border border-brand/20 soft-shadow-lg relative overflow-hidden space-y-8 bg-white"
          >
            {/* The standard dashboard UI (Hidden during printing) */}
            <div className="print:hidden space-y-8">

              <div className="flex flex-col md:flex-row items-center gap-6 border-b border-brand/10 pb-8">
                {(() => {
                  const isFailed = result.prediction.toLowerCase().includes("fail");
                  const isNormal = result.prediction.toLowerCase().includes("normal");
                  const isBenign = result.prediction.toLowerCase().includes("benign");

                  const gaugeColor = isNormal || isBenign
                    ? "var(--color-primary)"
                    : isFailed
                      ? "var(--color-muted-foreground)"
                      : "var(--color-secondary)";
                  const circumference = 2 * Math.PI * 36;
                  const dashOffset = circumference - (result.confidence / 100) * circumference;

                  return (
                    <div className="flex items-center gap-6 w-full">
                      {/* Circular Confidence Gauge */}
                      {!isFailed && (
                        <div className="relative shrink-0">
                          <svg width="88" height="88" viewBox="0 0 88 88" className="-rotate-90">
                            <circle cx="44" cy="44" r="36" fill="none" stroke="currentColor" strokeWidth="6"
                              className="text-muted/60" />
                            <motion.circle
                              cx="44" cy="44" r="36" fill="none" stroke={gaugeColor} strokeWidth="6"
                              strokeLinecap="round"
                              strokeDasharray={circumference}
                              initial={{ strokeDashoffset: circumference }}
                              animate={{ strokeDashoffset: dashOffset }}
                              transition={{ duration: 1.2, ease: "easeOut", delay: 0.2 }}
                            />
                          </svg>
                          <div className="absolute inset-0 flex flex-col items-center justify-center">
                            <span className="text-lg font-heading font-bold text-foreground leading-none">{result.confidence}</span>
                            <span className="text-[10px] text-muted-foreground font-sans mt-0.5">%</span>
                          </div>
                        </div>
                      )}

                      {/* Prediction Info */}
                      <div className="flex-1 space-y-2">
                        <div>
                          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide font-sans">Diagnostic result</p>
                          <div className="flex items-center gap-3 mt-1.5">
                            <span className={cn(
                              "text-base font-heading font-bold tracking-tight px-3 py-1 rounded-lg",
                              isFailed ? "text-muted-foreground bg-muted" :
                              isNormal ? "text-primary bg-primary/10" :
                              isBenign ? "text-primary bg-primary/8" :
                              "text-secondary-foreground bg-secondary/90"
                            )}>
                              {result.prediction}
                            </span>
                          </div>
                        </div>
                        {!isFailed && (
                          <p className="text-xs text-muted-foreground font-sans">
                            Model confidence for this classification
                          </p>
                        )}
                      </div>
                    </div>
                  );
                })()}
              </div>

              {reportStep === "hidden" && (
                <>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <div className="flex items-center gap-1.5">
                        <Eye className="h-3.5 w-3.5 text-primary" />
                        <p className="text-xs font-semibold text-foreground uppercase tracking-wide font-sans">Original scan</p>
                      </div>
                      <div className="rounded-2xl overflow-hidden bg-muted/50 flex items-center justify-center aspect-square border border-primary/10">
                        <img src={selectedImage!} alt="Original scan" className="max-h-full max-w-full object-contain" />
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center gap-1.5">
                        <Activity className="h-3.5 w-3.5 text-primary" />
                        <p className="text-xs font-semibold text-foreground uppercase tracking-wide font-sans">AI attention map</p>
                      </div>
                      <div className="rounded-2xl overflow-hidden bg-muted/50 flex items-center justify-center aspect-square border border-primary/10 relative">
                        {result.gradcam ? (
                          <img src={result.gradcam} alt="Heatmap" className="max-h-full max-w-full object-contain mix-blend-multiply" />
                        ) : (
                          <p className="text-xs text-muted-foreground p-4 text-center font-sans">Not available for this image</p>
                        )}
                      </div>
                    </div>
                  </div>

                  {result.triage && (
                    <div className="report-triage rounded-2xl p-5 border-l-4 border-l-primary bg-primary/5 space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-foreground uppercase tracking-wide font-sans">Risk assessment</span>
                        <span className={cn(
                          "px-3 py-1 rounded-full text-xs font-bold",
                          result.triage.tier === "high concern" ? "bg-red-50 text-red-700 border border-red-200" :
                          result.triage.tier === "moderate confidence" ? "bg-amber-50 text-amber-700 border border-amber-200" :
                          "bg-primary/10 text-primary border border-primary/20"
                        )}>
                          {result.triage.tier}
                        </span>
                      </div>
                      <p className="text-sm font-semibold text-foreground leading-snug">{result.triage.recommendation}</p>
                      <p className="text-sm text-muted-foreground font-sans leading-relaxed">{result.triage.rationale}</p>
                    </div>
                  )}
                </>
              )}

              {reportStep === "hidden" && (
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-t border-brand/10 pt-6">
                  <div className="flex items-start gap-3">
                    <ShieldCheck className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                    <p className="text-xs sm:text-sm text-muted-foreground font-sans leading-relaxed max-w-sm">
                      AI triage aid. Generate a clinical report to attach patient data and export for professional review.
                    </p>
                  </div>
                  <Button
                    onClick={() => setReportStep("form")}
                    className="w-full sm:w-auto bg-primary text-white hover:bg-primary/90 rounded-full text-sm px-6 py-5 flex items-center gap-2"
                  >
                    <FileText className="h-4 w-4" />
                    Export clinical report
                  </Button>
                </div>
              )}

              <AnimatePresence>
                {reportStep === "form" && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    className="bg-muted/50 rounded-2xl p-6 border border-brand/15 space-y-4"
                  >
                    <div className="flex items-center gap-2 mb-4 border-b border-brand/10 pb-3">
                      <User className="h-4 w-4 text-primary" />
                      <h4 className="font-bold font-sans text-sm text-foreground">Patient demographics</h4>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div className="space-y-1.5">
                        <Label htmlFor="patientName">Full Name</Label>
                        <Input type="text" id="patientName" name="patientName" placeholder="Full Name" value={patientData.patientName} onChange={handlePatientDataChange} />
                      </div>
                      <div className="space-y-1.5">
                        <Label htmlFor="patientAge">Age / DOB</Label>
                        <Input type="text" id="patientAge" name="patientAge" placeholder="Age / DOB" value={patientData.patientAge} onChange={handlePatientDataChange} />
                      </div>
                      <div className="space-y-1.5">
                        <Label htmlFor="patientId">Patient ID / Ref Number</Label>
                        <Input type="text" id="patientId" name="patientId" placeholder="Patient ID / Ref Number" value={patientData.patientId} onChange={handlePatientDataChange} />
                      </div>
                      <div className="space-y-1.5">
                        <Label htmlFor="clinicalNotes">Clinical Notes (Optional)</Label>
                        <Input type="text" id="clinicalNotes" name="clinicalNotes" placeholder="Initial Clinical Notes (Optional)" value={patientData.clinicalNotes} onChange={handlePatientDataChange} />
                      </div>
                    </div>
                    <div className="flex justify-end gap-3 pt-4">
                      <Button variant="outline" onClick={() => setReportStep("hidden")} className="rounded-full text-xs">Cancel</Button>
                      <Button onClick={() => setReportStep("preview")} className="rounded-full bg-primary hover:bg-primary/90 text-white text-xs flex items-center gap-2">
                        <FileOutput className="h-4 w-4" /> Generate report
                      </Button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* --- ACTUAL REPORT PREVIEW (Visible here AND during printing) --- */}
            <AnimatePresence>
              {reportStep === "preview" && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="report-preview bg-white rounded-none md:rounded-2xl border-t-4 border-t-primary shadow-sm border border-brand/10 p-6 sm:p-8 md:p-12 print:p-0 print:border-none print:shadow-none font-sans"
                >
                  <div className="report-accent-bar" />
                  <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-8 pb-4 border-b border-dashed border-brand/20 print:hidden">
                    <span className="text-xs font-mono font-bold text-muted-foreground">Document preview</span>
                    <div className="flex gap-3">
                      <Button variant="outline" size="sm" onClick={() => setReportStep("form")} className="rounded-full">Edit Details</Button>
                      <Button size="sm" onClick={() => window.print()} className="rounded-full bg-foreground hover:bg-foreground/90 flex gap-2">
                        <Printer className="h-4 w-4" /> Print / Save PDF
                      </Button>
                    </div>
                  </div>

                  <div className="flex flex-col sm:flex-row justify-between items-start gap-4 mb-8 report-section">
                    <div>
                      <p className="text-xs text-primary font-semibold uppercase tracking-wider mb-1">Breast Cancer Companion</p>
                      <h1 className="report-heading text-xl sm:text-2xl font-black text-foreground tracking-tight">AI-Assisted Diagnostic Report</h1>
                      <p className="report-subheading text-sm text-muted-foreground mt-1">Generated: {new Date().toLocaleDateString()} {new Date().toLocaleTimeString()}</p>
                    </div>
                    <div className="text-left sm:text-right">
                      <div className="inline-block px-4 py-2 border-2 border-primary text-primary font-bold uppercase tracking-widest rounded-lg">
                        {result.prediction}
                      </div>
                    </div>
                  </div>

                  <div className="report-grid grid grid-cols-2 md:grid-cols-4 gap-4 mb-8 bg-muted/50 p-4 rounded-xl border border-brand/10 break-inside-avoid report-section">
                    <div>
                      <p className="report-label text-xs text-brand font-bold uppercase tracking-wider">Patient Name</p>
                      <p className="report-value font-semibold text-foreground">{patientData.patientName || "N/A"}</p>
                    </div>
                    <div>
                      <p className="report-label text-xs text-brand font-bold uppercase tracking-wider">Age/DOB</p>
                      <p className="report-value font-semibold text-foreground">{patientData.patientAge || "N/A"}</p>
                    </div>
                    <div>
                      <p className="report-label text-xs text-brand font-bold uppercase tracking-wider">Patient ID</p>
                      <p className="report-value font-semibold text-foreground">{patientData.patientId || "N/A"}</p>
                    </div>
                    <div>
                      <p className="report-label text-xs text-brand font-bold uppercase tracking-wider">Confidence</p>
                      <p className="report-value font-semibold text-foreground">{result.confidence}% Match</p>
                    </div>
                  </div>

                  <div className="report-grid grid grid-cols-1 md:grid-cols-2 print:grid-cols-2 gap-6 mb-8 break-inside-avoid report-section">
                    <div className="space-y-2">
                      <p className="report-label text-xs font-bold text-muted-foreground font-mono border-b border-brand/10 pb-2">Original scan</p>
                      <div className="report-image-wrap bg-black/5 rounded-xl flex items-center justify-center p-2 aspect-square border border-brand/10">
                        <img src={selectedImage!} alt="Original" className="max-h-full max-w-full object-contain rounded-lg mix-blend-multiply" />
                      </div>
                    </div>
                    <div className="space-y-2">
                      <p className="report-label text-xs font-bold text-muted-foreground font-mono border-b border-brand/10 pb-2">Grad-CAM heatmap</p>
                      <div className="report-image-wrap bg-black/5 rounded-xl flex items-center justify-center p-2 aspect-square border border-brand/10">
                        {result.gradcam ? (
                          <img src={result.gradcam} alt="Heatmap" className="max-h-full max-w-full object-contain rounded-lg mix-blend-multiply" />
                        ) : (
                          <p className="report-body text-xs text-muted-foreground">No Heatmap Available</p>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="space-y-6 mb-8 break-inside-avoid report-section">
                    {result.triage && (
                      <div className="p-5 border border-brand/10 rounded-xl bg-white">
                        <h3 className="report-body text-sm font-bold text-foreground uppercase border-b border-brand/10 pb-2 mb-3">AI Triage Assessment</h3>
                        <p className="report-body text-sm text-foreground font-semibold mb-1">Level: <span className="uppercase text-brand">{result.triage.tier}</span></p>
                        <p className="report-body text-sm text-muted-foreground mb-2">{result.triage.recommendation}</p>
                        <p className="report-body text-xs text-brand italic">{result.triage.rationale}</p>
                      </div>
                    )}
                    {patientData.clinicalNotes && (
                      <div>
                        <h3 className="report-body text-sm font-bold text-foreground uppercase border-b border-brand/10 pb-2 mb-2">Physician Notes</h3>
                        <p className="report-body text-sm text-muted-foreground">{patientData.clinicalNotes}</p>
                      </div>
                    )}
                  </div>

                  <div className="report-disclaimer mt-8 sm:mt-12 p-4 border border-primary/20 bg-primary/5 rounded-xl break-inside-avoid">
                    <div className="flex gap-3 items-start">
                      <AlertCircle className="h-6 w-6 text-primary shrink-0" />
                      <div>
                        <h4 className="report-body text-sm font-bold text-foreground uppercase tracking-wide">Clinical Disclaimer</h4>
                        <p className="report-body text-xs text-muted-foreground mt-1 leading-relaxed font-semibold">
                          This document is generated by an Artificial Intelligence experimental model. It is <span className="font-black underline">NOT</span> a medical diagnosis. The predictions, heatmaps, and triage recommendations provided are strictly for educational and investigational aid. All findings MUST be reviewed, validated, and officially diagnosed by a board-certified radiologist or oncologist before any clinical decisions are made. Do not alter treatment plans based solely on this automated report.
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="report-signature mt-12 sm:mt-16 pt-8 border-t border-brand/20 flex justify-between items-end break-inside-avoid hidden print:flex">
                    <div className="report-body text-xs text-primary">Ref: BCC-{new Date().toISOString().slice(0,10).replace(/-/g,'')}-{new Date().toISOString().slice(11,19).replace(/:/g,'')}</div>
                    <div className="text-center">
                      <div className="w-48 border-b border-foreground mb-2"></div>
                      <span className="report-body text-xs font-bold text-muted-foreground">Reviewing Physician Signature</span>
                    </div>
                  </div>

                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ImageUploader;