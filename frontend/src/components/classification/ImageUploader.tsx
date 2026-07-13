import { useState, useCallback } from "react";
import {
  AlertCircle, FileOutput, Printer, User,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import UploadPanel from "./UploadPanel";
import ResultPanel from "./ResultPanel";
import HowItWorks from "./HowItWorks";
import { classifierApi, gradcamApi } from "@/lib/api";

interface ClassificationResult {
  prediction: string;
  confidence: number;
  details: string;
  gradcam?: string;
  inconclusive?: boolean;
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
  const [generatedGradcam, setGeneratedGradcam] = useState<string | undefined>(undefined);
  const [gradcamLoading, setGradcamLoading] = useState(false);

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
    if (file) processFile(file);
  };

  const processFile = (file: File) => {
    setSelectedFile(file);
    setResult(null);
    setReportStep("hidden");
    const reader = new FileReader();
    reader.onload = (e) => setSelectedImage(e.target?.result as string);
    reader.readAsDataURL(file);
  };

  const clearImage = () => {
    setSelectedImage(null);
    setSelectedFile(null);
    setResult(null);
    setGeneratedGradcam(undefined);
    setReportStep("hidden");
  };

  const handlePatientDataChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setPatientData({ ...patientData, [e.target.name]: e.target.value });
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setResult(null);
    setGeneratedGradcam(undefined);
    setReportStep("hidden");
    setAnalysisStep(0);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile, selectedFile.name);

      const res = await classifierApi("/predict", { method: "POST", body: formData });
      setAnalysisStep(3);

      if (!res.ok) {
        const text = await res.text();
        setResult({ prediction: "Analysis Failed", confidence: 0, details: `HTTP error: ${res.status} ${text}` });
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
        inconclusive: json.inconclusive ?? false,
        triage: json.triage,
      });
    } catch (err) {
      setResult({ prediction: "Analysis Failed", confidence: 0, details: String(err) });
    } finally {
      setIsAnalyzing(false);
      setAnalysisStep(-1);
    }
  };

  const handleGenerateGradcam = async () => {
    if (!selectedFile || gradcamLoading) return;
    setGradcamLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", selectedFile, selectedFile.name);
      const res = await gradcamApi("/gradcam", { method: "POST", body: formData });
      if (!res.ok) throw new Error("Grad-CAM failed");
      const json = await res.json();
      setGeneratedGradcam(json.gradcam_image);
    } catch {
      setGeneratedGradcam(undefined);
    } finally {
      setGradcamLoading(false);
    }
  };

  return (
    <div className="w-full space-y-8">
      {/* Pre-upload guidance — shown only when no image selected */}
      {!selectedImage && (
        <div className="space-y-4">
          <HowItWorks />

          <div className="rounded-2xl p-5 border border-primary/10 bg-white/60">
            <p className="text-xs font-semibold tracking-widest text-muted-foreground uppercase font-sans mb-3">
              Tips for best results
            </p>
            <ul className="space-y-2.5 text-xs text-muted-foreground font-sans">
              <li className="flex items-start gap-2">
                <span className="text-primary font-bold mt-0.5">1.</span>
                <span>Use high-resolution mammography or ultrasound images for most accurate classification.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary font-bold mt-0.5">2.</span>
                <span>Crop to the region of interest if the scan contains multiple areas.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary font-bold mt-0.5">3.</span>
                <span>Results below 60% confidence are flagged as inconclusive — always seek professional review.</span>
              </li>
            </ul>
          </div>
        </div>
      )}

      {/* Upload Panel */}
      <UploadPanel
        selectedImage={selectedImage}
        selectedFile={selectedFile}
        isDragging={isDragging}
        isAnalyzing={isAnalyzing}
        analysisStep={analysisStep}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onFileSelect={handleFileSelect}
        onClear={clearImage}
        onAnalyze={analyzeImage}
      />

      {/* Results */}
      <AnimatePresence>
        {result && (
          <>
            {reportStep === "hidden" ? (
              <ResultPanel
                selectedImage={selectedImage!}
                prediction={result.prediction}
                confidence={result.confidence}
                inconclusive={result.inconclusive}
                gradcam={generatedGradcam}
                triage={result.triage}
                onExportReport={() => setReportStep("form")}
                onGenerateGradcam={handleGenerateGradcam}
                isGradcamLoading={gradcamLoading}
              />
            ) : (
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
                        <Input type="text" id="patientName" name="patientName" placeholder="Full Name" autoComplete="name" value={patientData.patientName} onChange={handlePatientDataChange} />
                      </div>
                      <div className="space-y-1.5">
                        <Label htmlFor="patientAge">Age / DOB</Label>
                        <Input type="text" id="patientAge" name="patientAge" placeholder="Age / DOB" inputMode="numeric" value={patientData.patientAge} onChange={handlePatientDataChange} />
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

                {reportStep === "preview" && result && (() => {
                  const p = result.prediction.toLowerCase();
                  const isInc = result.inconclusive ?? false;
                  const isNormal = p.includes("normal");
                  const isBenign = p.includes("benign");
                  const isFail = p.includes("fail");

                  const colors = {
                    normal: { hex: "#059669", badge: "border-emerald-600 text-emerald-700 bg-emerald-50", text: "text-emerald-700", border: "border-emerald-200", bg: "bg-emerald-50" },
                    benign: { hex: "#2563eb", badge: "border-blue-600 text-blue-700 bg-blue-50", text: "text-blue-700", border: "border-blue-200", bg: "bg-blue-50" },
                    malignant: { hex: "#dc2626", badge: "border-red-600 text-red-700 bg-red-50", text: "text-red-700", border: "border-red-200", bg: "bg-red-50" },
                    inconclusive: { hex: "#d97706", badge: "border-amber-500 text-amber-700 bg-amber-50", text: "text-amber-700", border: "border-amber-200", bg: "bg-amber-50" },
                    fail: { hex: "#9ca3af", badge: "border-gray-400 text-gray-500 bg-gray-50", text: "text-gray-500", border: "border-gray-200", bg: "bg-gray-50" },
                  };
                  const c = colors[isFail ? "fail" : isInc ? "inconclusive" : isNormal ? "normal" : isBenign ? "benign" : "malignant"];

                  return (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="bg-white rounded-none md:rounded-2xl border-t-4 shadow-sm border border-brand/10 p-6 sm:p-8 md:p-12 print:p-0 print:border-none print:shadow-none font-sans"
                      style={{ borderTopColor: c.hex }}
                    >
                      <div className="report-accent-bar" style={{ background: `linear-gradient(90deg, ${c.hex}, ${c.hex}66)` }} />
                      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-8 pb-4 border-b border-dashed border-brand/20 print:hidden">
                        <span className="text-xs font-mono font-bold text-muted-foreground">Document preview</span>
                        <div className="flex gap-3">
                          <Button variant="outline" size="sm" onClick={() => setReportStep("form")} className="rounded-full">Edit Details</Button>
                          <Button size="sm" onClick={() => window.print()} className="rounded-full bg-foreground hover:bg-foreground/90 flex gap-2">
                            <Printer className="h-4 w-4" /> Print / Save PDF
                          </Button>
                        </div>
                      </div>

                      <div className="flex flex-col sm:flex-row justify-between items-start gap-4 mb-8" style={{ breakInside: "avoid" }}>
                        <div>
                          <p className={`text-xs ${c.text} font-semibold uppercase tracking-wider mb-1`}>Breast Cancer Companion</p>
                          <h1 className="text-xl sm:text-2xl font-black text-foreground tracking-tight">AI-Assisted Diagnostic Report</h1>
                          <p className="text-sm text-muted-foreground mt-1">Generated: {new Date().toLocaleDateString()} {new Date().toLocaleTimeString()}</p>
                        </div>
                        <div className="text-left sm:text-right">
                          <div className={`inline-block px-4 py-2 ${c.badge} font-bold uppercase tracking-widest rounded-lg`}>
                            {result.prediction}
                          </div>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8 bg-muted/50 p-4 rounded-xl border border-brand/10" style={{ breakInside: "avoid" }}>
                        <div>
                          <p className="text-xs text-brand font-bold uppercase tracking-wider">Patient Name</p>
                          <p className="font-semibold text-foreground">{patientData.patientName || "N/A"}</p>
                        </div>
                        <div>
                          <p className="text-xs text-brand font-bold uppercase tracking-wider">Age/DOB</p>
                          <p className="font-semibold text-foreground">{patientData.patientAge || "N/A"}</p>
                        </div>
                        <div>
                          <p className="text-xs text-brand font-bold uppercase tracking-wider">Patient ID</p>
                          <p className="font-semibold text-foreground">{patientData.patientId || "N/A"}</p>
                        </div>
                        <div>
                          <p className="text-xs text-brand font-bold uppercase tracking-wider">Confidence</p>
                          <p className="font-semibold text-foreground tabular-nums">{result.confidence}% Match</p>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 print:grid-cols-2 gap-6 mb-8 print:mb-2" style={{ breakInside: "avoid" }}>
                        <div className="space-y-2">
                          <p className="text-xs font-bold text-muted-foreground font-mono border-b border-brand/10 pb-2">Original scan</p>
                          <div className="bg-black/5 rounded-xl p-2 border border-brand/10 report-image-wrap flex items-center justify-center overflow-hidden">
                            <img src={selectedImage!} alt="Original" className="w-full max-h-full object-contain rounded-lg mix-blend-multiply" />
                          </div>
                        </div>
                        <div className="space-y-2">
                          <p className="text-xs font-bold text-muted-foreground font-mono border-b border-brand/10 pb-2">Grad-CAM heatmap</p>
                          <div className="bg-black/5 rounded-xl p-2 border border-brand/10 report-image-wrap flex items-center justify-center overflow-hidden">
                            {generatedGradcam ? (
                              <img src={generatedGradcam} alt="Heatmap" className="w-full max-h-full object-contain rounded-lg mix-blend-multiply" />
                            ) : (
                              <p className="text-xs text-muted-foreground">No Heatmap Available</p>
                            )}
                          </div>
                        </div>
                      </div>

                      <div className="space-y-6 print:space-y-2 mb-8 print:mb-2" style={{ breakInside: "avoid" }}>
                        {result.triage && (
                          <div className="p-5 border border-brand/10 rounded-xl bg-white">
                            <h3 className="text-sm font-bold text-foreground uppercase border-b border-brand/10 pb-2 mb-3">AI Triage Assessment</h3>
                            <p className="text-sm text-foreground font-semibold mb-1">Level: <span className="uppercase text-brand">{result.triage.tier}</span></p>
                            <p className="text-sm text-muted-foreground mb-2">{result.triage.recommendation}</p>
                            <p className="text-xs text-brand italic">{result.triage.rationale}</p>
                          </div>
                        )}
                        {patientData.clinicalNotes && (
                          <div>
                            <h3 className="text-sm font-bold text-foreground uppercase border-b border-brand/10 pb-2 mb-2">Physician Notes</h3>
                            <p className="text-sm text-muted-foreground">{patientData.clinicalNotes}</p>
                          </div>
                        )}
                      </div>

                      <div className={`mt-8 sm:mt-12 p-4 border ${c.border} ${c.bg} rounded-xl print:mt-2`} style={{ breakInside: "avoid" }}>
                        <div className="flex gap-3 items-start">
                          <AlertCircle className={`h-6 w-6 ${c.text} shrink-0`} />
                          <div>
                            <h4 className="text-sm font-bold text-foreground uppercase tracking-wide">Clinical Disclaimer</h4>
                            <p className="text-xs text-muted-foreground mt-1 leading-relaxed font-semibold">
                              This document is generated by an Artificial Intelligence experimental model. It is <span className="font-black underline">NOT</span> a medical diagnosis. The predictions, heatmaps, and triage recommendations provided are strictly for educational and investigational aid. All findings MUST be reviewed, validated, and officially diagnosed by a board-certified radiologist or oncologist before any clinical decisions are made. Do not alter treatment plans based solely on this automated report.
                            </p>
                          </div>
                        </div>
                      </div>

                      <div className="mt-12 sm:mt-16 pt-8 border-t border-brand/20 flex justify-between items-end hidden print:flex print:mt-2 print:pt-2" style={{ breakInside: "avoid" }}>
                        <div className={`text-xs ${c.text}`}>Ref: BCC-{new Date().toISOString().slice(0,10).replace(/-/g,'')}-{new Date().toISOString().slice(11,19).replace(/:/g,'')}</div>
                        <div className="text-center">
                            <div className="w-56 border-b-2 border-foreground pt-8 mb-1"></div>
                            <span className="text-xs font-bold text-muted-foreground">Reviewing Physician Signature</span>
                          </div>
                      </div>
                    </motion.div>
                  );
                })()}
              </AnimatePresence>
            )}
          </>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ImageUploader;
