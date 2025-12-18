import { useState, useCallback } from "react";
import { Upload, X, Image as ImageIcon, Loader2, AlertCircle, CheckCircle, Microscope } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ClassificationResult {
  prediction: string;
  confidence: number;
  details: string;
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
          prediction: "Error",
          confidence: 0,
          details: `Server error: ${res.status} ${text}`,
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
      });
    } catch (err) {
      setResult({
        prediction: "Error",
        confidence: 0,
        details: String(err),
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto space-y-6">
      {/* Upload Area */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={cn(
          "relative border-2 border-dashed rounded-2xl p-8 transition-all duration-300 bg-card",
          isDragging
            ? "border-primary bg-accent/50 scale-[1.02]"
            : "border-border hover:border-primary/50 hover:bg-accent/20",
          selectedImage && "border-primary/30"
        )}
      >
        {!selectedImage ? (
          <div className="flex flex-col items-center gap-4 py-8">
            <div className="flex h-20 w-20 items-center justify-center rounded-full bg-accent">
              <Upload className="h-10 w-10 text-primary" />
            </div>
            <div className="text-center">
              <p className="text-lg font-semibold text-foreground">
                Drop your image here
              </p>
              <p className="text-sm text-muted-foreground mt-1">
                or click to browse from your device
              </p>
            </div>
            <input
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <ImageIcon className="h-4 w-4" />
              <span>Supports: JPG, PNG</span>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="relative rounded-xl overflow-hidden bg-muted aspect-video">
              <img
                src={selectedImage}
                alt="Selected medical image"
                className="w-full h-full object-contain animate-fade-in"
              />
              <button
                onClick={clearImage}
                className="absolute top-3 right-3 p-2 rounded-full bg-foreground/80 text-background hover:bg-foreground transition-colors"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="flex items-center justify-between">
              <p className="text-sm text-muted-foreground truncate max-w-[200px]">
                {selectedFile?.name}
              </p>
              <Button
                onClick={analyzeImage}
                disabled={isAnalyzing}
                variant="medical"
                size="lg"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Microscope className="h-4 w-4" />
                    Analyze Image
                  </>
                )}
              </Button>
            </div>
          </div>
        )}
      </div>

      {/* Analysis Progress */}
      {isAnalyzing && (
        <div className="bg-card rounded-xl p-6 border border-border shadow-card animate-slide-up">
          <div className="flex items-center gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-accent">
              <Loader2 className="h-6 w-6 text-primary animate-spin" />
            </div>
            <div>
              <p className="font-semibold text-foreground">Processing Image</p>
              <p className="text-sm text-muted-foreground">
                Running deep learning analysis...
              </p>
            </div>
          </div>
          <div className="mt-4 h-2 bg-muted rounded-full overflow-hidden">
            <div className="h-full bg-primary rounded-full animate-pulse" style={{ width: "70%" }} />
          </div>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="bg-card rounded-xl p-6 border border-border shadow-card animate-slide-up">
          <div className="flex items-start gap-4">
            {(() => {
              const status = result.prediction?.toLowerCase() ?? "";
              const isBenign = status.includes("benign");
              const isNormal = status.includes("normal");
              const isMalignant = status.includes("malig");

              const circleClass = isBenign
                ? "bg-success-light"
                : isNormal
                  ? "bg-muted"
                  : "bg-medical-coral-light";
              const icon = isMalignant ? (
                <AlertCircle className="h-6 w-6 text-medical-coral" />
              ) : (
                <CheckCircle className="h-6 w-6 text-success" />
              );

              const badgeClass = isBenign
                ? "bg-success-light text-success"
                : isNormal
                  ? "bg-muted text-foreground"
                  : "bg-medical-coral-light text-medical-coral";

              return (
                <>
                  <div
                    className={cn(
                      "flex h-12 w-12 items-center justify-center rounded-full",
                      circleClass
                    )}
                  >
                    {icon}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="text-xl font-bold text-foreground">
                        {result.prediction}
                      </h3>
                      <span
                        className={cn(
                          "px-3 py-1 rounded-full text-sm font-medium",
                          badgeClass
                        )}
                      >
                        {result.confidence}% Confidence
                      </span>
                    </div>
                    {/* <p className="text-sm text-muted-foreground leading-relaxed">
                      {result.details}
                    </p> */}
                  </div>
                </>
              );
            })()}
          </div>

          <div className="mt-6 pt-4 border-t border-border">
            <p className="text-xs text-muted-foreground flex items-center gap-2">
              <AlertCircle className="h-3 w-3" />
              This is an AI-assisted analysis. Please consult a healthcare professional for diagnosis.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};




export default ImageUploader;
