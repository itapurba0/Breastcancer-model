import { Eye, Activity, Loader2 } from "lucide-react";

interface ImageComparisonProps {
  selectedImage: string;
  gradcam?: string;
  onGenerateGradcam?: () => void;
  isGradcamLoading?: boolean;
}

const ImageComparison = ({ selectedImage, gradcam, onGenerateGradcam, isGradcamLoading }: ImageComparisonProps) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div className="space-y-2">
        <div className="flex items-center gap-1.5">
          <Eye className="h-3.5 w-3.5 text-primary" />
          <p className="text-xs font-semibold text-foreground uppercase tracking-wide font-sans">Original scan</p>
        </div>
        <div className="rounded-2xl overflow-hidden bg-muted/50 border border-primary/10">
          <img
            src={selectedImage}
            alt="Uploaded breast tissue scan"
            loading="lazy"
            className="w-full h-auto object-contain"
          />
        </div>
      </div>
      <div className="space-y-2">
        <div className="flex items-center gap-1.5">
          <Activity className="h-3.5 w-3.5 text-primary" />
          <p className="text-xs font-semibold text-foreground uppercase tracking-wide font-sans">AI attention map</p>
        </div>
        <div className="rounded-2xl overflow-hidden bg-muted/50 border border-primary/10 relative min-h-[120px] flex items-center justify-center">
          {gradcam ? (
            <img
              src={gradcam}
              alt="Grad-CAM heatmap showing model attention regions"
              loading="lazy"
              className="w-full h-auto object-contain mix-blend-multiply"
            />
          ) : isGradcamLoading ? (
            <div className="flex flex-col items-center gap-2 py-6">
              <Loader2 className="h-6 w-6 animate-spin text-primary" />
              <p className="text-xs text-muted-foreground font-sans">Generating attention map...</p>
            </div>
          ) : onGenerateGradcam ? (
            <button
              onClick={onGenerateGradcam}
              className="flex flex-col items-center gap-2 py-6 px-4 text-primary hover:text-primary/80 transition-colors cursor-pointer"
            >
              <Activity className="h-6 w-6" />
              <span className="text-xs font-semibold font-sans">Show AI attention map</span>
            </button>
          ) : (
            <p className="text-xs text-muted-foreground p-4 text-center font-sans">Not available for this image</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default ImageComparison;
