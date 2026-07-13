import { AlertTriangle } from "lucide-react";
import { motion } from "framer-motion";
import ConfidenceHeader from "./ConfidenceHeader";
import ImageComparison from "./ImageComparison";
import TriageCard from "./TriageCard";
import FacilityRecommendation from "./FacilityRecommendation";
import ActionBar from "./ActionBar";

interface ResultPanelProps {
  selectedImage: string;
  prediction: string;
  confidence: number;
  inconclusive?: boolean;
  gradcam?: string;
  triage?: {
    tier: string;
    recommendation: string;
    rationale: string;
    confidence_score: number;
  };
  onExportReport: () => void;
}

const ResultPanel = ({
  selectedImage,
  prediction,
  confidence,
  inconclusive,
  gradcam,
  triage,
  onExportReport,
}: ResultPanelProps) => {
  const isFailed = prediction.toLowerCase().includes("fail");
  const isInconclusive = inconclusive ?? false;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, filter: "blur(6px)" }}
      animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      transition={{ duration: 0.4, ease: [0.25, 1, 0.5, 1] }}
      className="space-y-6"
    >
      {/* Full width: Confidence Header */}
      <div className="rounded-2xl p-6 border border-brand/20 bg-white">
        <ConfidenceHeader
          prediction={prediction}
          confidence={confidence}
          inconclusive={isInconclusive}
          isFailed={isFailed}
        />
      </div>

      {/* Full width: Image Comparison (large, 4:3 aspect ratio) */}
      <div className="rounded-2xl border border-brand/20 bg-white p-5">
        <ImageComparison
          selectedImage={selectedImage}
          gradcam={gradcam}
        />
      </div>

      {/* Inconclusive Warning (full width) */}
      {isInconclusive && (
        <div className="flex items-start gap-3 p-4 rounded-2xl bg-amber-50 border border-amber-200" role="alert">
          <AlertTriangle className="h-5 w-5 text-amber-600 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <p className="text-sm font-bold text-amber-800 font-sans">Confidence below safety threshold</p>
            <p className="text-sm text-amber-700 font-sans leading-relaxed">
              The model's confidence ({confidence}%) is below the 60% threshold for definitive classification.
              This result should be treated as inconclusive. Please consult a radiologist for clinical review.
            </p>
          </div>
        </div>
      )}

      {/* Desktop: Two-column for triage + notes (left) and facilities (right, sticky) */}
      <div className="lg:grid lg:grid-cols-5 lg:gap-8">
        {/* Left column (3/5): Triage + Clinical Notes + Action Bar */}
        <div className="lg:col-span-3 space-y-6">
          {/* Triage Card */}
          {triage && (
            <div className="rounded-2xl border border-brand/20 bg-white p-5">
              <TriageCard triage={triage} prediction={prediction} />
            </div>
          )}

          {/* Clinical Disclaimer */}
          <div className="rounded-2xl p-4 border border-primary/15 bg-primary/5">
            <p className="text-xs text-muted-foreground font-sans leading-relaxed">
              <span className="font-bold text-foreground">Clinical note:</span> AI-generated classifications are decision-support tools only.
              All findings must be validated by a board-certified radiologist before clinical action.
            </p>
          </div>

          {/* Action Bar */}
          <div className="rounded-2xl border border-brand/20 bg-white p-5">
            <ActionBar onExportReport={onExportReport} />
          </div>
        </div>

        {/* Right column (2/5): Facilities (sticky, scrollable) */}
        <div className="lg:col-span-2 space-y-6 mt-6 lg:mt-0">
          <div className="lg:sticky lg:top-24">
            <div className="rounded-2xl border border-brand/20 bg-white p-5">
              <FacilityRecommendation
                prediction={prediction}
                confidence={confidence / 100}
                inconclusive={isInconclusive}
              />
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default ResultPanel;
