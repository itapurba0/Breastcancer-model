import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface ConfidenceHeaderProps {
  prediction: string;
  confidence: number;
  inconclusive?: boolean;
  isFailed?: boolean;
}

const colorMap = {
  normal: { gauge: "#22c55e", bg: "bg-emerald-50", text: "text-emerald-700", border: "border-emerald-200" },
  benign: { gauge: "#3b82f6", bg: "bg-blue-50", text: "text-blue-700", border: "border-blue-200" },
  malignant: { gauge: "#ef4444", bg: "bg-red-50", text: "text-red-700", border: "border-red-200" },
  inconclusive: { gauge: "#f59e0b", bg: "bg-amber-50", text: "text-amber-700", border: "border-amber-200" },
  failed: { gauge: "#9ca3af", bg: "bg-muted", text: "text-muted-foreground", border: "border-muted-foreground/20" },
};

const ConfidenceHeader = ({ prediction, confidence, inconclusive, isFailed }: ConfidenceHeaderProps) => {
  const isInconclusive = inconclusive ?? false;
  const isNormal = prediction.toLowerCase().includes("normal");
  const isBenign = prediction.toLowerCase().includes("benign");

  const colors = isFailed ? colorMap.failed : isInconclusive ? colorMap.inconclusive : isNormal ? colorMap.normal : isBenign ? colorMap.benign : colorMap.malignant;
  const circumference = 2 * Math.PI * 36;
  const dashOffset = circumference - (confidence / 100) * circumference;

  return (
    <div className="flex items-center gap-6 w-full">
      {!isFailed && (
        <div className="relative shrink-0">
          <svg width="88" height="88" viewBox="0 0 88 88" className="-rotate-90" aria-hidden="true">
            <circle cx="44" cy="44" r="36" fill="none" stroke="currentColor" strokeWidth="6"
              className="text-muted/60" />
            <motion.circle
              cx="44" cy="44" r="36" fill="none" stroke={colors.gauge} strokeWidth="6"
              strokeLinecap="round"
              strokeDasharray={circumference}
              initial={{ strokeDashoffset: circumference }}
              animate={{ strokeDashoffset: dashOffset }}
              transition={{ duration: 1, ease: [0.25, 1, 0.5, 1], delay: 0.15 }}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-lg font-heading font-bold text-foreground leading-none tabular-nums">{confidence}</span>
            <span className="text-[11px] text-muted-foreground font-sans mt-0.5">%</span>
          </div>
        </div>
      )}

      <div className="flex-1 space-y-2">
        <div>
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide font-sans">Diagnostic result</p>
          <div className="flex items-center gap-3 mt-1.5">
            <span className={cn(
              "text-base font-heading font-bold tracking-tight px-3 py-1 rounded-lg border",
              colors.bg, colors.text, colors.border
            )}>
              {isInconclusive ? "Inconclusive" : prediction}
            </span>
            {isInconclusive && (
              <span className="text-xs font-semibold text-amber-600 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded-full">
                Uncertain
              </span>
            )}
          </div>
        </div>
        {!isFailed && (
          <p className="text-xs text-muted-foreground font-sans">
            {isInconclusive
              ? "Model confidence is below the safety threshold — clinical review required"
              : "Model confidence for this classification"
            }
          </p>
        )}
      </div>
    </div>
  );
};

export default ConfidenceHeader;
