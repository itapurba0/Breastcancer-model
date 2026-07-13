import { cn } from "@/lib/utils";

interface Triage {
  tier: string;
  recommendation: string;
  rationale: string;
  confidence_score: number;
}

interface TriageCardProps {
  triage: Triage;
  prediction: string;
}

const TriageCard = ({ triage, prediction }: TriageCardProps) => {
  const isNormal = prediction.toLowerCase().includes("normal");
  const isBenign = prediction.toLowerCase().includes("benign");
  const isInconclusiveTriage = triage.tier === "Further Evaluation Required";
  const triageBorder = isInconclusiveTriage ? "border-l-amber-500" : isNormal ? "border-l-emerald-500" : isBenign ? "border-l-blue-500" : "border-l-red-500";

  return (
    <div className={cn("rounded-2xl p-5 border-l-4 bg-primary/5 space-y-3", triageBorder)}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-foreground uppercase tracking-wide font-sans">Risk assessment</span>
        <span className={cn(
          "px-3 py-1 rounded-full text-xs font-bold",
          triage.tier === "High Concern" || triage.tier === "high concern" ? "bg-red-50 text-red-700 border border-red-200" :
          triage.tier === "Moderate Concern" || triage.tier === "moderate confidence" ? "bg-amber-50 text-amber-700 border border-amber-200" :
          triage.tier === "Further Evaluation Required" ? "bg-amber-50 text-amber-700 border border-amber-200" :
          "bg-primary/10 text-primary border border-primary/20"
        )}>
          {triage.tier}
        </span>
      </div>
      <p className="text-sm font-semibold text-foreground leading-snug">{triage.recommendation}</p>
      <p className="text-sm text-muted-foreground font-sans leading-relaxed">{triage.rationale}</p>
    </div>
  );
};

export default TriageCard;
