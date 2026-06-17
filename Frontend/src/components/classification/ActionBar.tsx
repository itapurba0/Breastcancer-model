import { ShieldCheck, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ActionBarProps {
  onExportReport: () => void;
}

const ActionBar = ({ onExportReport }: ActionBarProps) => {
  return (
    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-t border-brand/10 pt-6">
      <div className="flex items-start gap-3">
        <ShieldCheck className="h-5 w-5 text-primary shrink-0 mt-0.5" />
        <p className="text-xs sm:text-sm text-muted-foreground font-sans leading-relaxed max-w-sm">
          AI triage aid. Generate a clinical report to attach patient data and export for professional review.
        </p>
      </div>
      <Button
        onClick={onExportReport}
        className="w-full sm:w-auto bg-primary text-white hover:bg-primary/90 rounded-full text-sm px-6 py-5 flex items-center gap-2"
      >
        <FileText className="h-4 w-4" />
        Export clinical report
      </Button>
    </div>
  );
};

export default ActionBar;
