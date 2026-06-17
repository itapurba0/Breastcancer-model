import { Upload, Cpu, CheckCircle } from "lucide-react";

const steps = [
  {
    icon: Upload,
    title: "Upload",
    description: "Drop a mammography or ultrasound image",
  },
  {
    icon: Cpu,
    title: "Analyze",
    description: "Neural layers process the tissue in real time",
  },
  {
    icon: CheckCircle,
    title: "Results",
    description: "Classification, heatmap, and triage assessment",
  },
];

const HowItWorks = () => {
  return (
    <div className="rounded-2xl p-5 sm:p-6 border border-primary/10 bg-white/60">
      <p className="text-xs font-semibold tracking-widest text-muted-foreground uppercase font-sans mb-4">
        How it works
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 sm:gap-6">
        {steps.map((step, i) => {
          const Icon = step.icon;
          return (
            <div key={i} className="flex items-start gap-3">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary text-xs font-bold">
                {i + 1}
              </div>
              <div className="space-y-0.5 pt-1">
                <div className="flex items-center gap-1.5">
                  <Icon className="h-3.5 w-3.5 text-primary" />
                  <span className="text-sm font-semibold text-foreground">{step.title}</span>
                </div>
                <p className="text-xs text-muted-foreground leading-relaxed">{step.description}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default HowItWorks;
