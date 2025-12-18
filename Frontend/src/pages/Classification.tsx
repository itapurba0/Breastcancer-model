import Header from "@/components/layout/Header";
import ImageUploader from "@/components/classification/ImageUploader";
import { Shield, Clock, CheckCircle } from "lucide-react";

const Classification = () => {
  const benefits = [
    { icon: Shield, text: "Secure & encrypted uploads" },
    { icon: Clock, text: "Results in seconds" },
    { icon: CheckCircle, text: "High accuracy AI model" },
  ];

  return (
    <div className="min-h-screen bg-background">
      <Header />

      <main className="container mx-auto px-4 py-8 md:py-12">
        {/* Page Header */}
        <div className="text-center mb-10">
          <h1 className="text-3xl md:text-4xl font-bold text-foreground mb-3">
            Breast Cancer Classification
          </h1>
          <p className="text-muted-foreground max-w-xl mx-auto">
            Upload a Ultrasound image for AI-powered analysis. Our deep learning model will classify the image and provide insights.
          </p>

          {/* Benefits */}
          <div className="flex flex-wrap items-center justify-center gap-6 mt-6">
            {benefits.map((benefit, index) => {
              const Icon = benefit.icon;
              return (
                <div key={index} className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Icon className="h-4 w-4 text-primary" />
                  <span>{benefit.text}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Upload Section */}
        <ImageUploader />

        {/* Information Section */}
        <div className="mt-12 max-w-2xl mx-auto">
          <div className="bg-accent/50 rounded-2xl p-6 border border-border">
            <h3 className="font-semibold text-foreground mb-3">How It Works</h3>
            <ol className="space-y-3">
              <li className="flex items-start gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs font-bold">1</span>
                <p className="text-sm text-muted-foreground">Upload a Ultrasound image (JPG, PNG, or DICOM format)</p>
              </li>
              <li className="flex items-start gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs font-bold">2</span>
                <p className="text-sm text-muted-foreground">Our AI model analyzes the image using deep learning techniques</p>
              </li>
              <li className="flex items-start gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs font-bold">3</span>
                <p className="text-sm text-muted-foreground">Receive a classification result with confidence score</p>
              </li>
            </ol>
            <p className="text-xs text-muted-foreground mt-4 pt-4 border-t border-border">
              <strong>Disclaimer:</strong> This tool is for informational purposes only and should not replace professional medical diagnosis. Always consult with a healthcare provider.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Classification;
