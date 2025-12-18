import { Link } from "react-router-dom";
import { ArrowRight, Microscope, MessageCircle, Shield, Zap, Heart } from "lucide-react";
import { Button } from "@/components/ui/button";
import Header from "@/components/layout/Header";

const Index = () => {
  const features = [
    {
      icon: Microscope,
      title: "AI-Powered Classification",
      description: "Upload mammography images and receive instant analysis using advanced deep learning models.",
    },
    {
      icon: MessageCircle,
      title: "Medical Chatbot",
      description: "Get answers to your health questions 24/7 with our intelligent medical assistant.",
    },
    {
      icon: Shield,
      title: "Secure & Private",
      description: "Your medical data is encrypted and protected with enterprise-grade security.",
    },
    {
      icon: Zap,
      title: "Instant Results",
      description: "Receive classification results in seconds, helping speed up the screening process.",
    },
  ];

  return (
    <div className="min-h-screen bg-background">
      <Header />

      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-accent/10" />
        <div className="absolute top-20 right-0 w-96 h-96 bg-primary/10 rounded-full blur-3xl" />
        <div className="absolute bottom-0 left-0 w-96 h-96 bg-accent/20 rounded-full blur-3xl" />

        <div className="container mx-auto px-4 py-20 md:py-32 relative">
          <div className="max-w-3xl mx-auto text-center">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-accent text-accent-foreground text-sm font-medium mb-6 animate-fade-in">
              <Heart className="h-4 w-4 text-primary" />
              Advanced AI for Early Detection
            </div>

            <h1 className="text-4xl md:text-6xl font-bold text-foreground leading-tight mb-6 animate-slide-up">
              Breast Cancer
              <span className="block text-primary">Classification & Support</span>
            </h1>

            <p className="text-lg md:text-xl text-muted-foreground mb-10 leading-relaxed animate-slide-up" style={{ animationDelay: "100ms" }}>
              Harness the power of artificial intelligence for early breast cancer detection.
              Upload medical images for instant analysis and chat with our medical assistant for support.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 animate-slide-up" style={{ animationDelay: "200ms" }}>
              <Link to="/classification">
                <Button variant="medical" size="lg" className="gap-2">
                  <Microscope className="h-5 w-5" />
                  Start Classification
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
              <Link to="/chatbot">
                <Button variant="outline" size="lg" className="gap-2">
                  <MessageCircle className="h-5 w-5" />
                  Chat with Assistant
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-muted/30">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
              Comprehensive Medical Support
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Our platform combines cutting-edge AI technology with user-friendly design to provide reliable medical assistance.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <div
                  key={index}
                  className="bg-card rounded-2xl p-6 border border-border shadow-card hover:shadow-card-hover hover:-translate-y-1 transition-all duration-300 animate-slide-up"
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-accent mb-4">
                    <Icon className="h-6 w-6 text-primary" />
                  </div>
                  <h3 className="text-lg font-semibold text-foreground mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <div className="bg-gradient-to-r from-primary to-primary/80 rounded-3xl p-8 md:p-12 text-center relative overflow-hidden">
            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiNmZmYiIGZpbGwtb3BhY2l0eT0iMC4xIj48cGF0aCBkPSJNMzYgMzRjMC0yIDItNCAyLTRzLTItMi00LTItNCAwLTQgMiAwIDIgMiA0IDQgNCA0LTIgNC00IDIgMCAwLTItMi0yLTQtMnoiLz48L2c+PC9nPjwvc3ZnPg==')] opacity-30" />

            <div className="relative">
              <h2 className="text-2xl md:text-4xl font-bold text-primary-foreground mb-4">
                Ready to Get Started?
              </h2>
              <p className="text-primary-foreground/80 mb-8 max-w-xl mx-auto">
                Take the first step towards early detection. Our AI-powered tools are here to support you.
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <Link to="/classification">
                  <Button variant="secondary" size="lg" className="gap-2 bg-primary-foreground text-primary hover:bg-primary-foreground/90">
                    Upload Image
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </Link>
                <Link to="/chatbot">
                  <Button variant="outline" size="lg" className="gap-2 border-primary-foreground/30 text-primary-foreground hover:bg-primary-foreground/10">
                    Ask a Question
                  </Button>
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 border-t border-border">
        <div className="container mx-auto px-4">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
                <Heart className="h-4 w-4 text-primary-foreground" />
              </div>
              <span className="font-semibold text-foreground">Classifier</span>
            </div>
            <p className="text-sm text-muted-foreground text-center">
              For informational purposes only. Not a substitute for professional medical advice.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
