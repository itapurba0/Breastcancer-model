import HeroCanvas from "@/components/layout/HeroCanvas";
import Header from "@/components/layout/Header";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

const NotFound = () => {
  return (
    <>
      <HeroCanvas />
      <Header />
      <main id="main-content" className="relative z-10 min-h-screen flex items-center justify-center px-4">
        <div className="glass-panel p-12 text-center max-w-md">
          <h1 className="text-6xl font-heading font-bold text-foreground mb-4">404</h1>
          <p className="text-xl text-muted-foreground mb-2">Page not found</p>
          <p className="text-sm text-muted-foreground/70 mb-8">
            The page you're looking for doesn't exist or has been moved.
          </p>
          <Button asChild className="bg-primary text-primary-foreground rounded-3xl">
            <Link to="/">Return home</Link>
          </Button>
        </div>
      </main>
    </>
  );
};

export default NotFound;
