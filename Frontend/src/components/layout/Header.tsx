import { Link, useLocation } from "react-router-dom";
import { Activity, MessageCircle, Microscope, ShieldCheck, LogIn, LogOut } from "lucide-react";
import { cn } from "@/lib/utils";
import { useAuth } from "@/contexts/AuthContext";

const Header = () => {
  const location = useLocation();
  const { isAuthenticated, user, logout } = useAuth();

  const navItems = [
    { path: "/", label: "Home", icon: Activity },
    { path: "/classification", label: "Diagnostic Scan", icon: Microscope },
    { path: "/chatbot", label: "Neural Chat", icon: MessageCircle },
  ];

  return (
    <header className="sticky top-6 z-50 w-full max-w-5xl mx-auto px-6">
      <div className="glass-panel px-6 py-3.5 flex items-center justify-between soft-shadow-sm relative overflow-hidden">
        <div className="absolute -left-8 top-1/2 -translate-y-1/2 w-32 h-8 bg-highlight/10 rounded-full blur-xl pointer-events-none" />

        <Link to="/" className="flex items-center gap-3 group relative z-10">
          <div className="flex h-10 w-10 items-center justify-center rounded-3xl bg-white border border-brand/10 transition-transform duration-300 group-hover:scale-102">
            <ShieldCheck className="h-5 w-5 text-brand stroke-[2px]" />
          </div>
          <span className="text-base font-bold text-foreground font-heading tracking-tight">
            Classifier<span className="text-brand">AI</span>
          </span>
          <span className="text-[9px] uppercase tracking-widest font-mono text-brand border border-brand/20 px-2 py-0.5 rounded-lg ml-2 bg-white/60 hidden sm:inline-block font-bold">
            Clinical Hub
          </span>
        </Link>

        <nav className="flex items-center gap-1 sm:gap-2 relative z-10">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={cn(
                  "flex items-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-2.5 min-h-[44px] rounded-3xl text-xs font-semibold tracking-wide transition-transform duration-300",
                  isActive
                    ? "bg-secondary/40 text-foreground border border-secondary/60 soft-shadow-sm"
                    : "text-muted-foreground hover:text-foreground hover:bg-brand/5"
                )}
              >
                <Icon className={cn("h-4 w-4", isActive ? "text-foreground" : "text-muted-foreground")} />
                <span className="text-[10px] sm:text-xs">{item.label}</span>
              </Link>
            );
          })}

          {isAuthenticated ? (
            <div className="flex items-center gap-1.5 sm:gap-2 ml-1 sm:ml-2 pl-2 sm:pl-3 border-l border-brand/10">
              <span className="text-[10px] sm:text-xs font-mono text-muted-foreground hidden md:inline truncate max-w-[80px]">
                {user}
              </span>
              <button
                onClick={logout}
                className="flex items-center gap-1.5 px-3 py-2.5 min-h-[44px] rounded-3xl text-xs font-semibold text-muted-foreground hover:text-foreground hover:bg-brand/5 transition-colors duration-300"
              >
                <LogOut className="h-4 w-4" />
                <span className="text-[10px] sm:text-xs hidden sm:inline">Logout</span>
              </button>
            </div>
          ) : (
            <Link
              to="/login"
              className={cn(
                "flex items-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-2.5 min-h-[44px] rounded-3xl text-xs font-semibold tracking-wide transition-colors duration-300 ml-1 sm:ml-2",
                location.pathname === "/login"
                  ? "bg-secondary/40 text-foreground border border-secondary/60 soft-shadow-sm"
                  : "text-muted-foreground hover:text-foreground hover:bg-brand/5"
              )}
            >
              <LogIn className="h-4 w-4" />
              <span className="text-[10px] sm:text-xs">Login</span>
            </Link>
          )}
        </nav>

        <div className="hidden md:flex items-center gap-2 text-xs font-mono text-muted-foreground relative z-10 font-bold">
          <span className="w-2.5 h-2.5 rounded-full bg-sage animate-pulse" />
          <span>TRIAGE_NODE_ACTIVE</span>
        </div>
      </div>
    </header>
  );
};

export default Header;
