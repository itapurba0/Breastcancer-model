import { useState, FormEvent } from "react";
import { useNavigate, Link } from "react-router-dom";
import { motion } from "framer-motion";
import { Heart, Mail, Lock, AlertCircle, Eye, EyeOff, CheckCircle2 } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import HeroCanvas from "@/components/layout/HeroCanvas";

const Auth = () => {
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const { login, signup } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      if (mode === "login") {
        await login(email, password);
      } else {
        await signup(email, password);
      }
      navigate("/chatbot", { replace: true });
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-transparent text-foreground relative selection:bg-secondary/40 selection:text-foreground">
      <HeroCanvas />

      <main className="container mx-auto px-4 sm:px-6 py-16 md:py-24 max-w-lg relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 15, filter: "blur(10px)" }}
          animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
          transition={{ duration: 0.55, ease: "easeOut" }}
          className="glass-panel rounded-[2.5rem] p-8 sm:p-10 relative overflow-hidden shadow-lg border border-brand/15"
        >
          <div className="absolute -top-40 -right-40 w-96 h-96 bg-secondary/10 rounded-full blur-3xl pointer-events-none" />
          <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-sage/30 rounded-full blur-3xl pointer-events-none" />

          <div className="relative z-10 space-y-6">
            <div className="text-center space-y-3">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-muted/80 border border-brand/25 text-xs font-mono text-foreground tracking-wide font-bold">
                <Heart className="h-3.5 w-3.5 animate-pulse text-brand" />
                <span>{mode === "login" ? "SECURE_AUTH_GATE" : "NEW_USER_REGISTRATION"}</span>
              </div>

              <h1 className="text-2xl sm:text-3xl font-extrabold text-foreground font-heading tracking-[-0.04em]">
                {mode === "login" ? "Welcome Back" : "Create Account"}
              </h1>
              <p className="text-sm text-muted-foreground font-sans leading-relaxed">
                {mode === "login"
                  ? "Sign in to access the neural assistant terminal."
                  : "Register to start chatting with the clinical support AI."}
              </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-5">
              {error && (
                <div className="flex items-start gap-2.5 p-3 rounded-xl bg-red-50 border border-red-200 text-red-700 text-xs font-sans font-semibold">
                  <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
                  <span>{error}</span>
                </div>
              )}

              <div className="space-y-2">
                <label className="text-xs font-bold text-foreground font-mono uppercase tracking-wide">Email</label>
                <div className="relative">
                  <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="you@example.com"
                    required
                    className="w-full rounded-2xl border border-brand/15 bg-white pl-10 pr-4 py-3 text-sm focus:outline-none focus:ring-1 focus:ring-brand transition-all duration-300"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-xs font-bold text-foreground font-mono uppercase tracking-wide">Password</label>
                <div className="relative">
                  <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <input
                    type={showPw ? "text" : "password"}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    required
                    minLength={6}
                    className="w-full rounded-2xl border border-brand/15 bg-white pl-10 pr-10 py-3 text-sm focus:outline-none focus:ring-1 focus:ring-brand transition-all duration-300"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPw(!showPw)}
                    className="absolute right-3.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                  >
                    {showPw ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
              </div>

              <button
                type="submit"
                disabled={submitting}
                className="w-full h-12 rounded-full bg-foreground text-white font-bold font-sans tracking-wide text-sm hover:opacity-90 transition-all duration-300 disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {submitting ? (
                  <span className="flex items-center gap-2">
                    <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    {mode === "login" ? "Signing in..." : "Creating account..."}
                  </span>
                ) : mode === "login" ? (
                  "Sign In"
                ) : (
                  "Create Account"
                )}
              </button>
            </form>

            <div className="text-center pt-2">
              <p className="text-xs text-muted-foreground font-sans">
                {mode === "login" ? (
                  <>Don&apos;t have an account?{" "}
                    <button onClick={() => { setMode("signup"); setError(""); }} className="text-brand font-bold hover:underline">
                      Sign up
                    </button>
                  </>
                ) : (
                  <>Already have an account?{" "}
                    <button onClick={() => { setMode("login"); setError(""); }} className="text-brand font-bold hover:underline">
                      Sign in
                    </button>
                  </>
                )}
              </p>
            </div>

            <div className="flex items-center justify-center gap-2 pt-2 border-t border-brand/10">
              <CheckCircle2 className="h-3.5 w-3.5 text-brand" />
              <p className="text-[10px] text-muted-foreground font-sans font-semibold">
                End-to-end encrypted. Your data stays private.
              </p>
            </div>
          </div>
        </motion.div>

        <div className="text-center mt-6">
          <Link to="/" className="text-xs text-muted-foreground font-mono hover:text-foreground transition-colors">
            &larr; Back to Home
          </Link>
        </div>
      </main>
    </div>
  );
};

export default Auth;
