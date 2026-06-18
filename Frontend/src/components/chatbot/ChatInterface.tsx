import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Loader2, ChevronDown, LogOut, BookOpen } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useAuth } from "@/contexts/AuthContext";

interface Source {
  id: number;
  source: string;
  title: string;
  chunk_index: number;
  score: number;
  text_preview: string;
}

interface Message {
  id: string;
  content: string;
  role: "user" | "assistant";
  timestamp: Date;
  sources?: Source[];
  status?: string;
}

const MAX_MESSAGES = 50;

const suggestedQuestions = [
  "What are the early signs of breast cancer?",
  "How should I prepare for a mammogram?",
  "What do benign results mean?",
  "What are the recommended screening frequencies?",
];

const SourcesPanel = ({ sources }: { sources: Source[] }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!sources || sources.length === 0) {
    return null;
  }

  return (
    <div className="mt-3 text-xs">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-1.5 text-primary font-sans font-medium hover:opacity-80 transition-opacity"
        aria-label={isExpanded ? "Close sources" : "View sources"}
      >
        <ChevronDown
          className={cn(
            "h-3.5 w-3.5 transition-transform",
            isExpanded ? "rotate-0" : "-rotate-90"
          )}
        />
        <BookOpen className="h-3.5 w-3.5" />
        <span>View sources</span>
      </button>

      {isExpanded && (
        <div className="mt-3 space-y-2 border-t border-primary/10 pt-3">
          {sources.map((source) => (
            <div
              key={source.id}
              className="bg-white rounded-xl p-3 border border-primary/10 space-y-1.5"
            >
              <div className="flex items-start justify-between">
                <div className="font-semibold text-foreground text-sm">
                  {source.title}
                </div>
                <span className="inline-block bg-primary/10 text-primary text-xs px-2 py-0.5 rounded-full">
                  {(source.score * 100).toFixed(0)}% relevant
                </span>
              </div>
              <p className="text-muted-foreground font-sans text-xs">
                {source.source} &middot; chunk #{source.chunk_index}
              </p>
              <p className="text-muted-foreground font-sans leading-relaxed text-xs bg-white/50 p-2 rounded-lg border border-primary/5">
                &ldquo;{source.text_preview}&rdquo;
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const WELCOME_MESSAGE: Message = {
  id: "1",
  content: "Hello! I'm your medical assistant. I can help you understand breast cancer diagnosis, treatment options, and screening guidelines. What would you like to know?",
  role: "assistant",
  timestamp: new Date(),
};

const ChatInterface = () => {
  const { token, logout } = useAuth();

  const [messages, setMessages] = useState<Message[]>([WELCOME_MESSAGE]);
  const [chatLoading, setChatLoading] = useState(true);

  useEffect(() => {
    if (!token) {
      setChatLoading(false);
      return;
    }
    fetch("/chat/history", {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.messages && data.messages.length > 0) {
          const loadedMessages = data.messages
            .slice(-MAX_MESSAGES)
            .map((m: Record<string, unknown>) => ({
              ...m,
              id: m.id || Date.now().toString() + Math.random(),
              timestamp: new Date(m.timestamp || Date.now()),
            }));
          setMessages(loadedMessages);
        }
      })
      .catch(() => {})
      .finally(() => setChatLoading(false));
  }, [token]);

  const saveMessages = (msgs: Message[]) => {
    if (!token) return;
    fetch("/chat/save", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ messages: msgs }),
    }).catch(() => {});
  };

  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (content: string = inputValue) => {
    if (!content.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: content.trim(),
      role: "user",
      timestamp: new Date(),
    };

    setMessages((prev) => {
      const updated = [...prev, userMessage].slice(-MAX_MESSAGES);
      saveMessages(updated);
      return updated;
    });
    setInputValue("");
    setIsTyping(true);

    try {
      const conversationHistory = [...messages, userMessage].map((msg) => ({
        role: msg.role,
        content: msg.content,
      }));

      const response = await fetch("/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: conversationHistory,
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`API error: ${response.status}`);
      }

      const assistantMessageId = (Date.now() + 1).toString();
      setMessages((prev) => {
        const updated = [
          ...prev,
          {
            id: assistantMessageId,
            content: "",
            role: "assistant" as const,
            timestamp: new Date(),
            sources: [] as Source[],
            status: "success" as const,
          },
        ].slice(-MAX_MESSAGES);
        return updated;
      });

      setIsTyping(false);

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let done = false;
      let streamedText = "";

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;

        if (value) {
          const chunk = decoder.decode(value, { stream: true });
          streamedText += chunk;

          setMessages((prev) => {
            const updated = prev.map((msg) =>
              msg.id === assistantMessageId
                ? { ...msg, content: streamedText }
                : msg
            ).slice(-MAX_MESSAGES);
            return updated;
          });
        }
      }

      setMessages((prev) => {
        const limited = prev.slice(-MAX_MESSAGES);
        saveMessages(limited);
        return limited;
      });
    } catch (error) {
      console.error("Chat error:", error);

      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: "I encountered an error connecting to the local AI backend. Please ensure the FastAPI server is running on port 8000.",
        role: "assistant",
        timestamp: new Date(),
        sources: [],
        status: "error",
      };

      setMessages((prev) => [...prev, errorMessage].slice(-MAX_MESSAGES));
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex flex-col h-[60vh] min-h-[400px] max-h-[80vh] lg:max-h-[700px] glass-panel rounded-3xl border border-primary/10 soft-shadow-lg overflow-hidden bg-white">
      <div className="px-4 sm:px-6 py-4 border-b border-primary/10 bg-white flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-primary/10 border border-primary/20 soft-shadow-sm">
            <Bot className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h3 className="font-heading font-semibold text-foreground text-sm">Medical Assistant</h3>
            <p className="text-xs text-muted-foreground flex items-center gap-1.5 font-sans">
              <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
              Connected
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className="hidden sm:flex items-center gap-2 px-3 py-1 rounded-full bg-primary/5 border border-primary/10 text-[10px] font-sans text-muted-foreground">
            <span>{messages.filter(m => m.role === 'user').length} messages</span>
          </div>
          <button
            onClick={logout}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-[10px] font-sans text-muted-foreground hover:text-foreground border border-transparent hover:border-primary/10 hover:bg-primary/5 transition-all duration-300"
            title="Sign out"
            aria-label="Logout"
          >
            <LogOut className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Sign out</span>
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6 bg-gradient-to-b from-[#FEFDFB] to-[#F7F6F3]" role="log" aria-live="polite">
        {chatLoading ? (
          <div className="flex items-center justify-center h-full">
            <div className="flex items-center gap-2 text-muted-foreground">
              <Loader2 className="h-5 w-5 animate-spin text-primary" />
              <span className="text-xs font-sans font-bold">Loading history...</span>
            </div>
          </div>
        ) : messages.map((message) => (
          <motion.div
            key={message.id}
            initial={{ opacity: 0, y: 15, filter: "blur(5px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{ type: "tween", ease: "easeOut", duration: 0.45 }}
            className={cn(
              "flex gap-3 sm:gap-4",
              message.role === "user" ? "flex-row-reverse" : ""
            )}
          >
            <div
              className={cn(
                "flex h-8 w-8 shrink-0 items-center justify-center rounded-2xl border",
                message.role === "user"
                  ? "bg-primary/10 border-primary/20 text-primary"
                  : "bg-primary/10 border-primary/20 text-primary"
              )}
            >
              {message.role === "user" ? (
                <User className="h-4 w-4" />
              ) : (
                <Bot className="h-4 w-4" />
              )}
            </div>

            <div
              className={cn(
                "max-w-[90%] sm:max-w-[82%] md:max-w-[78%] rounded-3xl px-4 sm:px-5 py-3.5 border relative bg-white",
                message.role === "user"
                  ? "bg-primary/8 text-foreground border-primary/10 rounded-br-lg"
                  : "bg-white text-foreground border-primary/10 rounded-bl-lg soft-shadow-sm"
              )}
            >
              {message.role === "assistant" && (
                <div className="absolute -left-[1px] top-0 bottom-0 w-[2px] bg-gradient-to-b from-primary to-transparent rounded-l" />
              )}

              {message.role === "user" ? (
                <p className="text-sm sm:text-base whitespace-pre-wrap leading-relaxed font-sans">
                  {message.content}
                </p>
              ) : message.content === "" ? (
                <div className="flex items-center gap-1.5 py-2 px-1">
                  <div className="w-1.5 h-1.5 rounded-full bg-primary/60 animate-pulse" style={{ animationDelay: "0ms" }} />
                  <div className="w-1.5 h-1.5 rounded-full bg-primary/60 animate-pulse" style={{ animationDelay: "150ms" }} />
                  <div className="w-1.5 h-1.5 rounded-full bg-primary/60 animate-pulse" style={{ animationDelay: "300ms" }} />
                </div>
              ) : (
                    <div className="text-sm sm:text-base leading-relaxed font-sans space-y-3">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          p: ({ node, ...props }) => <p className="whitespace-pre-wrap" {...props} />,
                          ul: ({ node, ...props }) => <ul className="list-disc pl-5 space-y-1" {...props} />,
                          ol: ({ node, ...props }) => <ol className="list-decimal pl-5 space-y-1" {...props} />,
                          li: ({ node, ...props }) => <li className="pl-1 marker:text-primary/70" {...props} />,
                          strong: ({ node, ...props }) => <strong className="font-semibold text-foreground" {...props} />,
                          h1: ({ node, ...props }) => <h1 className="text-lg font-bold mb-2 mt-4" {...props} />,
                          h2: ({ node, ...props }) => <h2 className="text-base font-bold mb-2 mt-4" {...props} />,
                          h3: ({ node, ...props }) => <h3 className="text-sm font-bold mb-2 mt-3" {...props} />,
                          table: ({ children, ...props }) => (
                            <div className="bg-muted/50 p-4 rounded-lg border border-primary/10 overflow-x-auto font-mono text-xs" {...props}>
                              {children}
                            </div>
                          ),
                          thead: ({ children, ...props }) => <div className="font-bold border-b border-primary/20 pb-2 mb-2" {...props}>{children}</div>,
                          tbody: ({ children, ...props }) => <div {...props}>{children}</div>,
                          tr: ({ children, ...props }) => <div className="flex gap-4 py-1" {...props}>{children}</div>,
                          th: ({ children, ...props }) => <span className="font-semibold w-1/2" {...props}>{children}</span>,
                          td: ({ children, ...props }) => <span className="w-1/2" {...props}>{children}</span>,
                        }}
                      >
                        {message.content}
                      </ReactMarkdown>
                    </div>
              )}

              {message.role === "assistant" && message.sources && (
                <SourcesPanel sources={message.sources} />
              )}

              <p
                className={cn(
                  "text-[10px] mt-2 font-mono tracking-wide text-muted-foreground",
                  message.role === "user" ? "text-right" : "text-left"
                )}
              >
                {message.timestamp.toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </p>
            </div>
          </motion.div>
        ))}

        {isTyping && (
          <div className="flex gap-3 sm:gap-4">
            <div className="flex h-8 w-8 items-center justify-center rounded-2xl bg-primary/10 border border-primary/20 soft-shadow-sm">
              <Bot className="h-4 w-4 text-primary" />
            </div>
            <div className="bg-white rounded-3xl rounded-bl-lg px-5 py-3 border border-primary/10 soft-shadow-sm">
              <div className="flex items-center gap-1.5 py-1">
                <div className="w-1.5 h-1.5 rounded-full bg-primary/60 animate-pulse" style={{ animationDelay: "0ms" }} />
                <div className="w-1.5 h-1.5 rounded-full bg-primary/60 animate-pulse" style={{ animationDelay: "150ms" }} />
                <div className="w-1.5 h-1.5 rounded-full bg-primary/60 animate-pulse" style={{ animationDelay: "300ms" }} />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {messages.length === 1 && (
        <div className="px-4 sm:px-6 pb-4 pt-2">
          <p className="text-xs text-muted-foreground mb-3 font-sans font-medium">
            How can I help you today?
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {suggestedQuestions.map((question, index) => (
              <button
                key={index}
                onClick={() => handleSendMessage(question)}
                className="text-left text-xs sm:text-sm px-4 py-3 rounded-2xl bg-white border border-primary/10 text-foreground hover:bg-primary/5 hover:border-primary/20 transition-all duration-200 font-sans min-h-[44px] flex items-center"
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="p-4 sm:p-6 border-t border-primary/10 bg-white">
        <div className="flex items-end gap-3 sm:gap-4">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about breast cancer diagnosis, treatment, or screening..."
              rows={1}
              className="w-full resize-none rounded-3xl border border-primary/15 bg-white px-4 py-3.5 text-sm sm:text-base text-foreground placeholder:text-muted-foreground/60 focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary/30 transition-all duration-200"
              style={{ minHeight: "48px", maxHeight: "120px" }}
            />
          </div>
          <Button
            onClick={() => handleSendMessage()}
            disabled={!inputValue.trim() || isTyping}
            className="h-12 w-12 shrink-0 bg-primary text-white hover:bg-primary/90 rounded-full transition-colors duration-200 soft-shadow-sm"
            aria-label="Send message"
          >
            {isTyping ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <Send className="h-5 w-5" />
            )}
          </Button>
        </div>
        <p className="text-[10px] sm:text-xs text-muted-foreground/60 mt-3 text-center font-sans italic">
          This assistant provides educational information only. Always consult a healthcare professional for medical advice.
        </p>
      </div>
    </div>
  );
};

export default ChatInterface;