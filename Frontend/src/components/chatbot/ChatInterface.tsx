import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Loader2, Sparkles, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

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

const suggestedQuestions = [
  "Early warning indicators of tissue changes?",
  "Standard frequency recommendations for scanning?",
  "What is the mathematical definition of benign triage?",
  "Known genetic risk markers for breast cancer?",
];

// Frosted Citations Panel
const SourcesPanel = ({ sources }: { sources: Source[] }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!sources || sources.length === 0) {
    return null;
  }

  return (
    <div className="mt-3 text-[10px]">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-1.5 text-primary text-glow-teal font-mono tracking-wide hover:opacity-80 transition-opacity"
      >
        <ChevronDown
          className={cn(
            "h-3.5 w-3.5 transition-transform",
            isExpanded ? "rotate-0" : "-rotate-90"
          )}
        />
        <span>[{sources.length}_SOURCE_CITATIONS_VERIFIED]</span>
      </button>

      {isExpanded && (
        <div className="mt-3 space-y-2 border-t border-white/5 pt-3">
          {sources.map((source) => (
            <div
              key={source.id}
              className="bg-black/30 rounded-xl p-3 border border-white/5 space-y-1.5"
            >
              <div className="flex items-start justify-between">
                <div className="font-bold text-white font-mono uppercase tracking-wide">
                  [{source.id}] {source.title}
                </div>
                <span className="inline-block px-2 py-0.5 rounded bg-primary/10 border border-primary/20 text-primary text-[8px] font-mono text-glow-teal">
                  {(source.score * 100).toFixed(0)}% MATCH
                </span>
              </div>
              <p className="text-muted-foreground font-sans">
                {source.source} • chunk #{source.chunk_index}
              </p>
              <p className="text-muted-foreground italic font-serif leading-relaxed text-[11px] bg-white/[0.01] p-2 rounded-lg border border-white/[0.03]">
                "{source.text_preview}"
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const ChatInterface = () => {
  const [messages, setMessages] = useState<Message[]>(() => {
    const saved = sessionStorage.getItem("classifierAIChat");
    if (saved) {
      // We have to turn the saved string dates back into real Date objects
      const parsed = JSON.parse(saved);
      return parsed.map((m: Omit<Message, "timestamp"> & { timestamp: string }) => ({
        ...m,
        timestamp: new Date(m.timestamp)
      }));
    } // <--- THIS WAS THE MISSING BRACKET!

    // If no history exists, load the default welcome message
    return [
      {
        id: "1",
        content: "Welcome to the ClassifierAI neural assistant hub. I am integrated with our RAG scientific indexes. I can help resolve clinical queries regarding breast cancer, diagnostic scans, and model explanations. How can I assist you today?",
        role: "assistant",
        timestamp: new Date(),
      },
    ];
  });

  // Add this effect to auto-save every time the chat updates
  useEffect(() => {
    sessionStorage.setItem("classifierAIChat", JSON.stringify(messages));
  }, [messages]);

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

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");

    // Turn on the cool pulsing loading dots while Qdrant searches the database
    setIsTyping(true);

    try {
      // ---------------------------------------------------------
      // HERE IS WHERE YOUR NEW CODE GOES! 
      // 1. Package up the history
      const conversationHistory = [...messages, userMessage].map((msg) => ({
        role: msg.role,
        content: msg.content,
      }));

      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        // 2. Send the full array using the 'messages' key
        body: JSON.stringify({
          messages: conversationHistory,
        }),
      });
      // ---------------------------------------------------------

      if (!response.ok || !response.body) {
        throw new Error(`API error: ${response.status}`);
      }

      // 1. Create a blank assistant message instantly
      const assistantMessageId = (Date.now() + 1).toString();
      setMessages((prev) => [
        ...prev,
        {
          id: assistantMessageId,
          content: "", // Starts completely empty!
          role: "assistant",
          timestamp: new Date(),
          sources: [],
          status: "success",
        },
      ]);

      // 2. Hide the pulsing dots because the AI is about to start typing
      setIsTyping(false);

      // 3. Open the "catcher's mitt" (The Streams API)
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let done = false;
      let streamedText = "";

      // 4. Read the text chunk-by-chunk as it arrives from Python
      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;

        if (value) {
          // Decode the raw bytes into a readable string chunk
          const chunk = decoder.decode(value, { stream: true });
          streamedText += chunk;

          // Reactively update ONLY the new assistant message with the growing text
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessageId
                ? { ...msg, content: streamedText }
                : msg
            )
          );
        }
      }
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

      setMessages((prev) => [...prev, errorMessage]);
      setIsTyping(false); // Turn off the loading dots if it crashes
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-12rem)] max-h-[700px] glass-panel rounded-3xl border border-brand/10 soft-shadow-lg overflow-hidden bg-white">
      {/* 1. Header (frosted pill top) */}
      <div className="px-6 py-4 border-b border-brand/8 bg-white flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-white border border-brand/10 soft-shadow-sm">
            <Bot className="h-5 w-5 text-brand" />
          </div>
          <div>
            <h3 className="font-bold text-foreground font-heading text-sm">NEURAL_MED_CHAT</h3>
            <p className="text-[10px] text-muted-foreground flex items-center gap-1.5 font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-sage animate-pulse" />
              RAG_MODEL_ONLINE • ACTIVE
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-muted/60 border border-brand/10 text-[9px] font-mono text-muted-foreground">
          <span>QUERY_COUNT: {messages.filter(m => m.role === 'user').length}</span>
        </div>
      </div>

      {/* 2. Messages Display Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-white">
        {messages.map((message) => (
          <motion.div
            key={message.id}
            initial={{ opacity: 0, y: 15, filter: "blur(5px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{ type: "tween", ease: "easeOut", duration: 0.45 }}
            className={cn(
              "flex gap-4",
              message.role === "user" ? "flex-row-reverse" : ""
            )}
          >
            <div
              className={cn(
                "flex h-8 w-8 shrink-0 items-center justify-center rounded-2xl border text-xs",
                message.role === "user"
                  ? "bg-highlight/20 border-highlight/30 text-foreground"
                  : "bg-white border-brand/10 text-foreground"
              )}
            >
              {message.role === "user" ? (
                <User className="h-4 w-4" />
              ) : (
                <Bot className="h-4 w-4 text-brand" />
              )}
            </div>

            <div
              className={cn(
                "max-w-[78%] rounded-3xl px-5 py-3.5 border relative bg-white",
                message.role === "user"
                  ? "bg-highlight/10 text-foreground border-highlight/20 rounded-tr-sm"
                  : "bg-white text-foreground border-brand/10 rounded-tl-sm soft-shadow-sm"
              )}
            >
              {/* Highlight glowing border for assistant */}
              {message.role === "assistant" && (
                <div className="absolute -left-[1px] top-0 bottom-0 w-[2px] bg-gradient-to-b from-brand to-transparent rounded-l" />
              )}

              <p className="text-xs md:text-sm whitespace-pre-wrap leading-relaxed font-sans">
                {message.content}
              </p>

              {/* RAG citations Accordion */}
              {message.role === "assistant" && message.sources && (
                <SourcesPanel sources={message.sources} />
              )}

              <p
                className={cn(
                  "text-[9px] mt-2 font-mono tracking-wide text-muted-foreground"
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
          <div className="flex gap-4">
            <div className="flex h-8 w-8 items-center justify-center rounded-2xl bg-white border border-brand/10 soft-shadow-sm">
              <Bot className="h-4 w-4 text-brand" />
            </div>
            <div className="bg-white rounded-3xl rounded-tl-sm px-5 py-3 border border-brand/10 soft-shadow-sm">
              <div className="flex items-center gap-1.5 py-1">
                <div className="w-1.5 h-1.5 rounded-full bg-brand animate-pulse" style={{ animationDelay: "0ms" }} />
                <div className="w-1.5 h-1.5 rounded-full bg-brand animate-pulse" style={{ animationDelay: "150ms" }} />
                <div className="w-1.5 h-1.5 rounded-full bg-brand animate-pulse" style={{ animationDelay: "300ms" }} />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* 3. Suggested Prompt Shortcuts */}
      {messages.length === 1 && (
        <div className="px-6 pb-4 pt-2">
          <p className="text-[10px] text-muted-foreground mb-3 flex items-center gap-1.5 font-mono">
            <Sparkles className="h-3.5 w-3.5 text-brand" />
            [QUICK_QUERY_PROMPTS]
          </p>
          <div className="flex flex-wrap gap-2.5">
            {suggestedQuestions.map((question, index) => (
              <button
                key={index}
                onClick={() => handleSendMessage(question)}
                className="text-[10px] px-3 py-2 rounded-2xl bg-muted/60 border border-brand/10 text-foreground hover:soft-shadow-sm hover:bg-muted/70 transition-all duration-300 font-sans text-left"
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* 4. Frosted Input Console */}
      <div className="p-6 border-t border-brand/8 bg-white">
        <div className="flex items-end gap-4">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Query deep diagnostic patterns..."
              rows={1}
              className="w-full resize-none rounded-3xl border border-brand/10 bg-white px-4 py-3.5 text-xs md:text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-brand focus:border-brand transition-all duration-300"
              style={{ minHeight: "48px", maxHeight: "120px" }}
            />
          </div>
          <Button
            onClick={() => handleSendMessage()}
            disabled={!inputValue.trim() || isTyping}
            variant="medical"
            size="icon"
            className="h-12 w-12 shrink-0 bg-brand text-white hover:scale-105 transition-transform duration-300 soft-shadow-sm"
          >
            {isTyping ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <Send className="h-5 w-5" />
            )}
          </Button>
        </div>
        <p className="text-[9px] text-muted-foreground mt-3 text-center font-sans">
          <strong>[DISCLAIMER]</strong> AI assistant summaries are strictly educational. Direct all clinical triages to specialized oncological staff.
        </p>
      </div>
    </div>
  );
};

export default ChatInterface;