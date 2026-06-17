# Chatbot + Classification + Report Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the chatbot into a premium clinical assistant, restructure the classification page into a diagnostic dashboard, and create a professional clinical print report.

**Architecture:** Three focused visual redesigns sharing a common CSS foundation. Print CSS overhaul first (both pages depend on it), then chatbot redesign, then classification restructure. All backend connections, API calls, streaming, and classification logic remain exactly as-is.

**Tech Stack:** React 18, Tailwind CSS 3.4, Framer Motion 12, shadcn/ui (Radix), ReactMarkdown, Lucide icons

## Global Constraints

- **No functional changes:** All API endpoints, chat streaming, image upload, Grad-CAM, classification pipeline, auth logic remain identical
- **Light mode only:** Palette: sage primary `#2D6A4F`, warm amber secondary `#E8B86D`, mint accent `#D8F3DC`
- **Fonts:** Inter (body), Plus Jakarta Sans (headings), Inconsolata (data values only)
- **Print CSS:** Must produce clean clinical document at 9pt, sage palette throughout
- **Jargon removal:** All monospaced uppercase labels replaced with professional copy (see spec jargon table)
- **Accessibility:** `aria-label` on icon-only buttons, `role="log"` + `aria-live="polite"` on chat messages
- **Verification:** Every phase ends with `npm run lint` + `npm run build` passing

---

## Phase 1: Print CSS Foundation

### Task 1: Overhaul print CSS in index.css

**Files:**
- Modify: `Frontend/src/index.css` (the `@media print` block and report-specific styles)

**What changes:** Replace all hardcoded print colors (`#000`, `#fff`, `#ccc`) with CSS custom properties. Add new report-specific print styles for the letterhead, patient grid, images, triage callout, disclaimer, and signature footer.

- [ ] **Step 1: Read current print CSS**

Read `Frontend/src/index.css` lines 160-249 to understand the current `@media print` block.

- [ ] **Step 2: Replace the @media print block**

Replace the entire `@media print` block with:

```css
@media print {
  html {
    font-size: 9pt !important;
  }

  body {
    font-size: 9pt;
    line-height: 1.35;
    color: hsl(var(--foreground));
    background: #fff;
  }

  .glass-panel,
  .glass-panel-hover {
    background: #fff !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    box-shadow: none !important;
    border: 1px solid hsl(var(--border)) !important;
    border-radius: 0 !important;
  }

  .soft-shadow-sm,
  .soft-shadow-md,
  .soft-shadow-lg {
    box-shadow: none !important;
  }

  .print\:hidden {
    display: none !important;
  }

  /* Report letterhead */
  .report-accent-bar {
    height: 4px;
    background: hsl(var(--primary));
    margin-bottom: 1rem;
  }

  /* Report preview container */
  .report-preview {
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    border-radius: 0 !important;
  }

  /* Report sections */
  .report-section {
    margin-bottom: 0.5cm !important;
    break-inside: avoid;
  }

  /* Patient info grid */
  .report-grid {
    gap: 0.3cm !important;
  }

  /* Report images */
  .report-image-wrap {
    aspect-ratio: auto !important;
    height: 3.5cm !important;
    max-height: 3.5cm !important;
    padding: 0.2cm !important;
  }

  .report-image-wrap img {
    max-height: 3.1cm !important;
  }

  /* Report typography */
  .report-heading {
    font-size: 14pt !important;
    margin-bottom: 0.2cm !important;
  }

  .report-subheading {
    font-size: 8pt !important;
  }

  .report-label {
    font-size: 7pt !important;
    color: hsl(var(--primary)) !important;
  }

  .report-value {
    font-size: 9pt !important;
    font-weight: 600 !important;
  }

  .report-body {
    font-size: 9pt !important;
    line-height: 1.35 !important;
  }

  /* Triage callout */
  .report-triage {
    border: 1px solid hsl(var(--primary) / 0.3) !important;
    background: hsl(var(--primary) / 0.03) !important;
    padding: 0.3cm !important;
    break-inside: avoid;
  }

  /* Disclaimer */
  .report-disclaimer {
    margin-top: 0.35cm !important;
    padding: 0.25cm !important;
    border: 1px solid hsl(var(--primary) / 0.3) !important;
    background: hsl(var(--primary) / 0.03) !important;
    break-inside: avoid;
  }

  .report-disclaimer .report-body {
    font-size: 8pt !important;
  }

  /* Signature footer */
  .report-signature {
    margin-top: 1cm !important;
    padding-top: 0.3cm !important;
    border-top: 1px solid hsl(var(--border)) !important;
    break-inside: avoid;
  }
}
```

- [ ] **Step 3: Add report-specific utility classes**

After the `@media print` block (but still in the CSS file), add these screen+print utilities:

```css
/* Report accent bar (screen + print) */
.report-accent-bar {
  height: 4px;
  background: linear-gradient(90deg, hsl(var(--primary)), hsl(var(--primary) / 0.4));
  border-radius: 2px;
}

/* Triage callout (screen + print) */
.report-triage {
  border: 1px solid hsl(var(--primary) / 0.2);
  background: hsl(var(--primary) / 0.03);
  border-radius: 0.75rem;
  padding: 1rem;
}

/* Signature footer (screen + print) */
.report-signature {
  border-top: 1px solid hsl(var(--border));
  padding-top: 1rem;
  margin-top: 2rem;
}
```

- [ ] **Step 4: Verify**

Run: `cd Frontend && npm run lint && npm run build`
Expected: Both pass

- [ ] **Step 5: Commit**

```bash
cd Frontend && git add src/index.css && git commit -m "feat(css): overhaul print CSS with sage palette and report styles"
```

---

## Phase 2: Chatbot Redesign

### Task 2: Redesign ChatInterface.tsx

**Files:**
- Modify: `Frontend/src/components/chatbot/ChatInterface.tsx`

**What changes:** Full visual redesign of the chat interface. All streaming logic, API calls, state management, and save/load history remain identical. Only visual presentation and copy change.

- [ ] **Step 1: Read the current file**

Read `Frontend/src/components/chatbot/ChatInterface.tsx` (456 lines) to understand the full component structure.

- [ ] **Step 2: Update imports**

Add `BookOpen` to lucide imports (for sources panel). The rest stays the same.

- [ ] **Step 3: Update suggestedQuestions**

Replace the array with professional medical questions:

```typescript
const suggestedQuestions = [
  "What are the early signs of breast cancer?",
  "How should I prepare for a mammogram?",
  "What do benign results mean?",
  "What are the recommended screening frequencies?",
];
```

- [ ] **Step 4: Update WELCOME_MESSAGE**

Replace the welcome message content:

```typescript
const WELCOME_MESSAGE: Message = {
  id: "1",
  content: "Hello! I'm your medical assistant. I can help you understand breast cancer diagnosis, treatment options, and screening guidelines. What would you like to know?",
  role: "assistant",
  timestamp: new Date(),
};
```

- [ ] **Step 5: Update SourcesPanel**

Replace the SourcesPanel component. Key changes:
- Replace `[SOURCE_CITATIONS_VERIFIED]` with "View sources" + `BookOpen` icon
- Replace monospaced uppercase labels with clean typography
- Source title: `font-semibold text-foreground text-sm` (not mono uppercase)
- Relevance badge: `bg-primary/10 text-primary text-xs px-2 py-0.5 rounded-full` (sage)
- Source text preview: remove `font-mono`, use clean quote styling

- [ ] **Step 6: Update top bar**

Key changes:
- Bot icon: sage circle `bg-primary/10 border border-primary/20` with `text-primary`
- "Medical Assistant" title: `font-heading font-semibold text-foreground text-sm`
- Status dot: sage `bg-primary animate-pulse` (not `bg-sage`)
- Connection text: "Connected" (remove "Data encrypted end-to-end" — that's for the footer)
- Query count: `{messages.filter(m => m.role === 'user').length} messages` (remove `QUERY_COUNT:` prefix)
- Logout button: add `aria-label="Logout"` (keep existing)

- [ ] **Step 7: Update message area background**

Change the messages container from flat white to subtle gradient:

```tsx
<div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6 bg-gradient-to-b from-[#FEFDFB] to-[#F7F6F3]" role="log" aria-live="polite">
```

- [ ] **Step 8: Update user message styling**

Change user message bubble:
```tsx
"bg-primary/8 text-foreground border border-primary/10 rounded-3xl rounded-br-lg"
```
(Replace `bg-primary/10` with `bg-primary/8`, add `rounded-br-lg` for asymmetric corners)

- [ ] **Step 9: Update bot message styling**

Change bot message bubble:
```tsx
"bg-white text-foreground border border-primary/10 rounded-3xl rounded-bl-lg soft-shadow-sm"
```
(Replace `bg-muted` with `bg-white`, add `rounded-bl-lg`, replace `border-brand/10` with `border-primary/10`)

- [ ] **Step 10: Update user avatar**

Change user avatar circle:
```tsx
"bg-primary/10 border-primary/20 text-primary"
```
(Replace `bg-secondary/20 border-secondary/30`)

- [ ] **Step 11: Update bot avatar**

Change bot avatar circle:
```tsx
"bg-primary/10 border-primary/20 text-primary"
```
(Replace `bg-white border-brand/10 text-brand`)

- [ ] **Step 12: Update typing indicator**

Change the typing dots from `bg-brand/60` to `bg-primary/60`:
```tsx
<div className="w-1.5 h-1.5 rounded-full bg-primary/60 animate-pulse" style={{ animationDelay: "0ms" }} />
```
(Update all three dots)

- [ ] **Step 13: Update suggested questions section**

Replace the section below messages (when `messages.length === 1`):

```tsx
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
```

- [ ] **Step 14: Update input area**

Change textarea placeholder:
```tsx
placeholder="Ask about breast cancer diagnosis, treatment, or screening..."
```

Change textarea border on focus:
```tsx
className="w-full resize-none rounded-3xl border border-primary/15 bg-white px-4 py-3.5 text-sm sm:text-base text-foreground placeholder:text-muted-foreground/60 focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary/30 transition-all duration-200"
```

Change send button:
```tsx
className="h-12 w-12 shrink-0 bg-primary text-white hover:bg-primary/90 rounded-full transition-colors duration-200 soft-shadow-sm"
```
(Replace `variant="medical"` with explicit classes, remove `hover:scale-105`)

- [ ] **Step 15: Update disclaimer**

Replace the disclaimer text:
```tsx
<p className="text-[10px] sm:text-xs text-muted-foreground/60 mt-3 text-center font-sans italic">
  This assistant provides educational information only. Always consult a healthcare professional for medical advice.
</p>
```
(Remove `<strong>[DISCLAIMER]</strong>`, use italic)

- [ ] **Step 16: Verify**

Run: `cd Frontend && npm run lint && npm run build`
Expected: Both pass

- [ ] **Step 17: Commit**

```bash
cd Frontend && git add src/components/chatbot/ChatInterface.tsx && git commit -m "feat(chatbot): premium clinical redesign with clean copy and refined styling"
```

---

### Task 3: Update Chatbot.tsx layout

**Files:**
- Modify: `Frontend/src/pages/Chatbot.tsx`

**What changes:** Update the page layout to match the new chatbot design. Remove info badges, add subtitle.

- [ ] **Step 1: Read current Chatbot.tsx**

Read to understand current structure.

- [ ] **Step 2: Update layout**

Replace the 3-column info badges section with a single subtitle:

```tsx
<main id="main-content" className="relative z-10 min-h-screen px-4 py-12 md:py-16">
  <div className="max-w-4xl mx-auto space-y-8">
    <div className="text-center space-y-2">
      <h1 className="text-2xl md:text-3xl font-heading font-bold text-foreground tracking-tight">
        Medical Chat
      </h1>
      <p className="text-sm text-muted-foreground font-sans max-w-md mx-auto">
        Ask me about breast cancer diagnosis, treatment options, or screening guidelines.
      </p>
    </div>
    <ChatInterface />
  </div>
</main>
```

- [ ] **Step 3: Verify**

Run: `cd Frontend && npm run lint && npm run build`
Expected: Both pass

- [ ] **Step 4: Commit**

```bash
cd Frontend && git add src/pages/Chatbot.tsx && git commit -m "feat(chatbot): clean page layout with subtitle"
```

---

## Phase 3: Classification Restructure

### Task 4: Restructure ImageUploader.tsx

**Files:**
- Modify: `Frontend/src/components/classification/ImageUploader.tsx`

**What changes:** Complete restructure of the classification UI. All API calls, image processing, Grad-CAM, and state management remain identical. Only visual presentation and copy change.

- [ ] **Step 1: Read the current file**

Read `Frontend/src/components/classification/ImageUploader.tsx` (560 lines) to understand the full structure.

- [ ] **Step 2: Update upload zone**

Key changes:
- Increase min-height: `min-h-[280px]`
- Icon: Use `Microscope` instead of `Upload`, sage circle `bg-primary/10 border-primary/20`
- Heading: "Upload mammography image" (keep — already clean)
- Subtitle: "Drag and drop or click to browse" (keep)
- Support badge: remove `font-mono`, use clean `text-xs text-muted-foreground`
- Drag state: `border-primary/40 bg-primary/5` (sage, not `border-highlight bg-muted/50`)
- Remove the yellow gradient blob (`bg-[#FFE082]/5`) — replace with subtle sage: `bg-primary/5`

- [ ] **Step 3: Update analysis progress panel**

Replace the current analysis panel with step-by-step progress:

```tsx
{isAnalyzing && (
  <motion.div
    initial={{ opacity: 0, y: 15 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0, y: -15 }}
    className="glass-panel rounded-3xl p-6 border border-primary/10 bg-white"
  >
    <div className="space-y-3">
      {["Receiving image", "Processing neural layers", "Generating heatmap", "Compiling results"].map((step, i) => (
        <div key={step} className="flex items-center gap-3">
          <div className={cn(
            "flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-bold",
            analysisStep > i ? "bg-primary text-white" : analysisStep === i ? "bg-primary/10 text-primary animate-pulse" : "bg-muted text-muted-foreground"
          )}>
            {analysisStep > i ? "✓" : i + 1}
          </div>
          <span className={cn("text-sm font-sans", analysisStep >= i ? "text-foreground" : "text-muted-foreground")}>
            {step}
          </span>
        </div>
      ))}
    </div>
  </motion.div>
)}
```

This requires adding an `analysisStep` state variable that increments as the analysis progresses. Update the `analyzeImage` function to set steps:

```typescript
const [analysisStep, setAnalysisStep] = useState(-1);

// Inside analyzeImage, after setting isAnalyzing(true):
const steps = [0, 1, 2, 3];
for (const step of steps) {
  await new Promise(r => setTimeout(r, 600));
  setAnalysisStep(step);
}
// Then proceed with the actual API call
```

- [ ] **Step 4: Update result panel — prediction card**

Replace the prediction section with a cleaner layout:

```tsx
<div className="flex items-start gap-4 p-5 rounded-2xl border-l-4 border-l-primary bg-primary/5">
  <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-primary/10">
    {isNormal || isBenign ? <CheckCircle className="h-6 w-6 text-primary" /> : <AlertCircle className="h-6 w-6 text-secondary" />}
  </div>
  <div className="flex-1 space-y-3">
    <div>
      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Diagnostic result</p>
      <div className="flex items-center gap-3 mt-1">
        <Badge className={cn("text-sm font-bold", badgeClass)}>{result.prediction}</Badge>
      </div>
    </div>
    {!isFailed && (
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Confidence</span>
          <span className="text-xs font-semibold text-foreground">{result.confidence}%</span>
        </div>
        <div className="h-2 rounded-full bg-muted overflow-hidden">
          <div className="h-full bg-primary rounded-full transition-all duration-700" style={{ width: `${result.confidence}%` }} />
        </div>
      </div>
    )}
  </div>
</div>
```

- [ ] **Step 5: Update Grad-CAM section**

Replace the Grad-CAM display:

```tsx
<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
  <div className="space-y-2">
    <p className="text-xs font-medium text-muted-foreground">Original scan</p>
    <div className="rounded-2xl overflow-hidden bg-muted/50 flex items-center justify-center aspect-square border border-primary/10">
      <img src={selectedImage!} alt="Original scan" className="max-h-full max-w-full object-contain" />
    </div>
  </div>
  <div className="space-y-2">
    <p className="text-xs font-medium text-muted-foreground">Activation heatmap</p>
    <div className="rounded-2xl overflow-hidden bg-muted/50 flex items-center justify-center aspect-square border border-primary/10 relative">
      {result.gradcam ? (
        <img src={result.gradcam} alt="Heatmap" className="max-h-full max-w-full object-contain mix-blend-multiply" />
      ) : (
        <p className="text-xs text-muted-foreground p-4 text-center">Heatmap not available for this image</p>
      )}
    </div>
  </div>
</div>
```

- [ ] **Step 6: Update triage card**

Replace the triage section with a callout box:

```tsx
{result.triage && (
  <div className="report-triage rounded-2xl">
    <div className="flex items-center justify-between mb-3">
      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Risk assessment</span>
      <span className={cn(
        "px-3 py-1 rounded-full text-xs font-semibold",
        result.triage.tier === "high concern" ? "bg-red-50 text-red-700 border border-red-200" :
        result.triage.tier === "moderate confidence" ? "bg-amber-50 text-amber-700 border border-amber-200" :
        "bg-primary/10 text-primary border border-primary/20"
      )}>
        {result.triage.tier}
      </span>
    </div>
    <p className="text-sm font-semibold text-foreground mb-1">{result.triage.recommendation}</p>
    <p className="text-sm text-muted-foreground">{result.triage.rationale}</p>
  </div>
)}
```

- [ ] **Step 7: Update report generation button**

Replace "GENERATE CLINICAL REPORT" button:

```tsx
<Button
  onClick={() => setReportStep("form")}
  className="w-full sm:w-auto bg-primary text-white hover:bg-primary/90 rounded-full text-sm px-6 py-5 soft-shadow-sm flex items-center gap-2"
>
  <FileText className="h-4 w-4" />
  Export clinical report
</Button>
```

- [ ] **Step 8: Update report form**

Replace jargon in the form:
- `GENERATE CLINICAL REPORT` → keep as "Generate report" on the submit button
- `CANCEL` → "Cancel" (sentence case)
- `COMPILE REPORT` → "Generate report"

- [ ] **Step 9: Update report preview**

Replace the report preview with professional clinical document structure:
- Add `report-accent-bar` div at top
- Replace "AI Diagnostic Scan Report" heading with: "Breast Cancer Companion" (small) + "AI-Assisted Diagnostic Report" (large)
- Replace random system ID with timestamp-based: `BCC-${new Date().toISOString().slice(0,10).replace(/-/g,'')}-${new Date().toISOString().slice(11,19).replace(/:/g,'')}`
- Update disclaimer border from `border-red-200 bg-red-50` to use `report-disclaimer` class (sage)
- Update signature footer with `report-signature` class

- [ ] **Step 10: Remove jargon throughout**

Apply all jargon replacements from the spec:
- `FILE: {name}` → just `{name}`
- `Explainable Grad-CAM map` → "Heatmap explanation"
- `Heatmap saliency view` → remove
- `Mapping visual tissue saliencies...` → "Analyzing tissue patterns..."
- `Risk triage index` → "Risk assessment"
- `DIAGNOSTIC PREDICTION` → "Diagnostic result"

- [ ] **Step 11: Verify**

Run: `cd Frontend && npm run lint && npm run build`
Expected: Both pass

- [ ] **Step 12: Commit**

```bash
cd Frontend && git add src/components/classification/ImageUploader.tsx && git commit -m "feat(classification): complete restructure with dashboard layout and clean copy"
```

---

### Task 5: Update Classification.tsx layout

**Files:**
- Modify: `Frontend/src/pages/Classification.tsx`

**What changes:** Update the page layout to match the new classification design.

- [ ] **Step 1: Read current Classification.tsx**

Read to understand current structure.

- [ ] **Step 2: Update layout if needed**

Ensure the page has a clean centered layout with:
- HeroCanvas background
- Header
- Main content area with max-w-3xl

- [ ] **Step 3: Verify**

Run: `cd Frontend && npm run lint && npm run build`
Expected: Both pass

- [ ] **Step 4: Commit**

```bash
cd Frontend && git add src/pages/Classification.tsx && git commit -m "feat(classification): clean page layout"
```

---

## Phase 4: Final Verification

### Task 6: Final verification

- [ ] **Step 1: Full lint**

Run: `cd Frontend && npm run lint`
Expected: 0 errors

- [ ] **Step 2: Full build**

Run: `cd Frontend && npm run build`
Expected: Successful production build

- [ ] **Step 3: Check no jargon remains**

Run: `cd Frontend && grep -r "SOURCE_CITATIONS\|QUERY_COUNT\|MAMMOGRAPHY_INF\|COGNITIVE_SCANNER\|TRIAGE_NODE\|CLASSIFIER_AI_LABS\|GENERATE CLINICAL REPORT\|COMPILE REPORT" src/ --include="*.tsx" --include="*.ts"`
Expected: No results

- [ ] **Step 4: Check no hardcoded print colors**

Run: `cd Frontend && grep -n "#000\|#fff\|#ccc" src/index.css`
Expected: No results in @media print block (only in base styles if any)

- [ ] **Step 5: Check console.log removed**

Run: `cd Frontend && grep -r "console.log" src/ --include="*.tsx" --include="*.ts"`
Expected: No results

---

## Summary

| Phase | Tasks | What it delivers |
|-------|-------|-----------------|
| 1. Print CSS | 1 | Sage palette print styles, report-specific utilities |
| 2. Chatbot | 2-3 | Premium clinical chat with clean copy, gradient background, 2x2 suggested questions |
| 3. Classification | 4-5 | Dashboard layout with step-by-step progress, side-by-side images, clean triage |
| 4. Verification | 6 | Final lint, build, jargon check |
