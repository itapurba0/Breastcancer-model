# Chatbot + Classification + Report Redesign

**Date:** 2026-06-17
**Status:** Approved — Ready for implementation
**Scope:** Premium clinical chatbot, classification page restructure, professional print report. Zero functional changes to backend.

---

## 1. Goal

Three focused improvements:
1. **Chatbot:** Transform from basic functional chat to a premium clinical assistant (Ada Health quality)
2. **Classification:** Complete restructure from flat layout to diagnostic dashboard with richer visual hierarchy
3. **Print report:** Professional clinical document with letterhead, sage palette, and clean structure

**Constraints:**
- All backend connections, API calls, streaming, classification pipeline remain exactly as-is
- No new features — visual/UX redesign only
- Existing functionality (chat streaming, RAG sources, image upload, Grad-CAM, report generation) must work identically

---

## 2. Premium Clinical Chatbot

### 2.1 Chat container
- Keep glass panel wrapper
- Message area background: subtle sage-tinted gradient (`from-[#FEFDFB] to-[#F7F6F3]`) instead of flat white
- Add very subtle dot-grid pattern at 3% opacity (medical chart paper feel)

### 2.2 Top bar
- Bot icon in sage circle with subtle pulse animation when connected
- "Medical Assistant" title + green "Connected" dot (sage, not generic green)
- Query count → clean "X messages" label (remove `QUERY_COUNT:` prefix)
- Logout button: clean icon + "Sign out" text, subtle hover

### 2.3 Messages
- **User messages:** `bg-primary/8` (sage tint), right-aligned, asymmetric border-radius (larger bottom-right: `rounded-3xl rounded-br-lg`)
- **Bot messages:** White background with left sage accent border (2px gradient from primary to transparent), left-aligned, `rounded-3xl rounded-bl-lg`
- **Avatars:** Bot gets refined sage circle with Bot icon, user gets initials circle with `bg-secondary/20`
- **Streaming:** Keep existing 3-dot pulse indicator
- **Markdown rendering:** Code blocks get `bg-muted` + `font-mono`, lists get proper spacing, headings use `font-heading`, tables get clean borders

### 2.4 Suggested questions (when only welcome message)
- Replace "Suggested questions" label with "How can I help you today?"
- Show 4 cards in 2x2 grid (not flex-wrap chips)
- Each card: glass-panel, icon (Microscope, Heart, Brain, Shield), question text, hover lift
- Professional medical questions (no jargon):
  - "What are the early signs of breast cancer?"
  - "How should I prepare for a mammogram?"
  - "What do benign results mean?"
  - "What are the recommended screening frequencies?"

### 2.5 Input area
- Textarea with auto-grow (min 48px, max 120px)
- Placeholder: "Ask about breast cancer diagnosis, treatment, or screening..."
- Send button: sage primary, rounded-full, with soft shadow
- Enter to send, Shift+Enter for newline (existing behavior)
- Subtle character count when > 200 chars

### 2.6 Sources panel
- Replace `[SOURCE_CITATIONS_VERIFIED]` with "View sources" + BookOpen icon
- Each source card: title in bold, relevance as small sage badge (`85% match`), text preview in quote block
- Remove all monospaced uppercase labels

### 2.7 Disclaimer
- Replace `[DISCLAIMER] AI assistant summaries are strictly educational...` with:
- Clean italic: "This assistant provides educational information only. Always consult a healthcare professional for medical advice."

### 2.8 Welcome message
- Replace "Welcome to the ClassifierAI neural assistant hub..." with:
- "Hello! I'm your medical assistant. I can help you understand breast cancer diagnosis, treatment options, and screening guidelines. What would you like to know?"

---

## 3. Classification Page — Complete Restructure

### 3.1 Upload zone (top section)
- Larger drop zone (min-h-[280px]) with centered content
- Medical cross/heart icon in sage circle (not generic Upload icon)
- "Upload mammography image" heading
- "Drag and drop or click to browse" subtitle
- "Supports: JPEG, PNG" as clean caption
- Sage border on hover/drag (not brand blue-gray)
- File name displayed cleanly below upload zone after selection

### 3.2 Analysis progress (replaces current spinner)
- Animated progress steps with sage checkmarks:
  1. "Receiving image..." → ✓
  2. "Processing neural layers..." → ✓
  3. "Generating heatmap..." → ✓
  4. "Compiling results..." → ✓
- Each step lights up sage as it completes
- Subtle animation between steps

### 3.3 Results panel (restructured)

**Prediction card (top):**
- Large prediction text with color-coded left border:
  - Benign: sage (`border-l-4 border-l-primary`)
  - Malignant: amber (`border-l-4 border-l-secondary`)
  - Normal: muted (`border-l-4 border-l-muted-foreground/30`)
- Confidence as circular SVG gauge (arc from 0-100%) — visually striking
- Prediction class badge (sage/amber/muted)

**Grad-CAM section:**
- Side-by-side on desktop (md+): original image (left) + heatmap (right)
- On mobile: stacked
- Clean labels: "Original scan" / "Activation heatmap"
- Subtle scale-on-hover (1.02x) for interactivity

**Triage card:**
- Color-coded border based on severity
- Clean risk level badge (sage/amber/red)
- Recommendation in a sage callout box
- Rationale in smaller text below

**Report generation (bottom):**
- Replace "GENERATE CLINICAL REPORT" with "Export clinical report" + Printer icon
- Button: sage primary, not `bg-foreground`
- Form fields in clean 2-column grid with shadcn Input + Label
- Preview shows professional clinical document

### 3.4 Jargon removal
| Old | New |
|-----|-----|
| `GENERATE CLINICAL REPORT` | "Export clinical report" |
| `COMPILE REPORT` | "Generate report" |
| `FILE: {name}` | "{name}" (just the filename) |
| `Explainable Grad-CAM map` | "Heatmap explanation" |
| `Heatmap saliency view` | Remove (redundant) |
| `Mapping visual tissue saliencies...` | "Analyzing tissue patterns..." |
| `Risk triage index` | "Risk assessment" |
| `DIAGNOSTIC PREDICTION` | "Diagnostic result" |

---

## 4. Professional Clinical Report (Print)

### 4.1 Report structure

**Letterhead header:**
- Sage accent bar (4px) across top of page
- Left: "Breast Cancer Companion" + "AI-Assisted Diagnostic Report"
- Right: Date, time, document reference ID (timestamp-based, not random)
- Clean horizontal rule below

**Patient information grid:**
- 2-column grid on print, 4-column on screen
- Labels in sage (`text-primary`)
- Values in bold foreground
- Subtle background tint

**Diagnostic results:**
- Prediction badge (color-coded)
- Confidence as both text AND visual gauge
- Side-by-side images: original + Grad-CAM heatmap
- Clean labels

**Triage assessment:**
- Color-coded callout box (sage for low, amber for moderate, red for high)
- Recommendation in bold
- Rationale in regular text

**Physician notes (if provided):**
- Clean section with notes in regular text

**Disclaimer:**
- Sage-bordered callout (not red — less alarming for professional document)
- "This AI-generated report is for screening purposes only. All findings require review by a certified radiologist."

**Signature footer:**
- Signature line for reviewing physician
- System reference ID (timestamp-based: `BCC-YYYYMMDD-HHMMSS`)
- "Generated by Breast Cancer Companion AI"

### 4.2 Print CSS updates

Replace all hardcoded colors with CSS custom properties:
```css
@media print {
  html { font-size: 9pt !important; }
  body { color: hsl(var(--foreground)); background: #fff; }
  
  .glass-panel { background: #fff; backdrop-filter: none; border: 1px solid #e5e3de; }
  
  .report-accent-bar { 
    height: 4px; 
    background: hsl(var(--primary)); 
    margin-bottom: 1rem; 
  }
  
  .report-label { color: hsl(var(--primary)); }
  .report-value { color: hsl(var(--foreground)); font-weight: 600; }
  
  .report-disclaimer { 
    border: 1px solid hsl(var(--primary) / 0.3); 
    background: hsl(var(--primary) / 0.03); 
  }
}
```

### 4.3 System ID
- Replace `Math.random().toString(36).substr(2, 9)` with timestamp-based ID
- Format: `BCC-YYYYMMDD-HHMMSS` (e.g., `BCC-20260617-143052`)
- Generated once on report preview, not on every render

---

## 5. Files Modified

| File | Changes |
|------|---------|
| `Frontend/src/components/chatbot/ChatInterface.tsx` | Full redesign: messages, input, sources, suggested questions, welcome message, disclaimer |
| `Frontend/src/pages/Chatbot.tsx` | Update layout if needed |
| `Frontend/src/components/classification/ImageUploader.tsx` | Complete restructure: upload, analysis progress, results, report, jargon removal |
| `Frontend/src/pages/Classification.tsx` | Update layout if needed |
| `Frontend/src/index.css` | Print CSS overhaul: replace hardcoded colors, add report-specific styles |

### No changes to
- Backend API endpoints
- Chat streaming logic
- Image upload/processing logic
- Grad-CAM generation
- Auth logic
- RAG sources functionality

---

## 6. Verification

1. `npm run lint` — 0 errors
2. `npm run build` — successful
3. Manual check: Chatbot — send message, see streaming, check sources, suggested questions
4. Manual check: Classification — upload image, see analysis, view results, generate report
5. Manual check: Print — open report preview, print to PDF, verify clean clinical document
6. Manual check: Responsive — mobile, tablet, desktop
