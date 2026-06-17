# Premium Clinical UI Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the Breast Cancer Companion frontend from a developer-jargon-heavy glassmorphism prototype into a premium clinical interface — warm sage palette, clean typography, professional copy, page transitions, and full design system consistency.

**Architecture:** Visual-only rebuild. All backend connections, API calls, data flows, chat streaming, classification pipeline, and auth logic remain exactly as-is. Changes are confined to CSS tokens, Tailwind config, page components, layout components, and dead code removal.

**Tech Stack:** React 18, Tailwind CSS 3.4, Framer Motion 12, shadcn/ui (Radix), Three.js (React Three Fiber), Vite

## Global Constraints

- **No functional changes:** All API endpoints, data flows, auth logic, chat streaming, classification pipeline, and Grad-CAM remain identical
- **Light mode only:** `darkMode: false` in tailwind config, CSS `color-scheme: light` lock
- **Fonts:** Inter (body), Plus Jakarta Sans (headings), Inconsolata (data values only)
- **Palette:** Deep sage `#2D6A4F` primary, warm amber `#E8B86D` secondary, mint wash `#D8F3DC` accent, warm white `#FEFDFB` background
- **Print CSS:** Must remain functional for clinical report PDF export
- **Accessibility:** `aria-label` on all icon-only buttons, skip-to-content link, `role="log"` + `aria-live="polite"` on chat messages
- **Verification:** Every phase ends with `npm run lint` + `npm run build` passing

---

## Phase 1: Foundation — Color Palette & Dark Mode Lock

### Task 1: Update CSS custom properties in index.css

**Files:**
- Modify: `Frontend/src/index.css:9-47` (the `:root` block)

**What changes:** Replace all HSL values in `:root` with the new warm clinical palette. Add dark mode CSS lock. Remove dead utilities (`.text-glow-teal`, `.animate-float`). Update glass panel and shadow colors from blue-gray to sage.

- [ ] **Step 1: Replace :root CSS variables**

Replace the entire `:root` block (lines 9-47) with:

```css
:root {
  --background: 40 20% 99%;
  --foreground: 0 0% 10%;

  --card: 40 20% 99%;
  --card-foreground: 0 0% 10%;

  --popover: 40 20% 99%;
  --popover-foreground: 0 0% 10%;

  --primary: 152 40% 30%;
  --primary-foreground: 0 0% 100%;

  --secondary: 38 65% 60%;
  --secondary-foreground: 0 0% 10%;

  --muted: 40 15% 97%;
  --muted-foreground: 0 0% 38%;

  --accent: 145 50% 85%;
  --accent-foreground: 152 40% 20%;

  --destructive: 0 84% 60%;
  --destructive-foreground: 0 0% 100%;

  --border: 40 12% 90%;
  --input: 40 12% 93%;
  --ring: 152 35% 40%;

  --radius: 1.75rem;

  --font-sans: 'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif;
  --font-heading: 'Plus Jakarta Sans', ui-sans-serif, system-ui, -apple-system, sans-serif;
  --font-mono: 'Inconsolata', ui-monospace, SFMono-Regular, monospace;

  --brand: 152 40% 30%;
  --highlight: 38 65% 60%;
  --sage: 145 50% 85%;
}
```

- [ ] **Step 2: Add dark mode CSS lock after the :root block**

Add this after the closing `}` of `:root`:

```css
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: light !important;
  }
}
```

- [ ] **Step 3: Update glass-panel utility to use sage-tinted borders/shadows**

Replace the `.glass-panel` block (lines 81-90) with:

```css
.glass-panel {
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(20px) saturate(120%);
  -webkit-backdrop-filter: blur(20px) saturate(120%);
  border: 1px solid hsl(var(--primary) / 0.1);
  box-shadow:
    0 20px 50px hsl(var(--primary) / 0.05),
    inset 0 1px 2px rgb(255 255 255 / 0.9);
  border-radius: 1.75rem;
}
```

- [ ] **Step 4: Update glass-panel-hover to use sage**

Replace the `.glass-panel-hover:hover` block (lines 96-103) with:

```css
.glass-panel-hover:hover {
  transform: translateY(-2px);
  background: rgba(255, 255, 255, 0.95);
  border-color: hsl(var(--primary) / 0.2);
  box-shadow:
    0 30px 60px hsl(var(--primary) / 0.08),
    inset 0 1px 2px rgb(255 255 255 / 1);
}
```

- [ ] **Step 5: Update shadow utilities to use sage**

Replace the `.soft-shadow-*` blocks (lines 105-115) with:

```css
.soft-shadow-sm {
  box-shadow: 0 4px 15px hsl(var(--primary) / 0.03);
}

.soft-shadow-md {
  box-shadow: 0 10px 30px hsl(var(--primary) / 0.05);
}

.soft-shadow-lg {
  box-shadow: 0 20px 50px hsl(var(--primary) / 0.08);
}
```

- [ ] **Step 6: Remove dead utilities**

Delete these blocks entirely:
- `.text-glow-teal` (lines 117-119) — no-op (`text-shadow: none`)
- `.animate-float` (lines 125-127) — never applied to any element
- `@keyframes float` (lines 130-140) — associated dead keyframe

- [ ] **Step 7: Update scrollbar colors to use sage**

Replace scrollbar styles (lines 142-158) with:

```css
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}

::-webkit-scrollbar-track {
  background: hsl(var(--primary) / 0.02);
}

::-webkit-scrollbar-thumb {
  background: hsl(var(--primary) / 0.2);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: hsl(var(--primary) / 0.4);
}
```

- [ ] **Step 8: Update body selection color**

In the `body` rule (line 56), change `selection:bg-secondary/40` to `selection:bg-primary/20` to use the sage primary for text selection.

- [ ] **Step 9: Tighten heading letter-spacing**

In the `h1-h6, .font-heading` rule (line 69), change `letter-spacing: -0.04em` to `letter-spacing: -0.03em`.

- [ ] **Step 10: Verify**

Run: `cd Frontend && npm run lint && npm run build`
Expected: Both pass with no errors

---

### Task 2: Update Tailwind config and dark mode lock

**Files:**
- Modify: `Frontend/tailwind.config.cjs`

**What changes:** Add `darkMode: false`, update shadow colors from blue-gray to sage, update `--input` token reference.

- [ ] **Step 1: Add darkMode: false**

Add `darkMode: false,` after `media: false,` on line 4:

```js
module.exports = {
    darkMode: false,
    media: false,
```

- [ ] **Step 2: Update boxShadow values to sage**

Replace the `boxShadow` block (lines 101-106) with:

```js
boxShadow: {
    "soft": "0 14px 40px rgba(45, 106, 79, 0.06)",
    "soft-md": "0 20px 60px rgba(45, 106, 79, 0.08)",
    "soft-lg": "0 30px 80px rgba(45, 106, 79, 0.10)",
    "none": "none",
},
```

- [ ] **Step 3: Verify**

Run: `cd Frontend && npm run lint && npm run build`
Expected: Both pass

---

## Phase 2: Cleanup — Dead Code, Unused Components, Bug Fixes

### Task 3: Delete dead files

**Files:**
- Delete: `Frontend/src/components/layout/GlobalLayout.tsx`
- Delete: `Frontend/src/components/particles/ParticleCanvas.tsx`
- Delete: `Frontend/src/components/NavLink.tsx`
- Delete: `Frontend/src/hooks/use-mobile.tsx`

**Verification:** None of these files are imported anywhere in application code. HeroCanvas.tsx is self-contained (has its own `ParticleNet` inline).

- [ ] **Step 1: Delete the four files**

```bash
rm Frontend/src/components/layout/GlobalLayout.tsx
rm Frontend/src/components/particles/ParticleCanvas.tsx
rm Frontend/src/components/NavLink.tsx
rm Frontend/src/hooks/use-mobile.tsx
```

- [ ] **Step 2: Verify no broken imports**

Run: `cd Frontend && npm run lint && npm run build`
Expected: Both pass (these files were not imported by any active code)

---

### Task 4: Remove unused shadcn/ui components

**Files:**
- Delete: All files in `Frontend/src/components/ui/` EXCEPT: `button.tsx`, `toaster.tsx`, `sonner.tsx`, `tooltip.tsx`, `input.tsx`, `label.tsx`, `skeleton.tsx`, `badge.tsx`, `toast.tsx` (needed by toaster)

**Keep list (9 files):**
- `button.tsx` — used in Index, ImageUploader, ChatInterface
- `toaster.tsx` — used in App.tsx
- `sonner.tsx` — used in App.tsx
- `tooltip.tsx` — used in App.tsx (TooltipProvider)
- `input.tsx` — needed for Auth rebuild
- `label.tsx` — needed for Auth rebuild
- `skeleton.tsx` — needed for loading states
- `badge.tsx` — needed for color-coded prediction labels
- `toast.tsx` — dependency of toaster.tsx

**Delete (38 files):**
`accordion.tsx`, `alert.tsx`, `alert-dialog.tsx`, `aspect-ratio.tsx`, `avatar.tsx`, `breadcrumb.tsx`, `calendar.tsx`, `card.tsx`, `carousel.tsx`, `checkbox.tsx`, `collapsible.tsx`, `command.tsx`, `context-menu.tsx`, `dialog.tsx`, `drawer.tsx`, `dropdown-menu.tsx`, `form.tsx`, `hover-card.tsx`, `input-otp.tsx`, `menubar.tsx`, `navigation-menu.tsx`, `pagination.tsx`, `popover.tsx`, `progress.tsx`, `radio-group.tsx`, `resizable.tsx`, `scroll-area.tsx`, `select.tsx`, `separator.tsx`, `sheet.tsx`, `sidebar.tsx`, `switch.tsx`, `table.tsx`, `tabs.tsx`, `textarea.tsx`, `toggle.tsx`, `toggle-group.tsx`

- [ ] **Step 1: Delete unused component files**

```bash
cd Frontend/src/components/ui
rm accordion.tsx alert.tsx alert-dialog.tsx aspect-ratio.tsx avatar.tsx \
   breadcrumb.tsx calendar.tsx card.tsx carousel.tsx checkbox.tsx \
   collapsible.tsx command.tsx context-menu.tsx dialog.tsx drawer.tsx \
   dropdown-menu.tsx form.tsx hover-card.tsx input-otp.tsx menubar.tsx \
   navigation-menu.tsx pagination.tsx popover.tsx progress.tsx \
   radio-group.tsx resizable.tsx scroll-area.tsx select.tsx \
   separator.tsx sheet.tsx sidebar.tsx switch.tsx table.tsx tabs.tsx \
   textarea.tsx toggle.tsx toggle-group.tsx
```

- [ ] **Step 2: Verify build**

Run: `cd Frontend && npm run lint && npm run build`
Expected: Both pass. If build fails due to missing imports in ui/ internal cross-references, those files are already deleted — the build should succeed because no application code imports them.

---

### Task 5: Fix sonner.tsx — remove next-themes

**Files:**
- Modify: `Frontend/src/components/ui/sonner.tsx`

**What changes:** Remove `next-themes` import and `useTheme()` call. Hardcode theme to "light".

- [ ] **Step 1: Replace sonner.tsx content**

```tsx
import { Toaster as Sonner, toast } from "sonner";

type ToasterProps = React.ComponentProps<typeof Sonner>;

const Toaster = ({ ...props }: ToasterProps) => {
  return (
    <Sonner
      theme="light"
      className="toaster group"
      toastOptions={{
        classNames: {
          toast:
            "group toast group-[.toaster]:bg-background group-[.toaster]:text-foreground group-[.toaster]:border-border group-[.toaster]:shadow-lg",
          description: "group-[.toast]:text-muted-foreground",
          actionButton: "group-[.toast]:bg-primary group-[.toast]:text-primary-foreground",
          cancelButton: "group-[.toast]:bg-muted group-[.toast]:text-muted-foreground",
        },
      }}
      {...props}
    />
  );
};

export { Toaster, toast };
```

- [ ] **Step 2: Verify**

Run: `cd Frontend && npm run lint && npm run build`
Expected: Both pass

---

### Task 6: Fix App.tsx — remove duplicate Toaster, console.log, add AnimatePresence

**Files:**
- Modify: `Frontend/src/App.tsx`

**What changes:** Remove Radix `Toaster` (keep only Sonner), remove `console.log`, add `useLocation` + `AnimatePresence` for page transitions.

- [ ] **Step 1: Replace App.tsx content**

```tsx
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import Index from "./pages/Index";
import Classification from "./pages/Classification";
import Chatbot from "./pages/Chatbot";
import Auth from "./pages/Auth";
import NotFound from "./pages/NotFound";
import { AuthProvider } from "@/contexts/AuthContext";
import ProtectedRoute from "@/components/auth/ProtectedRoute";

const queryClient = new QueryClient();

const pageVariants = {
  initial: { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -8 },
};

const pageTransition = {
  type: "tween" as const,
  ease: [0.25, 1, 0.5, 1],
  duration: 0.2,
};

const AnimatedRoutes = () => {
  const location = useLocation();
  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={location.pathname}
        variants={pageVariants}
        initial="initial"
        animate="animate"
        exit="exit"
        transition={pageTransition}
      >
        <Routes location={location}>
          <Route path="/" element={<Index />} />
          <Route path="/classification" element={<Classification />} />
          <Route path="/login" element={<Auth />} />
          <Route path="/chatbot" element={<ProtectedRoute><Chatbot /></ProtectedRoute>} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </motion.div>
    </AnimatePresence>
  );
};

const App = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Sonner />
        <BrowserRouter>
          <AuthProvider>
            <AnimatedRoutes />
          </AuthProvider>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
};

export default App;
```

- [ ] **Step 2: Verify**

Run: `cd Frontend && npm run lint && npm run build`
Expected: Both pass

---

### Task 7: Fix main.tsx — remove console.log

**Files:**
- Modify: `Frontend/src/main.tsx`

- [ ] **Step 1: Read main.tsx, remove any console.log statements**

Read the file, remove debug logging, keep the rest.

- [ ] **Step 2: Verify**

Run: `cd Frontend && npm run lint && npm run build`
Expected: Both pass

---

## Phase 3: Page Rebuilds

### Task 8: Rebuild Index page

**Files:**
- Modify: `Frontend/src/pages/Index.tsx`

**What changes:** Left-aligned hero (not centered), solid heading (no gradient), 3 horizontal feature cards, clean copy, remove `AI_TRIAGE_CORE_v2.1` badge.

- [ ] **Step 1: Read current Index.tsx**

Read to understand current structure before modifying.

- [ ] **Step 2: Rewrite Index.tsx**

Replace with a clean premium clinical layout:
- HeroCanvas background
- Header
- Left-aligned hero: subtitle "AI-powered breast cancer screening", heading "Early detection saves lives", 1-2 sentence description, two CTA buttons (primary sage "Start scan" + secondary outline "Chat with assistant")
- 3 horizontal feature cards (icon left, text right): "Image Analysis", "Medical Chat", "Privacy & Security"
- Subtle disclaimer footer note
- Footer with attribution and pulsing status dot

Key styling rules:
- Use `glass-panel` on feature cards
- Headings: `font-heading font-bold tracking-tight text-foreground` (no gradient)
- Buttons: primary uses `bg-primary text-primary-foreground`, secondary uses `border border-primary/20 text-primary`
- Feature cards: horizontal flex layout, icon in a sage-tinted circle, text left-aligned

- [ ] **Step 3: Verify**

Run: `cd Frontend && npm run lint && npm run build`
Expected: Both pass

---

### Task 9: Rebuild Header component

**Files:**
- Modify: `Frontend/src/components/layout/Header.tsx`

**What changes:** Clean copy (remove `TRIAGE_NODE_ACTIVE`, `NEURAL_MED_CHAT`, `Clinical Hub` labels), add skip-to-content link, add `aria-label` to icon-only buttons, update nav labels.

- [ ] **Step 1: Read current Header.tsx**

Read to understand current structure.

- [ ] **Step 2: Rewrite Header.tsx**

Key changes:
- Add `<a href="#main-content" className="sr-only focus:not-sr-only ...">Skip to content</a>` as first child
- Replace "ClassifierAI" branding with clean "Breast Cancer Companion" or keep "ClassifierAI" but clean the `Clinical Hub` badge
- Nav labels: "Home", "Diagnostic Scan", "Medical Chat" (remove `NEURAL_MED_CHAT`)
- Replace `TRIAGE_NODE_ACTIVE` with "System ready" and a sage dot
- Add `aria-label="Logout"` to logout button
- Add `aria-label="Login"` to login link
- Update active nav state to use sage primary instead of gold secondary

- [ ] **Step 3: Verify**

Run: `cd Frontend && npm run lint && npm run build`
Expected: Both pass

---

### Task 10: Rebuild Classification page and ImageUploader

**Files:**
- Modify: `Frontend/src/pages/Classification.tsx`
- Modify: `Frontend/src/components/classification/ImageUploader.tsx`

**What changes:** Clean copy (remove `MAMMOGRAPHY_IMAGE_DROP_ZONE`, `MAMMOGRAPHY_INF_SCAN`), add skeleton loading, color-coded prediction badges, progress bar for confidence, use shadcn Input/Label for report form.

- [ ] **Step 1: Read current files**

Read both Classification.tsx and ImageUploader.tsx.

- [ ] **Step 2: Update ImageUploader.tsx copy and styling**

Key changes:
- Upload zone: "Upload mammography image" heading, "Drag and drop or click to browse" subtitle, "Supports: JPEG, PNG" caption
- Replace `MAMMOGRAPHY_INF_SCAN` with "Image analysis"
- Replace `COGNITIVE_SCANNER_CALC` with "Analyzing..."
- Add skeleton placeholder div during analysis (use Tailwind `animate-pulse bg-muted rounded-2xl`)
- Prediction result: use `Badge` component with color classes (sage for benign, amber for malignant, muted for normal)
- Confidence: render as a progress bar div (width based on percentage) instead of raw text
- Patient form: use shadcn `Input` + `Label` components instead of raw HTML inputs

- [ ] **Step 3: Update Classification.tsx if needed**

Check if Classification.tsx has any copy or styling that needs updating. Apply same clean copy principles.

- [ ] **Step 4: Verify**

Run: `cd Frontend && npm run lint && npm run build`
Expected: Both pass

---

### Task 11: Rebuild Chatbot page and ChatInterface

**Files:**
- Modify: `Frontend/src/pages/Chatbot.tsx`
- Modify: `Frontend/src/components/chatbot/ChatInterface.tsx`

**What changes:** Remove 3-column info badges, add subtitle, clean message styling, accessible chat attributes, clean suggested prompts.

- [ ] **Step 1: Read current files**

Read both Chatbot.tsx and ChatInterface.tsx.

- [ ] **Step 2: Update Chatbot.tsx**

Key changes:
- Remove the 3-column info badges grid (`VIRTUAL_SCAN_NODE_ONLINE`, etc.)
- Replace with a single subtitle: "Ask me about breast cancer diagnosis, treatment options, or screening guidelines."
- Keep the ChatInterface component below

- [ ] **Step 3: Update ChatInterface.tsx**

Key changes:
- Top bar: "Medical Assistant" title (remove `NEURAL_MED_CHAT`), sage status dot, query count, logout button with `aria-label="Logout"`
- Add `role="log"` and `aria-live="polite"` to the message list container
- Message styling: user messages in `bg-primary/10` (sage tint), bot messages in `bg-muted` (warm white)
- Suggested prompts: clean card-style labels, remove any emoji
- Add `aria-label="Send message"` to send button
- Add `aria-label="Close sources"` / `aria-label="Show sources"` to sources toggle

- [ ] **Step 4: Verify**

Run: `cd Frontend && npm run lint && npm run build`
Expected: Both pass

---

### Task 12: Rebuild Auth page

**Files:**
- Modify: `Frontend/src/pages/Auth.tsx`

**What changes:** Add Header, use shadcn Input/Button, sage submit button, add left branding panel on desktop, clean copy.

- [ ] **Step 1: Read current Auth.tsx**

Read to understand current structure.

- [ ] **Step 2: Rewrite Auth.tsx**

Key changes:
- Import and render `<Header />` at top
- Desktop: two-column layout — left branding panel (project name, tagline, sage accent), right glass-panel form
- Mobile: stacked (form only)
- Form uses shadcn `Input` + `Label` components (not raw HTML)
- Submit button: `bg-primary text-primary-foreground` (sage, not dark)
- Copy: "Sign in to access your chat history" / "Create an account"
- Error display: keep existing logic
- Footer links: "Don't have an account? Sign up" / "Already have an account? Sign in"
- Encryption badge: "Data encrypted end-to-end"
- Add `aria-label` to password toggle button

- [ ] **Step 3: Verify**

Run: `cd Frontend && npm run lint && npm run build`
Expected: Both pass

---

### Task 13: Rebuild NotFound page

**Files:**
- Modify: `Frontend/src/pages/NotFound.tsx`

**What changes:** Add HeroCanvas + Header + glass panel with "404" heading and "Return home" button.

- [ ] **Step 1: Read current NotFound.tsx**

Read to understand current structure.

- [ ] **Step 2: Rewrite NotFound.tsx**

```tsx
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
```

- [ ] **Step 3: Verify**

Run: `cd Frontend && npm run lint && npm run build`
Expected: Both pass

---

## Phase 4: Final Polish

### Task 14: Update HeroCanvas colors to match new palette

**Files:**
- Modify: `Frontend/src/components/layout/HeroCanvas.tsx`

**What changes:** Update the `ParticleNet` material color from `#78909C` (blue-gray) to `#2D6A4F` (sage). Update the fallback gradient colors.

- [ ] **Step 1: Update ParticleNet color**

Change line 115: `color="#78909C"` → `color="#2D6A4F"`

- [ ] **Step 2: Update fallback gradient**

Update the radial gradient in `FallbackVoid` (line 136) to use sage tint:
```js
background: "radial-gradient(circle at center, rgba(45, 106, 79, 0.04) 0%, rgba(247, 246, 243, 0.95) 80%)"
```

- [ ] **Step 3: Update main gradient**

Update the main `div` gradient (line 150) to use warm white:
```jsx
className="fixed inset-0 z-[-1] pointer-events-none bg-gradient-to-b from-[#FEFDFB] via-[#F7F6F3] to-[#FEFDFB]"
```

- [ ] **Step 4: Verify**

Run: `cd Frontend && npm run lint && npm run build`
Expected: Both pass

---

### Task 15: Final verification

- [ ] **Step 1: Full lint**

Run: `cd Frontend && npm run lint`
Expected: No errors

- [ ] **Step 2: Full build**

Run: `cd Frontend && npm run build`
Expected: Successful production build

- [ ] **Step 3: Count remaining shadcn/ui files**

Run: `ls Frontend/src/components/ui/ | wc -l`
Expected: 9 files (button, toaster, sonner, tooltip, input, label, skeleton, badge, toast)

- [ ] **Step 4: Check no dead imports remain**

Run: `cd Frontend && grep -r "GlobalLayout\|ParticleCanvas\|NavLink\|use-mobile" src/ --include="*.tsx" --include="*.ts"`
Expected: No results (all dead imports removed)

- [ ] **Step 5: Check no console.log in src/**

Run: `cd Frontend && grep -r "console.log" src/ --include="*.tsx" --include="*.ts"`
Expected: No results

---

## Summary

| Phase | Tasks | What it delivers |
|-------|-------|-----------------|
| 1. Foundation | 1-2 | New color palette, dark mode lock, sage-tinted glass/shadows |
| 2. Cleanup | 3-7 | Dead files removed, unused components pruned, AnimatePresence wired |
| 3. Pages | 8-13 | All 5 pages rebuilt with clean copy, consistent design system |
| 4. Polish | 14-15 | HeroCanvas color update, final verification |
