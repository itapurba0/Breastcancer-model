# Premium Clinical UI Redesign

**Date:** 2026-06-17
**Status:** Approved — Ready for implementation
**Scope:** Full visual rebuild of all frontend pages. Zero functional changes.

---

## 1. Goal

Transform the Breast Cancer Companion frontend from a developer-jargon-heavy glassmorphism prototype into a premium clinical interface that patients and doctors would trust. The result should feel like Apple Health meets medical portal — refined, warm, professional.

**Constraints:**
- All backend connections, API calls, data flows, chat streaming, classification pipeline, and auth logic remain exactly as-is
- No new features — this is a visual/design overhaul only
- Existing functionality (image upload, Grad-CAM, chat streaming, RAG sources, print reports) must work identically

---

## 2. Color Palette — "Warm Clinical"

Shift from blue-gray + gold to a sage-based health palette.

| Token | Old | New | Usage |
|-------|-----|-----|-------|
| `--background` | `0 0% 100%` (#FFF) | `40 20% 99%` (#FEFDFB) | Page background — warm white |
| `--foreground` | `0 0% 20%` (#333) | `0 0% 10%` (#1A1A1A) | Primary text — near-black |
| `--primary` | `183 8% 55%` (#82979A) | `152 40% 30%` (#2D6A4F) | Deep sage — trust, healing |
| `--secondary` | `46 100% 75%` (#FFD966) | `38 65% 60%` (#E8B86D) | Warm amber — soft warmth |
| `--accent` | `173 40% 92%` (#D4EDED) | `145 50% 85%` (#D8F3DC) | Mint wash — fresh accent |
| `--muted` | `60 50% 98%` (#FAF9F6) | `40 15% 97%` (#F7F6F3) | Off-white surfaces |
| `--border` | `200 6% 94%` (#EDF0F2) | `40 12% 90%` (#E5E3DE) | Warm-tinted borders |
| `--brand` | `200 15% 54%` (#78909C) | `152 40% 30%` (#2D6A4F) | Aligned with primary |
| `--highlight` | `46 100% 75%` (#FFD966) | `38 65% 60%` (#E8B86D) | Aligned with secondary |
| `--sage` | `173 40% 92%` (#D4EDED) | `145 50% 85%` (#D8F3DC) | Aligned with accent |
| `--ring` | `200 12% 44%` (#6B7E85) | `152 35% 40%` (#40916C) | Focus ring — medium sage |

**Glass panel updates:**
- Border: `hsl(var(--primary) / 0.1)` (sage-tinted)
- Shadow: `0 20px 50px hsl(var(--primary) / 0.05)` (sage shadow)
- Inset highlight: `inset 0 1px 2px rgb(255 255 255 / 0.9)` (unchanged)

---

## 3. Typography

### Font families (unchanged)
- **Headings:** Plus Jakarta Sans (300–800)
- **Body:** Inter (300–800)
- **Mono:** Inconsolata (400, 700) — repurposed for data values only

### Type scale
| Element | Classes | Notes |
|---------|---------|-------|
| Hero heading | `text-4xl md:text-5xl font-heading font-bold tracking-tight` | Solid color, no gradient |
| Section heading | `text-2xl md:text-3xl font-heading font-semibold` | |
| Card heading | `text-lg font-heading font-semibold` | |
| Body | `text-base font-sans leading-relaxed` | |
| Label | `text-sm font-medium` | |
| Caption | `text-xs font-sans text-muted-foreground` | |
| Data value | `font-mono text-sm` | Confidence %, timestamps, system IDs |

### Letter-spacing
- Headings: `-0.03em` (Plus Jakarta Sans default is tight, just refine)
- Body: `0` (default)

### Key copy changes
| Old | New |
|-----|-----|
| `MAMMOGRAPHY_IMAGE_DROP_ZONE` | "Upload mammography image" |
| `MAMMOGRAPHY_INF_SCAN` | "Image analysis" |
| `HIPAA_DATA_SECURE` | "Data encrypted end-to-end" |
| `TRIAGE_NODE_ACTIVE` | "System ready" |
| `COGNITIVE_SCANNER_CALC` | "Analyzing..." |
| `AI_TRIAGE_CORE_v2.1` | "Breast Cancer Detection" |
| `NEURAL_MED_CHAT` | "Medical Assistant" |
| `VIRTUAL_SCAN_NODE_ONLINE` | "Connected" |
| `CLASSIFIER_AI_LABS` | Project author attribution |
| `CLASSIFIER_AI_LABS_FOOTER` | © {year} Breast Cancer Companion |

---

## 4. Page Designs

### 4.1 Index (`/`)

**Structure:**
```
HeroCanvas (Three.js particle background)
Header (sticky glass nav)
main
  hero section (left-aligned, not centered)
    subtitle: "AI-powered breast cancer screening"
    heading: "Early detection saves lives"
    description: 1-2 sentences about the tool
    CTA row: [Start scan — primary sage] [Chat with assistant — secondary outline]
  feature section (3 cards, horizontal layout)
    card 1: icon + "Image Analysis" + description
    card 2: icon + "Medical Chat" + description
    card 3: icon + "Privacy & Security" + description
  disclaimer (subtle footer note, small text)
footer
  attribution + copyright
  pulsing status dot (sage)
```

**Key changes:**
- Remove gradient heading text → solid `text-foreground`
- Remove `AI_TRIAGE_CORE_v2.1` badge
- Reduce from 4 feature cards to 3 (combine Cpu + Shield)
- Feature cards: horizontal layout (icon left, text right) instead of icon-above-heading
- Left-aligned hero instead of centered
- Remove `CLASSIFIER_AI_LABS` → real attribution

### 4.2 Classification (`/classification`)

**Structure:**
```
HeroCanvas
Header
main > ImageUploader
  upload zone (glass panel)
    "Upload mammography image" heading
    "Drag and drop or click to browse" subtitle
    "Supports: JPEG, PNG, DICOM" caption
    file input (opacity-0, accessible)
  image preview (when loaded)
    scan animation overlay (keep gradient sweep)
  analysis progress (when analyzing)
    skeleton placeholder for results
    spinner + "Analyzing..." text
  results panel (when complete)
    prediction: "Benign" / "Malignant" / "Normal" — color-coded badge
    confidence: progress bar + percentage
    Grad-CAM heatmap (keep as-is)
  triage risk card (keep logic, clean styling)
  clinical report flow
    patient form (use shadcn Input + Label components)
    report preview (keep print CSS)
    Print/Save buttons
```

**Key changes:**
- Replace raw HTML labels with shadcn Input + Label
- Confidence shown as progress bar, not raw number
- Color-coded prediction badges: sage (benign), amber (malignant), muted (normal)
- Skeleton loading state during analysis

### 4.3 Chatbot (`/chatbot`)

**Structure:**
```
HeroCanvas
Header
main
  subtitle: "Ask me about breast cancer diagnosis, treatment, or screening"
  ChatInterface
    top bar: "Medical Assistant" + status dot + query count + logout
    message list (streaming)
    typing indicator (keep 3-dot bounce)
    suggested prompts (clean card-style, no emoji)
    textarea input + send button
    SourcesPanel (collapsible, clean typography)
    disclaimer footer
```

**Key changes:**
- Remove 3-column technical badges (`VIRTUAL_SCAN_NODE_ONLINE` etc.)
- Replace with single subtitle line
- Clean message styling: user = sage tint, bot = warm white
- Suggested prompts: professional labels, no emoji
- Accessible: `role="log"`, `aria-live="polite"` on message container

### 4.4 Auth (`/login`)

**Structure:**
```
HeroCanvas
Header (with nav — consistent with other pages)
main
  desktop: two-column layout
    left: branding panel (project name, tagline, sage accent)
    right: glass-panel form
  mobile: stacked (form only, no branding panel)
  form
    mode toggle: "Sign in" / "Create account"
    email input (shadcn Input + Label)
    password input (shadcn Input + Label + show/hide toggle)
    submit button (sage primary, full-width)
    error display (keep existing)
    footer link: "Don't have an account? Sign up" / "Already have an account? Sign in"
    encryption badge: "Data encrypted end-to-end"
```

**Key changes:**
- Add Header (was missing)
- Use shadcn Input + Button (was raw HTML)
- Submit button: sage primary (was dark `bg-foreground`)
- Add left branding panel on desktop
- Consistent with design system

### 4.5 NotFound (`*`)

**Structure:**
```
HeroCanvas
Header
main
  glass-panel
    "404" — large heading
    "Page not found" — subtitle
    "The page you're looking for doesn't exist or has been moved."
    [Return home — sage primary button]
```

**Key changes:**
- Add HeroCanvas + Header (was plain bg-muted)
- Glass panel with brand typography
- Sage primary button

---

## 5. Animations & Interactions

### Page transitions
Wire up AnimatePresence directly in `App.tsx` using `useLocation()` from react-router-dom:
- Import `useLocation` and `AnimatePresence` from `framer-motion`
- Get `location = useLocation()` inside the `BrowserRouter` wrapper
- Wrap `<Routes location={location} key={location.pathname}>` with `<AnimatePresence mode="wait">`
- Exit: fade out + slide down 8px (150ms)
- Enter: fade in + slide up 8px (200ms)
- Easing: `ease-out-quart` (exponential deceleration)
- **Critical:** The `key={location.pathname}` on `<Routes>` is required — without it, AnimatePresence cannot detect route changes and exit animations will snap instantly

### Page enter
- Fade in 300ms with 8px upward slide
- Stagger children: 100ms delay between elements

### Hover effects
- Cards: `y: -2` only (remove `scale: 1.01`)
- Buttons: background color transition 200ms (remove `scale(1.02)`)

### Loading states
- Add skeleton placeholders for image results and chat messages
- Use shadcn Skeleton component or custom `animate-pulse` divs

### Three.js background
- Keep on: Index, Classification, Chatbot
- Remove from: Auth (login doesn't need particle effects)

### What we remove
- `text-glow-teal` CSS utility (no-op: `text-shadow: none`)
- `animate-float` keyframe (defined but never applied)
- Blur-in page transitions (replace with fade + slide)
- Scale transforms on hover

---

## 6. Cleanup & Fixes

### Dead code removal
| File | Action |
|------|--------|
| `GlobalLayout.tsx` | Delete — inline transition logic in `App.tsx` |
| `ParticleCanvas.tsx` | Delete — unused (only imported by GlobalLayout) |
| `NavLink.tsx` | Delete — unused (Header uses raw `<Link>`) |
| `use-mobile.tsx` | Delete — unused hook |

### Bundle cleanup
- Remove ~36 unused shadcn/ui components from `src/components/ui/`
- Keep only: `Button`, `Toaster`, `Sonner`, `TooltipProvider`, `Input`, `Label`, `Skeleton`, `Badge`
- `Badge` needed for color-coded prediction labels in Classification page
- Remove `next-themes` import from `sonner.tsx`
- Remove duplicate Radix `Toaster` from `App.tsx` (keep only Sonner)

### Bug fixes
- **Lock light mode explicitly:** Add `darkMode: false` to `tailwind.config.cjs` (prevents system dark mode override). Add `@media (prefers-color-scheme: dark) { :root { color-scheme: light; } }` to `index.css` as a CSS-level safeguard — forces light mode even if OS is dark.
- Wire up `AnimatePresence` page transitions in `App.tsx` (see Section 5 for critical `useLocation()` details)
- Add `<Header />` to Auth page
- Fix NotFound page (add HeroCanvas + Header + glass panel)
- Replace raw HTML in Auth with shadcn Input + Button
- Remove `console.log` debug statements from `App.tsx` and `main.tsx`

### Accessibility (basics)
- Add `aria-label` to icon-only buttons (close, send, logout, password toggle)
- Add skip-to-content link in Header
- Add `role="log"` and `aria-live="polite"` to chat message container
- Ensure file input in upload zone is keyboard-focusable with visible focus ring

---

## 7. Files Modified

| File | Changes |
|------|---------|
| `Frontend/src/index.css` | New HSL palette, remove dead utilities, update glass panel styles, add dark mode CSS lock |
| `Frontend/src/App.tsx` | Add AnimatePresence transitions, remove duplicate Toaster, remove console.log |
| `Frontend/src/main.tsx` | Remove console.log |
| `Frontend/src/pages/Index.tsx` | New layout (left-aligned hero, 3 horizontal cards, clean copy) |
| `Frontend/src/pages/Classification.tsx` | Clean copy, skeleton loading, progress bar for confidence |
| `Frontend/src/pages/Chatbot.tsx` | Remove info badges, clean subtitle, accessible chat |
| `Frontend/src/pages/Auth.tsx` | Full rebuild with shadcn components, Header, branding panel |
| `Frontend/src/pages/NotFound.tsx` | Full rebuild with HeroCanvas + Header + glass panel |
| `Frontend/src/components/layout/Header.tsx` | Clean copy, skip-to-content link, aria-labels |
| `Frontend/src/components/classification/ImageUploader.tsx` | Clean copy, shadcn form components, skeleton loading |
| `Frontend/src/components/chatbot/ChatInterface.tsx` | Clean copy, accessible attributes, suggested prompts |
| `Frontend/tailwind.config.cjs` | Add `darkMode: false`, update color tokens, update shadow colors to sage |
| `Frontend/src/components/ui/button.tsx` | Update primary/secondary variants to new palette |
| `Frontend/sonner.tsx` | Remove next-themes import |

### Files deleted
| File | Reason |
|------|--------|
| `Frontend/src/components/layout/GlobalLayout.tsx` | Unused — logic inlined in App.tsx |
| `Frontend/src/components/three/ParticleCanvas.tsx` | Unused — only imported by GlobalLayout |
| `Frontend/src/components/NavLink.tsx` | Unused — Header uses raw Link |
| `Frontend/src/hooks/use-mobile.tsx` | Unused hook |
| ~36 unused shadcn/ui components | Bundle bloat |

---

## 8. Verification

After implementation:
1. `npm run lint` — no errors
2. `npm run build` — successful production build
3. Manual check: all 5 pages render correctly
4. Manual check: image upload → classification → results → report still works
5. Manual check: chatbot streaming, suggested prompts, sources panel
6. Manual check: auth login/signup/logout flow
7. Manual check: print CSS produces clean report
8. Manual check: responsive on mobile (< 640px), tablet (768px), desktop (1024px+)
