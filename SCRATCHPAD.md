# SCRATCHPAD.md

## 1. Current Goal
Desktop layout refactor (component split appraoch), two-column results layout on lg+, UI/UX improvements guided by ui-ux-pro-max + critique + web-design-guidelines + frontend-design skills.

## 2. Key Decisions
- **Layout**: Classification page uses `max-w-7xl` with `lg:grid-cols-5` (3/5 main + 2/5 sticky sidebar). Sidebar holds How It Works + Tips card.
- **Result Panel**: On desktop, results split into two columns (gauge/images left, triage/facilities right). On mobile, single column preserved.
- **Component split**: ImageUploader (616→~230 lines) is now a thin orchestrator. Extracted: UploadPanel, ResultPanel, ConfidenceHeader, ImageComparison, TriageCard, ActionBar, HowItWorks.
- **Accessibility**: Added `prefers-reduced-motion` CSS, `focus-visible:ring-2` on all interactive elements, `role="alert"` on errors, `aria-label` on icon buttons.
- **Motion**: Exponential easing (`cubic-bezier(0.25, 1, 0.5, 1)`), durations follow 100/300/500ms rule, no `transition: all`, decorative infinite animation removed.
- **Anti-slop**: No `backdrop-filter: blur` on result panels, `whileHover` scale replaced with border transitions, left-aligned headers (not centered).
- **Typography**: `tabular-nums` on numbers, `text-balance` on headings, `text-[10px]` upgraded to `text-xs` minimum.

## 3. File State
| File | Status |
|------|--------|
| `backend/auth/deps.py` | Done — JWT + bcrypt utils |
| `backend/auth/routes.py` | Done — signup/login/me endpoints |
| `backend/database.py` | Done — MongoDB connection |
| `backend/api.py` | Done — auth router + /chat/history + /chat/save |
| `backend/model_utils.py` | Done — CONFIDENCE_THRESHOLD, inconclusive flag |
| `backend/facilities.json` | Done — 20 Indian hospitals/facilities dataset |
| `Frontend/src/contexts/AuthContext.tsx` | Done — login/signup/logout |
| `Frontend/src/pages/Auth.tsx` | Done — login/signup page |
| `Frontend/src/pages/Classification.tsx` | Done — two-column grid, left-aligned header, tips card |
| `Frontend/src/components/auth/ProtectedRoute.tsx` | Done — route guard |
| `Frontend/src/App.tsx` | Done — AuthProvider, /login route, ProtectedRoute |
| `Frontend/src/components/chatbot/ChatInterface.tsx` | Done — backend API for history, logout button |
| `Frontend/src/index.css` | Done — prefers-reduced-motion, tabular-nums, text-balance, fixed transition:all |
| `Frontend/src/pages/Index.tsx` | Done — semantic tokens |
| `Frontend/src/pages/Chatbot.tsx` | Done — light theme unified |
| `Frontend/src/components/layout/Header.tsx` | Done — touch targets |
| `Frontend/src/components/layout/GlobalLayout.tsx` | Done — container wrapper removed |
| `Frontend/vite.config.ts` | Done — added /auth + /facilities proxy |

### New Components
| Component | File | Purpose |
|-----------|------|---------|
| `UploadPanel` | `classification/UploadPanel.tsx` | Drag-drop, file select, progress steps |
| `ResultPanel` | `classification/ResultPanel.tsx` | Desktop two-column orchestrator |
| `ConfidenceHeader` | `classification/ConfidenceHeader.tsx` | SVG gauge + prediction badge |
| `ImageComparison` | `classification/ImageComparison.tsx` | Side-by-side original + Grad-CAM |
| `TriageCard` | `classification/TriageCard.tsx` | Risk assessment card |
| `ActionBar` | `classification/ActionBar.tsx` | Export button + disclaimer |
| `HowItWorks` | `classification/HowItWorks.tsx` | 3-step explainer card |
| `FacilityRecommendation` | `classification/FacilityRecommendation.tsx` | Location-aware facility finder |

## 4. Immediate Next Steps
1. Test full classification flow (upload → results → triage → facilities → report)
2. Test confidence thresholding (use an image that produces <60% confidence)
3. Verify facility recommendation works with city input and geolocation
4. Test prefers-reduced-motion in browser DevTools

## 5. Warnings
- `backend/chatbot/engine.py` loads `AsyncOpenAI` at module level with `OPENROUTER_API_KEY` — import fails if env var not set in `backend/chatbot/.env`.
- Backend venv is at `backend/.venv` (Python 3.12). Run backend with `.venv/bin/python` or `uvicorn`.
- `@tailwindcss/typography` was uninstalled (unused). `App.css` deleted (dead scaffold).
- `no-unused-vars` is OFF in ESLint. `strictNullChecks: false` in tsconfig.
- `GOOGLE_PLACES_API_KEY` env var optional — facility search degrades gracefully with curated dataset only.
- ResultPanel receives `selectedImage` as prop but ImageUploader wraps it in AnimatePresence — the check `reportStep === "hidden"` controls visibility.
