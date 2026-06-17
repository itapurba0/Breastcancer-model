# SCRATCHPAD.md

## 1. Current Goal
Polish the frontend design/responsiveness and add chatbot-only auth with MongoDB chat history persistence.

## 2. Key Decisions
- **Fonts**: Consolidated to Inter (body) + Plus Jakarta Sans (headings) + Inconsolata (mono). Dropped Manrope, Outfit, Lora (4→2 sans-serif families).
- **Colors**: `brand`, `highlight`, `sage` converted from hex to HSL CSS vars for Tailwind `/opacity` modifier support.
- **Auth scope**: Chatbot only — classification pages remain public (target users are patients wanting quick scans).
- **Database**: MongoDB Atlas (`medicalChat` DB) — `users` + `chat_sessions` collections.
- **Password hashing**: `bcrypt` directly (not passlib — passlib 1.7.4 has noisy warning with bcrypt 5.x).
- **JWT**: Stored in `sessionStorage` (clears on tab close). 30-day expiry. `HS256` algorithm.
- **Print CSS**: `html { font-size: 7.5pt }` in `@media print` — overrides ALL Tailwind `rem` spacing. `print:grid-cols-2` on image grid (no viewport in print).

## 3. File State
| File | Status |
|------|--------|
| `backend/auth/deps.py` | Done — JWT + bcrypt utils |
| `backend/auth/routes.py` | Done — signup/login/me endpoints |
| `backend/database.py` | Done — MongoDB connection |
| `backend/api.py` | Done — auth router + /chat/history + /chat/save |
| `Frontend/src/contexts/AuthContext.tsx` | Done — login/signup/logout, `isAuthenticated` added |
| `Frontend/src/pages/Auth.tsx` | Done — login/signup page |
| `Frontend/src/components/auth/ProtectedRoute.tsx` | Done — route guard |
| `Frontend/src/App.tsx` | Done — AuthProvider, /login route, ProtectedRoute |
| `Frontend/src/components/chatbot/ChatInterface.tsx` | Done — backend API for history, logout button |
| `Frontend/src/index.css` | Done — print styles, HSL vars, font consolidation |
| `Frontend/src/pages/Index.tsx` | Done — semantic tokens |
| `Frontend/src/pages/Classification.tsx` | Done — semantic tokens |
| `Frontend/src/pages/Chatbot.tsx` | Done — light theme unified |
| `Frontend/src/components/classification/ImageUploader.tsx` | Done — print/report responsive |
| `Frontend/src/components/layout/Header.tsx` | Done — touch targets |
| `Frontend/src/components/layout/GlobalLayout.tsx` | Done — container wrapper removed |
| `Frontend/vite.config.ts` | Done — added /auth proxy |

## 4. Immediate Next Steps
1. Test login → chatbot redirect end-to-end (user reported it wasn't navigating after login)
2. Verify chat history saves to MongoDB `chat_sessions` collection after first conversation
3. Add chatbot link in Header nav pointing to `/login` when unauthenticated

## 5. Warnings
- `backend/chatbot/engine.py` loads `AsyncOpenAI` at module level with `OPENROUTER_API_KEY` — import fails if env var not set in `backend/chatbot/.env`.
- Backend venv is at `backend/.venv` (Python 3.12). Run backend with `.venv/bin/python` or `uvicorn`.
- `@tailwindcss/typography` was uninstalled (unused). `App.css` deleted (dead scaffold).
- `no-unused-vars` is OFF in ESLint. `strictNullChecks: false` in tsconfig.
