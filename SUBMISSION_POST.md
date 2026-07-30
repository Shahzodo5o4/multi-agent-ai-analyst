# Telegram submission post

Two versions — English and Uzbek. Same content. Copy one, paste, done.
Telegram renders `**bold**` only in Markdown mode; if yours posts plain text the
formatting simply disappears and the post still reads fine.

---

## English

🤖 **Multi-Agent AI Analyst — Capstone Submission**

A supervisor agent that reads a question, routes it to the right specialist,
and runs a critic over the draft before you ever see the answer.

**Try it live:**
🔗 App — https://shahzodo5o4.github.io/multi-agent-ai-analyst/
⚙️ API — https://multi-agent-analyst-api.onrender.com
💻 Code — https://github.com/Shahzodo5o4/multi-agent-ai-analyst

**4 specialists + a critic**
📄 Retriever — RAG over the company documents
🌐 Web — live search for anything outside them
🗄 Data — text-to-SQL against a real 240-customer database
🐍 Code — Python in a sandbox with a hard timeout
✅ Critic — verifies the draft and can send it back for revision

**Results (14 questions × 2 arms, fully graded)**
• Routing accuracy — 100% across 28 runs
• Answer accuracy — 92.9%, judge score 4.43/5
• RAGAS — faithfulness 1.000, answer relevancy 0.898
• Every step traced in Langfuse (29 observations on one question)

**Stack:** LangGraph · Gemini · Qdrant · SQLite · FastAPI · Langfuse
**Cost:** $0 — free tiers only, no credit card anywhere.

Reported honestly in the README: 8 real failures found during testing, each
traced to a specific agent, 6 fixed and 2 documented as open. The critic
ablation showed no quality difference — and the README says so rather than
claiming a win the data doesn't support.

⏳ First request may take ~30s — the free tier sleeps when idle.

---

## Uzbek

🤖 **Multi-Agent AI Analyst — Capstone loyihasi**

Supervisor agent savolni o'qiydi, kerakli mutaxassis agentga yo'naltiradi va
javob foydalanuvchiga yetib borishidan oldin critic uni tekshiradi.

**Sinab ko'ring:**
🔗 Ilova — https://shahzodo5o4.github.io/multi-agent-ai-analyst/
⚙️ API — https://multi-agent-analyst-api.onrender.com
💻 Kod — https://github.com/Shahzodo5o4/multi-agent-ai-analyst

**4 ta mutaxassis agent + critic**
📄 Retriever — hujjatlar bo'yicha RAG
🌐 Web — hujjatlarda yo'q ma'lumot uchun jonli qidiruv
🗄 Data — 240 mijozli real bazaga text-to-SQL
🐍 Code — sandbox ichida Python, qat'iy vaqt chegarasi bilan
✅ Critic — javobni tekshiradi va qayta ishlashga qaytara oladi

**Natijalar (14 savol × 2 rejim, to'liq baholangan)**
• Routing aniqligi — 28 ta sinovda 100%
• Javob aniqligi — 92.9%, baho 4.43/5
• RAGAS — faithfulness 1.000, answer relevancy 0.898
• Har bir qadam Langfuse'da kuzatilgan (bitta savolga 29 ta observation)

**Texnologiyalar:** LangGraph · Gemini · Qdrant · SQLite · FastAPI · Langfuse
**Xarajat:** $0 — faqat bepul tariflar, karta talab qilinmaydi.

README'da hammasi ochiq yozilgan: test paytida 8 ta real xatolik topilgan, har
biri aniq agentga bog'langan, 6 tasi tuzatilgan, 2 tasi ochiq deb qoldirilgan.
Critic ablatsiyasi sifat farqini ko'rsatmadi — README buni yashirmay, borligicha
aytadi.

⏳ Birinchi so'rov ~30s olishi mumkin — bepul tarif bo'sh turganda uxlaydi.

---

## Suggested demo questions

Paste one of these into the app if someone asks for a demo:

- *How many customers churned in Q3 2024, and what does the postmortem say
  caused it?* — routes to data, retriever and code in a single run
- *What is the latest stable version of Python?* — routes to web
- *What is the refund policy?* — routes to the retriever
