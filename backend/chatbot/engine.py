import os
import time
import warnings
from qdrant_client import QdrantClient
from fastembed import TextEmbedding  
from openai import AsyncOpenAI

from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
COLLECTION_NAME = "medical_knowledge_base"

# Mute warnings
warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")

print("Loading lightweight embedding model...")
# Initialize FastEmbed (Uses ONNX runtime instead of PyTorch)
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Connect to Qdrant
try:
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=15)
    print("✅ Connected to Qdrant")
except Exception as e:
    print(f"❌ Qdrant Connection Error: {e}")
    qdrant = None

ai_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def retrieve_context(user_question: str, top_k: int = 1):
    if not qdrant:
        return ""

    try:
        start = time.time()

        # FastEmbed generates a generator, so we cast to list and grab the first item
        query_vector = list(embed_model.embed([user_question]))[0].tolist()

        search_response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            score_threshold=0.5,  # Optional: filter out low-relevance results
            with_payload=True
        )

        print(f"⚡ Retrieval Time: {time.time()-start:.2f}s")

        context_chunks = []
        for hit in search_response.points:
            if hit.payload and "text" in hit.payload:
                context_chunks.append(hit.payload["text"][:1200])
        print(f"🔍 Retrieved {len(context_chunks)} chunks from Qdrant.")
        return "\n\n---\n\n".join(context_chunks)
    except Exception as e:
        print(f"\n❌ Retrieval Error: {e}")
        return ""

# Change the function to act as a generator
# 3. NEW: Add 'async' to the function definition
async def generate_rag_response(messages):
    # --- BULLETPROOF DATA EXTRACTION ---
    if isinstance(messages, str):
        latest_user_question = messages
        ai_memory = [{"role": "user", "content": messages}]
    else:
        last_msg = messages[-1]
        if isinstance(last_msg, dict):
            latest_user_question = last_msg.get("content", "")
        else:
            latest_user_question = getattr(last_msg, "content", str(last_msg))

        ai_memory = []
        for msg in messages:
            if isinstance(msg, dict):
                ai_memory.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
            else:
                ai_memory.append({"role": getattr(msg, "role", "user"), "content": getattr(msg, "content", "")})
    # ------------------------------------

    print(f"\n[1/2] Retrieving context for: '{latest_user_question}'...")
   
    context = retrieve_context(latest_user_question)
    print(f"\n[Context Retrieved]:\n{context}\n")
    if not context.strip():
        yield "I'm sorry, but I don't have enough information in my current medical files to answer that safely."
        return

    system_prompt = f"""
You are a warm, empathetic Breast Cancer Awareness Companion.

Your purpose is to help users understand breast cancer using ONLY the information found in the provided documents.

=========================
CORE RULES
=========================

1. NEVER use your own medical knowledge.
2. NEVER guess or infer information.
3. ONLY answer using information explicitly present in the provided context.
4. If the answer is not found in the provided context, say:

"I'm sorry, but I couldn't find information about that in the documents I have. I can only answer questions using the breast cancer information available in these resources."

Do NOT add explanations from your own knowledge.

=========================
SCOPE
=========================

You ONLY answer questions related to breast cancer.

If asked about:
- another cancer
- another disease
- general medicine
- unrelated topics

Reply politely:

"I'm designed specifically to provide breast cancer awareness and educational information. I can't answer questions outside that topic."

=========================
STYLE
=========================

- Be warm and supportive.
- Use simple everyday language.
- Explain things like you're talking to someone with no medical background.
- Never sound like a doctor.
- Keep answers concise unless the user asks for detail.

=========================
RESPONSE RULES
=========================

Before answering:

Step 1:
Check whether the information exists in the provided context.

Step 2:
If YES:
Answer ONLY using that information.

Step 3:
If NO:
State that the information is not available in the documents.
Do not use outside knowledge.

Never mention facts, medicines, statistics, or treatments that are absent from the provided context.

=========================
MEDICAL CONTEXT
=========================

{context}
"""

    ai_memory.insert(0, {"role": "system", "content": system_prompt})

    print("\n[2/2] Generating response from AI...\n🤖 AI: ", end="", flush=True)

    try:
        # 4. NEW: Add 'await' so Python knows it can handle other users while OpenRouter thinks
        # Request the stream with strict consistency parameters
        stream = await ai_client.chat.completions.create(
            model="nvidia/nemotron-3-super-120b-a12b:free", # 1. Force a specific model instead of openrouter/free
            messages=ai_memory,
            temperature=0.3, # 2. NEW: Drops randomness to absolute zero
            stream=True
        )

        # 5. NEW: Change 'for' to 'async for' to handle the asynchronous stream
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                # print(content, end="", flush=True)
                yield content
            
        print("\n")

    except Exception as e:
        print(f"\n❌ OpenRouter Error: {e}")
        yield "System Error: Unable to connect to the cloud AI engine."