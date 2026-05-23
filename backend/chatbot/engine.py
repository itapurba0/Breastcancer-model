# import os
# import warnings
# from dotenv import load_dotenv
# from qdrant_client import QdrantClient
# import ollama

# # Mute the harmless Qdrant deprecation warnings to keep the terminal clean
# warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")

# # Load environment variables
# load_dotenv()

# # 1. Configuration
# QDRANT_URL = os.getenv("QDRANT_URL")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# COLLECTION_NAME = "medical_knowledge_base"

# # 2. Connect to Qdrant
# try:
#     qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=15)
#     qdrant.set_model("BAAI/bge-small-en-v1.5") 
# except Exception as e:
#     print(f"Backend Error: Failed to connect to Qdrant: {e}")
#     qdrant = None

# def retrieve_context(user_question: str, top_k: int = 2) -> str:
#     """Searches Qdrant for the most relevant indexed medical chunks."""
#     if not qdrant:
#         return ""

#     try:
#         # Reverted back to the stable query method that your cloud cluster supports
#         search_results = qdrant.query(
#             collection_name=COLLECTION_NAME,
#             query_text=user_question,
#             limit=top_k
#         )
        
#         # Combine the chunks into a single context string
#         context_pieces = [hit.document for hit in search_results if hasattr(hit, 'document')]
#         return "\n\n---\n\n".join(context_pieces)
#     except Exception as e:
#         print(f"\n[Retrieval Error]: {e}")
#         return ""

# def generate_rag_response(user_question: str):
#     """Combines context and streams the response directly to the console."""
    
#     print("\n[1/2] Searching Qdrant Database for context...")
#     context = retrieve_context(user_question)
    
#     if not context.strip():
#         print("[!] No context found in database. The AI is answering from memory.")
        
#     system_prompt = f"""
#     You are a compassionate, highly accurate medical AI assistant for ClassifierAI. 
#     Your goal is to explain medical information so that an everyday person with no medical background can easily understand it.

#     STRICT RULES:
#     1. USE ONLY THE CONTEXT: Answer the user's question using ONLY the medical context provided below. Do not invent or guess medical facts.
#     2. PLAIN ENGLISH: Translate all complex medical jargon from the context into simple, comforting, and clear language. Use analogies if it helps.
#     3. SAFETY FIRST: If the context does not contain the answer, politely state: "I don't have enough information in my current medical files to answer that safely."
#     4. NO INTERNAL THINKING: Provide ONLY your final, polished response to the user. DO NOT output your internal thinking process, reasoning steps, or drafting notes. Start your answer directly.

#     Context:
#     {context}
#     """
    
#     print("[2/2] Generating answer (Streaming)...\n")
#     print("🤖 AI: ", end="", flush=True)
    
#     try:
#         stream = ollama.chat(
#             model='qwen3.5:4b', 
#             messages=[
#                 {'role': 'system', 'content': system_prompt},
#                 {'role': 'user', 'content': user_question}
#             ],
#             stream=True 
#         )
        
#         for chunk in stream:
#             print(chunk['message']['content'], end='', flush=True)
            
#         print("\n") 
            
#     except Exception as e:
#         print(f"\n[Ollama Error]: Make sure Ollama is running and you have pulled the model! ({e})")

# if __name__ == "__main__":
#     print("=== ClassifierAI Local RAG Test ===")
#     test_q = "What is the early symptom of malignant breast cancer?"
#     print(f"\n👤 You: {test_q}")
#     generate_rag_response(test_q)




# import os
# import time
# import warnings
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# from dotenv import load_dotenv
# from qdrant_client import QdrantClient
# from regex import search
# from sentence_transformers import SentenceTransformer
# import ollama

# # ---------------------------------------------------
# # Clean terminal warnings
# # ---------------------------------------------------

# warnings.filterwarnings(
#     "ignore",
#     category=UserWarning,
#     module="qdrant_client"
# )

# # ---------------------------------------------------
# # Load Environment Variables
# # ---------------------------------------------------
# # ---------------------------------------------------
# # Load Environment Variables
# # ---------------------------------------------------
# load_dotenv()
# QDRANT_URL = os.getenv("QDRANT_URL")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# COLLECTION_NAME = "medical_knowledge_base"

# # ---------------------------------------------------
# # Lazy Initialization
# # ---------------------------------------------------
# # We set these to None initially to prevent the Segmentation Fault
# embed_model = None
# qdrant = None

# def init_services():
#     """Initializes the ML model and database only when needed."""
#     global embed_model, qdrant
    
#     if embed_model is None:
#         print("Loading embedding model safely...")
#         embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        
#     if qdrant is None:
#         try:
#             qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=15)
#             print("✅ Connected to Qdrant")
#         except Exception as e:
#             print(f"❌ Qdrant Connection Error: {e}")
#             qdrant = None

# # ---------------------------------------------------
# # Retrieve Context
# # ---------------------------------------------------
# def retrieve_context(user_question: str, top_k: int = 1):
#     # Ensure services are loaded before we try to search!
#     init_services()

#     if not qdrant or not embed_model:
#         return ""

#     try:
#         start = time.time()
        
#         # ... (Keep the rest of your retrieve_context logic the same) ...

#         # Create query embedding
#         query_vector = embed_model.encode(
#             user_question,
#             normalize_embeddings=True
#         ).tolist()

#         # Search vectors
#         search_response = qdrant.query_points(
#             collection_name=COLLECTION_NAME,
#             query=query_vector,
#             limit=top_k,
#             with_payload=True
#         )
#         results = search_response.points

#         print(
#             f"⚡ Retrieval Time: "
#             f"{time.time()-start:.2f}s"
#         )

#         # Extract the chunks safely, bypassing Pylance strict-typing errors
#         context_chunks = []

#         for hit in results:
#             # getattr() grabs the payload safely, and the comment forces Pylance to mute the error
#             payload = getattr(hit, "payload", {}) # type: ignore
            
#             # Ensure payload is a dictionary, then safely extract the text
#             if isinstance(payload, dict):
#                 text = payload.get("text", "")
                
#                 if text:
#                     context_chunks.append(text[:1200])

#         return "\n\n---\n\n".join(context_chunks)

#     except Exception as e:

#         print(f"\n❌ Retrieval Error: {e}")

#         return ""

# # ---------------------------------------------------
# # Generate RAG Response
# # ---------------------------------------------------

# def generate_rag_response(user_question: str) -> str:
#     print("\n[1/2] Retrieving medical context...")
#     context = retrieve_context(user_question)

#     if not context.strip():
#         print("⚠️ No relevant context found. Short-circuiting.")
#         # Return the safe answer directly to the API!
#         return "I'm sorry, but I don't have enough information in my current medical files to answer that safely."

#     system_prompt = f"""
#     You are a compassionate and accurate medical AI assistant.
#     Your job is to explain medical information in simple, clear, beginner-friendly language.
    
#     STRICT RULES:
#     1. ONLY use the provided medical context.
#     2. Do NOT invent medical facts.
#     3. Keep answers short and understandable.
#     4. Do NOT reveal internal reasoning.
    
#     MEDICAL CONTEXT:
#     {context}
#     """

#     print("\n[2/2] Generating response...\n🤖 AI: ", end="", flush=True)

#     try:
#         stream = ollama.chat(
#             model="phi3:mini",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_question}
#             ],
#             stream=True
#         )

#         full_response = ""
        
#         for chunk in stream:
#             content = chunk["message"]["content"]
#             full_response += content  # Save the word to our final string
#             print(content, end="", flush=True) # Keep printing it to the backend terminal for debugging
            
#         print("\n")
        
#         # Hand the completed paragraph back to api.py
#         return full_response

#     except Exception as e:
#         print(f"\n❌ Ollama Error: {e}")
#         return "System Error: Unable to connect to the local AI engine."

# # ---------------------------------------------------
# # Main
# # ---------------------------------------------------

# if __name__ == "__main__":

#     print("=" * 50)
#     print("🩺 ClassifierAI Local RAG Test")
#     print("=" * 50)

#     while True:

#         user_question = input("\n👤 You: ")

#         if user_question.lower() in ["exit", "quit"]:
#             print("\n👋 Goodbye!")
#             break

#         generate_rag_response(user_question)


import os
import time
import warnings
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from fastembed import TextEmbedding  
import ollama

# Load Environment Variables
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
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
            with_payload=True
        )

        print(f"⚡ Retrieval Time: {time.time()-start:.2f}s")

        context_chunks = []
        for hit in search_response.points:
            if hit.payload and "text" in hit.payload:
                context_chunks.append(hit.payload["text"][:1200])

        return "\n\n---\n\n".join(context_chunks)
    except Exception as e:
        print(f"\n❌ Retrieval Error: {e}")
        return ""

def generate_rag_response(user_question: str):
    print("\n[1/2] Retrieving medical context...")
    context = retrieve_context(user_question)

    if not context.strip():
        return "I'm sorry, but I don't have enough information in my current medical files to answer that safely."

    system_prompt = f"""
    You are a compassionate and accurate medical AI assistant.
    Your job is to explain medical information in simple, clear, beginner-friendly language.
    
    STRICT RULES:
    1. ONLY use the provided medical context.
    2. Do NOT invent medical facts.
    3. Keep answers short and understandable.
    4. Do NOT reveal internal reasoning.
    
    MEDICAL CONTEXT:
    {context}
    """

    print("\n[2/2] Generating response...\n")
    print("🤖 AI: ", end="", flush=True)

    try:
        stream = ollama.chat(
            model="phi3:mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            stream=True
        )

        full_response = ""
        for chunk in stream:
            content = chunk["message"]["content"]
            full_response += content
            print(content, end="", flush=True)
            
        print("\n")
        return full_response

    except Exception as e:
        print(f"\n❌ Ollama Error: {e}")
        return "System Error: Unable to connect to the local AI engine."