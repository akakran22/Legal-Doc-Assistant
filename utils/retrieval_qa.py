import os
from groq import Groq
from .chunk_embeddings import HuggingFaceEmbeddings

class RetrievalQA:
    def __init__(self, vectordb):
        self.vdb = vectordb
        self.embeddings = HuggingFaceEmbeddings()
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY required")
        self.client = Groq(api_key=groq_api_key)

    def ask(self, query, top_k=6, max_tokens=1200, temperature=0.2):
        try:
            print(f"Processing: {query[:50]}...")

            q_lower = query.strip().lower()

            # Greeting phrases
            greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

            # Small talk phrases
            small_talks = ["how are you", "how is it going", "what's up", "how do you do", "are you there"]

            # Check for greetings
            if q_lower in greetings:
                return "Hello! How can I help you?", []

            # Check for small talk
            if q_lower in small_talks:
                return "I'm here to assist you with legal queries. How can I help?", []

            # Embed query
            q_emb = self.embeddings.encode([query], show_progress=False)
            hits = self.vdb.search(q_emb, top_k=top_k)

            # Filter hits with a relevance threshold (example: 10%)
            relevant_hits = [h for h in hits if h.get('score', 0) >= 10.0]

            # Build context if any relevant hits found
            context = ""
            if relevant_hits:
                context_parts = []
                for h in relevant_hits:
                    source = h['payload'].get('source', 'Unknown')
                    page = h['payload'].get('page', 'Unknown')
                    text = h['text'][:1000]
                    context_parts.append(f"[Source: {source}, Page: {page}]\n{text}")
                context = "\n\n---\n\n".join(context_parts)
                print(f"Found {len(relevant_hits)} relevant docs")
            else:
                print("No sufficiently relevant documents found, using LLM base knowledge")

            # System prompt
            system_prompt = """You are an expert Indian legal assistant. Provide accurate, well-structured legal information.

Rules:
1. Prefer context from Indian legal documents if available.
2. If context is insufficient, use your base knowledge as an Indian law expert.
3. Always provide proper citations if context exists.
4. Format:
   - ## for main sections
   - ### for subsections
   - **Bold** for important terms
   - Always quote sections/acts exactly if available"""

            # User prompt includes context only if available
            user_prompt = f"Legal Query: {query}\n"
            if context:
                user_prompt += f"\nContext from Indian Legal Documents:\n{context}\n"
            user_prompt += "\nProvide a clear, comprehensive answer."

            print("Generating response...")
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            answer = completion.choices[0].message.content if completion.choices else "Sorry, couldn't generate response."
            print("Response generated")
            return answer, relevant_hits

        except Exception as e:
            print(f"Error: {e}")
            return f"Error processing question: {str(e)}. Please try again.", []

