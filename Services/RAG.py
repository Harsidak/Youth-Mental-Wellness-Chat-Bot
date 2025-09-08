from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pinecone import Pinecone
from typing import List, Dict, Optional

# Assuming Credentials.py is in the same Services/ folder.
from Credentials import GEMINI_API_KEY, PINECONE_API_KEY

# --- INITIALIZE MODELS AND SERVICES (Done once when the server starts) ---

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY
)

pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("gemini-index")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    google_api_key=GEMINI_API_KEY,
    convert_system_message_to_human=True
)


# --- CORE RAG FUNCTION (Upgraded to handle initial turn) ---

def get_rag_response(user_data: dict, history: List[Dict[str, str]], user_query: Optional[str] = None) -> str:
    """
    Processes user data and chat history. If a user_query is provided, it's a conversational turn.
    If not, it's the initial turn and generates the specific welcoming prompt.
    """
    print("--- Starting RAG Pipeline ---")
    try:
        # **FIXED**: This block now handles the initial message as you requested.
        if not user_query:
            intensity = user_data.get('intensity', 'this')
            emotions_str = ", ".join(user_data.get('emotions', ['these emotions']))

            # This is the new, specific opening message.
            response_text = f"Hi I am Sahaara AI Chat-Bot, I see that you're experiencing {emotions_str} with a {intensity}/10 intensity. Thank you for sharing that. Can you please describe the situation you are dealing with in as much detail as you're comfortable with, so I can understand better?"

            print("--- Generated initial prompt for user. ---")
            return response_text

        # --- This is the logic for all subsequent conversational turns ---
        formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        extraction_prompt = f"""
        Based on the user's latest message, extract the key psychological topics.
        List them as a concise, comma-separated string.
        Latest User Message: "{user_query}"
        Extracted Keywords:
        """
        extracted_keywords = llm.invoke(extraction_prompt).content
        print(f"LLM Extracted Keywords for Retrieval: {extracted_keywords}")

        query_vector = embedding_model.embed_query(extracted_keywords)
        results = pinecone_index.query(vector=query_vector, top_k=4, include_metadata=True)
        retrieved_context = "\n\n".join([match['metadata']['text_snippet'] for match in results.get('matches', [])])

        final_prompt = f"""
        You are Sahaara AI, a compassionate and safe AI mental wellness assistant. Your knowledge is based on clinical guides, but you are NOT a doctor.
        **Strict Safety Rules:** Never give a diagnosis or prescribe medication. If self-harm is mentioned, strongly advise contacting a crisis hotline immediately.

        A user is talking to you. Here is the conversation history:
        <history>{formatted_history}</history>

        Here is the user's latest message:
        <user_query>{user_query}</user_query>

        You have retrieved this relevant knowledge:
        <retrieved_knowledge>{retrieved_context}</retrieved_knowledge>

        Your Task: Respond to the user's latest message empathetically, integrating the retrieved knowledge naturally and ending with an open-ended question. Follow all safety rules.
        """

        final_response = llm.invoke(final_prompt).content
        print("--- RAG Pipeline Complete ---")
        return final_response

    except Exception as e:
        print(f"An error occurred in the RAG pipeline: {e}")
        return "I'm sorry, I encountered a technical issue. Could you please try again?"

