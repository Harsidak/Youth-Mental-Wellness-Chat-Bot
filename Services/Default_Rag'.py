from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from Credentials import GEMINI_API_KEY, PINECONE_API_KEY


#Following update by me, Suryansh, uncomment it to see if it works, then comment it out again if it doesn't :-
#from googletrans import Translator
# My edits end here

# 1️⃣ Initialize embedding model
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY
)

# 2️⃣ Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("gemini-index")  # same index created in indexing.py

# 3️⃣ Initialize LLM (Gemini chat model)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=GEMINI_API_KEY
)

# 4️⃣ User input
user_input = input("Describe what you're feeling or experiencing: ")

# 5️⃣ Classification prompt – extract symptoms and possible disorders
classification_prompt = f"""
You are a mental health expert helping to identify possible disorders based on user descriptions.
Read the user's input and extract:
1. Relevant symptoms mentioned.
2. Possible disorders these symptoms might indicate.
3. Confidence level for each disorder.

User Input:
\"\"\"{user_input}\"\"\"

Return the output in this format:
Symptoms: [list of symptoms]
Possible Disorders: [list of disorders with confidence score]
If uncertain, mention: "Insufficient information to confidently classify."
"""

classification_response = llm.predict(classification_prompt)
print("\n--- Classification Output ---")
print(classification_response)

# 6️⃣ Extract top disorder for querying Pinecone (basic parsing, refine as needed)
# For this example, we'll assume you manually pick the top disorder from the output
# In production, you should parse this programmatically
top_disorder = "Anxiety Disorder"  # Example placeholder

# 7️⃣ Retrieve context from Pinecone using disorder name
query_vector = embedding.embed_query(top_disorder)
results = index.query(
    vector=query_vector,
    top_k=5,
    include_metadata=True
)
context = "\n".join([match['metadata']['text_snippet'] for match in results['matches']])

print("\n--- Retrieved Context ---")
print(context[:500], "...")  # Print preview

# 8️⃣ Generate final response using user input, classification, and retrieved context
final_prompt = f"""
A youth has described their mental health concerns as follows:
\"\"\"{user_input}\"\"\"

Based on this description, the possible disorders are:
{classification_response}

Here is additional information about these disorders:
{context}

Provide a compassionate, supportive, and step-by-step response explaining how the youth can address these issues. If uncertain, advise contacting a healthcare professional.
"""

final_response = llm.predict(final_prompt)

print("\n=== FINAL RESPONSE ===")
print(final_response)