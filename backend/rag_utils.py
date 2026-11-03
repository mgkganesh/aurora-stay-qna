
from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 5,vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks, model) -> str:

    # Join the chunks into one readable block (instead of passing a Python list)
    context = "\n\n---\n\n".join(retrieved_chunks)

    prompt = f"""
    You are an internal assistant for AuroraStay Hotels front-desk and reservations staff.
    Answer the staff member's question using only the retrieved AuroraStay policy information below.

    Staff Question:
    {query}

    Retrieved Policy Information:
    {context}

    Rules:
    1. Use only facts stated in the retrieved policy information. Do not use outside knowledge or assumptions.
    2. Answer every part of the question. If it asks about several things (for example a fee and a refund rule, or two properties), address each one.
    3. When a property is named (by code such as ASH-CHI or by city or name such as "Chicago" or "Miami Brickell"), use only that property's row. Never apply one property's fee, time, or rule to another property.
    4. Quote amounts, times, limits, and deadlines exactly as written (for example "$50 per night, $150 stay cap" or "6:00 PM local property time one day before arrival").
    5. Include conditions and exceptions that change the answer (for example "unless the confirmation states a different rule", or that service animals are not charged a pet fee).
    6. Start with the direct answer in the first sentence, then add only the supporting details needed. Do not add unrelated policy text.
    7. Keep the answer concise: 2 to 5 sentences or a short bullet list.
    8. Do not invent fees, deadlines, amenities, exceptions, or approvals.
    9. If only part of the question is covered, answer the covered part and clearly state which part is not covered, so staff can check the live property profile or contact the property team.
       Only if nothing relevant is retrieved, reply exactly:
       "The available policy documents do not contain enough information to answer this question."
    10. If the retrieved information conflicts, clearly mention the conflict.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 5,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
