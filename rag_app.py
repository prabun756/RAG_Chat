import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from typing import List, Dict

# ----------------------
# Global Setup
# ----------------------
@st.cache_resource
def setup_rag():
    # Embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load document
    loader = TextLoader("data.txt")  # Replace with your file
    docs = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    # Build FAISS index
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # LLaMA3 via Ollama
    llm = OllamaLLM(model="llama3")

    # QA prompt
    qa_prompt = PromptTemplate.from_template(
        "Use the following pieces of context to answer the question at the end. "
        "If you don't know the answer, say that you don't know. "
        "Context: {context}\nQuestion: {question}\nAnswer:"
    )

    base_qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": qa_prompt}
    )

    return llm, retriever, base_qa_chain

# ----------------------
# 🔁 Corrective RAG
# ----------------------
def corrective_rag(question: str, llm, base_qa_chain) -> str:
    result = base_qa_chain.invoke({"query": question})
    initial_answer = result["result"]

    feedback_prompt = f"""
    Evaluate the quality of the following answer to the question.
    If it's incomplete or incorrect, refine it.

    Question: {question}
    Initial Answer: {initial_answer}

    Is the answer sufficient? If not, improve it based on the question and context.
    Improve and return the final answer.
    """
    refined_answer = llm.invoke(feedback_prompt).strip()
    return refined_answer

# ----------------------
# 🤔 Self-RAG
# ----------------------
def self_rag(question: str, llm, retriever) -> Dict[str, str]:
    retrieved_docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    reflection_prompt = f"""
    You are performing Self-RAG. Evaluate if the provided context is relevant to the question.
    
    Question: {question}
    Context: {context}
    
    Think: Does the context contain information useful for answering the question?
    Grade the retrieval on relevance (High/Medium/Low), then generate an answer.
    """
    reflection_response = llm.invoke(reflection_prompt).strip()

    grade = "Medium"
    if "Low" in reflection_response:
        grade = "Low"
    elif "High" in reflection_response:
        grade = "High"

    final_answer = base_qa_chain.invoke({"query": question})["result"]

    return {
        "reflection": reflection_response,
        "retrieval_grade": grade,
        "final_answer": final_answer
    }

# ----------------------
# 🔗 Fusion RAG
# ----------------------
def hybrid_queries(question: str, llm) -> List[str]:
    prompt = f"""
    Given the question: "{question}", generate 3 alternative versions of this question that might help retrieve different perspectives or facts.
    Return only the list of questions separated by newlines.
    """
    response = llm.invoke(prompt).strip()
    return response.split("\n")

def fusion_rag(question: str, llm, retriever) -> str:
    alternate_queries = hybrid_queries(question, llm)
    all_docs = []
    for q in alternate_queries:
        docs = retriever.get_relevant_documents(q)
        all_docs.extend(docs)

    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
    combined_context = "\n\n".join([doc.page_content for doc in unique_docs])

    final_prompt = f"""
    Use the following context to answer the question.

    Context: {combined_context}
    Question: {question}
    Answer:
    """
    answer = llm.invoke(final_prompt).strip()
    return answer

# ----------------------
# 🖥️ Streamlit UI
# ----------------------
st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("📚 RAG Chat App - LLaMA3 + Streamlit")
st.markdown("Explore **Corrective**, **Self**, and **Fusion RAG** techniques.")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# Setup RAG
llm, retriever, base_qa_chain = setup_rag()

# Input form
with st.form(key='rag_form'):
    user_question = st.text_input("Enter your question:")
    col1, col2, col3 = st.columns(3)
    run_corrective = col1.checkbox("Run Corrective RAG")
    run_self = col2.checkbox("Run Self RAG")
    run_fusion = col3.checkbox("Run Fusion RAG")
    submit_button = st.form_submit_button(label='Submit')

if submit_button and user_question:
    result = {}

    if run_corrective:
        result['corrective'] = corrective_rag(user_question, llm, base_qa_chain)

    if run_self:
        result['self'] = self_rag(user_question, llm, retriever)

    if run_fusion:
        result['fusion'] = fusion_rag(user_question, llm, retriever)

    # Save to history
    st.session_state.history.append({
        "question": user_question,
        "result": result
    })

# Display history
for idx, item in enumerate(st.session_state.history):
    st.markdown(f"### Q{idx+1}: {item['question']}")
    if 'corrective' in item['result']:
        st.markdown("**✅ Corrective RAG Answer:**")
        st.write(item['result']['corrective'])
    if 'self' in item['result']:
        st.markdown("**🤔 Self RAG Answer:**")
        st.write(item['result']['self']['final_answer'])
        st.markdown("_Reflection:_")
        st.write(item['result']['self']['reflection'])
    if 'fusion' in item['result']:
        st.markdown("**🔗 Fusion RAG Answer:**")
        st.write(item['result']['fusion'])
    st.divider()