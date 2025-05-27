<!DOCTYPE html>
<html lang="en">
<body>
  <h1>📚 RAG Chat App with LLaMA3 and Streamlit</h1>

  <p>This repository contains a local RAG chat application using:</p>
  <ul>
    <li><strong>LLaMA3</strong> via <strong>Ollama</strong></li>
    <li><strong>LangChain</strong> for RAG logic</li>
    <li><strong>Streamlit</strong> for UI</li>
    <li><strong>FAISS</strong> + HuggingFace embeddings for retrieval</li>
  </ul>

  <h2>🛠️ Features</h2>
  <ul>
    <li>✅ Corrective RAG</li>
    <li>🤔 Self-RAG (reflection-based)</li>
    <li>🔗 Fusion RAG (multi-query retrieval)</li>
    <li>📄 Local document support (<code>data.txt</code>)</li>
    <li>🧠 Fully runs offline – no API keys required</li>
  </ul>

  <h2>📦 Requirements</h2>
  <pre>pip install langchain langchain-community langchain-huggingface faiss-cpu ollama streamlit</pre>

  <h2>🦙 Ollama Setup</h2>
  <p>Make sure you have Ollama installed and running locally:</p>
  <pre>ollama pull llama3</pre>

  <h2>📂 Project Structure</h2>
  <pre>
rag-app/
├── rag_app.py          # Main app
├── data.txt            # Knowledge source
└── README.md           # This file
  </pre>

  <h2>🎮 How to Run</h2>
  <ol>
    <li>Create a <code>data.txt</code> file with your knowledge base.</li>
    <li>Run the app:
      <pre>streamlit run rag_app.py</pre>
    </li>
    <li>Open your browser and interact with the web interface.</li>
  </ol>

  <h2>📝 Example data.txt</h2>
  <pre>
World War I began in 1914 after the assassination of Archduke Franz Ferdinand of Austria.
The major causes included militarism, alliances, imperialism, and nationalism.
Tensions had been building in Europe for years due to rivalries among nations and arms races.
  </pre>

  <h2>✨ Tips</h2>
  <ul>
    <li>Use checkboxes in the UI to choose which RAG method(s) to run per question.</li>
    <li>History is saved in session state during use.</li>
  </ul>

  <h2>📌 License</h2>
  <p>MIT License – Feel free to modify and redistribute.</p>

  <h2>🚀 Deploy?</h2>
  <p>If you'd like to deploy this publicly (e.g., on Streamlit Cloud or Hugging Face Spaces), let me know!</p>
</body>
</html>
