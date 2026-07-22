# Serverless Nutrition RAG Assistant

**Author:** Manu M

A client-side Retrieval-Augmented Generation (RAG) assistant that lets you search and query a 1200-page Human Nutrition textbook directly in your browser.

## 🚀 Live Demo
🌐 **[https://manu-mitt.github.io/NutriRAG-Assistant/](https://manu-mitt.github.io/NutriRAG-Assistant/)**

---

## 💻 Local Host Execution
If you want to run the pipeline locally (bypassing online API restrictions or regional DNS blocks):
1. Start the Python server: `python app_server.py`
2. Access the Web UI dashboard on your computer: **[http://localhost:8000](http://localhost:8000)**
3. Access on your mobile device (connected to same Wi-Fi): **[http://10.50.148.236:8000](http://10.50.148.236:8000)**

---

## 🛠️ Architecture & Optimization

This version is optimized for **zero-cost serverless deployment** on static hosting (like GitHub Pages):

1. **Lightweight Payload Optimization**:
   - The original textbook data (21 MB CSV) was split and compressed.
   - **Text Chunks**: Stored in a raw JSON file (1.36 MB).
   - **Vector Embeddings**: Packed into a raw binary `float32` file (`embeddings.bin` - 4.92 MB).
   - Total loading size is reduced by **72%** (from 21 MB to 6.28 MB), resulting in instant page loads in the browser.

2. **In-Browser Semantic Search**:
   - The user query is converted to a vector embedding using the `sentence-transformers/all-mpnet-base-v2` model hosted on the Hugging Face Inference API.
   - Cosine similarity matching against the 1,680 textbook chunk embeddings is computed **instantly in client-side JavaScript** (~15ms).

3. **Serverless Language Generation**:
   - The top-K context passages and query are formatted and sent to the `Qwen/Qwen2.5-7B-Instruct` model via Hugging Face's serverless Inference API.
   - The browser displays the generated answer along with page citations.

---

## 📦 How to Deploy to your GitHub Pages

1. **Create a new repository** on GitHub (e.g. `nutrichat`).
2. **Commit and push** the files in this folder to your repository:
   ```bash
   git init
   git add .
   git commit -m "Deploy serverless RAG dashboard"
   git branch -M main
   git remote add origin https://github.com/your-username/nutrichat.git
   git push -u origin main
   ```
3. **Enable GitHub Pages**:
   - Go to your repository settings on GitHub.
   - Navigate to the **Pages** section in the sidebar.
   - Under **Build and deployment**, select the `main` branch and `/ (root)` folder, then click **Save**.
4. Your application will be live at `https://<your-username>.github.io/nutrichat/` within a couple of minutes!
