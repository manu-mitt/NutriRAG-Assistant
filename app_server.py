import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HOME"] = "f:/simple-local-rag-main/.hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import sys
import torch
import pandas as pd
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure stdout encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[API Server] Using device: {device}")

# 1. Load data and models once
print("[API Server] Loading pre-computed embeddings...")
df = pd.read_csv("f:/simple-local-rag-main/simple-local-rag-main/simple-local-rag/text_chunks_and_embeddings_df.csv")
df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(str(x).strip("[]"), sep=" "))
embeddings = torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32).to(device)
pages_and_chunks = df.to_dict(orient="records")

print("[API Server] Loading models...")
embed_model = SentenceTransformer("f:/simple-local-rag-main/models/all-mpnet-base-v2", device=device)
tokenizer = AutoTokenizer.from_pretrained("f:/simple-local-rag-main/models/Qwen2.5-0.5B-Instruct")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
llm_model = AutoModelForCausalLM.from_pretrained(
    "f:/simple-local-rag-main/models/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=False
).to(device)

print("[API Server] Models loaded successfully!")

def prompt_formatter(query: str, context_items: list) -> str:
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])
    base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.

Example 1:
Query: What are the fat-soluble vitamins?
Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.

Example 2:
Query: What are the causes of type 2 diabetes?
Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.

Example 3:
Query: What is the importance of hydration for physical performance?
Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.

Now use the following context items to answer the user query:
{context}

Relevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""
    formatted_prompt = base_prompt.format(context=context, query=query)
    dialogue_template = [{"role": "user", "content": formatted_prompt}]
    return tokenizer.apply_chat_template(conversation=dialogue_template, tokenize=False, add_generation_prompt=True)

def perform_rag_query(query: str, temperature: float, top_k: int):
    query_embedding = embed_model.encode(query, convert_to_tensor=True)
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    scores, indices = torch.topk(dot_scores, k=top_k)
    
    context_items = [pages_and_chunks[idx.item()] for idx in indices]
    for i, item in enumerate(context_items):
        item["score"] = float(scores[i].item())
        
    prompt = prompt_formatter(query=query, context_items=context_items)
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    
    outputs = llm_model.generate(
        **input_ids,
        temperature=temperature,
        do_sample=True if temperature > 0.0 else False,
        max_new_tokens=512
    )
    
    output_text = tokenizer.decode(outputs[0])
    answer = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").strip()
    if "<|im_end|>" in answer:
        answer = answer.split("<|im_end|>")[0].strip()
        
    return answer, context_items

class RAGServerHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        # Serve the UI page
        if self.path in ["/", "/index.html"]:
            try:
                with open("index.html", "r", encoding="utf-8") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(content.encode("utf-8"))
            except FileNotFoundError:
                self.send_response(404)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"index.html not found. Please build it first.")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/query":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                query = data.get("query", "")
                temperature = float(data.get("temperature", 0.7))
                top_k = int(data.get("top_k", 5))

                if not query:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Empty query"}).encode('utf-8'))
                    return

                print(f"[API Server] Query: {query} (temp={temperature}, top_k={top_k})")
                answer, context_items = perform_rag_query(query, temperature, top_k)
                
                response_data = {
                    "answer": answer,
                    "context_items": [
                        {
                            "page_number": int(item["page_number"]),
                            "sentence_chunk": item["sentence_chunk"],
                            "score": float(item["score"])
                        } for item in context_items
                    ]
                }
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, RAGServerHandler)
    print(f"\n==========================================================================")
    print(f"RAG Local API Server running at http://localhost:{port}/")
    print(f"To open the Web UI dashboard, visit: http://localhost:{port}/")
    print(f"==========================================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == '__main__':
    run()
