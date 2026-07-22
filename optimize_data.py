import pandas as pd
import numpy as np
import json
import os

csv_path = "f:/simple-local-rag-main/simple-local-rag-main/simple-local-rag/text_chunks_and_embeddings_df.csv"
output_dir = "f:/simple-local-rag-main/simple-local-rag-main/simple-local-rag/github_pages"

print(f"Reading CSV from {csv_path}...")
df = pd.read_csv(csv_path)

# Extract text chunks
print("Extracting text chunks...")
text_chunks = []
for idx, row in df.iterrows():
    text_chunks.append({
        "page": int(row["page_number"]),
        "text": str(row["sentence_chunk"])
    })

# Write text chunks to JSON
json_path = os.path.join(output_dir, "text_chunks.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(text_chunks, f, ensure_ascii=False, indent=2)
print(f"Saved text chunks to {json_path} (size: {os.path.getsize(json_path) / 1024 / 1024:.2f} MB)")

# Extract and convert embeddings to float32 binary
print("Converting embeddings to binary...")
embeddings_list = []
for idx, row in df.iterrows():
    # Convert string representation of list to numpy array
    emb = np.fromstring(str(row["embedding"]).strip("[]"), sep=" ")
    embeddings_list.append(emb)

embeddings_array = np.array(embeddings_list, dtype=np.float32)
bin_path = os.path.join(output_dir, "embeddings.bin")
embeddings_array.tofile(bin_path)
print(f"Saved binary embeddings to {bin_path} (shape: {embeddings_array.shape}, size: {os.path.getsize(bin_path) / 1024 / 1024:.2f} MB)")
print("Optimization complete!")
