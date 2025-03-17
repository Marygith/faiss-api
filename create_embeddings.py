from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np

app = Flask(__name__)

all_passages_df = pd.read_parquet('D:\\diplom\\data\\all_passages.parquet')
passage_texts = all_passages_df['passage_text'].tolist()
passage_ids = all_passages_df['id'].tolist()
model = SentenceTransformer('all-MiniLM-L6-v2')

passage_embeddings = model.encode(passage_texts, show_progress_bar=True)
passage_embeddings = np.array(passage_embeddings, dtype=np.float16)

# Save embeddings in CSV format
output_csv_path = 'D:\\diplom\\data_v2\\passage_data.csv'
with open(output_csv_path, 'w') as csv_file:
    csv_file.write("id," + ",".join([f"dim{i}" for i in range(384)]) + "\n")  # Header
    for id_, embedding in zip(passage_ids, passage_embeddings):
        csv_file.write(f"{id_}," + ",".join(map(str, embedding)) + "\n")

print(f"Data successfully saved to {output_csv_path}")


output_embeddings_path = 'D:\\diplom\\data_v2\\new_passage_embeddings.npy'
# output_ids_path = 'D:\\diplom\\shared\\new_passage_ids.npy'

np.save(output_embeddings_path, passage_embeddings)
# np.save(output_ids_path, passage_ids)

dimension = passage_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index_with_ids = faiss.IndexIDMap(index)

index_with_ids.add_with_ids(passage_embeddings, passage_ids)

faiss_index_path = 'D:\\diplom\\data_v2\\faiss_index_with_ids'
faiss.write_index(index_with_ids, faiss_index_path)

print(f"FAISS index with IDs saved to {faiss_index_path}.")