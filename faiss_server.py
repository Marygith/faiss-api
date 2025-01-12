from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np

app = Flask(__name__)

# all_passages_df = pd.read_parquet('D:\\diplom\\shared\\all_passages.parquet')
# passage_texts = all_passages_df['passage_text'].tolist()
# passage_ids = all_passages_df['id'].tolist()
model = SentenceTransformer('msmarco-distilbert-base-v3')

# passage_embeddings = np.load('D:\\diplom\\shared\\passage_embeddings_distilibert.npy')

# dimension = passage_embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(passage_embeddings)
index = faiss.read_index('D:\\diplom\\shared\\faiss_index_with_ids')
@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    print(data)
    query = data.get('query')
    k = int(data.get('k', 100))
    normalized = data.get('normalized', False)

    # Encode the query
    query_embedding = model.encode([query], convert_to_tensor=False)
    query_embedding = query_embedding.astype('float32')

    # Perform the search
    distances, ids = index.search(query_embedding, k)
    distances = distances[0]
    ids = ids[0]

    if normalized:
        max_distance = max(distances)
        min_distance = min(distances)
        normalized_distances = [
            1 - (dist - min_distance) / (max_distance - min_distance) for dist in distances
        ]
        results = {int(ids[idx]): float(normalized_distances[idx]) for idx in range(len(ids))}
    else:
        results = {int(ids[idx]): float(distances[idx]) for idx in range(len(ids))}

    return jsonify(results)

if __name__ == '__main__':
    app.run(port=5000)