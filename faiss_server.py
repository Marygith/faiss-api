from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = Flask(__name__)

model = SentenceTransformer('msmarco-distilbert-base-v3')

index = faiss.read_index('D:\\diplom\\shared\\faiss_index_with_ids')

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    print("request: ", data)
    query = data.get('query')
    k = int(data.get('k', 100))
    normalized = data.get('normalized', False)

    query_embedding = model.encode([query], convert_to_tensor=False)
    query_embedding = query_embedding.astype('float32')

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
    print("response: ", jsonify(results))
    return jsonify(results)

@app.route('/getSimilarityScores', methods=['POST'])
def get_similarity_scores():
    try:
        data = request.get_json()
        print("request: ", data)
        query = data.get('query')
        doc_map = data.get('idsToText')
        print("query: ", query)

        if not isinstance(query, str) or not isinstance(doc_map, dict):
            return jsonify({"error": "Invalid input format"}), 400

        query_embedding = model.encode([query])[0].astype(np.float32)

        results = {}

        for doc_id, passage_text in doc_map.items():
            if not isinstance(passage_text, str):
                results[doc_id] = None
                continue

            passage_embedding = model.encode([passage_text])[0].astype(np.float32)

            l2_distance = np.sum((query_embedding - passage_embedding) ** 2)
            results[doc_id] = float(l2_distance)
        print("response: ", jsonify(results))
        return jsonify(results)

    except Exception as e:
        print(jsonify({"error": str(e)}), 500)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000)