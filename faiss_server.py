from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = Flask(__name__)

model = SentenceTransformer('msmarco-distilbert-base-v3')

index = faiss.read_index('D:\\diplom\\data\\faiss_index_with_ids')

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
        ids_to_vector = data.get('idsToVector')
        print("query: ", query)

        if not isinstance(query, str) or not isinstance(ids_to_vector, dict):
            return jsonify({"error": "Invalid input format"}), 400

        # Encode the query into a vector
        query_embedding = model.encode([query])[0].astype(np.float32)

        results = {}

        for doc_id, passage_vector in ids_to_vector.items():
            try:
                # Ensure the passage_vector is a valid list of floats
                passage_embedding = np.array(passage_vector, dtype=np.float32)

                if passage_embedding.shape != (768,):
                    raise ValueError(f"Invalid vector shape for doc_id {doc_id}, vector shape is {passage_embedding.shape}")

                # Calculate the L2 distance
                l2_distance = np.sum((query_embedding - passage_embedding) ** 2)
                results[doc_id] = float(l2_distance)

            except Exception as e:
                # Handle malformed vector input
                print(f"Error processing doc_id {doc_id}: {e}")
                results[doc_id] = None

        print("response: ", results)
        return jsonify(results)

    except Exception as e:
        print(jsonify({"error": str(e)}), 500)
        return jsonify({"error": str(e)}), 500


@app.route('/getMinScore', methods=['POST'])
def getMinScore():
    data = request.get_json()
    print("request: ", data)
    query = data.get('query')
    max_distance = find_least_similar_doc_range(query)
    return jsonify({
        'distance': float(max_distance)
    })


def find_least_similar_doc_range(query):
    query_embedding = model.encode([query])[0].astype(np.float32)
    """
    Find the least similar document in the FAISS index using range search.

    :param query_vector: The query vector (1D numpy array, shape [dimension])
    :return: A tuple of (doc_id, distance)
    """
    # Perform range search with a very large distance threshold
    distances, ids = index.search(np.array([query_embedding]), k=index.ntotal)

    max_distance_idx = np.argmax(distances[0])
    least_similar_doc_id = ids[0][max_distance_idx]
    max_distance = distances[0][max_distance_idx]

    print(f"Least similar document ID: {least_similar_doc_id}, Distance: {max_distance}")
    return max_distance


if __name__ == '__main__':
    app.run(port=5000)