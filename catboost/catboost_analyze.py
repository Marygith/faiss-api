import pandas as pd
from catboost import CatBoostClassifier

# Load the trained model
model = CatBoostClassifier()
model.load_model("D:\\diplom\\data_v2\\catboost_model.bin")

# Define feature names (same as used during training)
feature_cols = ["faiss_score", "lucene_score", "query_length", "doc_length"]

# Get feature importance
feature_importance = model.get_feature_importance(prettified=True)
print(feature_importance)
# Convert to DataFrame for better visualization
# feature_importance_df = pd.DataFrame({
#     "Feature": feature_cols,
#     "Importance": feature_importance
# })
#
# # Sort features by importance
# feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
#
# # Print or save feature importance
# print(feature_importance_df)
