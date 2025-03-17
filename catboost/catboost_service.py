import pandas as pd
from catboost import CatBoostClassifier, Pool

# Load the dataset
df = pd.read_csv("D:\\diplom\\data_v2\\feature_store.csv", delimiter="^")

# Define features and target
feature_cols = ["faiss_score", "lucene_score", "query_length", "doc_length"]
target_col = "relevant"

# Split dataset into training and test sets
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])

# Convert data into CatBoost format
train_pool = Pool(train_df[feature_cols], label=train_df[target_col])
test_pool = Pool(test_df[feature_cols], label=test_df[target_col])

# Train CatBoost model
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    verbose=100
)

model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

# Save model in binary and JSON formats (Java requires JSON)
model.save_model("D:\\diplom\\data_v2\\catboost_model.bin")
model.save_model("D:\\diplom\\data_v2\\catboost_model.json", format="json")


