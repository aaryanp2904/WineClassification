import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Load dataset and remove instances missing items
data = pd.read_csv("./data/raw/wine_quality_1000.csv")
data = data.dropna(subset=["country", "description", "price", "points"])

# Label the countries and define inputs and outputs
le = LabelEncoder()
data['country'] = le.fit_transform(data['country'])
X = data[['description', 'price', 'points']]
y = data['country']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessor with TfidfVectorizer
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=5000, stop_words='english'), 'description'),
        ('num', StandardScaler(), ['price', 'points'])
    ]
)

# Use RandomOversampler for balancing classes since we saw there is a class imbalance
oversampler = RandomOverSampler(random_state=42)

# Define RandomForest pipeline
rf_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('oversampler', oversampler),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Define Neural Network pipeline
nn_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('oversampler', oversampler),
    ('classifier', MLPClassifier(hidden_layer_sizes=(100,), random_state=42, max_iter=500))
])

# Train models
rf_pipeline.fit(X_train, y_train)
nn_pipeline.fit(X_train, y_train)


def create_prompt(X_train, y_train):

    # Convert the series to a df
    X_train = pd.DataFrame(X_train)

    examples = []

    # Get names of classes rather than just labels
    y_train_classes = pd.DataFrame(le.inverse_transform(y_train), index=X_train.index, columns=["country"])

    # Sample examples
    sampled_data = (
        X_train.join(y_train_classes)
        .groupby("country", group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), 200), random_state=42))
    )

    # Form desc and country string
    for _, (desc, country) in sampled_data.iterrows():
        examples.append(f"Description: {desc}\nCountry: {country}")
    return "\n".join(examples)


def llm_batch_pred(test_descriptions, batch_id):

    prompt_examples = create_prompt(X_train['description'], y_train)
    formatted_test_cases = [{"id": i, "description": desc} for i, desc in enumerate(test_descriptions)]
    system_prompt = (
        "You are an analyst who predicts the origin of a wine based on its description. "
        "The origin can only be 'US', 'Italy', 'Spain', or 'France'. Respond with a JSON object containing predictions for "
        "each wine description. DO NOT output anything apart from the JSON object. DO NOT output in markdown format. ALWAYS respond with JSON that "
        "matches the following format:\n"
        "{\n"
        "    \"predictions\": [\n"
        "        {\"id\": 0, \"prediction\": \"country\", \"confidence\": \"HIGH|MEDIUM|LOW\"},\n"
        "        {\"id\": 1, \"prediction\": \"country\", \"confidence\": \"HIGH|MEDIUM|LOW\"},\n"
        "        ...\n"
        "    ]\n"
        "}\n"
        "Examples from the training data are provided below:\n\n"
        f"{prompt_examples}"
    )

    user_prompt = (
            f"Batch {batch_id}: Predict the origin for the following wine descriptions:\n" +
            json.dumps({"test_cases": formatted_test_cases})
    )

    response = requests.post(
        'https://candidate-llm.extraction.artificialos.com/v1/chat/completions',
        headers={
            'x-api-key': os.getenv("OPENAI_API_KEY")
        },
        json={
            'model': 'gpt-4o',
            'messages': [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
    )

    # Get response
    response_json = response.json()
    predictions = response_json.get('choices', [])[0].get('message', {}).get('content', '{}')
    data = json.loads(predictions)

    # Offset the id in each prediction (otherwise we would have multiple e.g., id 0s and 1s
    for prediction in data["predictions"]:
        prediction["id"] += (batch_id - 1) * 20

    return data


# Split test data into 10 batches, we are doing this as doing a single request per test instance causes a 429 (cost)
# error and doing all tests in 1 go cause a timeout
test_descriptions = X_test['description'].tolist()
batch_size = len(test_descriptions) // 10
batches = [test_descriptions[i:i + batch_size] for i in range(0, len(test_descriptions), batch_size)]

# Add the remainder to the last batch
if len(batches) > 10:
    batches[-2].extend(batches[-1])
    batches = batches[:-1]

# Get predictions for all batches
all_predictions = []
for batch_id, batch in enumerate(batches, start=1):
    batch_predictions = llm_batch_pred(batch, batch_id=batch_id)
    all_predictions.extend(batch_predictions["predictions"])

actual_origins = pd.Series(le.inverse_transform(y_test), index=y_test.index)

# Create mapping from X_test index to IDs used in all_predictions
test_index_to_id = {idx: i for i, idx in enumerate(X_test.index)}


final_predictions = []
for idx, instance in X_test.iterrows():

    # Convert Series to DataFrame
    instance_df = pd.DataFrame([instance])

    # Predictions from RandomForest, Neural Network
    rf_pred = le.inverse_transform([rf_pipeline.predict(instance_df)[0]])[0]
    nn_pred = le.inverse_transform([nn_pipeline.predict(instance_df)[0]])[0]

    # Get GPT pred, using an iterator here for lazy checks on each iteration
    llm_pred = next(
        pred["prediction"] for pred in all_predictions if pred["id"] == test_index_to_id[idx]
    )

    # Using majority voting
    votes = Counter([rf_pred, nn_pred, llm_pred])
    majority_vote = votes.most_common(1)[0][0]
    final_predictions.append(majority_vote)

print("Classification Report:")
print(classification_report(actual_origins, final_predictions))

# 88% accuracy
