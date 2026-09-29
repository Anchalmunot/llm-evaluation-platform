import pandas as pd
import requests
import time

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------
# 1. Load evaluation dataset
# -----------------------------------

df = pd.read_csv("data/evaluation_dataset.csv")


# -----------------------------------
# 2. Load embedding model
# -----------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# -----------------------------------
# 3. Ollama configuration
# -----------------------------------

url = "http://localhost:11434/api/generate"

model = "qwen2.5:3b"


# -----------------------------------
# 4. Ask the LLM
# -----------------------------------

def ask_llm(question):

    data = {
        "model": model,
        "prompt": question,
        "stream": False
    }

    start_time = time.time()

    response = requests.post(url, json=data)

    end_time = time.time()

    answer = response.json()["response"]

    latency = end_time - start_time

    return answer, latency


# -----------------------------------
# 5. Calculate semantic similarity
# -----------------------------------

def calculate_similarity(expected, actual):

    embeddings = embedding_model.encode(
        [expected, actual]
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return float(similarity)


# -----------------------------------
# 6. Store results
# -----------------------------------

results = []


# -----------------------------------
# 7. Evaluate each question
# -----------------------------------

for index, row in df.iterrows():

    question = row["question"]

    expected_answer = row["expected_answer"]

    print("\nQuestion:", question)

    answer, latency = ask_llm(question)

    similarity_score = calculate_similarity(
        expected_answer,
        answer
    )

    print("Qwen Answer:", answer)

    print(
        "Similarity Score:",
        round(similarity_score, 3)
    )

    print(
        "Latency:",
        round(latency, 2),
        "seconds"
    )


    results.append({

        "id": row["id"],

        "question": question,

        "expected_answer": expected_answer,

        "model_answer": answer,

        "similarity_score": similarity_score,

        "latency": latency

    })


# -----------------------------------
# 8. Convert results to DataFrame
# -----------------------------------

results_df = pd.DataFrame(results)


# -----------------------------------
# 9. Save results
# -----------------------------------

results_df.to_csv(
    "results.csv",
    index=False
)


print("\nEvaluation completed!")

print("Results saved to results.csv")