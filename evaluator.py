import pandas as pd
import requests
import time

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load evaluation dataset

df = pd.read_csv("data/evaluation_dataset.csv")


# Load embedding model

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# Ollama configuration

url = "http://localhost:11434/api/generate"

model = "qwen2.5:3b"


# Ask the LLM

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



# Calculate correctness
# -----------------------------------
# 6. LLM-as-a-Judge correctness
# -----------------------------------

def calculate_correctness(question, expected, actual):

    judge_prompt = f"""
You are an expert evaluator of AI-generated answers.

Determine whether the model answer is factually correct.

Question:
{question}

Expected Answer:
{expected}

Model Answer:
{actual}

Rules:
1. Compare the model answer with the expected answer.
2. Different wording is acceptable.
3. Additional correct information is acceptable.
4. If the model answer contains a factual error, mark it INCORRECT.
5. Pay special attention to definitions, names, abbreviations, and factual claims.
6. Do not judge based only on wording similarity.

Return exactly one word:
CORRECT
or
INCORRECT

Your final answer must be exactly CORRECT or INCORRECT.
"""

    data = {
        "model": model,
        "prompt": judge_prompt,
        "stream": False
    }

    response = requests.post(
        url,
        json=data
    )

    judge_response = response.json()["response"].strip().upper()

    if judge_response == "CORRECT":
        return 1

    return 0





# Store results

results = []


# Evaluate each question

for index, row in df.iterrows():

    question = row["question"]

    expected_answer = row["expected_answer"]

    print("\nQuestion:", question)

    answer, latency = ask_llm(question)

    similarity_score = calculate_similarity(
        expected_answer,
        answer
    )
    correctness_score = calculate_correctness(
    question,
    expected_answer,
    answer
)

    print("Qwen Answer:", answer)

    print(
        "Similarity Score:",
        round(similarity_score, 3)
    )
    print(
    "Correctness:",
    "Correct" if correctness_score == 1 else "Incorrect"
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

    "category": row["category"],

    "similarity_score": similarity_score,

    "correctness_score": correctness_score,

    "latency": latency

})


# Convert results to DataFrame

results_df = pd.DataFrame(results)


# Save results

results_df.to_csv(
    "results.csv",
    index=False
)


print("\nEvaluation completed!")

print("Results saved to results.csv")