# LLM Evaluation Platform

An interactive platform for evaluating and comparing Large Language Models (LLMs) using semantic similarity, automated correctness evaluation, and response latency.

The platform runs models locally using Ollama and provides a Streamlit dashboard to analyze their performance across a common set of questions.

## Features

- **Multi-model evaluation:** Compare Qwen2.5-3B and SmallThinker.
- **Semantic similarity:** Compare generated answers with expected answers using sentence embeddings.
- **Automated correctness evaluation:** Use an LLM-based judge to assess factual correctness.
- **Response latency:** Measure the time taken to generate each answer.
- **Interactive dashboard:** Explore evaluation results using Streamlit and Plotly.
- **Filtering:** Filter results by model and question category.
- **CSV export:** Download filtered evaluation results.
- **Detailed results:** Inspect expected answers and model-generated responses.

## Technologies Used

- Python
- Ollama
- Qwen2.5-3B
- SmallThinker
- Sentence Transformers
- Scikit-learn
- Pandas
- Streamlit
- Plotly

## Project Structure

```text
llm-evaluation-platform/
├── dashboard/
│   └── app.py
├── data/
│   └── evaluation_dataset.csv
├── evaluator.py
├── test_ollama.py
├── results.csv
├── requirements.txt
└── README.md
```

## Evaluation Metrics

### 1. Semantic Similarity

Uses the `all-MiniLM-L6-v2` sentence embedding model to compare the semantic similarity between an expected answer and a generated answer.

Scores closer to 1 indicate greater semantic similarity. Similarity does not guarantee factual correctness.

### 2. Correctness Score

Uses Qwen2.5-3B as an automated judge to compare each generated answer against the expected answer.

- `1`: Judged correct
- `0`: Judged incorrect

These scores are automated estimates and may contain errors.

### 3. Response Latency

Measures the elapsed time for the model-generation request, in seconds.

Latency may vary depending on the model, response length, and system workload.

## Sample Evaluation Results

The following results were observed in one local evaluation run using 10 questions per model.

| Metric | Qwen2.5-3B | SmallThinker |
|---|---:|---:|
| Questions Evaluated | 10 | 10 |
| Average Semantic Similarity | 0.752 | 0.667 |
| Automated Correctness Rate | 90% | 90% |
| Average Response Latency | 7.73 sec | 41.71 sec |

**Observation:** In this evaluation run, Qwen2.5-3B achieved higher average semantic similarity and lower average latency. Both models received the same automated correctness rate.

These results are specific to the current dataset and run. They should not be interpreted as a general ranking of the models.

## Getting Started

### Prerequisites

Install the following:

- Python 3.12 or a compatible Python version
- Git
- Ollama

Install Ollama from its official website:

https://ollama.com/

### 1. Clone the repository

```bash
git clone https://github.com/Anchalmunot/llm-evaluation-platform.git
cd llm-evaluation-platform
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

Activate it on macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the required models

```bash
ollama pull qwen2.5:3b
ollama pull smallthinker:latest
```

Ensure Ollama is running before evaluating the models.

### 5. Run the evaluation pipeline

```bash
python evaluator.py
```

The script evaluates both models against the dataset and saves the results to `results.csv`.

**Note:** Running the evaluation pipeline again performs new model inferences and may take several minutes.

### 6. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open the local URL displayed in the terminal to explore the dashboard.

## Dataset

The evaluation dataset is stored in `data/evaluation_dataset.csv`.

It contains 10 questions covering topics such as:

- Computer Science
- Database Management Systems
- Programming
- Artificial Intelligence
- Operating Systems
- Machine Learning
- Object-Oriented Programming
- Web Technologies

Each question includes an expected answer and a category.

## Limitations

- The evaluation dataset currently contains only 10 questions.
- Automated correctness judgments may be inaccurate.
- The same model is used as the correctness judge for both evaluated models.
- Semantic similarity is not a direct measure of factual accuracy.
- Latency measurements depend on local hardware and system workload.
- The platform requires Ollama and the selected models to be available locally.

## Future Improvements

- Expand the evaluation dataset.
- Add additional language models.
- Improve correctness evaluation with independent judges.
- Track token generation speed and memory usage.
- Add repeated evaluation runs and statistical comparisons.
- Support evaluation history and result visualization.

## Author

**Anchal Munot**

B.Tech Computer Science Engineering

GitHub: https://github.com/Anchalmunot