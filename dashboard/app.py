import streamlit as st
import pandas as pd
import plotly.express as px


# -----------------------------------
# Page configuration
# -----------------------------------

st.set_page_config(
    page_title="LLM Evaluation Platform",
    layout="wide"
)


# -----------------------------------
# Title
# -----------------------------------

st.title("LLM Evaluation Platform")

st.write(
    "Compare large language models using semantic similarity, "
    "correctness, and response latency."
)


# -----------------------------------
# Load results
# -----------------------------------

@st.cache_data
def load_results(file_modified_time):
    return pd.read_csv("results.csv")


# Use file modification time to refresh the dashboard
# whenever results.csv changes.
import os

file_modified_time = os.path.getmtime("results.csv")
df = load_results(file_modified_time)


# -----------------------------------
# Validate results
# -----------------------------------

required_columns = [
    "id",
    "model",
    "question",
    "expected_answer",
    "model_answer",
    "category",
    "similarity_score",
    "correctness_score",
    "latency"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        f"Missing columns in results.csv: {', '.join(missing_columns)}"
    )
    st.stop()


if df.empty:
    st.warning("No evaluation results found. Run evaluator.py first.")
    st.stop()


# -----------------------------------
# Sidebar filters
# -----------------------------------

st.sidebar.header("Filters")

models = ["All"] + sorted(df["model"].dropna().unique().tolist())

selected_model = st.sidebar.selectbox(
    "Select Model",
    models
)

categories = ["All"] + sorted(
    df["category"].dropna().unique().tolist()
)

selected_category = st.sidebar.selectbox(
    "Select Category",
    categories
)


# Apply model filter
if selected_model != "All":
    df = df[df["model"] == selected_model]


# Apply category filter
if selected_category != "All":
    df = df[df["category"] == selected_category]


if df.empty:
    st.warning("No results match the selected filters.")
    st.stop()


# -----------------------------------
# Overall model comparison
# -----------------------------------

st.subheader("Model Comparison")

comparison = (
    df.groupby("model")
    .agg(
        Questions_Evaluated=("id", "nunique"),
        Average_Similarity=("similarity_score", "mean"),
        Correctness_Rate=("correctness_score", "mean"),
        Average_Latency=("latency", "mean")
    )
    .reset_index()
)

comparison["Correctness_Rate"] *= 100

comparison = comparison.rename(
    columns={
        "model": "Model",
        "Questions_Evaluated": "Questions Evaluated",
        "Average_Similarity": "Average Similarity",
        "Correctness_Rate": "Correctness Rate (%)",
        "Average_Latency": "Average Latency (sec)"
    }
)

st.dataframe(
    comparison.style.format({
        "Average Similarity": "{:.3f}",
        "Correctness Rate (%)": "{:.1f}%",
        "Average Latency (sec)": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


st.divider()


# -----------------------------------
# KPI metrics
# -----------------------------------

total_evaluations = len(df)

unique_questions = df["id"].nunique()

number_of_models = df["model"].nunique()

average_similarity = df["similarity_score"].mean()

average_correctness = df["correctness_score"].mean() * 100

average_latency = df["latency"].mean()


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Questions Evaluated",
        unique_questions
    )

with col2:
    st.metric(
        "Total Evaluations",
        total_evaluations
    )

with col3:
    st.metric(
        "Correctness Rate",
        f"{average_correctness:.1f}%"
    )

with col4:
    st.metric(
        "Average Latency",
        f"{average_latency:.2f} sec"
    )


st.caption(
    f"Comparing {number_of_models} model(s) across "
    f"{unique_questions} unique question(s) in the filtered results."
)


st.divider()


# -----------------------------------
# Semantic similarity chart
# -----------------------------------

st.subheader("Semantic Similarity by Question")

fig_similarity = px.bar(
    df,
    x="id",
    y="similarity_score",
    color="model",
    barmode="group",
    hover_data=[
        "question",
        "model_answer",
        "expected_answer"
    ],
    labels={
        "id": "Question ID",
        "similarity_score": "Similarity Score",
        "model": "Model"
    },
    title="Comparison of Semantic Similarity"
)

fig_similarity.update_yaxes(range=[0, 1])

st.plotly_chart(
    fig_similarity,
    use_container_width=True
)


# -----------------------------------
# Correctness chart
# -----------------------------------

st.subheader("Correctness by Question")

fig_correctness = px.bar(
    df,
    x="id",
    y="correctness_score",
    color="model",
    barmode="group",
    hover_data=["question"],
    labels={
        "id": "Question ID",
        "correctness_score": "Correctness (0 = Incorrect, 1 = Correct)",
        "model": "Model"
    },
    title="Correctness Comparison"
)

fig_correctness.update_yaxes(
    range=[0, 1],
    tickvals=[0, 1],
    ticktext=["Incorrect", "Correct"]
)

st.plotly_chart(
    fig_correctness,
    use_container_width=True
)


# -----------------------------------
# Response latency chart
# -----------------------------------

st.subheader("Response Latency")

fig_latency = px.bar(
    df,
    x="id",
    y="latency",
    color="model",
    barmode="group",
    hover_data=["question"],
    labels={
        "id": "Question ID",
        "latency": "Latency (seconds)",
        "model": "Model"
    },
    title="Response Time Comparison"
)

st.plotly_chart(
    fig_latency,
    use_container_width=True
)


# -----------------------------------
# Results table
# -----------------------------------

st.subheader("Detailed Evaluation Results")

display_columns = [
    "id",
    "model",
    "question",
    "expected_answer",
    "model_answer",
    "similarity_score",
    "correctness_score",
    "latency",
    "category"
]

st.dataframe(
    df[display_columns].sort_values(
        by=["id", "model"]
    ),
    use_container_width=True,
    hide_index=True
)


# -----------------------------------
# Download results
# -----------------------------------

st.subheader("Export Results")

csv_data = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Results as CSV",
    data=csv_data,
    file_name="llm_evaluation_results.csv",
    mime="text/csv"
)