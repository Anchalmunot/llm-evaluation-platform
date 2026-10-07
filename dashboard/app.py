import streamlit as st
import pandas as pd
import plotly.express as px


# -----------------------------------
# Page configuration
# -----------------------------------

st.set_page_config(
    page_title="LLM Evaluation Platform",
    page_icon="🤖",
    layout="wide"
)


# -----------------------------------
# Title
# -----------------------------------

st.title("LLM Evaluation Platform")

st.write(
    "Evaluate LLM responses using semantic similarity "
    "and response latency."
)


# -----------------------------------
# Load results
# -----------------------------------

@st.cache_data
def load_results():

    return pd.read_csv("results.csv")


df = load_results()


# -----------------------------------
# Sidebar
# -----------------------------------

st.sidebar.header("Filters")

categories = ["All"] + sorted(
    df["category"].dropna().unique().tolist()
) if "category" in df.columns else ["All"]

selected_category = st.sidebar.selectbox(
    "Select Category",
    categories
)


if selected_category != "All":
    df = df[df["category"] == selected_category]


# -----------------------------------
# KPI metrics
# -----------------------------------

total_questions = len(df)

average_similarity = df["similarity_score"].mean()

average_correctness = df["correctness_score"].mean() * 100

average_latency = df["latency"].mean()


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric("Questions Evaluated", total_questions)

with col2:
    st.metric(
        "Average Similarity",
        f"{average_similarity:.3f}"
    )

with col3:
    st.metric(
        "Correctness",
        f"{average_correctness:.1f}%"
    )

with col4:
    st.metric(
        "Average Latency",
        f"{average_latency:.2f} sec"
    )


st.divider()


# -----------------------------------
# Similarity chart
# -----------------------------------

st.subheader("Semantic Similarity by Question")


fig_similarity = px.bar(
    df,
    x="id",
    y="similarity_score",
    hover_data=["question"],
    labels={
        "id": "Question ID",
        "similarity_score": "Similarity Score"
    }
)

fig_similarity.update_yaxes(
    range=[0, 1]
)

st.plotly_chart(
    fig_similarity,
    use_container_width=True
)

st.subheader("Correctness by Question")

fig_correctness = px.bar(
    df,
    x="id",
    y="correctness_score",
    hover_data=["question"],
    labels={
        "id": "Question ID",
        "correctness_score": "Correctness"
    }
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
# Latency chart
# -----------------------------------

st.subheader("Response Latency")


fig_latency = px.bar(
    df,
    x="id",
    y="latency",
    hover_data=["question"],
    labels={
        "id": "Question ID",
        "latency": "Latency (seconds)"
    }
)

st.plotly_chart(
    fig_latency,
    use_container_width=True
)


# -----------------------------------
# Results table
# -----------------------------------

st.subheader("Evaluation Results")


display_columns = [
    "id",
    "question",
    "expected_answer",
    "model_answer",
    "similarity_score",
    "correctness_score",
    "latency"
]


available_columns = [
    column
    for column in display_columns
    if column in df.columns
]


st.dataframe(
    df[available_columns],
    use_container_width=True,
    hide_index=True
)

