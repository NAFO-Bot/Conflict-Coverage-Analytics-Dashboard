import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import chi2_contingency

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Emotion Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title(" Conflict Coverage Analytics Dashboard")
st.caption("Comparative analysis of emotional framing across selected actors and events")

# ---------------------------------------------------
# LOAD WORKBOOK
# ---------------------------------------------------

FILE = "Dissertation Dataset.xlsx"

xls = pd.ExcelFile(FILE)

# Get all worksheet names
sheet_names = xls.sheet_names

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Dashboard Controls")

selected_actor = st.sidebar.selectbox(
    "Select Actor",
    sheet_names
)

# ---------------------------------------------------
# LOAD SELECTED SHEET
# ---------------------------------------------------

df = pd.read_excel(FILE, sheet_name=selected_actor)

df.columns = ["Emotion", "Articles"]

df = df.fillna(0)

df["Emotion"] = pd.to_numeric(df["Emotion"])
df["Articles"] = pd.to_numeric(df["Articles"])
# ---------------------------------------------------
# EMOTION CATEGORIES
# ---------------------------------------------------

def classify_emotion(score):

    if score <= -10:
        return "Very Negative"

    elif score < 0:
        return "Negative"

    elif score == 0:
        return "Neutral"

    elif score < 10:
        return "Positive"

    else:
        return "Very Positive"


df["Category"] = df["Emotion"].apply(classify_emotion)
# ---------------------------------------------------
# NARRATIVE CLASSIFICATION
# ---------------------------------------------------

def classify_frame(mean, overall_mean, overall_std):

    if mean <= overall_mean - overall_std:

        return (
            "Strongly Negative",
            "This actor's coverage is substantially more negative than the dataset average."
        )

    elif mean < overall_mean:

        return (
            "Moderately Negative",
            "This actor's coverage is slightly more negative than the dataset average."
        )

    elif mean <= overall_mean + overall_std:

        return (
            "Near Average",
            "This actor falls within the typical emotional range observed across the dataset."
        )

    else:

        return (
            "Relatively Positive",
            "This actor's coverage is more positive than the dataset average."
        )

# ---------------------------------------------------
# CALCULATIONS
# ---------------------------------------------------

total_articles = df["Articles"].sum()

weighted_mean = np.average(
    df["Emotion"],
    weights=df["Articles"]
)

weighted_variance = np.average(
    (df["Emotion"] - weighted_mean) ** 2,
    weights=df["Articles"]
)

weighted_std = np.sqrt(weighted_variance)

mode_emotion = df.loc[
    df["Articles"].idxmax(),
    "Emotion"
]

emotion_bins = len(df)
category_summary = (
    df.groupby("Category")["Articles"]
      .sum()
      .reset_index()
)
# ---------------------------------------------------
# ACTOR RANKING TABLE
# ---------------------------------------------------

ranking = []

for sheet in sheet_names:

    temp = pd.read_excel(FILE, sheet_name=sheet)

    temp.columns = ["Emotion", "Articles"]

    temp = temp.fillna(0)

    temp["Emotion"] = pd.to_numeric(temp["Emotion"])
    temp["Articles"] = pd.to_numeric(temp["Articles"])

    weighted_mean_actor = np.average(
        temp["Emotion"],
        weights=temp["Articles"]
    )

    total_actor_articles = temp["Articles"].sum()

    weighted_variance_actor = np.average(
        (temp["Emotion"] - weighted_mean_actor) ** 2,
        weights=temp["Articles"]
    )

    weighted_std_actor = np.sqrt(weighted_variance_actor)

    ranking.append({

        "Actor": sheet,

        "Weighted Mean": weighted_mean_actor,

        "Std Dev": weighted_std_actor,

        "Total Articles": total_actor_articles

    })
ranking_df = pd.DataFrame(ranking)
overall_mean = ranking_df["Weighted Mean"].mean()
overall_std = ranking_df["Weighted Mean"].std()

def classify_frame(mean, overall_mean, overall_std):

    if mean <= overall_mean - overall_std:

        return (
            "Strongly Negative",
            "This actor is substantially more negative than the average actor."
        )

    elif mean < overall_mean:

        return (
            "Moderately Negative",
            "This actor is slightly more negative than the dataset average."
        )

    elif mean <= overall_mean + overall_std:

        return (
            "Near Average",
            "This actor falls within the normal emotional range of the dataset."
        )

    else:

        return (
            "Relatively Positive",
            "This actor is more positive than most actors in the dataset."
        )
frame, explanation = classify_frame(
    weighted_mean,
    overall_mean,
    overall_std
)
st.subheader("Dataset Baseline")

c1, c2 = st.columns(2)

c1.metric(
    "Dataset Mean",
    f"{overall_mean:.2f}"
)

c2.metric(
    "Dataset Std Dev",
    f"{overall_std:.2f}"
)
ranking_df = pd.DataFrame(ranking)

ranking_df = ranking_df.sort_values("Weighted Mean")
ranking_df.reset_index(drop=True, inplace=True)
ranking_df.index += 1

# Dataset baseline
overall_mean = ranking_df["Weighted Mean"].mean()
overall_std = ranking_df["Weighted Mean"].std()

# Narrative classification
frame, explanation = classify_frame(
    weighted_mean,
    overall_mean,
    overall_std
)

# ---------------------------------------------------
# KPI CARDS
# ---------------------------------------------------

st.subheader(selected_actor)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Total Articles",
    f"{int(total_articles):,}"
)

c2.metric(
    "Weighted Mean",
    f"{weighted_mean:.2f}"
)

c3.metric(
    "Std Deviation",
    f"{weighted_std:.2f}"
)

c4.metric(
    "Most Frequent Emotion",
    mode_emotion
)

c5.metric(
    "Emotion Bins",
    emotion_bins
)

st.divider()

# ---------------------------------------------------
# VISUALIZATION LAYOUT
# ---------------------------------------------------

left, right = st.columns([2, 1])
with left:

    st.subheader("Weighted Emotion Distribution")

    fig = px.bar(
        df,
        x="Emotion",
        y="Articles",
        text="Articles",
        color="Articles",
        labels={
            "Emotion": "Emotion Score",
            "Articles": "Article Count"
        }
    )

    fig.update_layout(
        height=550,
        xaxis_title="Emotion Score",
        yaxis_title="Number of Articles",
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
st.subheader("Coverage by Emotional Category")

pie = px.pie(
    category_summary,
    names="Category",
    values="Articles",
    hole=0.55
)

pie.update_layout(
    height=450
)

st.plotly_chart(
    pie,
    use_container_width=True
)
with right:

    st.subheader("Summary")

    st.metric(
        "Weighted Mean",
        f"{weighted_mean:.2f}"
    )

    st.metric(
        "Std Deviation",
        f"{weighted_std:.2f}"
    )

    st.metric(
        "Most Common Emotion",
        mode_emotion
    )

    st.metric(
        "Total Articles",
        int(total_articles)
    )

    st.metric(
        "Emotion Bins",
        emotion_bins
    )
st.divider()

st.header("📊 Actor Ranking")
st.subheader("Weighted Mean Emotion by Actor")

fig_rank = px.bar(

    ranking_df,

    x="Weighted Mean",

    y="Actor",

    orientation="h",

    color="Weighted Mean",

    text="Weighted Mean"

)

fig_rank.update_layout(

    height=700,

    yaxis=dict(categoryorder="total ascending")

)

st.plotly_chart(

    fig_rank,

    use_container_width=True

)
# ---------------------------------------------------
# HEATMAP DATA
# ---------------------------------------------------

heatmap_data = []

for sheet in sheet_names:

    temp = pd.read_excel(FILE, sheet_name=sheet)

    temp.columns = ["Emotion", "Articles"]

    temp = temp.fillna(0)

    temp["Emotion"] = pd.to_numeric(temp["Emotion"])
    temp["Articles"] = pd.to_numeric(temp["Articles"])

    temp["Actor"] = sheet

    heatmap_data.append(temp)

heatmap_df = pd.concat(heatmap_data, ignore_index=True)
st.divider()

st.header(" Emotional Coverage Heatmap")

# Create pivot table
heatmap_matrix = heatmap_df.pivot_table(
    index="Actor",
    columns="Emotion",
    values="Articles",
    aggfunc="sum",
    fill_value=0
)

fig = px.imshow(
    heatmap_matrix,
    aspect="auto",
    color_continuous_scale="RdYlBu_r",
    text_auto=True
)

fig.update_xaxes(side="top")

fig.update_layout(
    height=800,
    xaxis_title="Emotion Score",
    yaxis_title="Actor"
)

st.plotly_chart(fig, use_container_width=True)
# ---------------------------------------------------
# CHI-SQUARE TEST
# ---------------------------------------------------

st.divider()

st.header("Chi-Square Test of Independence")

contingency = []

for sheet in sheet_names:

    temp = pd.read_excel(FILE, sheet_name=sheet)

    temp.columns = ["Emotion", "Articles"]

    temp = temp.fillna(0)

    contingency.append(
        temp["Articles"].values
    )

# ---------------------------------------------------
# BUILD CONTINGENCY TABLE
# ---------------------------------------------------

contingency_df = pd.DataFrame()

for sheet in sheet_names:

    temp = pd.read_excel(FILE, sheet_name=sheet)

    temp.columns = ["Emotion", "Articles"]

    temp = temp.fillna(0)

    temp["Emotion"] = pd.to_numeric(temp["Emotion"])
    temp["Articles"] = pd.to_numeric(temp["Articles"])

    temp = temp.set_index("Emotion")

    contingency_df[sheet] = temp["Articles"]

# Fill missing emotion bins with zero
contingency_df = contingency_df.fillna(0)

# Chi-square requires rows = actors
contingency = contingency_df.T
st.subheader("Table")

st.write(contingency)

st.write("Shape:", contingency.shape)
chi2, p, dof, expected = chi2_contingency(contingency)
import pandas as pd
from scipy.stats import chi2_contingency

# Read the exported CSV
chi_df = pd.read_csv("2026-07-10T10-21_export.csv")

# Actor names
actors = chi_df.iloc[:, 0]

# Contingency table (all emotion columns)
contingency = chi_df.iloc[:, 1:]

# Run Chi-Square
chi2, p, dof, expected = chi2_contingency(contingency)

st.header("Chi-Square Test of Independence")

c1, c2, c3 = st.columns(3)

c1.metric("Chi-Square Statistic", f"{chi2:.2f}")
c2.metric("Degrees of Freedom", dof)
c3.metric("p-value", f"{p:.6f}")

alpha = 0.05

if p < alpha:
    st.success(
        f"p = {p:.6f} < {alpha}. Reject the null hypothesis. "
        "The emotional distribution differs significantly across actors."
    )
else:
    st.info(
        f"p = {p:.6f} ≥ {alpha}. Fail to reject the null hypothesis."
    )

with st.expander("Expected Frequencies"):
    expected_df = pd.DataFrame(
        expected,
        index=actors,
        columns=contingency.columns
    )

    st.dataframe(expected_df, use_container_width=True)
expected_df = pd.DataFrame(
    expected,
    index=actors,
    columns=contingency.columns
)

st.dataframe(expected_df)
small_expected = (expected < 5).sum()

st.write("Expected frequencies < 5:", small_expected)
st.divider()

st.header(" Narrative Classification")

st.subheader(frame)

st.info(explanation)

st.download_button(
    "Download Ranking",
    ranking_df.to_csv(index=False),
    "actor_rankings.csv"
)
# ---------------------------------------------------
# EXECUTIVE SUMMARY
# ---------------------------------------------------

st.divider()

st.header("📄 Executive Summary")

dataset_position = "below"

if weighted_mean > overall_mean:
    dataset_position = "above"

variability = "low"

if weighted_std > overall_std:
    variability = "high"

significance = "not statistically significant"

if p < 0.05:
    significance = "statistically significant"

summary = f"""
The selected actor (**{selected_actor}**) exhibits a weighted
emotional score of **{weighted_mean:.2f}**, which is **{dataset_position}**
the dataset average (**{overall_mean:.2f}**).

The emotional distribution is characterised by **{variability} variability**
(Standard Deviation = **{weighted_std:.2f}**), suggesting a
**{frame.lower()}** and pattern of coverage.

Across all analysed actors, the Chi-Square test indicates that
differences in emotional distributions are **{significance}**
(p < **{p:.4f}**).
"""

st.info(summary)

st.markdown(
    """
    <div style='text-align:center;color:grey;font-size:14px'>
    <b>Sagnik Nath</b><br>
    MSc Media Science Dissertation Dashboard</br>
<br> Institute of Management Study (MAKAUT) 2026 </br>
 <br> Framing Patterns in Conflict Journalism: Content Analysis of News Media's </br>Russia-Ukraine War Coverage<br>
    © 2026
    </div>
    """,
    unsafe_allow_html=True
)
