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

ranking_df = ranking_df.sort_values(
    "Weighted Mean"
)

ranking_df.reset_index(
    drop=True,
    inplace=True
)

ranking_df.index += 1
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