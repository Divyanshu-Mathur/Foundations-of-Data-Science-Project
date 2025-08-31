import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Training.csv")

# Preprocess
symptom_df = df.drop(columns=['prognosis'])

# Top 25 symptoms (by frequency)
symptom_counts = symptom_df.sum().sort_values(ascending=False)
top_25_symptoms = symptom_counts.index[:25]
top_20_symptoms = symptom_counts.index[:20]
top_10_symptoms = symptom_counts.index[:10]


st.set_page_config(page_title="Disease Prediction Dashboard", layout="wide")

st.title("Disease Prediction Dataset Dashboard")

# 1. Distribution of Diseases
st.header("Distribution of Diseases")
disease_counts = df['prognosis'].value_counts()

fig1, ax1 = plt.subplots(figsize=(14,6))
disease_counts.plot(kind='bar', ax=ax1, color="lightcoral")

ax1.set_title("Distribution of Diseases")
ax1.set_xlabel("Disease")
ax1.set_ylabel("Count")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, ha="right") 
st.pyplot(fig1)
st.markdown("The dataset has an equal number of records for each disease (120), ensuring a balanced distribution across all prognosis categories")


# 2. Top 25 Most Common Symptoms
st.header("Top 25 Most Common Symptoms")

total_patients = df.shape[0]
symptom_percent = (symptom_counts / total_patients) * 100

fig2, ax2 = plt.subplots(figsize=(12,6))
symptom_percent[:25].plot(kind='bar', ax=ax2, color="skyblue")

ax2.set_title("Top 25 Most Common Symptoms")
ax2.set_ylabel("Percentage of Patients (%)")

st.pyplot(fig2)
st.markdown("The data shows that while some symptoms are very common and general, others provide specific diagnostic clues that can help narrow down possible diseases.")

# 3. Correlation Heatmap of Top 20 Symptoms
st.header("Correlation Heatmap of Top 20 Most Common Symptoms")
fig3, ax3 = plt.subplots(figsize=(15,10))
corr = df[top_20_symptoms].corr()
sns.heatmap(corr, cmap="YlOrRd", cbar=True, ax=ax3,annot=True,fmt=".2f")
ax3.set_title("Correlation Heatmap (Top 20 Symptoms)")
st.pyplot(fig3)

# 4. Symptom Co-occurrence Heatmap (Top 20 Symptoms)
st.header("Symptom Co-occurrence Heatmap (Top 20 Symptoms)")
from sklearn.metrics import jaccard_score

co_matrix = pd.DataFrame(index=top_20_symptoms, columns=top_20_symptoms)

for col1 in top_20_symptoms:
    for col2 in top_20_symptoms:
        co_matrix.loc[col1, col2] = jaccard_score(symptom_df[col1], symptom_df[col2])

co_matrix = co_matrix.astype(float)

fig4, ax4 = plt.subplots(figsize=(15,10))
sns.heatmap(co_matrix, cmap="YlOrRd", ax=ax4,annot=True,fmt=".2f")
ax4.set_title("Symptom Co-occurrence Heatmap (Top 20 Symptoms)")
st.pyplot(fig4)

# 5. Top 10 Symptom Frequency Across Diseases
st.header("Top 10 Symptom Frequency Across Diseases")
symptom_disease_matrix = df.groupby("prognosis").sum()

fig5, ax5 = plt.subplots(figsize=(15,10))
sns.heatmap(symptom_disease_matrix[top_10_symptoms], cmap="YlOrRd", cbar=True, ax=ax5,annot=True,fmt="d")
ax5.set_title("Top 10 Symptom Frequency Across Diseases")
ax5.set_xlabel("Symptoms")
ax5.set_ylabel("Disease")
st.pyplot(fig5)

