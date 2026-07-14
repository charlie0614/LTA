
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/DigiHero/2026-06-02_data_symptoms_DigiHero_Bonn_preprocessed.csv")

age_cols = ["18-34", "35-59", "60-79"]
age_df = df[age_cols]

# Count of 1s per column
ones_per_column = age_df.sum(axis=0)

# comorbidity: Anzahl der 1en und Rest
comorbidity_yes = int(df["comorbidity"].sum())
comorbidity_rest = len(df) - comorbidity_yes

# female: Anzahl der 1en und Rest
female_yes = int(df["female"].sum())
female_rest = len(df) - female_yes

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Alter
ones_per_column.plot(kind="bar", ax=axs[0])
axs[0].set_xlabel("Age group")
axs[0].set_ylabel("Number of individuals")

# Comorbidity
axs[1].bar(["comorbidity", "None"], [comorbidity_yes, comorbidity_rest], color=["tab:blue", "tab:gray"])
axs[1].set_xlabel("comorbidities")
axs[1].set_ylabel("Number of individuals")

# Female
axs[2].bar(["female", "male"], [female_yes, female_rest], color=["tab:blue", "tab:gray"])
axs[2].set_xlabel("sex")
axs[2].set_ylabel("Number of individuals")

plt.tight_layout()
plt.savefig("age_comorbidity_female_counts.png")
plt.show()