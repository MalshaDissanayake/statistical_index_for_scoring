import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# =========================================================
# 1. LOAD THE EXCEL FILE
# =========================================================
file_path = r"CAI Test.xlsx"

# Read without headers because your headers span 3 rows
raw = pd.read_excel(file_path, header=None)

# ---------------------------------------------------------
# Your structure:
# Row 0 = category
# Row 1 = Now / Pref
# Row 2 = Collaborate / Create / Compete / Control
# Row 3 onward = data
# ---------------------------------------------------------

header_row_1 = raw.iloc[0].copy()
header_row_2 = raw.iloc[1].copy()
header_row_3 = raw.iloc[2].copy()

# Forward-fill merged header cells
header_row_1 = header_row_1.ffill()
header_row_2 = header_row_2.ffill()

# Build clean column names
new_columns = []
for i in range(raw.shape[1]):
    h1 = header_row_1[i]
    h2 = header_row_2[i]
    h3 = header_row_3[i]

    if i == 0:
        new_columns.append("Participant_No")
    else:
        if pd.isna(h1) and pd.isna(h2) and pd.isna(h3):
            new_columns.append(f"EMPTY_{i}")
        else:
            new_columns.append(f"{h1}_{h2}_{h3}")

# Assign column names
df = raw.iloc[3:].copy()
df.columns = new_columns

# Drop fully empty columns
df = df.loc[:, ~df.columns.str.startswith("EMPTY_")]

# Drop fully empty rows
df = df.dropna(how="all").reset_index(drop=True)

# Convert score columns to numeric
for col in df.columns:
    if col != "Participant_No":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows where participant number is missing
df = df.dropna(subset=["Participant_No"]).reset_index(drop=True)

print("Dataset loaded successfully.")
print(f"Rows (participants): {len(df)}")
print(f"Columns: {len(df.columns)}")
print("\nFirst 10 columns:")
print(df.columns[:10].tolist())


# =========================================================
# 2. IDENTIFY THE SIX MAIN CATEGORIES
# =========================================================
categories = [
    "Dominant",
    "Leadership",
    "Employee Management",
    "Organisation Glue",
    "Strategic Emphasis",
    "Criteria for Success"
]

dimensions = ["Collaborate", "Create", "Compete", "Control"]


# =========================================================
# 3. CREATE PARTICIPANT-LEVEL DIMENSION INDICES
#    Median across the 6 category scores for each dimension
# =========================================================
for dim in dimensions:
    now_cols = [f"{cat}_Now_{dim}" for cat in categories]
    pref_cols = [f"{cat}_Pref_{dim}" for cat in categories]

    missing_now = [c for c in now_cols if c not in df.columns]
    missing_pref = [c for c in pref_cols if c not in df.columns]

    if missing_now:
        raise ValueError(f"Missing NOW columns for {dim}: {missing_now}")
    if missing_pref:
        raise ValueError(f"Missing PREF columns for {dim}: {missing_pref}")

    df[f"{dim}_Now_Index"] = df[now_cols].median(axis=1)
    df[f"{dim}_Pref_Index"] = df[pref_cols].median(axis=1)

print("\nParticipant-level dimension indices created.")


# =========================================================
# 4. CREATE PARTICIPANT-LEVEL OVERALL COMPANY INDICES
#    Median across all 24 Now items / all 24 Pref items
# =========================================================
all_now_cols = [f"{cat}_Now_{dim}" for cat in categories for dim in dimensions]
all_pref_cols = [f"{cat}_Pref_{dim}" for cat in categories for dim in dimensions]

df["Overall_Now_Index"] = df[all_now_cols].median(axis=1)
df["Overall_Pref_Index"] = df[all_pref_cols].median(axis=1)

print("Overall company indices created.")


# =========================================================
# 5. CREATE DIFFERENCE COLUMNS
#    Difference = Pref - Now
# =========================================================
for dim in dimensions:
    df[f"Diff_{dim}"] = df[f"{dim}_Pref_Index"] - df[f"{dim}_Now_Index"]

df["Diff_Overall"] = df["Overall_Pref_Index"] - df["Overall_Now_Index"]

print("Difference columns created.")


# =========================================================
# 6. RUN WILCOXON SIGNED-RANK TEST
# =========================================================
def run_wilcoxon(now_series, pref_series, label):
    temp = pd.DataFrame({"Now": now_series, "Pref": pref_series}).dropna()

    x = temp["Now"]
    y = temp["Pref"]
    diff = y - x

    n_nonzero = np.sum(diff != 0)

    if n_nonzero == 0:
        return {
            "Test": label,
            "N_total_pairs": len(temp),
            "N_nonzero_pairs": int(n_nonzero),
            "Median_Now": x.median(),
            "Median_Pref": y.median(),
            "Median_Gap_Pref_minus_Now": diff.median(),
            "Wilcoxon_Statistic": np.nan,
            "P_Value": np.nan,
            "Interpretation": "All paired differences are zero; Wilcoxon test not applicable."
        }

    stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")

    return {
        "Test": label,
        "N_total_pairs": len(temp),
        "N_nonzero_pairs": int(n_nonzero),
        "Median_Now": x.median(),
        "Median_Pref": y.median(),
        "Median_Gap_Pref_minus_Now": diff.median(),
        "Wilcoxon_Statistic": stat,
        "P_Value": p,
        "Interpretation": "Significant difference" if p < 0.05 else "No significant difference"
    }


results = []

for dim in dimensions:
    results.append(
        run_wilcoxon(
            df[f"{dim}_Now_Index"],
            df[f"{dim}_Pref_Index"],
            f"{dim}: Now vs Pref"
        )
    )

results.append(
    run_wilcoxon(
        df["Overall_Now_Index"],
        df["Overall_Pref_Index"],
        "Overall Company: Now vs Pref"
    )
)

results_df = pd.DataFrame(results)

print("\n================ WILCOXON RESULTS ================\n")
print(results_df.to_string(index=False))


# =========================================================
# 7. APPLY BONFERRONI CORRECTION
#    For the 4 dimension-level tests only
# =========================================================
alpha = 0.05
bonferroni_alpha = alpha / 4

results_df["Alpha_Used"] = np.where(
    results_df["Test"].str.contains("Overall Company"),
    alpha,
    bonferroni_alpha
)

results_df["Significant_After_Correction"] = results_df["P_Value"] < results_df["Alpha_Used"]

results_df["Decision_After_Correction"] = np.where(
    results_df["P_Value"].isna(),
    "Test not applicable",
    np.where(
        results_df["Significant_After_Correction"],
        "Reject H0",
        "Fail to reject H0"
    )
)

results_df["Corrected_Interpretation"] = np.where(
    results_df["P_Value"].isna(),
    "Wilcoxon test not applicable",
    np.where(
        results_df["Significant_After_Correction"],
        "Significant difference after correction",
        "No significant difference after correction"
    )
)

print("\n================ BONFERRONI-CORRECTED RESULTS ================\n")
print(results_df.to_string(index=False))


# =========================================================
# 8. CALCULATE EFFECT SIZE (r) AND PRACTICAL MAGNITUDE
#    r = |Z| / sqrt(N)
#    Z is approximated from Wilcoxon W
# =========================================================
def add_effect_size(now_series, pref_series, label):
    temp = pd.DataFrame({"Now": now_series, "Pref": pref_series}).dropna()

    x = temp["Now"]
    y = temp["Pref"]
    diff = y - x

    # Exclude zero differences
    diff_nonzero = diff[diff != 0]
    n = len(diff_nonzero)

    if n == 0:
        return {
            "Test": label,
            "Effect_Size_r": np.nan,
            "Effect_Size_Magnitude": "Not applicable"
        }

    stat, _ = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")

    # Approximate Z from Wilcoxon statistic
    mean_w = n * (n + 1) / 4
    sd_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)

    if sd_w == 0:
        r = np.nan
    else:
        z = (stat - mean_w) / sd_w
        r = abs(z) / np.sqrt(n)

    if pd.isna(r):
        mag = "Not applicable"
    elif r < 0.1:
        mag = "Negligible"
    elif r < 0.3:
        mag = "Small"
    elif r < 0.5:
        mag = "Medium"
    else:
        mag = "Large"

    return {
        "Test": label,
        "Effect_Size_r": r,
        "Effect_Size_Magnitude": mag
    }


effect_results = []

for dim in dimensions:
    effect_results.append(
        add_effect_size(
            df[f"{dim}_Now_Index"],
            df[f"{dim}_Pref_Index"],
            f"{dim}: Now vs Pref"
        )
    )

effect_results.append(
    add_effect_size(
        df["Overall_Now_Index"],
        df["Overall_Pref_Index"],
        "Overall Company: Now vs Pref"
    )
)

effect_df = pd.DataFrame(effect_results)

results_df = results_df.merge(effect_df, on="Test", how="left")

print("\n================ EFFECT SIZE RESULTS ================\n")
print(results_df.to_string(index=False))


# =========================================================
# 9. ADD FINAL INTERPRETATION
# =========================================================
def final_interpretation(row):
    if pd.isna(row["P_Value"]):
        return "Test not applicable"

    if row["Median_Gap_Pref_minus_Now"] > 0:
        direction = "Preferred is higher than Now"
    elif row["Median_Gap_Pref_minus_Now"] < 0:
        direction = "Preferred is lower than Now"
    else:
        direction = "No median gap"

    if row["Significant_After_Correction"]:
        return f"{direction}; statistically significant after correction; {row['Effect_Size_Magnitude'].lower()} effect"
    else:
        return f"{direction}; not statistically significant after correction; {row['Effect_Size_Magnitude'].lower()} effect"

results_df["Final_Interpretation"] = results_df.apply(final_interpretation, axis=1)

print("\n================ FINAL INTERPRETATION RESULTS ================\n")
print(results_df.to_string(index=False))


# =========================================================
# 10. REORDER RESULT COLUMNS FOR CLEANER EXCEL OUTPUT
# =========================================================
results_df = results_df[
    [
        "Test",
        "N_total_pairs",
        "N_nonzero_pairs",
        "Median_Now",
        "Median_Pref",
        "Median_Gap_Pref_minus_Now",
        "Wilcoxon_Statistic",
        "P_Value",
        "Interpretation",
        "Alpha_Used",
        "Significant_After_Correction",
        "Decision_After_Correction",
        "Corrected_Interpretation",
        "Effect_Size_r",
        "Effect_Size_Magnitude",
        "Final_Interpretation"
    ]
]


# =========================================================
# 11. SAVE OUTPUTS
# =========================================================
df.to_excel("culture_scored_output.xlsx", index=False)
results_df.to_excel("culture_wilcoxon_results.xlsx", index=False)

print("\nFiles saved:")
print("1. culture_scored_output.xlsx")
print("2. culture_wilcoxon_results.xlsx")