import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# =========================================================
# 1. LOAD THE EXCEL FILE
# =========================================================
file_path = r"culture_score_dummy_2.xlsx"
df = pd.read_excel(file_path)


# Drop empty columns and rows
df = df.loc[:, ~df.columns.str.startswith("EMPTY_")]
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
# 4. CREATE AXIS SCORES
#    Correct opposing pairs:
#    Axis 1 = Collaborate - Compete
#    Axis 2 = Create - Control
# =========================================================
df["Axis1_Now"] = df["Collaborate_Now_Index"] - df["Compete_Now_Index"]
df["Axis1_Pref"] = df["Collaborate_Pref_Index"] - df["Compete_Pref_Index"]

df["Axis2_Now"] = df["Create_Now_Index"] - df["Control_Now_Index"]
df["Axis2_Pref"] = df["Create_Pref_Index"] - df["Control_Pref_Index"]

# Difference columns
df["Diff_Axis1"] = df["Axis1_Pref"] - df["Axis1_Now"]
df["Diff_Axis2"] = df["Axis2_Pref"] - df["Axis2_Now"]

# Optional overall directional index
df["Overall_Axis_Now"] = df[["Axis1_Now", "Axis2_Now"]].mean(axis=1)
df["Overall_Axis_Pref"] = df[["Axis1_Pref", "Axis2_Pref"]].mean(axis=1)
df["Diff_Overall_Axis"] = df["Overall_Axis_Pref"] - df["Overall_Axis_Now"]

print("Axis scores created.")


# =========================================================
# 5. RUN WILCOXON SIGNED-RANK TEST
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

results.append(
    run_wilcoxon(
        df["Axis1_Now"],
        df["Axis1_Pref"],
        "Axis 1 (Collaborate - Compete): Now vs Pref"
    )
)

results.append(
    run_wilcoxon(
        df["Axis2_Now"],
        df["Axis2_Pref"],
        "Axis 2 (Create - Control): Now vs Pref"
    )
)

results.append(
    run_wilcoxon(
        df["Overall_Axis_Now"],
        df["Overall_Axis_Pref"],
        "Overall Axis Index: Now vs Pref"
    )
)

results_df = pd.DataFrame(results)

print("\n================ WILCOXON RESULTS ================\n")
print(results_df.to_string(index=False))


# =========================================================
# 6. APPLY BONFERRONI CORRECTION
#    For the 2 axis-level tests only
# =========================================================
alpha = 0.05
bonferroni_alpha = alpha / 2  # 0.025

results_df["Alpha_Used"] = np.where(
    results_df["Test"].str.contains("Overall Axis Index"),
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
# 7. CALCULATE EFFECT SIZE (r)
#    r = |Z| / sqrt(N)
#    Z approximated from Wilcoxon W
# =========================================================
def add_effect_size(now_series, pref_series, label):
    temp = pd.DataFrame({"Now": now_series, "Pref": pref_series}).dropna()

    x = temp["Now"]
    y = temp["Pref"]
    diff = y - x

    diff_nonzero = diff[diff != 0]
    n = len(diff_nonzero)

    if n == 0:
        return {
            "Test": label,
            "Effect_Size_r": np.nan,
            "Effect_Size_Magnitude": "Not applicable"
        }

    stat, _ = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")

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

effect_results.append(
    add_effect_size(
        df["Axis1_Now"],
        df["Axis1_Pref"],
        "Axis 1 (Collaborate - Compete): Now vs Pref"
    )
)

effect_results.append(
    add_effect_size(
        df["Axis2_Now"],
        df["Axis2_Pref"],
        "Axis 2 (Create - Control): Now vs Pref"
    )
)

effect_results.append(
    add_effect_size(
        df["Overall_Axis_Now"],
        df["Overall_Axis_Pref"],
        "Overall Axis Index: Now vs Pref"
    )
)

effect_df = pd.DataFrame(effect_results)
results_df = results_df.merge(effect_df, on="Test", how="left")

print("\n================ EFFECT SIZE RESULTS ================\n")
print(results_df.to_string(index=False))


# =========================================================
# 8. ADD FINAL INTERPRETATION
# =========================================================
def final_interpretation(row):
    if pd.isna(row["P_Value"]):
        return "Test not applicable"

    if "Axis 1" in row["Test"]:
        if row["Median_Gap_Pref_minus_Now"] > 0:
            direction = "Preferred culture shifts toward Collaborate over Compete"
        elif row["Median_Gap_Pref_minus_Now"] < 0:
            direction = "Preferred culture shifts toward Compete over Collaborate"
        else:
            direction = "No median shift on Collaborate-Compete axis"

    elif "Axis 2" in row["Test"]:
        if row["Median_Gap_Pref_minus_Now"] > 0:
            direction = "Preferred culture shifts toward Create over Control"
        elif row["Median_Gap_Pref_minus_Now"] < 0:
            direction = "Preferred culture shifts toward Control over Create"
        else:
            direction = "No median shift on Create-Control axis"

    else:
        if row["Median_Gap_Pref_minus_Now"] > 0:
            direction = "Preferred overall axis balance is higher than current"
        elif row["Median_Gap_Pref_minus_Now"] < 0:
            direction = "Preferred overall axis balance is lower than current"
        else:
            direction = "No overall median shift"

    if row["Significant_After_Correction"]:
        return f"{direction}; statistically significant after correction; {row['Effect_Size_Magnitude'].lower()} effect"
    else:
        return f"{direction}; not statistically significant after correction; {row['Effect_Size_Magnitude'].lower()} effect"


results_df["Final_Interpretation"] = results_df.apply(final_interpretation, axis=1)

print("\n================ FINAL INTERPRETATION RESULTS ================\n")
print(results_df.to_string(index=False))


# =========================================================
# 9. REORDER RESULT COLUMNS
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
# 10. SAVE OUTPUTS
# =========================================================
results_df.to_excel("dummy_2_results.xlsx", index=False)

print("\nFiles saved:")
print("dummy_2_results.xlsx")