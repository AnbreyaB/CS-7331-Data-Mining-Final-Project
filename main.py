import pandas as pd
import numpy as np

from tabulate import tabulate

from preprocess import preprocess
from evaluation import evaluation_aggregate
from clean import (
    clean_binary,
    clean_disclosure,
    clean_support,
    clean_gender
)
from modeling import (
    run_random_forest, 
    run_adaboost, 
    run_decision_tree, 
    run_kmeans, 
    run_association_rules
)
from visualization import (
    plot_feature_importance,
    plot_confusion_matrix,
    plot_elbow, 
    plot_cluster_era, 
    plot_association_rules
)

np.set_printoptions(legacy='1.25')

df = preprocess()

print(df.notna().sum())
print("\nPercent filled:")
print((df.notna().mean()*100).round(2))

#combine disclosure variables
df["disclosure"] = df["reveal_disorder_coworkers"].combine_first(
    df["discuss_willing_coworkers"]
)

df["insurance_clean"] = df["insurance"].apply(clean_binary)
df["treatment_clean"] = df["sought_treatment"].apply(clean_binary)
df["self_employed_clean"] = df["self_employed"].apply(clean_binary)

df["disclosure_clean"] = df["disclosure"].apply(clean_disclosure)


df["support_clean"] = df["previous_employer_benefits"].apply(clean_support)

df["gender_clean"] = df["gender"].apply(clean_gender)

# clean age
df["age_clean"] = pd.to_numeric(df["age"], errors="coerce")

# restrict age to reasonable ages
df.loc[(df["age_clean"] < 18) | (df["age_clean"] > 100), "age_clean"] = None

# Era label: 0 = pre-pandemic (before 2020), 1 = post-shutdown (2020 and after)
df["era"] = df["year"].apply(lambda y: 0 if y < 2020 else 1)

clean_cols = [
    "insurance_clean",
    "treatment_clean",
    "self_employed_clean",
    "disclosure_clean",
    "support_clean",
    "gender_clean",
    "age_clean",
    "year"
]

for col in clean_cols:
    print("\n", col)
    print(df[col].value_counts(dropna=False))


df_final = df[[
    "treatment_clean",
    "insurance_clean",
    "support_clean",
    "self_employed_clean",
    "disclosure_clean",
    "gender_clean",
    "age_clean",
    "year",
    "era"
]].copy()

df_final.shape
#FINAL missing count for final clean dataset.
print(df_final.isna().mean().round(3))

rf_results = run_random_forest(df_final)
ada_results = run_adaboost(df_final)
dt_results = run_decision_tree(df_final)
km_results = run_kmeans(df_final)
arm_results = run_association_rules(df_final)

print()
print("Evaluation Metrics for Random Forest: ")
rf_evaluation = evaluation_aggregate(rf_results)
print(rf_evaluation)
print()

print("Evaluation Metrics for AdaBoost: ")
ada_evaluation = evaluation_aggregate(ada_results)
print(ada_evaluation)
print()

print("Evaluation Metrics for Decision Tree: ")
dt_evaluation = evaluation_aggregate(dt_results)
print(dt_evaluation)
print()

print("Evaluation Metrics for k-Means Cluster: ")
km_evaluation = evaluation_aggregate(km_results)
print(km_evaluation)
print()

print("Evaluation Metrics for Association Rules: ")
arm_evaluation = {
    'mean_support': round(arm_results['rules']['support'].mean(), 4),
    'mean_confidence': round(arm_results['rules']['confidence'].mean(), 4),
    'mean_lift': round(arm_results['rules']['lift'].mean(), 4),
}
print(arm_evaluation)
print()

metric_headers = ["Model", "Accuracy", "Precision", "Recall", "F1"]
metrics_table = [
    ["Random Forest", rf_evaluation["accuracy"], rf_evaluation["precision"], rf_evaluation["recall"], rf_evaluation["f1"]],
    ["AdaBoost", ada_evaluation["accuracy"], ada_evaluation["precision"], ada_evaluation["recall"], ada_evaluation["f1"]],
    ["Decision Tree", dt_evaluation["accuracy"], dt_evaluation["precision"], dt_evaluation["recall"], dt_evaluation["f1"]]
]
with open('outputs/metrics_table.txt', 'w', encoding="utf-8") as f:
    print(tabulate(tabular_data=metrics_table, headers=metric_headers, tablefmt="fancy_grid", maxcolwidths=[None, 20]), file=f)

plot_feature_importance(rf_results['feature_importances'], 'Random Forest')
plot_confusion_matrix(rf_results['y_test'], rf_results['y_pred'], 'Random Forest')
plot_confusion_matrix(ada_results['y_test'], ada_results['y_pred'], 'AdaBoost')
plot_feature_importance(dt_results['feature_importances'], 'Decision Tree')
plot_confusion_matrix(dt_results['y_test'], dt_results['y_pred'], 'Decision Tree')
plot_elbow(km_results['inertia_scores'], km_results['k_range'], km_results['best_k'])
plot_cluster_era(km_results['km_df'])
plot_association_rules(arm_results['rules'])
