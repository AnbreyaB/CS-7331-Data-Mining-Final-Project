import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
 
# Save all charts to outputs folder
base_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(base_dir, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Feature importance bar chart for Random Forest and Decision Tree
def plot_feature_importance(feature_importances, model_name):
    plt.figure(figsize=(8, 5))
    feature_importances.sort_values().plot(kind='barh', color='steelblue')
    plt.xlabel('Importance')
    plt.title(f'{model_name} Feature Importances')
    plt.tight_layout()
    fname = model_name.lower().replace(' ', '_') + '_feature_importance.png'
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
    plt.close()
 
# Confusion matrix for each classifier
def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Pre-Pandemic', 'Post-Shutdown'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{model_name} Confusion Matrix')
    plt.tight_layout()
    fname = model_name.lower().replace(' ', '_') + '_confusion_matrix.png'
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
    plt.close()

# KMeans elbow plot
def plot_elbow(inertia_scores, k_range, best_k):
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia_scores, marker='o', color='steelblue')
    plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best k={best_k}')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('KMeans Elbow Plot')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'kmeans_elbow.png'), dpi=150)
    plt.close()

# KMeans era breakdown per cluster
def plot_cluster_era(km_df):
    era_counts = km_df.groupby(['cluster', 'era']).size().unstack(fill_value=0)
    era_counts.columns = ['Pre-Pandemic', 'Post-Shutdown']
    era_counts.plot(kind='bar', figsize=(8, 5), color=['steelblue', 'darkorange'])
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.title('Era Breakdown per Cluster')
    plt.xticks(rotation=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'kmeans_cluster_era.png'), dpi=150)
    plt.close()

# Association Rules top rules by lift
def plot_association_rules(rules):
    top_rules = rules.head(6).copy()
    labels = [
        f"{', '.join(list(r['antecedents']))} -> {', '.join(list(r['consequents']))}"
        for _, r in top_rules.iterrows()
    ]
    plt.figure(figsize=(10, 6))
    plt.barh(labels, top_rules['lift'].values, color='steelblue')
    plt.xlabel('Lift')
    plt.title('Top Association Rules by Lift')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'association_rules_lift.png'), dpi=150)
    plt.close()
