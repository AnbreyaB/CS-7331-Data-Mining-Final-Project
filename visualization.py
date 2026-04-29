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
