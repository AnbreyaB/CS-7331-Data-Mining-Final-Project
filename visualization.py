import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
 
# Save all charts to outputs folder
base_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(base_dir, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
