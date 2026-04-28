import pandas as pd

from preprocess import preprocess
import modeling as model
import evaluation as eval
import visualization as vis

df = preprocess()

print(df.info())
print(df.head(20))