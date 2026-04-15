import pandas as pd

from preprocess import preprocess
import evaluation

df = preprocess()

print(df.info())
print(df.head(20))