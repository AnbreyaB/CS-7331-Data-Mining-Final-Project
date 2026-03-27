import pandas as pd
import os
import csv

dataset_folder = "data/"
surveys = []

# for file in os.listdir(dataset_folder):
#     df = pd.read_csv(dataset_folder+file)
#     with open("columns/combined.txt", "a") as f:
#         for column in df.columns:
#             f.write(f"{column}\n")
#         f.write(f"--------------------------------------------------------\n")

for file in os.listdir(dataset_folder):
    df = pd.read_csv(dataset_folder+file)
    surveys.append(df)
df_concat = pd.concat(surveys, axis=0)
with open("columns/combined.txt", "a") as f:
    for column in df_concat.columns:
        f.write(f"{column}\n")
    f.write(f"--------------------------------------------------------\n")

# def remove_special_characters(survey):
#     for column in survey.columns:
#         print()
#     return