import pandas as pd
import json
import os

def preprocess():
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
    # with open("columns/combined.txt", "a") as f:
    #     for column in df_concat.columns:
    #         f.write(f"{column}\n")
    #     f.write(f"--------------------------------------------------------\n")

    # def remove_special_characters(survey):
    #     for column in survey.columns:
    #         print()
    #     return
    column_mappings = {}
    with open('columns/mapping.json', 'r') as file:
        column_mappings = json.load(file)
    df_mapped = df_concat.rename(columns=column_mappings)

    final_columns = [
        "age",
        "gender",
        "country",
        "insurance",
        "sought_treatment",
        "self_employed",
        "reveal_disorder_client_contacts",
        "reveal_disorder_coworkers",
        "previous_employers",
        "previous_employer_benefits",
        "previous_employer_options_aware",
        "discuss_willing_coworkers",
        "discuss_willing_supervisor",
        "discuss_willing_previous_employer"
    ]

    return df_mapped[final_columns]