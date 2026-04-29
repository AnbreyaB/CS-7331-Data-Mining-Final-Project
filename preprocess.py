import pandas as pd
import json
import os

def preprocess():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_folder = os.path.join(base_dir, 'data')
    surveys = []

    for file in os.listdir(dataset_folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(dataset_folder, file), encoding='latin1')

            # -------------------------
            # FIXED YEAR LOGIC
            # -------------------------
            year = None
            file_lower = file.lower()

            if "2014" in file_lower:
                year = 2014
            elif "2016" in file_lower:
                year = 2016
            elif "2017" in file_lower:
                year = 2017
            elif "2018" in file_lower:
                year = 2018
            elif "2019" in file_lower:
                year = 2019
            elif "2020" in file_lower:
                year = 2020
            elif "2021" in file_lower:
                year = 2021
            elif "responses" in file_lower and "(1)" not in file_lower and "__1__" not in file_lower:
                year = 2022
            elif "responses" in file_lower and "(1)" in file_lower:
                year = 2023

            df["year"] = year
            # -------------------------

            for col in df.columns:
                col_lower = col.lower()

                #insurance variable
                if "medical coverage" in col_lower:
                    df["insurance"] = df[col]

                #treatment variable
                if "sought treatment" in col_lower:
                    df["sought_treatment"] = df[col]

                #disclosure variable
                if "reveal" in col_lower and "cowork" in col_lower:
                    df["reveal_disorder_coworkers"] = df[col]
                if "discuss" in col_lower and "cowork" in col_lower:
                    df["discuss_willing_coworkers"] = df[col]

                #employer support variable
                if "mental health benefits" in col_lower:
                    df["previous_employer_benefits"] = df[col]
                if "aware of the options" in col_lower:
                    df["previous_employer_options_aware"] = df[col]
                if "supervisor" in col_lower:
                    df["discuss_willing_supervisor"] = df[col]

                #age gender and country variables
                if "self-employed" in col_lower:
                    df["self_employed"] = df[col]
                if "age" in col_lower:
                    df["age"] = df[col]
                if "gender" in col_lower:
                    df["gender"] = df[col]
                if "country" in col_lower:
                    df["country"] = df[col]

            surveys.append(df)

    df_concat = pd.concat(surveys, axis=0, ignore_index=True)
    df_concat = df_concat.loc[:, ~df_concat.columns.duplicated()]

    final_columns = [
        "age",
        "gender",
        "country",
        "insurance",
        "sought_treatment",
        "self_employed",
        "reveal_disorder_coworkers",
        "discuss_willing_coworkers",
        "previous_employer_benefits",
        "previous_employer_options_aware",
        "discuss_willing_supervisor",
        "year"
    ]

    df_final = df_concat[[col for col in final_columns if col in df_concat.columns]]
    return df_final
