from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import numpy as np

# Shared utilities used by all three classifiers
FEATURES = [
    'treatment_clean',
    'support_clean',
    'self_employed_clean',
    'disclosure_clean',
    'gender_clean',
    'age_clean',
]

TARGET = 'era'

# Encode gender string to numeric for modeling
def encode_features(df):
    df = df.copy()
    le = LabelEncoder()
    df['gender_clean'] = le.fit_transform(df['gender_clean'].astype(str))
    return df

# Split features and target, drop nulls, encode
def prepare_classification_data(df_final):
    model_df = df_final[FEATURES + [TARGET]].dropna().copy()
    model_df = encode_features(model_df)
    X = model_df[FEATURES]
    y = model_df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model 1: Random Forest
def run_random_forest(df_final):
    X_train, X_test, y_train, y_test = prepare_classification_data(df_final)

    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced'],
    }

    # Tune with randomized search using F1 as scoring metric
    rf = RandomForestClassifier(random_state=42)
    search = RandomizedSearchCV(rf, param_distributions=param_dist,
                                n_iter=20, cv=5, scoring='f1',
                                random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # Rank features by importance
    feature_importances = pd.Series(
        best_model.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)

    print("\nRandom Forest: \n")
    print(f"Best params: {search.best_params_}")
    print(f"Feature importances:\n{feature_importances}")

    return {
        'model': best_model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'feature_importances': feature_importances,
    }


# Model 2: AdaBoost
def run_adaboost(df_final):
    X_train, X_test, y_train, y_test = prepare_classification_data(df_final)

    param_dist = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5, 1.0],
        'estimator': [DecisionTreeClassifier(max_depth=1),
                      DecisionTreeClassifier(max_depth=2)],
    }

    # Tune with randomized search
    ada = AdaBoostClassifier(random_state=42)
    search = RandomizedSearchCV(ada, param_distributions=param_dist,
                                n_iter=10, cv=5, scoring='f1',
                                random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    print("\nAdaBoost: \n")
    print(f"Best params: {search.best_params_}")

    return {
        'model': best_model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
    }

# Model 3: Decision Tree
def run_decision_tree(df_final):
    X_train, X_test, y_train, y_test = prepare_classification_data(df_final)

    param_dist = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy'],
        'class_weight': ['balanced'],
    }

    # Tune with randomized search
    dt = DecisionTreeClassifier(random_state=42)
    search = RandomizedSearchCV(dt, param_distributions=param_dist,
                                n_iter=20, cv=5, scoring='f1',
                                random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # Rank features by importance
    feature_importances = pd.Series(
        best_model.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)

    print("\nDecision Tree: \n")
    print(f"Best params: {search.best_params_}")
    print(f"Feature importances:\n{feature_importances}")

    return {
        'model': best_model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'feature_importances': feature_importances,
    }


# Model 4: KMeans
KM_FEATURES = ['treatment_clean', 'support_clean', 'disclosure_clean', 'age_clean', 'gender_clean']

def run_kmeans(df_final):
    km_df = df_final[KM_FEATURES + ['era']].dropna().copy()
    km_df = encode_features(km_df)
    X = km_df[KM_FEATURES].values

    # Compute inertia k=2-10 to find best k
    inertia_scores = []
    k_range = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertia_scores.append(km.inertia_)

    # Select best k using elbow curvature
    inertia_arr = np.array(inertia_scores)
    deltas = np.diff(inertia_arr)
    curvature = np.diff(deltas)
    best_k = list(k_range)[np.argmax(curvature) + 2]

    # Fit final model with best k
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km_final.fit_predict(X)
    km_df['cluster'] = labels

    print("\nKMeans: \n")
    print(f"Best k: {best_k}")
    print("Cluster distribution:")
    print(km_df['cluster'].value_counts().sort_index())
    print("Era breakdown per cluster:")
    print(km_df.groupby('cluster')['era'].value_counts())

    return {
        'model': km_final,
        'X': X,
        'labels': labels,
        'km_df': km_df,
        'best_k': best_k,
        'inertia_scores': inertia_scores,
        'k_range': list(k_range),
    }


# Model 5: Association Rule Mining
ARM_FEATURES = ['treatment_clean', 'support_clean', 'disclosure_clean', 'self_employed_clean']

def run_association_rules(df_final):
    arm_df = df_final[ARM_FEATURES].dropna().copy()

    # Convert columns to binary for Apriori
    binary_df = pd.DataFrame()
    binary_df['sought_treatment'] = arm_df['treatment_clean'] == 1
    binary_df['has_employer_support'] = arm_df['support_clean'] > 0
    binary_df['willing_to_disclose'] = arm_df['disclosure_clean'] == 1
    binary_df['self_employed'] = arm_df['self_employed_clean'] == 1

    # Find frequent itemsets
    freq_items = apriori(binary_df, min_support=0.1, use_colnames=True)

    # Generate rules filtered by lift
    rules = association_rules(freq_items, metric='lift', min_threshold=1.0)
    rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)

    print("Association Rule Mining: \n")
    print(f"Frequent itemsets found: {len(freq_items)}")
    print(f"Rules found: {len(rules)}")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

    return {
        'rules': rules,
        'freq_items': freq_items,
        'binary_df': binary_df,
    }
