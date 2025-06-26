import os
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
from scripts import transform, plot
import shap

warnings.filterwarnings("ignore")

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/ml_ready"
MODEL_OUTPUT_DIR = "data/model_output"

def load_data():
    accounts = pd.read_csv(os.path.join(RAW_DATA_DIR, "accounts.csv"), parse_dates=['start_date'])
    snapshots = pd.read_csv(os.path.join(RAW_DATA_DIR, "balance_snapshots.csv"), parse_dates=['check_date'], keep_default_na=True).set_index('snapshot_id',drop=False)
    bad_debts = pd.read_csv(os.path.join(RAW_DATA_DIR, "bad_debts.csv"), parse_dates=['closing_date'])
    return accounts, snapshots, bad_debts

def generate_features(snapshots):
    from pipeline_utils import generate_snapshots_with_aggregated_features
    return generate_snapshots_with_aggregated_features(snapshots)

def annotate_data(snapshots_data, bad_debts, min_cycle=6, spacing=3):
    from pipeline_utils import annotate_snapshots
    return annotate_snapshots(snapshots_data, bad_debts, min_cycle, spacing)

def normalize_data(snapshots_data):
    from pipeline_utils import normalize_snapshot_features_by_cycles
    norm_features = [
        'check_cycle_number', 'total_delinquencies', 'count_suspension',
        'count_actively_delinquent', 'total_penalties', 'count_streak_1',
        'count_streak_2_3', 'count_streak_4plus', 'count_full_misses',
        'count_major_misses', 'count_partial_misses', 'count_minor_misses'
    ]
    non_norm_features = ['check_cycle_number', 'max_delinquency_score', 'average_penalty_per_incident']
    return normalize_snapshot_features_by_cycles(snapshots_data, norm_features, non_norm_features)

def label_snapshots(features, metadata):
    from pipeline_utils import label_data
    return label_data(features, metadata)

def assign_splits(annotated, final_labels, train_size=0.7, val_size=0.15, test_size=0.15):
    all_ids = annotated.loc[final_labels.index, "account_id"].unique()
    y = final_labels

    ids_train, ids_temp, y_train, y_temp = train_test_split(
        all_ids, y, stratify=y, test_size=(1 - train_size), random_state=42
    )
    val_relative = val_size / (val_size + test_size)
    ids_val, ids_test, y_val, y_test = train_test_split(
        ids_temp, y_temp, stratify=y_temp, test_size=(1 - val_relative), random_state=42
    )

    annotated['purpose'] = 'unused'
    annotated.loc[(annotated['account_id'].isin(ids_train)) & annotated['is_live_snapshot'], 'purpose'] = 'train'
    annotated.loc[(annotated['account_id'].isin(ids_val)) & annotated['is_live_snapshot'], 'purpose'] = 'val'
    annotated.loc[(annotated['account_id'].isin(ids_test)) & annotated['is_live_snapshot'], 'purpose'] = 'test'
    annotated.loc[annotated['check_cycle_number'] < 6, 'purpose'] = 'unqualified'
    annotated.loc[final_labels.index, 'purpose'] = 'final_label'
    return annotated

def train_model(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rus = RandomUnderSampler(random_state=42, replacement=True)
    X_resampled, y_resampled = rus.fit_resample(X_train_scaled, y_train)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test_scaled)

    print("[Model Performance]")
    print(classification_report(y_test, y_pred))

    return model, scaler

def predict_all(model, scaler, features, annotated):
    qualified = annotated[annotated['purpose'] != 'unqualified']
    X_all = features.loc[qualified.index]
    X_scaled = scaler.transform(X_all)
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)

    prediction_df = pd.DataFrame({
        'pred_label': preds,
        'P[A]': probs[:, 0].round(3),
        'P[B]': probs[:, 1].round(3),
        'P[C]': probs[:, 2].round(3),
    }, index=X_all.index)
    return prediction_df

def save_outputs(data, filename):
    path = os.path.join(MODEL_OUTPUT_DIR, filename)
    data.to_csv(path, index=True)
    print(f"[Saved] {filename} to {path}")

def main():
    accounts, raw_snapshots, bad_debts = load_data()
    snapshots_data = generate_features(raw_snapshots)
    annotated_data = annotate_data(snapshots_data, bad_debts)
    normalized_data = normalize_data(snapshots_data)

    to_label = normalized_data[annotated_data['is_final_snapshot'] & (
        annotated_data['is_mature_non_bad_debt_account'] | annotated_data['is_bad_debt_account'])].copy()
    meta_data = annotated_data.loc[to_label.index]

    labels = label_snapshots(to_label, meta_data)
    annotated_data = annotated_data.merge(labels.rename('label'), on='account_id', how='left').fillna({'label': 'X'})
    annotated_data['is_labelled'] = annotated_data['label'] != 'X'

    final_labels = annotated_data.loc[annotated_data['is_final_snapshot'] & annotated_data['is_labelled'], 'label']
    annotated_data = assign_splits(annotated_data, final_labels)

    train_idx = annotated_data[annotated_data['purpose'] == 'train'].index
    test_idx = annotated_data[annotated_data['purpose'] == 'test'].index

    X_train, y_train = normalized_data.loc[train_idx], annotated_data.loc[train_idx, 'label']
    X_test, y_test = normalized_data.loc[test_idx], annotated_data.loc[test_idx, 'label']

    model, scaler = train_model(X_train, y_train, X_test, y_test)

    predictions = predict_all(model, scaler, normalized_data, annotated_data)
    save_outputs(predictions, "balance_snapshots_predictions.csv")
    save_outputs(annotated_data, "balance_snapshots_metadata.csv")

    print("[Pipeline completed]")

if __name__ == "__main__":
    main()
