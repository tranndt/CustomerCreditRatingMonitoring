
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore")


def generate_snapshots_with_aggregated_features(
    snapshots_data,
):
    """
    Generate snapshot features from raw delinquency data.
    """
    aggregated_snapshots_data = snapshots_data.copy()
    aggregated_snapshots_data.fillna({'delinquency_type': 'None'}, inplace=True)
    aggregated_snapshots_data = aggregated_snapshots_data.sort_values(['account_id', 'check_date'])
    aggregated_snapshots_data['check_cycle_number'] = aggregated_snapshots_data.groupby('account_id').cumcount() + 1
    aggregated_snapshots_data['is_delinquent'] = aggregated_snapshots_data['delinquency_type'].isin(['Full', 'Major', 'Partial', 'Minor'])

    def compute_snapshot_cumulative_stats(group):
        result = []
        for idx, row in group.iterrows():
            current_cycle = row['check_cycle_number']
            df_upto = group[group['check_cycle_number'] <= current_cycle]
            streaks = (df_upto['is_delinquent'] != df_upto['is_delinquent'].shift()).cumsum()
            streak_lengths = df_upto.groupby(streaks)['is_delinquent'].agg(['first', 'size'])
            true_streaks = streak_lengths[streak_lengths['first'] == True]['size']
            snapshot = {
                'snapshot_id': row.name,
                'account_id': row['account_id'],
                'check_date': row['check_date'],
                'check_cycle_number': current_cycle,
                'current_balance': row['unpaid_balance'],
                'current_delinquency_score': row['delinquency_score'],
            }
            snapshot.update({
                'is_delinquent': row['is_delinquent'],
                'total_delinquencies': df_upto['is_delinquent'].sum(),
                'count_suspension': (df_upto['account_action'] == 'Suspended Account').sum(),
                'count_actively_delinquent': df_upto['account_action'].isin(['Account Marked Delinquent', 'Account Continued Delinquency']).sum(),
                'max_delinquency_score': df_upto['delinquency_score'].max(),
                'total_penalties': df_upto['delinquency_penalty'].sum(),
                'average_penalty_per_incident': df_upto[df_upto['delinquency_penalty'] > 0]['delinquency_penalty'].mean() or 0,
                'count_streak_1': (true_streaks == 1).sum(),
                'count_streak_2_3': true_streaks[(true_streaks >= 2) & (true_streaks <= 3)].count(),
                'count_streak_4plus': (true_streaks >= 4).sum(),
                'count_full_misses': (df_upto['delinquency_type'] == 'Full').sum(),
                'count_major_misses': (df_upto['delinquency_type'] == 'Major').sum(),
                'count_partial_misses': (df_upto['delinquency_type'] == 'Partial').sum(),
                'count_minor_misses': (df_upto['delinquency_type'] == 'Minor').sum(),
                'rolling_avg_delinquency_score_3m': df_upto['delinquency_score'].tail(3).mean() if len(df_upto) >= 3 else None,
                'rolling_avg_delinquency_score_6m': df_upto['delinquency_score'].tail(6).mean() if len(df_upto) >= 6 else None,
                'rolling_avg_penalty_3m': df_upto['delinquency_penalty'].tail(3).mean() if len(df_upto) >= 3 else None,
                'rolling_avg_penalty_6m': df_upto['delinquency_penalty'].tail(6).mean() if len(df_upto) >= 6 else None,
                'max_delinquency_score_12m': df_upto['delinquency_score'].tail(12).max() if len(df_upto) >= 12 else df_upto['delinquency_score'].max() or 0,
            })
            delinquent_amounts = df_upto[df_upto['is_delinquent']]['unpaid_balance']
            snapshot.update({
                'total_delinquent_amount': delinquent_amounts.sum(),
                'average_delinquent_amount_per_incident': delinquent_amounts.mean() or 0,
                'max_delinquent_amount': delinquent_amounts.max() or 0
            })
            result.append(snapshot)
        return pd.DataFrame(result)

    all_snapshots = (
        aggregated_snapshots_data
        .groupby('account_id', group_keys=False)
        .apply(compute_snapshot_cumulative_stats)
        .reset_index(drop=True)
    )

    final_columns = [
        'snapshot_id', 'account_id', 'check_date', 'check_cycle_number', 'is_delinquent',
        'total_delinquencies', 'count_suspension', 'count_actively_delinquent',
        'max_delinquency_score', 'total_penalties', 'average_penalty_per_incident',
        'count_streak_1', 'count_streak_2_3', 'count_streak_4plus',
        'count_full_misses', 'count_major_misses', 'count_partial_misses', 'count_minor_misses',
        'total_delinquent_amount', 'average_delinquent_amount_per_incident',
        'max_delinquent_amount', 'current_balance', 'current_delinquency_score',
        'rolling_avg_delinquency_score_3m', 'rolling_avg_delinquency_score_6m',
        'rolling_avg_penalty_3m', 'rolling_avg_penalty_6m', 'max_delinquency_score_12m',
    ]
    snapshots_data = all_snapshots[final_columns]
    snapshots_data.fillna({
        'average_penalty_per_incident': 0,
        'average_delinquent_amount_per_incident': 0,
        'max_delinquent_amount': 0,
        'rolling_avg_delinquency_score_3m': 0,
        'rolling_avg_delinquency_score_6m': 0,
        'rolling_avg_penalty_3m': 0,
        'rolling_avg_penalty_6m': 0,
        'max_delinquency_score_12m': 0,
    }, inplace=True)
    return snapshots_data.set_index('snapshot_id', drop=False)

def annotate_snapshots(balance_snapshots, bad_debts, min_cycle_for_snapshot=6, snapshot_cycle_spacing=3):
    snapshots_data_annotated = balance_snapshots.copy()
    # snapshots_data_annotated['snapshot_id'] = snapshots_data_annotated['snapshot_id']
    snapshots_data_annotated['check_cycle_number'] = snapshots_data_annotated.groupby('account_id').cumcount() + 1
    snapshots_data_annotated['max_check_cycle'] = snapshots_data_annotated.groupby('account_id')['check_cycle_number'].transform('max')
    snapshots_data_annotated['is_final_snapshot'] = snapshots_data_annotated['check_cycle_number'] == snapshots_data_annotated['max_check_cycle']
    snapshots_data_annotated['is_live_snapshot'] = (
        (snapshots_data_annotated['check_cycle_number'] >= min_cycle_for_snapshot) &
        (snapshots_data_annotated['check_cycle_number'] % snapshot_cycle_spacing == 0) &
        (~snapshots_data_annotated['is_final_snapshot'])
    )
    bad_debts_accounts = bad_debts['account_id'].unique()
    mature_accounts = snapshots_data_annotated[snapshots_data_annotated['max_check_cycle'] >= 24]['account_id'].unique()
    snapshots_data_annotated['is_bad_debt_account'] = snapshots_data_annotated['account_id'].isin(bad_debts_accounts)
    snapshots_data_annotated['is_mature_non_bad_debt_account'] = snapshots_data_annotated['account_id'].isin(mature_accounts) & ~snapshots_data_annotated['is_bad_debt_account']
    # snapshots_data_annotated['tobe_labelled'] = snapshots_data_annotated['is_mature_account'] | snapshots_data_annotated['is_bad_debt']
    snapshots_data_annotated = snapshots_data_annotated[[
        'snapshot_id','account_id', 'check_date',
        'check_cycle_number','max_check_cycle', 'is_delinquent', 'is_final_snapshot', 'is_live_snapshot',
        'is_bad_debt_account', 'is_mature_non_bad_debt_account'
        ]].copy()
    return snapshots_data_annotated.set_index('snapshot_id', drop=False)


def normalize_snapshot_features_by_cycles(
    snapshots_data,
    norm_features,
    non_norm_features
):
    features_df = snapshots_data[norm_features].copy()
    account_age = features_df.pop('check_cycle_number').clip(lower=1)
    features_df_cycl_norm = features_df.div(account_age, axis=0)
    features_df_cycl_norm.rename(columns={col: f"{col}_norm" for col in features_df.columns}, inplace=True)
    result = pd.concat([snapshots_data[non_norm_features], features_df_cycl_norm], axis=1)
    return result



def label_data(final_snapshots_features,final_snapshots_meta):
    final_snapshots_features_copy = final_snapshots_features.copy().drop(columns=['check_cycle_number'])
    final_snapshots_meta_copy = final_snapshots_meta.copy()

    # Transform the features for clustering
    X = final_snapshots_features_copy
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_indexed = pd.DataFrame(X_scaled, index=final_snapshots_features_copy.index, columns=X.columns)
    
    # Identify candidates for A, B, and C labels and determine their centroids
    # C: Bad Debt Accounts 
    C_candidates = final_snapshots_meta_copy[final_snapshots_meta_copy['is_bad_debt_account']].index
    X_C_candidates = X_scaled_indexed.loc[C_candidates]
    centroid_C = X_C_candidates.mean(axis=0) if len(X_C_candidates) > 0 else np.zeros(X.shape[1])

    # A: Mature Clean Non-Bad Debt Accounts and 
    AB_candidates = final_snapshots_meta_copy[final_snapshots_meta_copy['is_mature_non_bad_debt_account']].index
    X_AB_candidates = X_scaled_indexed.loc[AB_candidates]
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            kmeans_A = KMeans(n_clusters=2, random_state=42)
            A_clusters = kmeans_A.fit_predict(X_AB_candidates)
        cluster0_mean = X_AB_candidates[A_clusters == 0].mean(axis=0)
        cluster1_mean = X_AB_candidates[A_clusters == 1].mean(axis=0)
        A_cluster_id = 0 if cluster0_mean[0] < cluster1_mean[0] else 1
        A_candidates = AB_candidates[A_clusters == A_cluster_id]
        centroid_A = kmeans_A.cluster_centers_[A_cluster_id]
    except Exception as e:
        print(f"[Warning] KMeans clustering in hybrid_labelling failed: {e}")
        # Fallback: assign all non-bad-debt to A
        A_candidates = AB_candidates[AB_candidates].index
        centroid_A = X_AB_candidates.mean(axis=0) if len(X_AB_candidates) > 0 else np.zeros(X.shape[1])

    # B: Mature Risky Non-Bad Debt Accounts
    B_candidates = final_snapshots_features_copy.index.difference(A_candidates.union(C_candidates))
    X_B_candidates = X_scaled_indexed.loc[B_candidates]
    centroid_B = X_B_candidates.mean(axis=0) if len(X_B_candidates) > 0 else np.zeros(X.shape[1])

    dist_to_A = cdist(X_B_candidates, [centroid_A]).flatten() if len(X_B_candidates) > 0 else np.array([])
    dist_to_C = cdist(X_B_candidates, [centroid_C]).flatten() if len(X_B_candidates) > 0 else np.array([])
    dist_to_B = cdist(X_B_candidates, [centroid_B]).flatten() if len(X_B_candidates) > 0 else np.array([])

    # Reclassify B candidates based on distances to A, B, and C centroids
    max_delinquency_scores = final_snapshots_features_copy.loc[B_candidates]['max_delinquency_score'].values if len(X_B_candidates) > 0 else np.array([])
    B_candidates_labels = {}
    for idx, (a, b, c), max_score in zip(B_candidates, zip(dist_to_A, dist_to_B, dist_to_C), max_delinquency_scores):
        if b < a and b < c:
            B_candidates_labels[idx] = 'B'
        elif a < c:
            if max_score <= 6:
                B_candidates_labels[idx] = 'A'
            else:
                B_candidates_labels[idx] = 'B'
        else:
            if max_score >= 8:
                B_candidates_labels[idx] = 'C'
            else:
                B_candidates_labels[idx] = 'B'

    # Create a Series to hold the hybrid labels
    labels = pd.concat([
        pd.Series(index=A_candidates, data='A', dtype='object'),
        pd.Series(index=C_candidates, data='C', dtype='object'),
        pd.Series(B_candidates_labels,dtype='object')
    ]).rename('label')
    return labels
