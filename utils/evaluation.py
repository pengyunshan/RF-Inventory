#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation metrics and monotonicity check module
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Evaluation metrics calculation class"""

    @staticmethod
    def calculate_monotonicity_metrics_from_csv(
        pred_csv_path: str,
        context_cols: Optional[List[str]] = None,
        periods_col: str = 'periods',
        cpm_col: str = 'cpm_thr',
        pv_pred_col: str = 'y_pv_pred',
        uv_pred_col: str = 'y_uv_pred'
    ) -> Dict:
        """Read data from predictions.csv and calculate monotonicity violation rate.

        Specification (both):
        - context = region * gender * age * frequency control
        - Within each context, construct a (periods, cpm_thr) 2D grid, check monotonicity in both directions:

          A) Fix periods: Check if pv/uv is non-decreasing as cpm increases
             If any adjacent cpm has y(c+1) < y(c), this periods is marked as violation=1, otherwise=0

          B) Fix cpm_thr: Check if pv/uv is non-decreasing as periods increases
             If any adjacent periods has y(t+1) < y(t), this cpm_thr is marked as violation=1, otherwise=0

        - For one context:
            m = number of unique periods values
            n = number of unique cpm values
            k_pv = violations in (periods rows + cpm columns) (same for uv)
            denominator changed to m+n
        - Global violation_rate = sum(k) / sum(m+n)

        Note:
        - If a periods row is missing some cpm points, skip this row by default (not counted in k, also deducted from denominator by 1).
        - If a cpm column is missing some periods points, skip this column by default (not counted in k, also deducted from denominator by 1).
        """
        if context_cols is None:
            context_cols = ['fre_region_province', 'age', 'gender', 'freq_limit_days', 'freq_limit_count']

        df = pd.read_csv(pred_csv_path)

        # 基础列检查
        required_cols = set(context_cols + [periods_col, cpm_col, pv_pred_col, uv_pred_col])
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"predictions.csv missing required columns: {sorted(missing)}")

        
        pv_total_checks = 0
        uv_total_checks = 0
        pv_monotonic_ok = 0
        uv_monotonic_ok = 0

        for context_key, context_df in df.groupby(context_cols):
            # (periods, cpm_thr) -> value grid
            grid_df = context_df.drop(context_cols, axis=1)
            grid_df = grid_df.set_index([periods_col, cpm_col])

            # UV: pivot to matrix (rows=periods, cols=cpm)
            uv_matrix = grid_df[[uv_pred_col]].unstack().sort_index(axis=0).sort_index(axis=1)

            # A) Fix periods: check cpm direction monotonicity
            periods_monotonic_flags_uv = uv_matrix.apply(lambda s: s.is_monotonic_increasing)
            # B) Fix cpm: check periods direction monotonicity
            cpm_monotonic_flags_uv = uv_matrix.T.apply(lambda s: s.is_monotonic_increasing)

            uv_total_checks += periods_monotonic_flags_uv.shape[0]
            uv_total_checks += cpm_monotonic_flags_uv.shape[0]
            uv_monotonic_ok += periods_monotonic_flags_uv.sum()
            uv_monotonic_ok += cpm_monotonic_flags_uv.sum()

            # PV: pivot to matrix (rows=periods, cols=cpm)
            pv_matrix = grid_df[[pv_pred_col]].unstack().sort_index(axis=0).sort_index(axis=1)

            periods_monotonic_flags_pv = pv_matrix.apply(lambda s: s.is_monotonic_increasing)
            cpm_monotonic_flags_pv = pv_matrix.T.apply(lambda s: s.is_monotonic_increasing)

            pv_total_checks += periods_monotonic_flags_pv.shape[0]
            pv_total_checks += cpm_monotonic_flags_pv.shape[0]
            pv_monotonic_ok += periods_monotonic_flags_pv.sum()
            pv_monotonic_ok += cpm_monotonic_flags_pv.sum()

        pv_ok_rate = pv_monotonic_ok / max(pv_total_checks, 1)
        uv_ok_rate = uv_monotonic_ok / max(uv_total_checks, 1)

        return {
            'violation_rate_pv': 1-float(pv_ok_rate),
            'violation_rate_uv': 1-float(uv_ok_rate),
            'pv_total_checks': int(pv_total_checks),
            'uv_total_checks': int(uv_total_checks)
        }
    
    @staticmethod
    def calculate_pointwise_metrics(y_true_pv: np.ndarray, y_pred_pv: np.ndarray,
                                  y_true_uv: np.ndarray, y_pred_uv: np.ndarray) -> Dict:
        """Calculate point-level prediction metrics"""
        metrics = {}
        
        # PV metrics
        metrics['pv_mae'] = mean_absolute_error(y_true_pv, y_pred_pv)
        metrics['pv_rmse'] = np.sqrt(mean_squared_error(y_true_pv, y_pred_pv))
        metrics['pv_mape'] = np.mean(np.abs((y_true_pv - y_pred_pv) / (y_true_pv + 1e-8))) * 100
        
        # UV metrics
        metrics['uv_mae'] = mean_absolute_error(y_true_uv, y_pred_uv)
        metrics['uv_rmse'] = np.sqrt(mean_squared_error(y_true_uv, y_pred_uv))
        metrics['uv_mape'] = np.mean(np.abs((y_true_uv - y_pred_uv) / (y_true_uv + 1e-8))) * 100
        
        return metrics
    
    @staticmethod
    def calculate_theoretical_constraint_violations(y_pred_pv: np.ndarray,
                                                  theoretical_max: pd.DataFrame) -> Dict:
        """Calculate theoretical constraint violations"""
        
        theoretical_max['pv_pred'] = y_pred_pv
        theoretical_max['violation_flag'] = theoretical_max.apply(lambda i : 1 if i['pv_pred'] >= i['pv_theoretical_max'] else 0,axis=1)
        return {
            'constraint_violation_rate': theoretical_max['violation_flag'].sum() / len(theoretical_max)
        }


class BaselineRetrieval:
    """Nearest neighbor retrieval baseline"""
    
    @staticmethod
    def nearest_fc_retrieval(train_data: Dict, test_data: Dict, 
                           feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Frequency control nearest neighbor retrieval baseline
        
        Args:
            train_data: Training data dictionary
            test_data: Test data dictionary
            feature_names: Feature name list
        
        Returns:
            Predicted PV and UV arrays
        """
        X_train = train_data['X_train']
        y_train_pv = train_data['y_pv_train']
        y_train_uv = train_data['y_uv_train']
        X_test = test_data['X_test']
        
        y_pred_pv = []
        y_pred_uv = []
        
        # Find CPM and period feature indices
        cpm_idx = feature_names.index('cpm_thr') if 'cpm_thr' in feature_names else -2
        periods_idx = feature_names.index('periods') if 'periods' in feature_names else -1
        
        for i, test_sample in enumerate(X_test):
            # Find most similar sample in training set (same targeting, same schedule, same W, nearest K)
            # Simplified implementation here, should group by context in practice
            distances = np.linalg.norm(X_train[:, [cpm_idx, periods_idx]] - 
                                     test_sample[[cpm_idx, periods_idx]], axis=1)
            nearest_idx = np.argmin(distances)
            
            y_pred_pv.append(y_train_pv[nearest_idx])
            y_pred_uv.append(y_train_uv[nearest_idx])
        
        return np.array(y_pred_pv), np.array(y_pred_uv)


def print_evaluation_report(results: Dict, split_type: str) -> None:
    """Print evaluation report"""
    print(f"\n{'='*60}")
    print(f"Split Type: {split_type}")
    print(f"{'='*60}")
    
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        print(f"  PV  - MAE: {metrics.get('pv_mae', 0):.4f}, RMSE: {metrics.get('pv_rmse', 0):.4f}")
        print(f"  UV  - MAE: {metrics.get('uv_mae', 0):.4f}, RMSE: {metrics.get('uv_rmse', 0):.4f}")
        print(f"  Monotonicity - Violation pv Rate: {metrics.get('violation_rate_pv', 0):.4f}")
        print(f"  Monotonicity - Violation uv Rate: {metrics.get('violation_rate_uv', 0):.4f}")
        print(f"  Constraints  - Violation Rate: {metrics.get('constraint_violation_rate', 0):.4f}")


def save_results_to_csv(results: Dict, filename: str) -> None:
    """Save results to CSV file"""
    import csv
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['model', 'split_type', 'pv_mae', 'pv_rmse', 'uv_mae', 'uv_rmse', 
                     'violation_rate_pv', 'violation_rate_uv','constraint_violation_rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for model_name, metrics in results.items():
            row = {
                'model': model_name,
                'split_type': metrics.get('split_type', ''),
                'pv_mae': metrics.get('pv_mae', 0),
                'pv_rmse': metrics.get('pv_rmse', 0),
                'uv_mae': metrics.get('uv_mae', 0),
                'uv_rmse': metrics.get('uv_rmse', 0),
                'violation_rate_pv': metrics.get('violation_rate_pv', 0),
                'violation_rate_uv': metrics.get('violation_rate_uv', 0),
                'constraint_violation_rate': metrics.get('constraint_violation_rate', 0)
            }
            writer.writerow(row)
    
    logger.info(f"Results saved to {filename}")

if __name__ == '__main__':
    pred_csv_path='/Users/pengyunshan/workspace/rf_dataset/experiments/gbdt_final/gbdt_predictions.csv'
    pred_csv_path='/Users/pengyunshan/workspace/rf_dataset/experiments/tf_smoothed_minmax_onehot_fixpred/tf_smoothed_minmax_predictions.csv'
    point_metrics = EvaluationMetrics.calculate_monotonicity_metrics_from_csv(
    pred_csv_path=pred_csv_path)

    

