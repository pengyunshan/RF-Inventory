#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main training and evaluation script
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Add project path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.data_loader import RFDataLoader
from utils.evaluation import EvaluationMetrics, BaselineRetrieval, print_evaluation_report, save_results_to_csv
from utils.visualization import visualize_predictions_from_csv
from models.model_factory import get_baseline_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='RF contract inventory forecasting model training and evaluation')
    parser.add_argument('--train_path', type=str, 
                       help='Training set file path (.parquet or .txt)')
    parser.add_argument('--test_path', type=str, 
                       help='Test set file path (.parquet or .txt)')
    parser.add_argument('--data_path', type=str,
                       help='Data file path (.parquet or .txt), for compatibility')
    parser.add_argument('--split_type', type=str, default='cpm',
                       choices=['random', 'cpm', 'schedule', 'joint'],
                       help='Data split type')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set proportion')
    parser.add_argument('--output_dir', type=str, default='experiments',
                       help='Output directory')
    parser.add_argument('--models', nargs='+', default=['ridge', 'gbdt', 'mlp', 'monotonic_mlp', 'logistic_regression', 'nearest_fc'],
                       help='List of models to train')
    parser.add_argument('--gbdt_config', type=str, default='optimized',
                       choices=['default', 'optimized', 'fast', 'deep', 'conservative', 'experimental'],
                       help='GBDT model configuration name (default: optimized)')
    parser.add_argument('--use_cached', dest='use_cached', action='store_true', default=True,
                       help='Whether to use cached preprocessed data (enabled by default)')
    parser.add_argument('--no-use_cached', dest='use_cached', action='store_false',
                       help='Force reprocessing data')
    parser.add_argument('--visualize', dest='visualize', action='store_true', default=True,
                       help='Whether to generate visualizations (enabled by default)')
    parser.add_argument('--no-visualize', dest='visualize', action='store_false',
                       help='Do not generate visualizations')
    parser.add_argument('--n_vis_contexts', type=int, default=3,
                       help='Number of contexts to generate visualizations for')
    parser.add_argument('--save_models', dest='save_models', action='store_true', default=True,
                       help='Whether to save trained models (enabled by default)')
    parser.add_argument('--no-save_models', dest='save_models', action='store_false',
                       help='Do not save models')
    parser.add_argument('--save_predictions', dest='save_predictions', action='store_true', default=True,
                       help='Whether to save prediction results (enabled by default)')
    parser.add_argument('--no-save_predictions', dest='save_predictions', action='store_false',
                       help='Do not save prediction results')
    parser.add_argument('--model_dir', type=str, default=None,
                       help='Model save directory (default to output_dir/models)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Starting training and evaluation")
    logger.info(f"Split type: {args.split_type}")
    logger.info(f"Model list: {args.models}")
    if 'gbdt' in args.models:
        logger.info(f"GBDT config: {args.gbdt_config}")
    
    # 1. Load and preprocess data
    logger.info("=" * 60)
    logger.info("1. Load and preprocess data")
    
    # Use pre-split data
    if args.train_path and args.test_path:
        data_loader = RFDataLoader(args.train_path, test_size=0.0)  # test_size doesn't matter
        logger.info(f"Using pre-split data:")
        logger.info(f"  Training set: {args.train_path}")
        logger.info(f"  Test set: {args.test_path}")
        logger.info(f"  Use cache: {args.use_cached}")
    elif args.data_path:
        # Compatibility support
        data_loader = RFDataLoader(args.data_path, test_size=args.test_size)
        df = data_loader.load_data()
        if len(df) > 100000:
            logger.info(f"Large dataset ({len(df)} rows), using small sample for testing")
            df = df.sample(n=100000, random_state=42)
        df_processed = data_loader.preprocess_features(df)
    else:
        raise ValueError("Must provide --train_path and --test_path or --data_path")
    
    # 2. Load pre-split train-test sets
    logger.info("=" * 60)
    logger.info("2. Load pre-split train-test sets")
    
    if args.train_path and args.test_path:
        # TF/DNN models need one-hot categorical features, determined by whether tf_ model is included
        need_one_hot = any(m.startswith('tf_') or m.startswith('TF_') for m in args.models)
        data_split = data_loader.load_train_test_data(
            args.train_path,
            args.test_path,
            args.use_cached,
            one_hot_categorical=need_one_hot
        )
    else:
        # Select corresponding train-test set files based on split type
        # periods split: df_train1/df_test1
        # cpm split: df_train2/df_test2  
        # joint split (periods+cpm): df_train3/df_test3
        if args.split_type == 'schedule':
            # schedule = periods split
            train_path = 'data/df_train1.pq'
            test_path = 'data/df_test1.pq'
        elif args.split_type == 'cpm':
            # cpm split
            train_path = 'data/df_train2.pq'
            test_path = 'data/df_test2.pq'
        elif args.split_type == 'joint':
            # joint = periods + cpm combined split
            train_path = 'data/df_train3.pq'
            test_path = 'data/df_test3.pq'
        elif args.split_type == 'random':
            # random split (use df_train1 by default)
            train_path = 'data/df_train1.pq'
            test_path = 'data/df_test1.pq'
        else:
            raise ValueError(f"Unsupported split type: {args.split_type}")
        
        logger.info(f"Auto-selected dataset based on split_type={args.split_type}:")
        logger.info(f"  Training set: {train_path}")
        logger.info(f"  Test set: {test_path}")
        need_one_hot = any(m.startswith('tf_') or m.startswith('TF_') for m in args.models)
        data_split = data_loader.load_train_test_data(
            train_path,
            test_path,
            args.use_cached,
            one_hot_categorical=need_one_hot
        )
    
    context_groups = data_loader.get_context_groups(data_split['test_data'])
    
    # 3. Train and evaluate models
    logger.info("=" * 60)
    logger.info("3. Train and evaluate models")
    
    results = {}
    all_predictions = {}
    
    # Get baseline models
    from models.model_factory import get_all_models
    from models.tf_model_factory import get_tf_model_by_name, is_tf_model
    
    # Use specified GBDT config
    baseline_models = get_all_models(gbdt_config=args.gbdt_config)
    model_map = {model.name.lower().replace('_', ''): model for model in baseline_models}
    
    # Add nearest neighbor retrieval baseline
    model_map['nearestfc'] = None
    
    # Add TensorFlow model support
    tf_model_names = [
        'tfmlp', 'tf_mlp', 'tfposnn', 'tf_posnn',
        'tfminmax', 'tf_minmax', 'tfsmoothedminmax', 'tf_smoothed_minmax',
        'tfconstrained', 'tf_constrained', 'tfscalable', 'tf_scalable',
        'tfhint', 'tf_hint', 'tfpwl', 'tf_pwl',
        'tfgcm', 'tf_gcm', 'tfigcm', 'tf_igcm'
    ]
    for tf_name in tf_model_names:
        model_map[tf_name.replace('_', '')] = tf_name  # Mark as TensorFlow model
    
    # Filter out valid models
    valid_models = [name for name in args.models if name in model_map or name.replace('_', '') in model_map]
    
    # Use progress bar to show training process
    with tqdm(total=len(valid_models), desc="Training models", unit="model", ncols=100) as pbar:
        for model_name in valid_models:
            model_name_key = model_name.replace('_', '')
            
            if model_name_key not in model_map:
                logger.warning(f"Unknown model: {model_name}, skipping")
                pbar.update(1)
                continue
                
            logger.info(f"\nTraining model: {model_name}")
            
            try:
                if model_name == 'nearestfc':
                    # Nearest neighbor retrieval baseline
                    y_pred_pv, y_pred_uv = BaselineRetrieval.nearest_fc_retrieval(
                        data_split, {'X_test': data_split['X_test']}, data_split['feature_names']
                    )
                    model = None
                    
                elif isinstance(model_map[model_name_key], str):
                    # TensorFlow model
                    logger.info(f"Loading TensorFlow model: {model_name}")
                    model = get_tf_model_by_name(
                        model_name, 
                        hidden_dim=64, 
                        r_dim=2,  # periods, cpm_thr
                        epochs=100
                    )
                    
                    if model is None:
                        logger.error(f"Unable to create TensorFlow model: {model_name}")
                        pbar.update(1)
                        continue
                    
                    # Train TensorFlow model
                    model.fit(
                        data_split['X_train'], 
                        data_split['y_pv_train'], 
                        data_split['y_uv_train'],
                        data_split['theoretical_max'],
                        epochs=10,
                        batch_size=1024,
                        learning_rate=0.001
                    )
                    
                    y_pred_pv, y_pred_uv = model.predict(data_split['X_test'])
                    
                else:
                    # Machine learning model (sklearn)
                    model = model_map[model_name_key]
                    model.fit(
                        data_split['X_train'], 
                        data_split['y_pv_train'], 
                        data_split['y_uv_train'],
                        data_split['theoretical_max']
                    )
                    
                    y_pred_pv, y_pred_uv = model.predict(data_split['X_test'])
                
                all_predictions[model_name] = {'pv': y_pred_pv, 'uv': y_pred_uv}
                
                # Save complete prediction results (features + labels + predictions)
                if args.save_predictions:
                    # When saving predictions, output "encoded original feature columns" (not one-hot features)
                    # This way, subsequent evaluation/visualization can reuse the same code.
                    base_feature_cols = [
                        'fre_region_province', 'age', 'gender',
                        'freq_limit_days', 'freq_limit_count',
                        'periods', 'cpm_thr'
                    ]
                    missing_base_cols = [c for c in base_feature_cols if c not in data_split['test_data'].columns]
                    if missing_base_cols:
                        raise ValueError(f"test_data missing required feature columns: {missing_base_cols}")

                    pred_df = data_split['test_data'][base_feature_cols].copy()
                    
                    # Add true labels and predictions (row-aligned, as they're based on the same array)
                    pred_df['y_pv_true'] = data_split['y_pv_test']
                    pred_df['y_pv_pred'] = y_pred_pv
                    pred_df['y_uv_true'] = data_split['y_uv_test']
                    pred_df['y_uv_pred'] = y_pred_uv
                    
                    pred_filename = os.path.join(args.output_dir, f'{model_name}_predictions.csv')
                    pred_df.to_csv(pred_filename, index=False)
                    logger.info(f"Prediction results saved to: {pred_filename} (with original features + labels + predictions)")
                # Calculate point-level metrics
                    point_metrics = EvaluationMetrics.calculate_pointwise_metrics(
                        data_split['y_pv_test'], y_pred_pv,
                        data_split['y_uv_test'], y_pred_uv
                    )
                    
                    # Calculate monotonicity metrics (requires grouped prediction by context)
                    # Simplified implementation here, should predict separately for each context
                    # mono_metrics = {'violation_rate': 0.0, 'violation_magnitude': 0.0}
                    mono_metrics = \
                        EvaluationMetrics.calculate_monotonicity_metrics_from_csv(pred_filename)
                    # Calculate theoretical constraint violations
                    constraint_metrics = EvaluationMetrics.calculate_theoretical_constraint_violations(
                        y_pred_pv, theoretical_max=data_split['theoretical_max']
                    )
                    
                    # Merge all metrics
                    metrics = {**point_metrics, **mono_metrics,# **constraint_metrics, 
                            'split_type': args.split_type}
                    
                    results[model_name] = metrics

                
                # Save trained model
                if args.save_models and model_name != 'nearestfc' and model is not None:
                    model_save_dir = args.model_dir if args.model_dir else os.path.join(args.output_dir, 'models')
                    os.makedirs(model_save_dir, exist_ok=True)
                    model_filename = os.path.join(model_save_dir, f'{model_name}_model.pkl')
                    model.save(model_filename)
                
                # Update progress bar
                pbar.set_postfix_str(f"{model_name}: PV_RMSE={metrics.get('pv_rmse', 0) if isinstance(metrics, dict) else 0:.0f}")
                pbar.update(1)
                
            except Exception as e:
                logger.error(f"Error training model {model_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                pbar.update(1)
                continue
    
    # 4. Print and save results
    logger.info("=" * 60)
    logger.info("4. Evaluation results")
    
    print_evaluation_report(results, args.split_type)
    
    # Save results to CSV
    results_filename = os.path.join(args.output_dir, f'results_{args.split_type}.csv')
    print('results_results: ', results)
    save_results_to_csv(results, results_filename)
    
    logger.info(f"All results saved to: {results_filename}")
    
    # 5. Generate visualizations
    if args.visualize and all_predictions:
        logger.info("=" * 60)
        logger.info("5. Generate visualizations")
        
        # Generate visualization for each model - read directly from predictions.csv
        for model_name, predictions in all_predictions.items():
            vis_output_dir = os.path.join(args.output_dir, f'visualizations_{model_name}')
            logger.info(f"Generating visualizations for model {model_name}...")
            
            # Prediction CSV file path
            pred_csv_path = os.path.join(args.output_dir, f'{model_name}_predictions.csv')
            
            # Call visualization function
            visualize_predictions_from_csv(
                pred_csv_path=pred_csv_path,
                output_dir=vis_output_dir,
                model_name=model_name,
                split_type=args.split_type,
                n_contexts=args.n_vis_contexts,
                cpm_values=[2.45,3.0]  # Specify CPM values to plot
            )
        
        logger.info("Visualization generation complete!")
    
    logger.info("Training and evaluation complete!")


if __name__ == "__main__":
    main()