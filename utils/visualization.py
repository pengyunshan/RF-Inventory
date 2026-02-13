#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization module - generate visualizations from predictions.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def visualize_predictions_from_csv(
    pred_csv_path: str,
    output_dir: str,
    model_name: str = "",
    split_type: str = "",
    n_contexts: int = 3,
    cpm_values: Optional[List[float]] = None
) -> None:
    """
    Generate visualizations from predictions.csv file
    
    Args:
        pred_csv_path: Prediction result CSV file path
        output_dir: Output directory
        model_name: Model name
        split_type: Dataset split type (schedule/cpm/joint)
        n_contexts: Number of contexts to generate charts for
        cpm_values: List of CPM values to plot (original values), e.g., [2.45, 3.0]. If None, automatically select first 2
    """
    if not os.path.exists(pred_csv_path):
        logger.warning(f"Prediction file does not exist: {pred_csv_path}, skipping visualization")
        return
    
    try:
        # Read data
        logger.info(f"Reading prediction data from CSV: {pred_csv_path}")
        pred_df = pd.read_csv(pred_csv_path)
        
        # Automatically select some representative contexts for visualization
        # Simple strategy: select first n_contexts different context combinations (including freq_limit)
        context_features = ['age', 'gender', 'fre_region_province', 'freq_limit_days', 'freq_limit_count']
        contexts_to_plot = []
        
        # Group by context and count
        grouped = pred_df.groupby(context_features).size().reset_index(name='count')
        grouped = grouped.sort_values('count', ascending=False)
        
        for idx, row in grouped.head(n_contexts).iterrows():
            context_filter = {feat: row[feat] for feat in context_features}
            contexts_to_plot.append(context_filter)
            logger.info(f"Context {idx+1}: {context_filter}, sample_count={row['count']}")
        
        # Generate charts for each context
        os.makedirs(output_dir, exist_ok=True)
        
        for ctx_idx, context_filter in enumerate(contexts_to_plot):
            _plot_context_comparison(
                pred_df=pred_df,
                context_filter=context_filter,
                output_dir=output_dir,
                ctx_idx=ctx_idx,
                model_name=model_name,
                split_type=split_type,
                cpm_values=cpm_values
            )
        
        logger.info(f"Visualization for model {model_name} complete")
        
    except Exception as e:
        logger.error(f"Error generating visualization for {model_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())


def _plot_context_comparison(
    pred_df: pd.DataFrame,
    context_filter: dict,
    output_dir: str,
    ctx_idx: int,
    model_name: str,
    split_type: str,
    cpm_values: Optional[List[float]] = None
) -> None:
    """
    Plot true vs predicted values comparison chart for a single context
    
    Args:
        pred_df: Prediction data DataFrame
        context_filter: Context filter conditions
        output_dir: Output directory
        ctx_idx: Context index
        model_name: Model name
        split_type: Split type
        cpm_values: List of CPM values to plot (original values), e.g., [2.45, 3.0]. If None, automatically select first 2
    """
    # Filter data
    filtered_df = pred_df.copy()
    for key, value in context_filter.items():
        filtered_df = filtered_df[filtered_df[key] == value]
    
    if len(filtered_df) == 0:
        logger.warning(f"Context {ctx_idx+1} has no data after filtering")
        return
    
    # Get all CPM values
    all_cpms = sorted(filtered_df['cpm_thr'].unique())
    logger.info(f"Context {ctx_idx+1} available CPM values: {all_cpms[:10]}...")  # Show first 10
    
    # Select CPM values to plot
    if cpm_values is not None:
        # User specified CPM values, use them directly
        selected_cpms = cpm_values
        logger.info(f"Context {ctx_idx+1} using specified CPM values: {selected_cpms}")
    else:
        # Automatically select first 2 CPM values
        selected_cpms = all_cpms[:2] if len(all_cpms) >= 2 else all_cpms
        logger.info(f"Context {ctx_idx+1} auto-selected CPM values: {selected_cpms}")
    
    # Filter by CPM
    mask = filtered_df['cpm_thr'].isin(selected_cpms)
    filtered_df = filtered_df[mask]
    if len(filtered_df) == 0:
        logger.warning(f"Context {ctx_idx+1} has no data after CPM filtering")
        return
    
    # Create chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot grouped by CPM
    for cpm_idx, cpm in enumerate(sorted(filtered_df['cpm_thr'].unique())):
        cpm_data = filtered_df[filtered_df['cpm_thr'] == cpm].sort_values('periods')
        
        if len(cpm_data) == 0:
            continue
        cpm_data.to_csv(f'./data/{cpm}_.csv')
        color = colors[cpm_idx % len(colors)]
        
        # Plot true values (solid line)
        # ax1.plot(cpm_data['periods'], cpm_data['y_pv_true'], 
        #         marker='o', linestyle='-', label=f'CPM={cpm:.2f} True',
        #         color=color, linewidth=2, markersize=6, alpha=0.8)
        
        # ax2.plot(cpm_data['periods'], cpm_data['y_uv_true'], 
        #         marker='o', linestyle='-', label=f'CPM={cpm:.2f} 真实值',
        #         color=color, linewidth=2, markersize=6, alpha=0.8)
        
        # Plot predicted values (dashed line)
        ax1.plot(cpm_data['periods'], cpm_data['y_pv_pred'], 
                marker='^', linestyle='--', label=f'CPM={cpm:.2f} Predicted',
                color=color, linewidth=2, markersize=5, alpha=0.6)
        
        ax2.plot(cpm_data['periods'], cpm_data['y_uv_pred'], 
                marker='^', linestyle='--', label=f'CPM={cpm:.2f} 预测值',
                color=color, linewidth=2, markersize=5, alpha=0.6)
    
    # Set up charts
    ax1.set_xlabel('Schedule (days)', fontsize=12)
    ax1.set_ylabel('Impressions (PV)', fontsize=12)
    ax1.set_title(f'Impressions-Schedule Curve - {model_name}\nSolid=True, Dashed=Predicted', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Schedule (days)', fontsize=12)
    ax2.set_ylabel('Unique Visitors (UV)', fontsize=12)
    ax2.set_title(f'Unique Visitors-Schedule Curve - {model_name}\nSolid=True, Dashed=Predicted', 
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # Add context information
    context_str = ', '.join([f'{k}={v}' for k, v in context_filter.items()])
    split_type_map = {
        'schedule': 'Schedule Split',
        'cpm': 'CPM Split',
        'joint': 'Joint Split'
    }
    split_info = split_type_map.get(split_type, split_type)
    fig.suptitle(f'Context: {context_str} | {split_info}', 
                fontsize=12, y=1.02, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    context_str_filename = '_'.join([f'{k}{v}' for k, v in context_filter.items()])
    output_path = os.path.join(output_dir, f'context_{ctx_idx+1}_{context_str_filename}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Chart saved: {output_path}")
    plt.close()
