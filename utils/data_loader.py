#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loading and preprocessing module
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from typing import Tuple, Dict, List, Optional
import logging
import os
import math

logger = logging.getLogger(__name__)


class RFDataLoader:
    """RF contract data loader"""
    
    def __init__(self, data_path: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize data loader
        
        Args:
            data_path: Data file path
            test_size: Test set proportion
            random_state: Random seed
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load data"""
        logger.info(f"Loading data: {self.data_path}")
        if self.data_path.endswith(('.parquet', '.pq')):
            df = pd.read_parquet(self.data_path)
        else:
            cols = [
                'p_date', 'fre_region_country', 'fre_region_province', 'age', 
                'gender', 'freq_limit', 'periods', 'cpm_thr', 'budget', 
                'pv', 'uv', 'tag1'
            ]
            df = pd.read_csv(self.data_path, header=None, names=cols, sep='\t')
            df = df[df['fre_region_province'] != '\\N']
            df['fre_region_province'] = df['fre_region_province'].astype(int)
        
        logger.info(f"Data loaded, shape: {df.shape}")
        return df
    
    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature preprocessing
        
        Feature description:
        - Category features (classification): fre_region_province, age, gender
        - Numerical features: freq_limit_days, freq_limit_count, periods, cpm_thr
        - Non-feature fields: tag, p_date (used only for marking, not as training features)
        """
        logger.info("Starting feature preprocessing...")
        
        # 1. Process freq_limit - split into two numerical features
        # freq_limit format like "7_3", means no more than 3 times in 7 days
        df['freq_limit_days'] = df['freq_limit'].str.split('_').str[0].astype(int)
        df['freq_limit_count'] = df['freq_limit'].str.split('_').str[1].astype(int)
        
        # 2. Process categorical features - LabelEncoder encoding
        categorical_features = ['fre_region_province', 'age', 'gender']
        
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                df[feature] = self.label_encoders[feature].fit_transform(df[feature])
            else:
                df[feature] = self.label_encoders[feature].transform(df[feature])
        
        logger.info(f"Feature preprocessing complete, features: {list(df.columns)}")
        return df
    
    def load_train_test_data(
        self,
        train_path: str,
        test_path: str,
        use_cached: bool = True,
        one_hot_categorical: bool = False
    ) -> Dict:
        """
        Load pre-split train-test data
        
        Args:
            train_path: Training set file path
            test_path: Test set file path
            use_cached: Whether to use cached preprocessed data
            one_hot_categorical: Whether to one-hot encode categorical features (recommended only for DNN/TF models)
        """
        logger.info(f"Loading training set: {train_path}")
        logger.info(f"Loading test set: {test_path}")
        
        # Generate cache file paths
        cache_dir = 'data/cache'
        os.makedirs(cache_dir, exist_ok=True)
        
        train_cache_path = os.path.join(cache_dir, f"train_processed_{os.path.basename(train_path)}.parquet")
        test_cache_path = os.path.join(cache_dir, f"test_processed_{os.path.basename(test_path)}.parquet")
        
        # Check if cache should be used
        if use_cached and os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
            logger.info("Using cached preprocessed data")
            train_processed = pd.read_parquet(train_cache_path)
            test_processed = pd.read_parquet(test_cache_path)
        else:
            logger.info("Processing raw data...")
            # Load training set
            train_df = self.load_data_from_path(train_path)
            train_processed = self.preprocess_features(train_df)
            
            # Load test set
            test_df = self.load_data_from_path(test_path)
            test_processed = self.preprocess_features(test_df)
            
            # Save cache
            logger.info("Saving preprocessed data to cache...")
            train_processed.to_parquet(train_cache_path, index=False)
            test_processed.to_parquet(test_cache_path, index=False)
            logger.info(f"Cache files saved: {train_cache_path}, {test_cache_path}")
        
        # Get feature columns (exclude target and label columns)
        feature_cols = [col for col in train_processed.columns 
                       if col not in ['pv', 'uv', 'tag1', 'p_date', 'fre_region_country', 'budget', 'freq_limit']]
        
        # Prepare training data
        X_train = train_processed[feature_cols]
        y_pv_train = train_processed['pv']
        y_uv_train = train_processed['uv']
        
        # Prepare test data
        X_test = test_processed[feature_cols]
        y_pv_test = test_processed['pv']
        y_uv_test = test_processed['uv']
        
        # Process categorical and numerical features separately
        categorical_features = ['fre_region_province', 'age', 'gender']
        numerical_features = ['freq_limit_days', 'freq_limit_count', 'periods', 'cpm_thr']
        
        # Standardize numerical features only
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # Standardize numerical features
        X_train_scaled[numerical_features] = self.scaler.fit_transform(X_train[numerical_features])
        X_test_scaled[numerical_features] = self.scaler.transform(X_test[numerical_features])
        
        # Keep categorical features as integer encoding (no standardization)
        logger.info("Feature standardization complete:")
        logger.info(f"  Categorical features kept as integer encoding: {categorical_features}")
        logger.info(f"  Numerical features standardized: {numerical_features}")
        
        # Optional: one-hot encode categorical features (mainly for DNN/TF models)
        if one_hot_categorical:
            # Only one-hot categorical features, keep numerical features as-is (already standardized)
            cat_cols = [c for c in categorical_features if c in feature_cols]
            num_cols = [c for c in numerical_features if c in feature_cols]

            # Use training set to fit OneHotEncoder, ensure train/test alignment
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            X_train_cat = ohe.fit_transform(X_train_scaled[cat_cols])
            X_test_cat = ohe.transform(X_test_scaled[cat_cols])

            # Numerical features
            X_train_num = X_train_scaled[num_cols].to_numpy(dtype=float)
            X_test_num = X_test_scaled[num_cols].to_numpy(dtype=float)

            # Concatenate features: one-hot(cats) + nums
            X_train_scaled = np.concatenate([X_train_cat, X_train_num], axis=1)
            X_test_scaled = np.concatenate([X_test_cat, X_test_num], axis=1)

            # Generate new feature_names
            ohe_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
            feature_cols = ohe_feature_names + num_cols

            logger.info("One-Hot encoding applied to categorical features:")
            logger.info(f"  Categorical features: {cat_cols} -> one-hot dim={X_train_cat.shape[1]}")
            logger.info(f"  Numerical feature dim={X_train_num.shape[1]}")
            logger.info(f"  Total feature dim={X_train_scaled.shape[1]}")
        else:
            # Convert to numpy array (maintain feature order)
            X_train_scaled = X_train_scaled[feature_cols].values
            X_test_scaled = X_test_scaled[feature_cols].values
        
        # Calculate theoretical maximum impressions for constraints
        theoretical_max = self._calculate_theoretical_max(test_processed)
        
        result = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_pv_train': y_pv_train.values,
            'y_pv_test': y_pv_test.values,
            'y_uv_train': y_uv_train.values,
            'y_uv_test': y_uv_test.values,
            'feature_names': feature_cols,
            'theoretical_max': theoretical_max,
            'train_data': train_processed,
            'test_data': test_processed
        }
        
        logger.info(f"Data loading complete, training set: {len(X_train_scaled)}, test set: {len(X_test_scaled)}")
        return result
    
    def load_data_from_path(self, path: str) -> pd.DataFrame:
        """Load data from specified path"""
        if path.endswith(('.parquet', '.pq')):
            df = pd.read_parquet(path)
        else:
            cols = [
                'p_date', 'fre_region_country', 'fre_region_province', 'age', 
                'gender', 'freq_limit', 'periods', 'cpm_thr', 'budget', 
                'pv', 'uv', 'tag1'
            ]
            df = pd.read_csv(path, header=None, names=cols, sep='\t')
            df = df[df['fre_region_province'] != '\\N']
            df['fre_region_province'] = df['fre_region_province'].astype(int)
        
        logger.info(f"数据加载完成，形状: {df.shape}")
        return df
    
    def _calculate_theoretical_max(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate theoretical maximum impressions (for constraint checking)"""
        # Group by context and calculate maximum impressions
        # Use encoded feature names
        # group_cols = ['fre_region_province', 'age', 'gender', 
        #              'freq_limit_days', 'freq_limit_count']
        # print('xxxx: ', df.columns)
        
        df_copy = df.copy()
        df_copy['pv_theoretical_max'] = df_copy.apply(
            lambda i : i['uv'] * i['freq_limit_count']* math.ceil(i['periods']/i['freq_limit_days']), axis=1 )
        return df_copy
    
    def get_context_groups(self, df: pd.DataFrame) -> Dict:
        """Get context groups for monotonicity checking"""
        # Group by targeting conditions and frequency control
        # Note: Feature names are already encoded (without _encoded suffix)
        group_cols = ['fre_region_province', 'age', 'gender', 
                     'freq_limit_days', 'freq_limit_count']
        df_grouped = df.groupby(group_cols)
        
        context_groups = {}
        for name, group in df_grouped:
            # Sort by CPM and schedule
            group_sorted = group.sort_values(['cpm_thr', 'periods'])
            context_groups[name] = group_sorted
        
        logger.info(f"Context grouping complete, total {len(context_groups)} contexts")
        return context_groups
    
    def inverse_transform_features(self, X_scaled: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
        """
        Inverse transform standardized features to original values
        
        Args:
            X_scaled: Standardized feature array (n_samples, n_features)
            feature_names: Feature name list
            
        Returns:
            DataFrame with original feature values
        """
        # Create DataFrame
        df = pd.DataFrame(X_scaled.copy(), columns=feature_names)
        
        # Categorical and numerical features
        categorical_features = ['fre_region_province', 'age', 'gender']
        numerical_features = ['freq_limit_days', 'freq_limit_count', 'periods', 'cpm_thr']
        
        # Inverse numerical features (inverse StandardScaler)
        if hasattr(self.scaler, 'mean_'):  # Check if scaler is fitted
            # Only inverse transform numerical features
            df[numerical_features] = self.scaler.inverse_transform(df[numerical_features])
        
        # Inverse categorical features (inverse LabelEncoder)
        for feature in categorical_features:
            if feature in self.label_encoders and feature in df.columns:
                # LabelEncoder encoded values are integers, need to convert to int first before inverse
                try:
                    encoded_values = df[feature].round().astype(int)
                    df[feature] = self.label_encoders[feature].inverse_transform(encoded_values)
                except Exception as e:
                    logger.warning(f"Cannot inverse feature {feature}: {e}")
        
        return df

if __name__ == "__main__":
    data_path = ''
    train_path = '/Users/pengyunshan/workspace/rf_dataset/data/df_train3.pq'
    test_path = '/Users/pengyunshan/workspace/rf_dataset/data/df_test3.pq'
    # use_cached = ''
    # need_one_hot = True
    # data_loader = RFDataLoader(data_path)
    # data_split = data_loader.load_train_test_data(
    #     train_path,
    #     test_path,
    #     use_cached,
    #     one_hot_categorical=need_one_hot
    # )
    
    df = pd.read_parquet(train_path)
    
    
    