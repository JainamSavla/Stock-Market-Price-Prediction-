"""
Machine learning model module for the stock prediction dashboard.
Handles training, saving, and loading XGBoost classification models.
"""

from typing import Dict, Optional, Tuple, Any

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils import (
    save_model,
    load_model,
    model_exists,
    save_training_data,
    get_feature_columns,
    validate_dataframe_for_training
)
from features import engineer_all_features, prepare_features_for_training


# Model configuration
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

# Test split ratio (chronological split)
TEST_SIZE = 0.2


def sanitize_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize feature data by replacing inf/-inf with NaN, then filling NaN with 0.

    This is a safety check before passing data to XGBoost to prevent
    "Input data contains inf or a value too large" errors.

    Args:
        X: Feature DataFrame

    Returns:
        Sanitized DataFrame safe for XGBoost
    """
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    return X


def create_xgboost_model(scale_pos_weight: float = 1.0) -> XGBClassifier:
    """
    Create a new XGBoost classifier with default parameters.

    Args:
        scale_pos_weight: Weight for positive class to handle imbalance.

    Returns:
        Configured XGBClassifier instance
    """
    params = {**MODEL_PARAMS, 'scale_pos_weight': scale_pos_weight}
    return XGBClassifier(**params)


def _compute_scale_pos_weight(y: pd.Series, imbalance_threshold: float = 0.6) -> float:
    """
    Compute scale_pos_weight if class distribution exceeds threshold.

    If the majority class is more than `imbalance_threshold` of the total,
    returns n_negative / n_positive so XGBoost can correct for the imbalance.
    Otherwise returns 1.0 (no correction).

    Args:
        y: Binary target series (0 or 1)
        imbalance_threshold: Fraction above which imbalance correction kicks in

    Returns:
        scale_pos_weight value
    """
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    total = n_pos + n_neg

    if total == 0:
        return 1.0

    majority_frac = max(n_pos, n_neg) / total

    if majority_frac > imbalance_threshold:
        # Apply correction
        return n_neg / n_pos if n_pos > 0 else 1.0

    return 1.0


def train_single_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float = 1.0
) -> XGBClassifier:
    """
    Train a single XGBoost classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        scale_pos_weight: Weight for positive class to handle imbalance.

    Returns:
        Trained XGBClassifier
    """
    model = create_xgboost_model(scale_pos_weight=scale_pos_weight)
    # Sanitize features before training to prevent inf errors
    X_train_safe = sanitize_features(X_train)
    model.fit(X_train_safe, y_train)
    return model


def train_all_models(
    df: pd.DataFrame,
    ticker: str
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Train all three prediction models (next day, 1 week, 1 month).

    Uses chronological train/test split (80/20) without shuffling.
    Automatically applies scale_pos_weight if class distribution is
    more imbalanced than 60/40.

    Args:
        df: Raw OHLCV DataFrame
        ticker: Stock ticker symbol (for saving)

    Returns:
        Tuple of (success, message, metrics_dict)
    """
    try:
        # Engineer all features
        df_features = engineer_all_features(df)

        # Validate data
        is_valid, error_msg = validate_dataframe_for_training(df_features)
        if not is_valid:
            return False, error_msg, {}

        # Prepare features and targets
        X, y = prepare_features_for_training(df_features)

        if len(X) < 100:
            return False, f"Insufficient data after preprocessing: {len(X)} rows", {}

        # Chronological train/test split (no shuffle for time series)
        split_idx = int(len(X) * (1 - TEST_SIZE))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        metrics = {}
        models = {}

        # Helper: train + evaluate for one target
        target_map = {
            'model_next_day': ('target_next_day', 'next_day_accuracy'),
            'model_1_week':   ('target_1_week',   '1_week_accuracy'),
            'model_1_month':  ('target_1_month',  '1_month_accuracy'),
        }

        for model_key, (target_col, acc_key) in target_map.items():
            # Check class distribution and compute weight
            spw = _compute_scale_pos_weight(y_train[target_col])
            n_up = int((y_train[target_col] == 1).sum())
            n_down = int((y_train[target_col] == 0).sum())
            total = n_up + n_down
            metrics[f'{acc_key}_class_up_pct'] = round(n_up / total * 100, 1) if total > 0 else 50.0
            metrics[f'{acc_key}_scale_pos_weight'] = round(spw, 3)

            model = train_single_model(X_train, y_train[target_col], scale_pos_weight=spw)
            models[model_key] = model

            # Sanitize test features before prediction
            X_test_safe = sanitize_features(X_test)
            y_pred = model.predict(X_test_safe)
            acc = float(accuracy_score(y_test[target_col], y_pred))
            metrics[acc_key] = acc

            # Warn if accuracy is outside the realistic range for financial ML
            if acc < 0.50 or acc > 0.65:
                print(
                    f"WARNING: {model_key} accuracy = {acc:.4f} is outside "
                    f"the expected 0.50–0.65 range for financial ML models."
                )

        # Save all models
        for model_name, model in models.items():
            success = save_model(model, ticker, model_name)
            if not success:
                return False, f"Failed to save {model_name}", {}

        # Save the feature-engineered training data
        save_training_data(df_features, ticker)

        metrics['training_samples'] = len(X_train)
        metrics['test_samples'] = len(X_test)

        return True, "All models trained successfully", metrics

    except Exception as e:
        return False, f"Training error: {str(e)}", {}


def load_all_models(ticker: str) -> Tuple[bool, Dict[str, XGBClassifier]]:
    """
    Load all trained models for a ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Tuple of (success, dict of loaded models)
    """
    models = {}
    model_names = ['model_next_day', 'model_1_week', 'model_1_month']

    for model_name in model_names:
        model = load_model(ticker, model_name)
        if model is None:
            return False, {}
        models[model_name] = model

    return True, models


def get_model_for_horizon(
    ticker: str,
    horizon: str
) -> Optional[XGBClassifier]:
    """
    Get the appropriate model for a prediction horizon.

    Args:
        ticker: Stock ticker symbol
        horizon: 'next_day', '1_week', or '1_month'

    Returns:
        Trained XGBClassifier or None if not found
    """
    horizon_to_model = {
        'next_day': 'model_next_day',
        '1_week': 'model_1_week',
        '1_month': 'model_1_month'
    }

    model_name = horizon_to_model.get(horizon)
    if model_name is None:
        return None

    return load_model(ticker, model_name)


def predict_with_model(
    model: XGBClassifier,
    features: pd.DataFrame
) -> Tuple[int, float]:
    """
    Make a prediction using a trained model.

    Args:
        model: Trained XGBClassifier
        features: Feature DataFrame (single row)

    Returns:
        Tuple of (prediction, probability_up)
    """
    # Ensure features are in the right format
    feature_cols = get_feature_columns()
    X = features[feature_cols]

    # Sanitize features before prediction to prevent inf errors
    X = sanitize_features(X)

    # Get prediction
    prediction = model.predict(X)[0]

    # Get probability of class 1 (positive return)
    probabilities = model.predict_proba(X)[0]
    prob_up = probabilities[1] if len(probabilities) > 1 else probabilities[0]

    return int(prediction), float(prob_up)


def get_feature_importance(model: XGBClassifier) -> Dict[str, float]:
    """
    Get feature importance scores from a trained model.

    Args:
        model: Trained XGBClassifier

    Returns:
        Dictionary mapping feature names to importance scores
    """
    feature_cols = get_feature_columns()
    importances = model.feature_importances_

    importance_dict = {}
    for col, importance in zip(feature_cols, importances):
        importance_dict[col] = float(importance)

    # Sort by importance descending
    importance_dict = dict(sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    ))

    return importance_dict


def retrain_models_with_new_data(
    ticker: str,
    new_df: pd.DataFrame
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Retrain all models with updated data.

    This is called when new data is available for a previously trained ticker.

    Args:
        ticker: Stock ticker symbol
        new_df: DataFrame with combined old + new data

    Returns:
        Tuple of (success, message, metrics_dict)
    """
    # Simply call train_all_models with the combined data
    # This will overwrite existing models with updated versions
    return train_all_models(new_df, ticker)


def check_models_trained(ticker: str) -> bool:
    """
    Check if all required models are trained for a ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        True if all models exist, False otherwise
    """
    return model_exists(ticker)


def get_model_metrics_summary(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Get a summary of model performance metrics.

    This loads the models and computes basic statistics.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with model metrics or None if models don't exist
    """
    success, models = load_all_models(ticker)
    if not success:
        return None

    summary = {}
    for model_name, model in models.items():
        importance = get_feature_importance(model)
        top_features = list(importance.keys())[:5]
        summary[model_name] = {
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'top_features': top_features
        }

    return summary
