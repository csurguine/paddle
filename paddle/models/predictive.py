"""
Predictive Models Module

Implement various predictive modeling techniques for business problems.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from typing import Dict, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class PredictiveModels:
    """Implement predictive modeling techniques."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize predictive models.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.encoders = {}
    
    def train_churn_prediction_model(
        self, 
        df: pd.DataFrame,
        target_col: str = 'churned'
    ) -> Dict[str, Any]:
        """
        Train a customer churn prediction model.
        
        Args:
            df: DataFrame with customer data
            target_col: Target column name
            
        Returns:
            Dictionary with model performance metrics
        """
        # Prepare features
        feature_cols = [
            'age', 'tenure_months', 'monthly_charges', 'total_charges',
            'num_products', 'support_calls', 'customer_satisfaction'
        ]
        
        # Encode categorical variables
        df_encoded = df.copy()
        for col in ['contract_type', 'payment_method']:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col])
                feature_cols.append(col + '_encoded')
                self.encoders[col] = le
        
        X = df_encoded[feature_cols]
        y = df_encoded[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['churn'] = scaler
        
        # Train Random Forest model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        self.models['churn'] = model
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'feature_importance': dict(zip(feature_cols, model.feature_importances_)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        return metrics
    
    def train_sales_forecast_model(
        self,
        df: pd.DataFrame,
        target_col: str = 'sales',
        forecast_horizon: int = 30
    ) -> Dict[str, Any]:
        """
        Train a sales forecasting model.
        
        Args:
            df: DataFrame with time series data
            target_col: Target column name
            forecast_horizon: Number of periods to forecast
            
        Returns:
            Dictionary with model performance and forecasts
        """
        # Prepare features
        feature_cols = ['day_of_week', 'month', 'is_weekend']
        
        # Add lag features
        for lag in [1, 7, 14, 30]:
            df[f'sales_lag_{lag}'] = df[target_col].shift(lag)
            feature_cols.append(f'sales_lag_{lag}')
        
        # Add rolling statistics
        for window in [7, 30]:
            df[f'sales_rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window).mean()
            df[f'sales_rolling_std_{window}'] = df[target_col].shift(1).rolling(window=window).std()
            feature_cols.extend([f'sales_rolling_mean_{window}', f'sales_rolling_std_{window}'])
        
        # Remove rows with NaN values
        df_clean = df.dropna()
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        # Split data chronologically
        train_size = int(len(df_clean) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train Gradient Boosting model
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        self.models['sales_forecast'] = model
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred),
            'feature_importance': dict(zip(feature_cols, model.feature_importances_)),
            'actual_vs_predicted': list(zip(y_test.values[-20:], y_pred[-20:]))
        }
        
        return metrics
    
    def train_fraud_detection_model(
        self,
        df: pd.DataFrame,
        target_col: str = 'is_fraud'
    ) -> Dict[str, Any]:
        """
        Train a fraud detection model.
        
        Args:
            df: DataFrame with transaction data
            target_col: Target column name
            
        Returns:
            Dictionary with model performance metrics
        """
        # Prepare features
        feature_cols = [
            'amount', 'hour_of_day', 'day_of_week', 'distance_from_home',
            'num_transactions_day', 'avg_transaction_amount', 'account_age_days'
        ]
        
        # Encode categorical variables
        df_encoded = df.copy()
        if 'merchant_category' in df_encoded.columns:
            le = LabelEncoder()
            df_encoded['merchant_category_encoded'] = le.fit_transform(df_encoded['merchant_category'])
            feature_cols.append('merchant_category_encoded')
            self.encoders['merchant_category'] = le
        
        X = df_encoded[feature_cols]
        y = df_encoded[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['fraud'] = scaler
        
        # Train Random Forest with class weights for imbalanced data
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        self.models['fraud'] = model
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'feature_importance': dict(zip(feature_cols, model.feature_importances_)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        return metrics
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of the trained model
            X: Features for prediction
            
        Returns:
            Predictions array
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Train the model first.")
        
        model = self.models[model_name]
        
        # Scale features if scaler exists
        if model_name in self.scalers:
            X_scaled = self.scalers[model_name].transform(X)
            return model.predict(X_scaled)
        
        return model.predict(X)
