"""
Funções auxiliares para preparação de dados e feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split


def prepare_data(df: pd.DataFrame, target_col: str, 
                 test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                   pd.Series, pd.Series]:
    """
    Prepara dados para treinamento: split e separação de features/target.
    
    Args:
        df: DataFrame completo
        target_col: Nome da coluna target
        test_size: Proporção de teste
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # TODO: Implementar preparação
    # X = df.drop(columns=[target_col])
    # y = df[target_col]
    # return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    pass


def calculate_feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
    """
    Calcula importância de features do modelo.
    
    Args:
        model: Modelo treinado (XGBoost ou RandomForest)
        feature_names: Lista de nomes das features
    
    Returns:
        Dict {feature_name: importance_score}
    """
    # TODO: Implementar cálculo de importância
    # if hasattr(model, 'feature_importances_'):
    #     importances = model.feature_importances_
    #     return dict(zip(feature_names, importances))
    # return {}
    pass


def get_shap_values(model, X: pd.DataFrame):
    """
    Calcula SHAP values para explicabilidade.
    
    Args:
        model: Modelo treinado
        X: Features para calcular SHAP
    
    Returns:
        SHAP values (formato depende do explainer usado)
    """
    # TODO: Implementar SHAP
    # import shap
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(X)
    # return shap_values
    pass
