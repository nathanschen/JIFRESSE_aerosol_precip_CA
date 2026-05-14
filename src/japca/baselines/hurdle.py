from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


@dataclass
class HurdleXGB:
    rainy_threshold: float = 0.1
    amount_transform: str = "log1p"
    classifier_params: dict | None = None
    regressor_params: dict | None = None
    occurrence_probability_threshold: float = 0.5
    random_state: int = 42

    def __post_init__(self) -> None:
        self.backend = "xgboost"
        self.constant_occurrence_probability: float | None = None
        try:
            from xgboost import XGBClassifier, XGBRegressor

            classifier_defaults = {
                "objective": "binary:logistic",
                "n_estimators": 200,
                "max_depth": 8,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "tree_method": "hist",
                "random_state": self.random_state,
                "eval_metric": "logloss",
            }
            regressor_defaults = {
                "objective": "reg:squarederror",
                "n_estimators": 300,
                "max_depth": 8,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "tree_method": "hist",
                "random_state": self.random_state,
            }
            self.classifier = XGBClassifier(**{**classifier_defaults, **(self.classifier_params or {})})
            self.regressor = XGBRegressor(**{**regressor_defaults, **(self.regressor_params or {})})
        except Exception:
            self.backend = "sklearn_fallback"
            classifier_defaults = {
                "learning_rate": 0.05,
                "max_depth": 8,
                "max_iter": 200,
                "random_state": self.random_state,
            }
            regressor_defaults = {
                "learning_rate": 0.05,
                "max_depth": 8,
                "max_iter": 300,
                "random_state": self.random_state,
            }
            classifier_params = {**classifier_defaults, **(self.classifier_params or {})}
            regressor_params = {**regressor_defaults, **(self.regressor_params or {})}
            classifier_params.pop("n_estimators", None)
            classifier_params.pop("subsample", None)
            classifier_params.pop("colsample_bytree", None)
            classifier_params.pop("tree_method", None)
            classifier_params.pop("objective", None)
            classifier_params.pop("eval_metric", None)
            regressor_params.pop("n_estimators", None)
            regressor_params.pop("subsample", None)
            regressor_params.pop("colsample_bytree", None)
            regressor_params.pop("tree_method", None)
            regressor_params.pop("objective", None)
            self.classifier = HistGradientBoostingClassifier(**classifier_params)
            self.regressor = HistGradientBoostingRegressor(**regressor_params)

    def _transform_amount(self, values: np.ndarray) -> np.ndarray:
        if self.amount_transform == "log1p":
            return np.log1p(values)
        if self.amount_transform == "cube_root":
            return np.cbrt(values)
        return values

    def _inverse_transform_amount(self, values: np.ndarray) -> np.ndarray:
        if self.amount_transform == "log1p":
            return np.expm1(values)
        if self.amount_transform == "cube_root":
            return values ** 3
        return values

    def fit(self, x_train: np.ndarray, y_amount_train: np.ndarray) -> "HurdleXGB":
        y_occ = (y_amount_train >= self.rainy_threshold).astype(np.int8)
        unique = np.unique(y_occ)
        if unique.size == 1:
            self.classifier = None
            self.constant_occurrence_probability = float(unique[0])
        else:
            self.classifier.fit(x_train, y_occ)
            self.constant_occurrence_probability = None
        rainy_mask = y_occ.astype(bool)
        if rainy_mask.any():
            transformed = self._transform_amount(y_amount_train[rainy_mask])
            self.regressor.fit(x_train[rainy_mask], transformed)
        else:
            self.regressor = None
        return self

    def predict_occurrence_probability(self, x_values: np.ndarray) -> np.ndarray:
        if self.classifier is None and self.constant_occurrence_probability is not None:
            return np.full(x_values.shape[0], self.constant_occurrence_probability, dtype=float)
        return self.classifier.predict_proba(x_values)[:, 1]

    def predict_amount(self, x_values: np.ndarray) -> np.ndarray:
        occurrence_probability = self.predict_occurrence_probability(x_values)
        if self.regressor is None:
            return np.zeros_like(occurrence_probability)
        transformed = self.regressor.predict(x_values)
        amount = np.maximum(self._inverse_transform_amount(transformed), 0.0)
        return np.where(occurrence_probability >= self.occurrence_probability_threshold, amount, 0.0)
