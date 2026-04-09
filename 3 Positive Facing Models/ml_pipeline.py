"""ML training pipeline for IS 455-style model comparison and deployment."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class MLPipeline:
    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        models: Sequence[str],
        tune: bool = True,
        output_path: str = "model.joblib",
        cat_strategy: str = "onehot",
        scale: bool = True,
        test_size: float = 0.2,
        random_state: int = 42,
        cv_folds: int = 5,
        verbose: bool = False,
    ) -> None:
        if cat_strategy != "onehot":
            raise ValueError("Only cat_strategy='onehot' is supported.")
        self.df = df
        self.target = target
        self.models = list(models)
        self.tune = tune
        self.output_path = output_path
        self.cat_strategy = cat_strategy
        self.scale = scale
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.verbose = verbose

        self.best_model_key: Optional[str] = None
        self.final_pipeline: Optional[Pipeline] = None
        self.y_test: Optional[pd.Series] = None
        self.X_test: Optional[pd.DataFrame] = None
        self._numeric_cols: List[str] = []

    def _split_features(self, X: pd.DataFrame) -> tuple[List[str], List[str]]:
        num_cols: List[str] = []
        cat_cols: List[str] = []
        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                num_cols.append(c)
            else:
                cat_cols.append(c)
        return num_cols, cat_cols

    def _build_preprocessor(
        self, num_cols: List[str], cat_cols: List[str]
    ) -> ColumnTransformer:
        num_steps: List[tuple] = [("imputer", SimpleImputer(strategy="median"))]
        if self.scale:
            num_steps.append(("scaler", StandardScaler()))
        num_pipe = Pipeline(num_steps)

        transformers: List[tuple] = []
        if num_cols:
            transformers.append(("num", num_pipe, num_cols))
        if cat_cols:
            cat_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", _one_hot_encoder()),
                ]
            )
            transformers.append(("cat", cat_pipe, cat_cols))

        if not transformers:
            raise ValueError("No feature columns in DataFrame.")

        try:
            return ColumnTransformer(
                transformers, remainder="drop", verbose_feature_names_out=False
            )
        except TypeError:
            return ColumnTransformer(transformers, remainder="drop")

    def _make_classifier(self, key: str):
        rs = self.random_state
        if key == "lr":
            return LogisticRegression(
                max_iter=2000, random_state=rs, class_weight="balanced"
            )
        if key == "dt":
            return DecisionTreeClassifier(random_state=rs, class_weight="balanced")
        if key == "knn":
            return KNeighborsClassifier()
        if key == "rf":
            return RandomForestClassifier(
                random_state=rs, class_weight="balanced", n_jobs=-1
            )
        if key == "gb":
            return GradientBoostingClassifier(random_state=rs)
        if key == "ada":
            return AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1, random_state=rs),
                random_state=rs,
            )
        raise ValueError(f"Unknown model key: {key}")

    def _param_distributions(self, key: str) -> Dict[str, Any]:
        if key == "lr":
            return {"classifier__C": np.logspace(-3, 2, 10)}
        if key == "dt":
            return {
                "classifier__max_depth": [3, 5, 7, 12, None],
                "classifier__min_samples_leaf": [1, 2, 4, 8],
            }
        if key == "knn":
            return {
                "classifier__n_neighbors": [3, 5, 7, 11, 15],
                "classifier__weights": ["uniform", "distance"],
            }
        if key == "rf":
            return {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 10, 20, 30],
                "classifier__min_samples_leaf": [1, 2, 4],
            }
        if key == "gb":
            return {
                "classifier__n_estimators": [80, 120, 200],
                "classifier__max_depth": [2, 3, 5],
                "classifier__learning_rate": [0.05, 0.1, 0.15],
                "classifier__subsample": [0.6, 0.8, 1.0],
                "classifier__min_samples_leaf": [1, 3, 5],
            }
        if key == "ada":
            return {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__learning_rate": [0.5, 1.0, 1.5],
            }
        return {}

    def run(self) -> Dict[str, Dict[str, Any]]:
        df = self.df.copy()
        y = df[self.target]
        X = df.drop(columns=[self.target])

        num_cols, cat_cols = self._split_features(X)
        self._numeric_cols = num_cols

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        self.X_test = X_test
        self.y_test = y_test

        skf = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        results: Dict[str, Dict[str, Any]] = {}

        for key in self.models:
            pre = self._build_preprocessor(num_cols, cat_cols)
            clf = self._make_classifier(key)
            pipeline = Pipeline([("preprocessor", pre), ("classifier", clf)])

            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=skf,
                scoring="roc_auc",
                n_jobs=-1,
            )
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            results[key] = {
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "roc_auc": float(roc_auc_score(y_test, y_proba)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "pipeline": pipeline,
                "y_pred": y_pred,
                "y_proba": y_proba,
            }
            if self.verbose:
                print(
                    f"[{key}] CV AUC: {cv_mean:.4f} +/- {cv_std:.4f} | "
                    f"Test AUC: {results[key]['roc_auc']:.4f}"
                )

        best_key = max(results, key=lambda k: results[k]["cv_mean"])
        self.best_model_key = best_key

        if self.tune:
            dist = self._param_distributions(best_key)
            tune_pre = self._build_preprocessor(num_cols, cat_cols)
            tune_clf = self._make_classifier(best_key)
            tune_pipe = Pipeline(
                [("preprocessor", tune_pre), ("classifier", tune_clf)]
            )
            n_iter = 25 if dist else 1
            search = RandomizedSearchCV(
                tune_pipe,
                param_distributions=dist or {"classifier__n_estimators": [100]},
                n_iter=n_iter,
                cv=skf,
                scoring="roc_auc",
                random_state=self.random_state,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)
            final = search.best_estimator_
            if self.verbose:
                print(f"Tuned {best_key}: {search.best_params_}")
        else:
            final = results[best_key]["pipeline"]

        self.final_pipeline = final

        y_pred_f = final.predict(X_test)
        y_proba_f = final.predict_proba(X_test)[:, 1]

        tuned_cv = cross_val_score(
            final, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1
        )

        results[best_key] = {
            "cv_mean": float(np.mean(tuned_cv)),
            "cv_std": float(np.std(tuned_cv)),
            "roc_auc": float(roc_auc_score(y_test, y_proba_f)),
            "f1": float(f1_score(y_test, y_pred_f, zero_division=0)),
            "pipeline": final,
            "y_pred": y_pred_f,
            "y_proba": y_proba_f,
        }

        joblib.dump(final, self.output_path)
        if self.verbose:
            print(f"Saved model to {self.output_path}")

        return results
