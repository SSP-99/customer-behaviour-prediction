"""TensorFlow/Keras MLP models for churn classification and CLV regression."""

from typing import Any

import numpy as np
import tensorflow as tf


class KerasChurnClassifier:
    """Scikit-learn-style wrapper around a TensorFlow/Keras MLP binary classifier."""

    def __init__(
        self,
        hidden_units: list[int] | None = None,
        dropout_rate: float = 0.30,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 50,
        patience: int = 7,
        seed: int = 42,
    ) -> None:
        """Initialize Keras churn classifier hyperparameters."""
        self.hidden_units = hidden_units or list((128, 64, 32))
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.model_: tf.keras.Model | None = None

    def _build_model(self, input_dim: int) -> tf.keras.Model:
        tf.keras.utils.set_random_seed(self.seed)
        inputs = tf.keras.Input(shape=(input_dim,))
        x = inputs
        for units in self.hidden_units:
            x = tf.keras.layers.Dense(units, activation="relu")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )
        return model

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KerasChurnClassifier":
        """Fit Keras MLP with balanced class weights and EarlyStopping on training split."""
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        n_pos = max(float(np.sum(y_arr == 1)), 1.0)
        n_neg = max(float(np.sum(y_arr == 0)), 1.0)
        total = n_pos + n_neg
        class_weight = {0: total / (2.0 * n_neg), 1: (total / (2.0 * n_pos)) * 1.15}

        self.model_ = self._build_model(x_arr.shape[-1])
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True,
        )
        self.model_.fit(
            x_arr,
            y_arr,
            validation_split=0.15,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weight,
            callbacks=[early_stop],
            verbose=0,
        )
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict 2-column class probabilities."""
        if self.model_ is None:
            raise RuntimeError("KerasChurnClassifier must be fitted before predict_proba.")
        x_arr = np.asarray(x, dtype=np.float32)
        p1 = self.model_.predict(x_arr, verbose=0).reshape(-1)
        p1 = np.clip(p1, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - p1, p1])

    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters."""
        return {
            "hidden_units": self.hidden_units,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
        }


class KerasCLVRegressor:
    """Scikit-learn-style wrapper around a TensorFlow/Keras MLP regressor for 90d CLV."""

    def __init__(
        self,
        hidden_units: list[int] | None = None,
        dropout_rate: float = 0.25,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 60,
        patience: int = 8,
        seed: int = 42,
    ) -> None:
        """Initialize Keras CLV regressor hyperparameters."""
        self.hidden_units = hidden_units or list((128, 64, 32))
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.model_: tf.keras.Model | None = None

    def _build_model(self, input_dim: int) -> tf.keras.Model:
        tf.keras.utils.set_random_seed(self.seed)
        inputs = tf.keras.Input(shape=(input_dim,))
        x = inputs
        for units in self.hidden_units:
            x = tf.keras.layers.Dense(units, activation="relu")(x)
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(1, activation="linear")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.Huber(delta=1.5),
            metrics=["mae"],
        )
        return model

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KerasCLVRegressor":
        """Fit Keras MLP regressor on log1p-transformed CLV target."""
        x_arr = np.asarray(x, dtype=np.float32)
        y_log = np.log1p(np.maximum(np.asarray(y, dtype=np.float32), 0.0))

        self.model_ = self._build_model(x_arr.shape[-1])
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True,
        )
        self.model_.fit(
            x_arr,
            y_log,
            validation_split=0.15,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=0,
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict non-negative 90-day CLV values on original currency scale."""
        if self.model_ is None:
            raise RuntimeError("KerasCLVRegressor must be fitted before predict.")
        x_arr = np.asarray(x, dtype=np.float32)
        pred_log = self.model_.predict(x_arr, verbose=0).reshape(-1)
        pred_log = np.clip(pred_log, 0.0, 15.0)
        return np.maximum(np.expm1(pred_log), 0.0)

    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters."""
        return {
            "hidden_units": self.hidden_units,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
        }
