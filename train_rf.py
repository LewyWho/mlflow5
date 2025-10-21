#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_rf.py — Практика 5: Обучение RandomForest и деплой через MLflow
"""

import json
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# Константы
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.10
DEFAULT_N_ESTIMATORS = 100
DEFAULT_MAX_DEPTH = 5

# Порядок признаков для Wine датасета
WINE_COLUMNS = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"
]

def save_test_as_json(X_test: np.ndarray, y_test: np.ndarray, out_path: Path) -> Path:
    """Сохранить тестовую часть в JSON формате для инференса"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Формат для MLflow: {"columns": [...], "data": [[...], [...]]}
    data = {
        "columns": WINE_COLUMNS,
        "data": X_test.tolist()
    }
    
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path

def plot_confusion_matrix(cm: np.ndarray, classes: list[str], out_png: Path) -> Path:
    """Создать и сохранить confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Добавляем числа в ячейки
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_png

def load_data() -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Загрузить Wine датасет"""
    data = load_wine()
    X, y = data.data, data.target
    class_names = list(data.target_names)
    print(f"📊 Загружен датасет Wine: {X.shape[0]} образцов, {X.shape[1]} признаков")
    print(f"🏷️  Классы: {class_names}")
    return X, y, class_names

def train_and_log(
    test_size: float = DEFAULT_TEST_SIZE,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> None:
    """Обучить модель и залогировать в MLflow"""
    
    # Включаем автологирование
    mlflow.sklearn.autolog(disable=False, log_models=False)
    
    # Загружаем данные
    X, y, class_names = load_data()
    
    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"📈 Разделение: train={X_train.shape[0]}, test={X_test.shape[0]}")
    
    # Создаем DataFrame для удобства
    X_train_df = pd.DataFrame(X_train, columns=WINE_COLUMNS)
    X_test_df = pd.DataFrame(X_test, columns=WINE_COLUMNS)
    
    with mlflow.start_run():
        # Логируем параметры
        mlflow.log_param("dataset", "sklearn.load_wine")
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Создаем и обучаем модель
        print("🌲 Обучаем RandomForest...")
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(X_train_df, y_train)
        
        # Предсказания и метрики
        y_pred = clf.predict(X_test_df)
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")
        
        print(f"📊 Метрики: accuracy={acc:.4f}, f1_macro={f1m:.4f}")
        
        # Логируем метрики
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("f1_macro", float(f1m))
        
        # Создаем артефакты
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Classification report
        report_txt = artifacts_dir / "classification_report.txt"
        with report_txt.open("w", encoding="utf-8") as f:
            f.write(classification_report(y_test, y_pred, target_names=class_names))
        mlflow.log_artifact(str(report_txt))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_png = artifacts_dir / "confusion_matrix.png"
        plot_confusion_matrix(cm, class_names, cm_png)
        mlflow.log_artifact(str(cm_png))
        
        # Сохраняем тестовые данные в JSON
        test_json_path = Path("test_data.json")
        save_test_as_json(X_test, y_test, test_json_path)
        mlflow.log_artifact(str(test_json_path))
        
        # Создаем signature и input_example для модели
        signature = infer_signature(X_train_df, clf.predict(X_train_df))
        input_example = X_train_df.head(3).to_dict('records')
        
        # Логируем модель
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )
        
        print(f"✅ Модель сохранена в MLflow")
        print(f"📁 Тестовые данные сохранены в: {test_json_path.resolve()}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train RandomForest on Wine dataset")
    parser.add_argument("--test_size", type=float, default=DEFAULT_TEST_SIZE, 
                       help="Test size fraction (default 0.10)")
    parser.add_argument("--n_estimators", type=int, default=DEFAULT_N_ESTIMATORS, 
                       help="Number of trees (default 100)")
    parser.add_argument("--max_depth", type=int, default=DEFAULT_MAX_DEPTH, 
                       help="Max depth of trees (default 5)")
    parser.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE, 
                       help="Random seed (default 42)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("🚀 Начинаем обучение RandomForest на датасете Wine")
    print(f"⚙️  Параметры: n_estimators={args.n_estimators}, max_depth={args.max_depth}")
    
    train_and_log(
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    
    print("🎉 Обучение завершено!")

if __name__ == "__main__":
    main()
