#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flask_server.py — Простой Flask сервер для инференса модели
"""

import json
import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# Загружаем модель из MLflow
def load_model():
    """Загружаем модель из последнего run"""
    try:
        # Находим последний run
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(experiment_ids=["0"], max_results=1)
        
        if not runs:
            raise Exception("Нет доступных runs")
        
        latest_run = runs[0]
        run_id = latest_run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        
        print(f"📦 Загружаем модель из run: {run_id}")
        model = mlflow.sklearn.load_model(model_uri)
        print("✅ Модель успешно загружена!")
        return model
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return None

# Загружаем модель при запуске
model = load_model()

@app.route('/health', methods=['GET'])
def health():
    """Проверка здоровья сервера"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/invocations', methods=['POST'])
def predict():
    """Эндпоинт для предсказаний"""
    try:
        if model is None:
            return jsonify({"error": "Модель не загружена"}), 500
        
        # Получаем данные из запроса
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Нет данных в запросе"}), 400
        
        # Проверяем формат данных
        if 'columns' not in data or 'data' not in data:
            return jsonify({"error": "Неверный формат данных. Ожидается {columns: [...], data: [[...], [...]]}"}), 400
        
        # Создаем DataFrame
        df = pd.DataFrame(data['data'], columns=data['columns'])
        
        # Делаем предсказания
        predictions = model.predict(df)
        
        # Возвращаем результат
        return jsonify(predictions.tolist())
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def info():
    """Информация о сервере"""
    return jsonify({
        "message": "MLflow Model Inference Server",
        "endpoints": {
            "/health": "GET - проверка здоровья",
            "/invocations": "POST - предсказания",
            "/": "GET - информация"
        },
        "model_loaded": model is not None
    })

if __name__ == '__main__':
    print("🚀 Запускаем Flask сервер для инференса модели")
    print("📡 Сервер будет доступен на http://127.0.0.1:5000")
    print("🔗 Эндпоинты:")
    print("   GET  /health      - проверка здоровья")
    print("   POST /invocations - предсказания")
    print("   GET  /            - информация")
    
    app.run(host='127.0.0.1', port=5000, debug=True)
