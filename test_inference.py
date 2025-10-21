#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_inference.py — Тестирование инференса модели через REST API
"""

import json
import requests
from pathlib import Path
import time

def test_model_inference():
    """Тестируем модель через REST API"""
    
    # URL для инференса
    url = "http://127.0.0.1:5000/invocations"
    headers = {"Content-Type": "application/json"}
    
    # Загружаем тестовые данные
    test_data_path = Path("test_data.json")
    if not test_data_path.exists():
        print(f"❌ Файл {test_data_path} не найден!")
        return
    
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    print(f"Загружено тестовых данных: {len(test_data['data'])} образцов")
    print(f"Колонки: {test_data['columns']}")
    
    # Берем первые 3 образца для тестирования
    sample_data = test_data['data'][:12]
    print(f"Тестируем на {len(sample_data)} образцах")
    
    # Формируем запрос в формате MLflow
    request_data = {
        "columns": test_data['columns'],
        "data": sample_data
    }
    
    print(f"Данные: {len(sample_data)} образцов")
    
    try:
        # Отправляем запрос
        response = requests.post(url, headers=headers, json=request_data, timeout=30)
        
        if response.status_code == 200:
            predictions = response.json()
            print("Успешный ответ от модели!")
            print(f"Предсказания: {predictions}")
            print(f"Формат ответа: {type(predictions)}")
            
            # Показываем детали
            if isinstance(predictions, list):
                print(f"\n  Результаты инференса:")
                for i, pred in enumerate(predictions):
                    print(f"  Образец {i+1}: предсказание = {pred}")
            else:
                print("    Неожиданный формат ответа")
                
        else:
            print(f"Ошибка HTTP {response.status_code}")
            print(f"Ответ: {response.text}")
            
    except requests.exceptions.ConnectionError:
        pass
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")

def test_with_curl():
    pass

if __name__ == "__main__":
    print("Тестирование инференса модели через REST API")
    print("=" * 50)
    
    # Ждем немного, чтобы сервер запустился
    time.sleep(3)
    
    test_model_inference()
    test_with_curl()