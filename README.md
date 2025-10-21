# Практика 5: Деплоймент и инференс модели через MLflow

## 📊 О проекте

- **Датасет**: Wine (встроенный датасет sklearn)
- **Задача**: Классификация на 3 класса
- **Модель**: RandomForestClassifier
- **Параметры**: n_estimators=100, max_depth=5
- **Метрики**: accuracy=1.0, f1_macro=1.0
- **Разделение**: 90% train, 10% test

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install mlflow scikit-learn pandas matplotlib flask requests
```

### 2. Обучение модели

```bash
python train_rf.py
```

### 3. Запуск сервера инференса

```bash
python flask_server.py
```

### 4. Тестирование инференса

```bash
python test_inference.py
```

## 📁 Структура проекта

```
├── train_rf.py          # Скрипт обучения модели
├── flask_server.py      # Flask сервер для инференса
├── test_inference.py    # Скрипт тестирования API
├── test_data.json       # Тестовые данные в JSON формате
├── README.md            # Документация
└── mlruns/              # MLflow эксперименты
```

## 🔧 Детальное описание

### Обучение модели (`train_rf.py`)

- Загружает датасет Wine из sklearn
- Разделяет данные на train/test (90%/10%)
- Обучает RandomForestClassifier
- Логирует параметры и метрики в MLflow
- Сохраняет тестовые данные в JSON формате

### Flask сервер (`flask_server.py`)

- Загружает модель из MLflow
- Предоставляет REST API для инференса
- Эндпоинт `/invocations` для предсказаний
- Эндпоинт `/health` для проверки состояния

### Тестирование (`test_inference.py`)

- Загружает тестовые данные из JSON
- Отправляет запросы к API
- Сравнивает предсказания с истинными метками
- Выводит результаты инференса

## 📊 Результаты

### Метрики модели

- **Accuracy**: 1.0 (100%)
- **F1-macro**: 1.0 (100%)

### Формат данных

**Входные данные (JSON):**

```json
{
  "columns": ["alcohol", "malic_acid", "ash", ...],
  "data": [
    [13.88, 1.89, 2.59, ...],
    [12.86, 1.35, 2.32, ...]
  ]
}
```

**Выходные данные:**

```json
[0, 2, 2]
```

## 🌐 API Endpoints

### GET `/health`

Проверка состояния сервера

```bash
curl http://127.0.0.1:5000/health
```

### POST `/invocations`

Получение предсказаний

```bash
curl -X POST http://127.0.0.1:5000/invocations \
     -H 'Content-Type: application/json' \
     -d @test_data.json
```

## 📈 MLflow UI

Для просмотра экспериментов запустите MLflow UI:

```bash
mlflow ui
```

Откройте http://localhost:5000 в браузере для просмотра:

- Параметры модели
- Метрики обучения
- Артефакты (confusion matrix, classification report)
- Модель для скачивания

## 🧪 Пример использования

```python
import requests
import json

# Загружаем тестовые данные
with open('test_data.json', 'r') as f:
    data = json.load(f)

# Отправляем запрос к API
response = requests.post(
    'http://127.0.0.1:5000/invocations',
    headers={'Content-Type': 'application/json'},
    json=data
)

# Получаем предсказания
predictions = response.json()
print(f"Предсказания: {predictions}")
```

## 📝 Требования

- Python 3.8+
- mlflow
- scikit-learn
- pandas
- matplotlib
- flask
- requests

## 🎓 Образовательные цели

Этот проект демонстрирует:

1. **MLOps практики**: версионирование моделей через MLflow
2. **Деплоймент**: создание REST API для инференса
3. **Тестирование**: автоматизированное тестирование API
4. **Документация**: полная документация процесса

## 📸 Скриншоты

![1761047291393](image/README/1761047291393.png)


![1761047352650](image/README/1761047352650.png)



![1761047386793](image/README/1761047386793.png)


![1761047398830](image/README/1761047398830.png)
