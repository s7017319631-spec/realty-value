"""
app.py - Веб-приложение для оценки квартир
"""

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)

# Загрузка модели
MODEL_PATH = "apartment_model_auezov.pkl"
PREDICTOR_PATH = "apartment_predictor_auezov.pkl"


import joblib
import pandas as pd
import numpy as np


class SimplePredictor:
    def __init__(self, model_path):
        self.model = None
        self.features = None
        self.label_encoders = {}
        self.scaler = None

        try:
            if os.path.exists(model_path):
                print(f"🔄 Загружаем модель из {model_path}")

                # Загружаем все данные модели
                model_data = joblib.load(model_path)
                print(f"Ключи в файле модели: {list(model_data.keys())}")

                # Извлекаем компоненты
                self.model = model_data.get("model")
                self.features = model_data.get("features", [])
                self.label_encoders = model_data.get("label_encoders", {})
                self.scaler = model_data.get("scaler")

                if self.model:
                    print(f"✅ Модель загружена: {type(self.model).__name__}")
                    print(f"Количество признаков: {len(self.features)}")
                    print(f"Кодировщики: {list(self.label_encoders.keys())}")
                    self.mode = "trained"
                else:
                    print("❌ Модель не найдена в файле")
                    self.mode = "demo"

            else:
                print(f"❌ Файл модели не найден: {model_path}")
                self.mode = "demo"

        except Exception as e:
            print(f"❌ Ошибка при загрузке модели: {e}")
            import traceback

            traceback.print_exc()
            self.mode = "demo"

    def predict(self, apartment_data):
        """Предсказание с обученной моделью"""
        if self.mode != "trained":
            # Используем демо-режим если модель не загружена
            return self._demo_predict(apartment_data)

        try:
            print(f"\n🔍 Прогнозирование для данных: {apartment_data}")

            # Создаем DataFrame
            df = pd.DataFrame([apartment_data])

            # 1. Добавляем вычисляемые признаки
            current_year = datetime.now().year
            if "year_build" in df.columns:
                df["building_age"] = current_year - df["year_build"]

            if "floor" in df.columns and "total_floors" in df.columns:
                df["floor_ratio"] = df["floor"] / df["total_floors"].replace(0, 1)

            # 2. Кодируем категориальные признаки
            for col, encoder in self.label_encoders.items():
                if col in df.columns:
                    try:
                        # Преобразуем значение в строку
                        value = (
                            str(df[col].iloc[0])
                            if not pd.isna(df[col].iloc[0])
                            else "unknown"
                        )

                        # Если значение есть в encoder, преобразуем его
                        if value in encoder.classes_:
                            df[col] = encoder.transform([value])[0]
                        else:
                            # Используем наиболее частый класс или 0
                            df[col] = 0
                            print(
                                f"⚠️ Значение '{value}' не найдено в encoder для {col}"
                            )
                    except Exception as e:
                        print(f"⚠️ Ошибка кодирования {col}: {e}")
                        df[col] = 0

            # 3. Добавляем отсутствующие признаки (заполняем 0 или средними)
            for feature in self.features:
                if feature not in df.columns:
                    df[feature] = 0
                    print(f"⚠️ Добавлен отсутствующий признак: {feature}")

            # 4. Убедимся что порядок признаков правильный
            X = df[self.features]

            # 5. Применяем масштабирование если есть scaler
            if self.scaler:
                X = self.scaler.transform(X)
                print(f"✅ Применено масштабирование")

            print(f"✅ Данные подготовлены. Форма: {X.shape}")
            print(f"Признаки: {list(X.columns) if hasattr(X, 'columns') else 'массив'}")

            # 6. Предсказание
            prediction = self.model.predict(X)
            price_per_sqm = float(prediction[0])

            print(f"✅ Предсказание модели: {price_per_sqm} ₸/м²")

            # 7. Проверка на реалистичность
            if price_per_sqm < 50000 or price_per_sqm > 2000000:
                print(f"⚠️ Подозрительная цена за м²: {price_per_sqm}")
                print("Используем демо-расчет")
                return self._demo_predict(apartment_data)

            # 8. Расчет общей стоимости
            area = apartment_data.get("area_total", 50)
            total_price = price_per_sqm * area

            result = {
                "price_per_sqm": round(price_per_sqm, 2),
                "total_price": round(total_price, 2),
                "confidence_interval": {
                    "lower": round(total_price * 0.92, 2),
                    "upper": round(total_price * 1.08, 2),
                    "margin_percent": 8,
                },
                "success": True,
                "model_mode": "trained",
                "model_type": type(self.model).__name__,
                "area_used": area,
            }

            print(f"📊 Итоговая стоимость: {total_price:,.0f} ₸")
            return result

        except Exception as e:
            print(f"❌ Ошибка при предсказании: {e}")
            import traceback

            traceback.print_exc()
            return self._demo_predict(apartment_data)

    def _demo_predict(self, apartment_data):
        """Демо-режим если модель не работает"""
        print("🔄 Используем демо-режим")

        base_price = 350000
        price_per_sqm = base_price

        # Простые корректировки
        if "building_type" in apartment_data:
            btype = str(apartment_data["building_type"]).lower()
            if "монолит" in btype:
                price_per_sqm *= 1.15
            elif "кирпич" in btype:
                price_per_sqm *= 1.05

        area = apartment_data.get("area_total", 50)
        total_price = price_per_sqm * area

        return {
            "price_per_sqm": round(price_per_sqm, 2),
            "total_price": round(total_price, 2),
            "confidence_interval": {
                "lower": round(total_price * 0.9, 2),
                "upper": round(total_price * 1.1, 2),
                "margin_percent": 10,
            },
            "success": True,
            "model_mode": "demo",
            "note": "Используется демо-модель",
        }


# Инициализация предсказателя
predictor = SimplePredictor(MODEL_PATH)


@app.route("/")
def index():
    """Главная страница"""
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API для оценки квартиры"""
    try:
        data = request.get_json()

        print(f"\n📥 Получен запрос с данными:")
        for key, value in data.items():
            print(f"  {key}: {value} (тип: {type(value)})")

        # Проверка типов данных
        for field in ["area_total", "floor", "total_floors", "year_build"]:
            if field in data:
                if isinstance(data[field], str):
                    try:
                        data[field] = float(data[field])
                    except:
                        return (
                            jsonify(
                                {
                                    "success": False,
                                    "error": f"Поле {field} должно быть числом",
                                }
                            ),
                            400,
                        )

        # Оценка
        result = predictor.predict(data)

        print(f"📤 Результат: {result}")

        # Добавляем timestamp
        result["timestamp"] = datetime.now().isoformat()

        return jsonify(result)

    except Exception as e:
        print(f"❌ Ошибка в API: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/microdistricts", methods=["GET"])
def api_microdistricts():
    """API для получения списка микрорайонов"""
    microdistricts = [
        "Мамыр-1",
        "Мамыр-2",
        "Мамыр-3",
        "Мамыр-4",
        "Мамыр-5",
        "Мамыр-6",
        "Мамыр-7",
        "Школьный",
        "Астана",
        "Дубок-1",
        "Дубок-2",
        "Таугуль-1",
        "Таугуль-2",
        "Таугуль-3",
        "Сайран",
        "Аксай-1",
        "Аксай-2",
        "Аксай-3",
        "Аксай-4",
        "Аксай-5",
        "Баян-Аул",
        "Алтын бесик",
        "Жетысу-1",
        "Жетысу-2",
        "Жетысу-3",
        "Жетысу-4",
        "Шанырак-1",
        "Шанырак-2",
        "Шанырак-3",
        "Шанырак-4",
        "Шанырак-5",
        "Кирово",
        "Айгерим",
        "Курылысшы",
        "Коккайнар",
    ]

    return jsonify({"success": True, "microdistricts": microdistricts})


@app.route("/debug")
def debug():
    """Отладочная информация"""
    test_data = {
        "area_total": 50,
        "floor": 5,
        "total_floors": 9,
        "year_build": 2015,
        "building_type": "монолитный",
        "rooms": 2,
        "microdistrict": "Мамыр-1",
    }

    result = predictor.predict(test_data)

    debug_info = {
        "model_mode": predictor.mode,
        "test_data": test_data,
        "prediction_result": result,
        "has_model_file": os.path.exists(MODEL_PATH),
        "has_predictor_file": os.path.exists(PREDICTOR_PATH),
        "current_dir": os.listdir("."),
    }

    return jsonify(debug_info)


@app.route("/api/model_info")
def model_info():
    """Информация о загруженной модели"""
    info = {
        "mode": predictor.mode,
        "has_model": predictor.model is not None,
        "features_count": len(predictor.features) if predictor.features else 0,
        "features_sample": predictor.features[:10] if predictor.features else [],
        "encoders": list(predictor.label_encoders.keys()),
        "has_scaler": predictor.scaler is not None,
    }

    # Попробуем тестовое предсказание
    test_data = {
        "area_total": 50,
        "floor": 5,
        "total_floors": 9,
        "year_build": 2015,
        "building_type": "монолитный",
    }

    try:
        test_result = predictor.predict(test_data)
        info["test_prediction"] = test_result
    except Exception as e:
        info["test_error"] = str(e)

    return jsonify(info)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
