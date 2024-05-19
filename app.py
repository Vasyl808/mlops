from flask import Flask, request, jsonify
import pandas as pd
import shemas
import utils

app = Flask(__name__)


@app.route('/api/v1/predict', methods=['POST'])
def predict():
    # Перевірка чи надійшли дані у форматі JSON
    if not request.is_json:
        return jsonify({"error": "Input data must be in JSON format"}), 400

    # Перевірка чи дані відповідають схемі
    try:
        data = shemas.HeartDiseasePredictionInputSchema().load(request.json)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    data_df = pd.DataFrame([data], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                              'thalach', 'restecg', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

    return jsonify({
        'result': utils.predict(data_df)
    })


if __name__ == '__main__':
    app.run(debug=True)
