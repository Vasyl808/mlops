import joblib
import mlflow
import pandas as pd
import sklearn.preprocessing
import sklearn
import numpy as np


logged_model = 'runs:/a73e7ab366e9439e8e54f1a463bc5f07/model'
loaded_model = mlflow.pyfunc.load_model(logged_model)


def scale_data(date_to_transform: np.array, path: str) -> np.array:
    scaler: sklearn.preprocessing = joblib.load(path)
    scaled_data = scaler.transform(date_to_transform)
    return scaled_data


def encode_data(data_to_encode: pd.DataFrame, columns: list, path: str) -> pd.DataFrame:
    encoder = joblib.load(path)

    encoded_data = encoder.transform(data_to_encode[columns])
    encoded_columns = encoder.get_feature_names_out(columns)

    encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoded_columns)
    data_without_categories = data_to_encode.drop(columns=columns)

    final_data: pd.DataFrame = pd.concat([data_without_categories, encoded_df], axis=1)
    return final_data


def predict(data_df: pd.DataFrame) -> bool:
    numeric_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    data_numeric_scaled = scale_data(data_df[numeric_columns], 'scaler.pkl')

    data_numeric_scaled_df = pd.DataFrame(data_numeric_scaled, columns=numeric_columns)
    for column in numeric_columns:
        data_df[column] = data_numeric_scaled_df[column]

    category_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    data_encoded_df = encode_data(data_df, category_columns, 'encoder.pkl')

    predictions = (loaded_model.predict(data_encoded_df) > 0.5).astype("bool")

    return bool(predictions[0])

