import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


import tensorflow as tf

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pylab as plt
import joblib
import mlflow
import mlflow.tensorflow
import json


def load_data(file_path):
    data = pd.read_csv(file_path)
    data.rename(columns={'condition': 'target'}, inplace=True)
    return data


def preprocess_data(data, categorys, numeric_columns):
    encoder = OneHotEncoder()
    encoder.fit(data[categorys])
    joblib.dump(encoder, 'encoder.pkl')
    encoded_data = encoder.transform(data[categorys])
    encoded_columns = encoder.get_feature_names_out(categorys)
    encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoded_columns)
    data_without_categories = data.drop(columns=categorys)
    final_data = pd.concat([data_without_categories, encoded_df], axis=1)

    scaler = MinMaxScaler()
    scaled_numeric = scaler.fit_transform(final_data[numeric_columns])
    scaled_df = pd.DataFrame(scaled_numeric, columns=numeric_columns)
    data = pd.concat([scaled_df, final_data[final_data.columns.difference(numeric_columns)]], axis=1)

    joblib.dump(scaler, 'scaler.pkl')

    return data


def split_data(data, target_column, test_size=0.2):
    X, y = data.drop(columns=[target_column]), data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


def train_model(X_train, y_train, model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=10, batch_size=16)
    return history


def evaluate_model(X, y, model):
    loss, accuracy = model.evaluate(X, y)
    return loss, accuracy


def save_artifacts(history, model, X_test, y_test):
    # Log loss and accuracy curves
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    mlflow.log_artifact('loss_curve.png')

    plt.figure()
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_curve.png')
    mlflow.log_artifact('accuracy_curve.png')

    loss, accuracy = evaluate_model(X_test, y_test, model)
    mlflow.log_metric('test_loss', loss)
    mlflow.log_metric('test_accuracy', accuracy)


def automated_model_retrain_validation_selection(data, categorys, numeric_columns, target_column):
    data = preprocess_data(data, categorys, numeric_columns)
    X_train, X_test, y_train, y_test = split_data(data, target_column)

    input_shape = (X_train.shape[1],)
    model = create_model(input_shape)

    with mlflow.start_run() as run:
        mlflow.set_tracking_uri(f"file://mlruns")
        mlflow.tensorflow.autolog()

        dataset_train = mlflow.data.from_pandas(
            data, source='https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci',
            name="heart-disease-cleveland-uci", targets="target"
        )
        mlflow.log_input(dataset_train)

        history = train_model(X_train, y_train, model)
        save_artifacts(history, model, X_test, y_test)

        mlflow.log_param('dataset', 'Heart_disease_cleveland_new')
        mlflow.log_param('tags', 'prod')

        predictions = (model.predict(X_test) > 0.5).astype("int32")

        mlflow.keras.log_model(
            model,
            artifact_path="model",
            registered_model_name="hear_disease_clevland",
        )
        report = classification_report(y_test, predictions, output_dict=True)
        with open("classification_report.json", "w") as f:
            json.dump(report, f)
        mlflow.log_artifact("classification_report.json")
        matrix = confusion_matrix(y_test, predictions)
        with open("confusion_matrix.csv", "w") as f:
            np.savetxt(f, matrix, delimiter=",")
        mlflow.log_artifact("confusion_matrix.csv")


file_path = 'Heart_disease_cleveland_new.csv'
categorys = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numeric_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
target_column = 'target'

data = load_data(file_path)
automated_model_retrain_validation_selection(data, categorys, numeric_columns, target_column)
