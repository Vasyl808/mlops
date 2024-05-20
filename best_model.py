from mlflow import MlflowClient


def select_best_model_version(model_name):
    client = MlflowClient()
    all_versions = client.search_model_versions(f"name='{model_name}'")
    best_accuracy = 0.0
    best_model_version = None

    for version in all_versions:
        run_id = version.run_id
        run = client.get_run(run_id)
        if run.data.metrics is not None:
            metrics = {metric: value for metric, value in run.data.metrics.items()}
            accuracy = metrics.get('test_accuracy')
            if accuracy is not None and accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_version = version

    return best_model_version


def get_model_path(model_name, model_version):
    client = MlflowClient()
    model_details = client.get_model_version(model_name, model_version)
    if model_details is not None:
        return model_details.run_id
    else:
        return None
