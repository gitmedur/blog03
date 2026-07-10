from kfp import dsl, compiler, kubernetes
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics


BASE_IMAGE = "python:3.12"
SECRET_NAME = "pipeline-artifacts-iris"

SECRET_KEY_TO_ENV = {
    "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
    "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
    "AWS_S3_BUCKET": "AWS_S3_BUCKET",
    "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
}


def attach_minio_secret(task):
    kubernetes.use_secret_as_env(
        task=task,
        secret_name=SECRET_NAME,
        secret_key_to_env=SECRET_KEY_TO_ENV,
    )
    return task


@component(base_image=BASE_IMAGE, packages_to_install=["boto3"])
def check_env():
    import os

    required = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_S3_ENDPOINT",
        "AWS_S3_BUCKET",
        "AWS_DEFAULT_REGION",
    ]

    missing = [key for key in required if not os.getenv(key)]

    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")

    print("All required MinIO/S3 environment variables are present.")


@component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas", "numpy", "boto3", "scikit-learn"],
)
def generate_current_data(current_data: Output[Dataset]):
    import os
    import boto3
    import numpy as np
    from sklearn.datasets import load_iris

    def normalize_columns(df):
        df = df.copy()
        df.columns = [
            c.replace(" (cm)", "").replace(" ", "_")
            for c in df.columns
        ]
        return df

    bucket = os.environ["AWS_S3_BUCKET"]
    endpoint = os.environ["AWS_S3_ENDPOINT"]

    iris = load_iris(as_frame=True)
    df = normalize_columns(iris.frame.copy())

    current = df.sample(80, random_state=42).copy()

    # Demo drift for the blog.
    # In production, replace this with real inference payload data.
    current["sepal_length"] = current["sepal_length"] + np.random.uniform(
        1.0, 2.0, len(current)
    )
    current["petal_length"] = current["petal_length"] + np.random.uniform(
        0.5, 1.2, len(current)
    )

    current = current.round(2)
    current.to_csv(current_data.path, index=False)

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
        verify=False,
    )

    s3.upload_file(current_data.path, bucket, "data/current_data.csv")
    print(f"Uploaded current data to s3://{bucket}/data/current_data.csv")


@component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas", "boto3", "evidently==0.7.21"],
)
def analyze_drift(
    current_data: Input[Dataset],
    drift_metrics: Output[Metrics],
    drift_flag: Output[Dataset],
    drift_threshold: float = 0.5,
) -> bool:
    import os
    import json
    import boto3
    import pandas as pd
    from botocore.exceptions import ClientError
    from evidently import Report
    from evidently.presets import DataDriftPreset

    feature_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]

    def normalize_columns(df):
        df = df.copy()
        df.columns = [
            c.replace(" (cm)", "").replace(" ", "_")
            for c in df.columns
        ]
        return df

    def validate_columns(df, dataset_name):
        missing_columns = [
            column
            for column in feature_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{dataset_name} is missing columns: {missing_columns}. "
                f"Found columns: {list(df.columns)}"
            )

    def ensure_reference_exists(s3_client, bucket_name, reference_key):
        try:
            s3_client.head_object(Bucket=bucket_name, Key=reference_key)
        except ClientError as e:
            error_code = e.response["Error"].get("Code")

            if error_code in ["404", "NoSuchKey", "NotFound"]:
                raise RuntimeError(
                    f"""
Reference dataset not found.

Expected object:
s3://{bucket_name}/{reference_key}

Create the baseline dataset by running Part 1 of this blog series,
or upload reference_data.csv to this location before running the
production drift pipeline.
""".strip()
                )

            raise

    def extract_drift_info(obj):
        drift_share = 0.0
        drifted_columns = []

        for metric in result_dict.get("metrics", []):
            metric_type = metric.get("config", {}).get("type")

            if metric_type == "evidently:metric_v2:DriftedColumnsCount":
                drift_share = float(metric.get("value", {}).get("share", 0.0))

            if metric_type == "evidently:metric_v2:ValueDrift":
                column = metric.get("config", {}).get("column")
                p_value = float(metric.get("value", 1.0))
                threshold = float(metric.get("config", {}).get("threshold", 0.05))

                if column in feature_columns and p_value < threshold:
                    drifted_columns.append(column)

        return drift_share, drifted_columns

    bucket = os.environ["AWS_S3_BUCKET"]
    endpoint = os.environ["AWS_S3_ENDPOINT"]
    reference_key = "data/reference_data.csv"

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
        verify=False,
    )

    ensure_reference_exists(s3, bucket, reference_key)

    reference_path = "/tmp/reference_data.csv"
    s3.download_file(bucket, reference_key, reference_path)

    reference_df = normalize_columns(pd.read_csv(reference_path))
    current_df = normalize_columns(pd.read_csv(current_data.path))

    validate_columns(reference_df, "Reference dataset")
    validate_columns(current_df, "Current dataset")

    report = Report(metrics=[DataDriftPreset()])

    snapshot = report.run(
        reference_data=reference_df[feature_columns],
        current_data=current_df[feature_columns],
    )

    try:
        result_dict = snapshot.as_dict()
    except AttributeError:
        result_dict = snapshot.dict()

    drift_share, drifted_columns = extract_drift_info(result_dict)
    drift_detected = drift_share >= drift_threshold

    result = {
        "drift_detected": drift_detected,
        "drift_share": float(drift_share),
        "drift_threshold": float(drift_threshold),
        "drifted_columns": drifted_columns,
    }

    with open(drift_flag.path, "w") as f:
        json.dump(result, f, indent=2)

    drift_metrics.log_metric("drift_share", float(drift_share))
    drift_metrics.log_metric("drift_threshold", float(drift_threshold))
    drift_metrics.log_metric("drift_detected", int(drift_detected))

    print(json.dumps(result, indent=2))

    return drift_detected


@component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas", "boto3", "evidently==0.7.21"],
)
def generate_html_report(current_data: Input[Dataset]):
    import os
    import boto3
    import pandas as pd
    from datetime import datetime, timezone
    from botocore.exceptions import ClientError
    from evidently import Report
    from evidently.presets import DataDriftPreset

    feature_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]

    def normalize_columns(df):
        df = df.copy()
        df.columns = [
            c.replace(" (cm)", "").replace(" ", "_")
            for c in df.columns
        ]
        return df

    def ensure_reference_exists(s3_client, bucket_name, reference_key):
        try:
            s3_client.head_object(Bucket=bucket_name, Key=reference_key)
        except ClientError as e:
            error_code = e.response["Error"].get("Code")

            if error_code in ["404", "NoSuchKey", "NotFound"]:
                raise RuntimeError(
                    f"""
Reference dataset not found.

Expected object:
s3://{bucket_name}/{reference_key}

Create the baseline dataset by running Part 1 of this blog series,
or upload reference_data.csv to this location before running the
production drift pipeline.
""".strip()
                )

            raise

    bucket = os.environ["AWS_S3_BUCKET"]
    endpoint = os.environ["AWS_S3_ENDPOINT"]
    reference_key = "data/reference_data.csv"

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
        verify=False,
    )

    ensure_reference_exists(s3, bucket, reference_key)

    reference_path = "/tmp/reference_data.csv"
    report_path = "/tmp/iris_drift_report.html"

    s3.download_file(bucket, reference_key, reference_path)

    reference_df = normalize_columns(pd.read_csv(reference_path))
    current_df = normalize_columns(pd.read_csv(current_data.path))

    report = Report(metrics=[DataDriftPreset()])

    snapshot = report.run(
        reference_data=reference_df[feature_columns],
        current_data=current_df[feature_columns],
    )

    snapshot.save_html(report_path)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")

    s3.upload_file(report_path, bucket, "reports/latest/iris_drift_report.html")
    s3.upload_file(report_path, bucket, f"reports/history/{timestamp}.html")

    print(f"Uploaded latest report to s3://{bucket}/reports/latest/iris_drift_report.html")
    print(f"Uploaded historical report to s3://{bucket}/reports/history/{timestamp}.html")


@component(base_image=BASE_IMAGE, packages_to_install=["boto3"])
def write_drift_metrics_for_prometheus(drift_flag: Input[Dataset]):
    import os
    import json
    import boto3
    from datetime import datetime, timezone

    bucket = os.environ["AWS_S3_BUCKET"]
    endpoint = os.environ["AWS_S3_ENDPOINT"]

    with open(drift_flag.path) as f:
        drift_result = json.load(f)

    metrics_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": "iris",
        "drift_detected": bool(drift_result["drift_detected"]),
        "drift_detected_value": 1 if drift_result["drift_detected"] else 0,
        "drift_share": float(drift_result["drift_share"]),
        "drift_threshold": float(drift_result["drift_threshold"]),
        "drifted_columns": drift_result.get("drifted_columns", []),
    }

    metrics_path = "/tmp/iris_metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
        verify=False,
    )

    s3.upload_file(metrics_path, bucket, "metrics/iris_metrics.json")

    print(json.dumps(metrics_payload, indent=2))
    print(f"Uploaded metrics to s3://{bucket}/metrics/iris_metrics.json")


@component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas", "numpy", "boto3", "scikit-learn"],
)
def retrain_model(model_output: Output[Model]) -> str:
    import os
    import json
    import pickle
    import boto3
    from datetime import datetime, timezone

    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    def normalize_columns(df):
        df = df.copy()
        df.columns = [
            c.replace(" (cm)", "").replace(" ", "_")
            for c in df.columns
        ]
        return df

    bucket = os.environ["AWS_S3_BUCKET"]
    endpoint = os.environ["AWS_S3_ENDPOINT"]

    feature_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]

    def backup_existing_latest(s3_client, bucket_name, latest_prefix, backup_prefix):
        objects = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=f"{latest_prefix}/",
        )

        for obj in objects.get("Contents", []):
            source_key = obj["Key"]

            if source_key.endswith("/"):
                continue

            file_name = source_key.split("/")[-1]
            backup_key = f"{backup_prefix}/{file_name}"

            s3_client.copy_object(
                Bucket=bucket_name,
                CopySource={"Bucket": bucket_name, "Key": source_key},
                Key=backup_key,
            )

            print(
                f"Backed up s3://{bucket_name}/{source_key} "
                f"to s3://{bucket_name}/{backup_key}"
            )

    iris = load_iris(as_frame=True)
    df = normalize_columns(iris.frame.copy())

    X = df[feature_columns]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    os.makedirs(model_output.path, exist_ok=True)

    model_path = os.path.join(model_output.path, "iris_model.pkl")
    settings_path = os.path.join(model_output.path, "model-settings.json")
    metadata_path = os.path.join(model_output.path, "model_metadata.json")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    model_settings = {
        "name": "iris",
        "implementation": "mlserver_sklearn.SKLearnModel",
        "parameters": {
            "uri": "./iris_model.pkl",
            "version": "v1",
        },
    }

    with open(settings_path, "w") as f:
        json.dump(model_settings, f, indent=2)

    version = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")

    version_prefix = f"models/iris_model/{version}"
    latest_prefix = "models/iris_model/latest"
    backup_prefix = f"models/iris_model/backups/before-{version}"
    model_storage_uri = f"s3://{bucket}/{version_prefix}/"

    metadata = {
        "version": version,
        "accuracy": float(accuracy),
        "storage_uri": model_storage_uri,
        "version_path": f"s3://{bucket}/{version_prefix}/",
        "latest_path": f"s3://{bucket}/{latest_prefix}/",
        "backup_of_previous_latest": f"s3://{bucket}/{backup_prefix}/",
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
        verify=False,
    )

    backup_existing_latest(
        s3_client=s3,
        bucket_name=bucket,
        latest_prefix=latest_prefix,
        backup_prefix=backup_prefix,
    )

    s3.upload_file(model_path, bucket, f"{version_prefix}/iris_model.pkl")
    s3.upload_file(settings_path, bucket, f"{version_prefix}/model-settings.json")
    s3.upload_file(metadata_path, bucket, f"{version_prefix}/model_metadata.json")

    s3.upload_file(model_path, bucket, f"{latest_prefix}/iris_model.pkl")
    s3.upload_file(settings_path, bucket, f"{latest_prefix}/model-settings.json")
    s3.upload_file(metadata_path, bucket, f"{latest_prefix}/model_metadata.json")

    print("\n===== New Model Version =====")
    print(json.dumps(metadata, indent=2))
    print("=============================\n")

    return model_storage_uri


@component(base_image=BASE_IMAGE, packages_to_install=["kubernetes"])
def promote_model_to_kserve(
    model_storage_uri: str,
    namespace: str = "iris",
    inferenceservice_name: str = "iris",
):
    from kubernetes import client, config
    import time

    config.load_incluster_config()
    custom_api = client.CustomObjectsApi()
    core_api = client.CoreV1Api()

    patch_body = {
        "spec": {
            "predictor": {
                "model": {
                    "storageUri": model_storage_uri,
                }
            }
        }
    }

    print(
        f"Patching InferenceService {namespace}/{inferenceservice_name} "
        f"with storageUri={model_storage_uri}"
    )

    custom_api.patch_namespaced_custom_object(
        group="serving.kserve.io",
        version="v1beta1",
        namespace=namespace,
        plural="inferenceservices",
        name=inferenceservice_name,
        body=patch_body,
    )

    print("InferenceService patched successfully.")

    # Give KServe chance to reconcile.
    print("Waiting 15 seconds for KServe reconciliation...")
    time.sleep(15)

    # Restart predictor pod so the storage initializer
    # downloads the newly promoted model.
    label_selector = (
        f"serving.kserve.io/inferenceservice={inferenceservice_name}"
    )

    pods = core_api.list_namespaced_pod(
        namespace=namespace,
        label_selector=label_selector,
    )

    if not pods.items:
        print("No predictor pods found.")
        return

    for pod in pods.items:
        pod_name = pod.metadata.name
        print(
            f"Deleting predictor pod "
            f"{pod_name}"
        )

        core_api.delete_namespaced_pod(
            namespace=namespace,
            name=pod_name,
        )
    print("Predictor pod restart requested.")


@dsl.pipeline(
    name="iris-drift-production-pipeline",
    description="Scheduled Iris drift detection, Evidently reporting, Prometheus metrics source, model backup, and KServe model promotion.",
)
def iris_drift_production_pipeline(
    drift_threshold: float = 0.5,
):
    env_task = check_env()
    attach_minio_secret(env_task)

    current_task = generate_current_data()
    current_task.set_caching_options(False)
    attach_minio_secret(current_task)
    current_task.after(env_task)

    drift_task = analyze_drift(
        current_data=current_task.outputs["current_data"],
        drift_threshold=drift_threshold,
    )
    drift_task.set_caching_options(False)
    attach_minio_secret(drift_task)

    report_task = generate_html_report(
        current_data=current_task.outputs["current_data"],
    )
    report_task.set_caching_options(False)
    attach_minio_secret(report_task)
    report_task.after(drift_task)

    prometheus_metrics_task = write_drift_metrics_for_prometheus(
        drift_flag=drift_task.outputs["drift_flag"],
    )
    prometheus_metrics_task.set_caching_options(False)
    attach_minio_secret(prometheus_metrics_task)
    prometheus_metrics_task.after(drift_task)

    with dsl.If(
        drift_task.outputs["Output"] == True,
        name="drift-detected",
    ):
        retrain_task = retrain_model()
        retrain_task.set_caching_options(False)
        attach_minio_secret(retrain_task)
        retrain_task.after(drift_task)

        promote_task = promote_model_to_kserve(
            model_storage_uri=retrain_task.outputs["Output"],
            namespace="iris",
            inferenceservice_name="iris",
        )
        promote_task.set_caching_options(False)
        promote_task.after(retrain_task)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=iris_drift_production_pipeline,
        package_path="iris_drift_production_pipeline.yaml",
    )

    print("Compiled pipeline: iris_drift_production_pipeline.yaml")
