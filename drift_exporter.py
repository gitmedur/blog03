from flask import Flask, Response
import boto3
import json
import os
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

BUCKET = os.environ["AWS_S3_BUCKET"]
ENDPOINT = os.environ["AWS_S3_ENDPOINT"]
REGION = os.environ["AWS_DEFAULT_REGION"]

METRICS_KEY = os.getenv(
    "AWS_S3_METRICS_KEY",
    "metrics/iris_metrics.json"
)

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=REGION,
    verify=False,
)


def load_metrics():
    obj = s3.get_object(
        Bucket=BUCKET,
        Key=METRICS_KEY,
    )

    return json.loads(obj["Body"].read())


@app.route("/healthz")
def health():

    return "ok\n", 200


@app.route("/metrics")
def metrics():

    try:

        data = load_metrics()

        drift_detected = int(
            data.get("drift_detected_value", 0)
        )

        drift_share = float(
            data.get("drift_share", 0)
        )

        drift_threshold = float(
            data.get("drift_threshold", 0)
        )

        output = f"""# HELP iris_drift_detected Whether drift has been detected
# TYPE iris_drift_detected gauge
iris_drift_detected {drift_detected}

# HELP iris_drift_share Share of drifted columns
# TYPE iris_drift_share gauge
iris_drift_share {drift_share}

# HELP iris_drift_threshold Configured drift threshold
# TYPE iris_drift_threshold gauge
iris_drift_threshold {drift_threshold}

# HELP iris_drift_exporter_up Exporter health
# TYPE iris_drift_exporter_up gauge
iris_drift_exporter_up 1
"""

        return Response(
            output,
            mimetype="text/plain",
        )

    except Exception as e:

        logging.exception(e)

        output = f"""# HELP iris_drift_exporter_up Exporter health
# TYPE iris_drift_exporter_up gauge
iris_drift_exporter_up 0

# Exporter Error
# {e}
"""

        return Response(
            output,
            mimetype="text/plain",
            status=500,
        )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
    )
