FROM registry.access.redhat.com/ubi9/python-312:latest

WORKDIR /opt/app-root/src

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY drift_exporter.py .

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["gunicorn","--bind","0.0.0.0:8080","--workers","1","--threads","2","drift_exporter:app"]
