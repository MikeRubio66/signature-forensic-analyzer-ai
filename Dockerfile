FROM python:3.10-slim

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libx11-6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=web/app.py
ENV FLASK_RUN_HOST=0.0.0.0
EXPOSE 8000

CMD ["python", "web/app.py"]
