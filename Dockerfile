FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY creditrisk ./creditrisk
COPY api ./api
COPY dashboard ./dashboard
COPY scripts ./scripts

# Download the public dataset and train at build time so the image is self-contained.
RUN python scripts/download_data.py && python -m creditrisk.train --no-mlflow

ENV LLM_ENABLED=0
EXPOSE 8000 8501
# API by default; run the dashboard with:
#   docker run -p 8501:8501 credit-risk streamlit run dashboard/app.py --server.address=0.0.0.0
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
