FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV TRAIN_SUBSET_SIZE=256
ENV VAL_SUBSET_SIZE=64
ENV TEST_SUBSET_SIZE=64

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["gunicorn", "--workers", "1", "--threads", "2", "--bind", "0.0.0.0:7860", "app:app"]
