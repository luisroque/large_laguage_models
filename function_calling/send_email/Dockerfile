FROM python:3.9-slim-buster

WORKDIR /app
ADD . /app

RUN pip install --no-cache-dir fastapi uvicorn

EXPOSE 1000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "1000"]
