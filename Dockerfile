
FROM python:3.9


WORKDIR /usr/src/app


COPY requirements.txt ./


RUN pip install --no-cache-dir -r requirements.txt


COPY . .

EXPOSE 8000

CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
