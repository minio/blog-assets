FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY generate-and-upload-fake-data.py .

CMD ["python", "generate-and-upload-fake-data.py"]