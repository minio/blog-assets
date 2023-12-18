# flaskapp.dockerfile
FROM python:3.8 
WORKDIR /app
COPY . .
RUN pip install Flask psycopg2-binary redis minio pydantic python-dotenv
EXPOSE 5000
ENV FLASK_ENV=development
CMD ["python", "flaskapp.py"]
