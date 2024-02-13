FROM python:3.9-slim
WORKDIR /app
RUN pip install weaviate-client
COPY . /app/
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
