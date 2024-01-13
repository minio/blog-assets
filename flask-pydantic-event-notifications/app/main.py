import json
import psycopg2
from flask import Flask, jsonify, request
from minio import Minio
from pydantic import BaseModel, ValidationError

# Pydantic configuration class for MinIO client
class MinioClientConfig(BaseModel):
    endpoint: str
    access_key: str
    secret_key: str
    secure: bool = False

# Pydantic configuration class for PostgreSQL client
class PostgresClientConfig(BaseModel):
    host: str
    port: int
    user: str
    password: str
    database: str

# Initialize configuration instances
minio_config = MinioClientConfig(
    endpoint='minio:9000',
    access_key='minio',
    secret_key='minio123'
)

postgres_config = PostgresClientConfig(
    host='postgres',
    port=5432,
    user='myuser',
    password='mypassword',
    database='postgres'
)

# Initialize MinIO and PostgreSQL clients
minio_client = Minio(
    minio_config.endpoint,
    access_key=minio_config.access_key,
    secret_key=minio_config.secret_key,
    secure=minio_config.secure
)

pg_conn = psycopg2.connect(
    host=postgres_config.host,
    port=postgres_config.port,
    user=postgres_config.user,
    password=postgres_config.password,
    dbname=postgres_config.database
)

# Flask app initialization
app = Flask(__name__)

@app.route('/minio-event', methods=['POST'])
def handle_minio_event():
    event_data = request.json
    try:
        with pg_conn.cursor() as cur:
            for record in event_data['Records']:
                try:
                    # Extract object key directly from record
                    object_key = record['s3']['object']['key']
                except KeyError as e:
                    print(f"Key error: {e}")
                    continue

                # Store the entire record as JSON
                json_data = json.dumps(record)

                cur.execute("""
                    INSERT INTO events (key, value) 
                    VALUES (%s, %s)
                    ON CONFLICT (key) 
                    DO UPDATE SET value = EXCLUDED.value;
                    """, (object_key, json_data))
            pg_conn.commit()
    except Exception as e:
        print(f"Error: {e}")
        pg_conn.rollback()
    finally:
        pg_conn.close()

    return "Event processed", 200


@app.route('/hello', methods=['GET'])
def hello():
    return "Hello MinIO!"


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
