import json
import psycopg2
from flask import Flask, jsonify, request
from minio import Minio

# Initialize MinIO and PostgreSQL clients
minio_client = Minio(
    endpoint='minio:9000',
    access_key='minio',
    secret_key='minio123',
    secure=True
)

pg_conn = psycopg2.connect(
    host='postgres',
    port=5432,
    user='myuser',
    password='mypassword',
    dbname='postgres'
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
