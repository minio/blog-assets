from pydantic import BaseModel
from dataclasses import dataclass
from flask import Flask, request
from minio import Minio
import psycopg2
import redis

# Pydantic configuration class
class MinioClientConfig(BaseModel):
    endpoint: str = 'localhost:9000'
    access_key: str = 'minio'
    secret_key: str = 'minio123'

# Pydantic configuration class
class PostgresClientConfig(BaseModel):
    host: str = 'localhost'
    port: int = 5432
    user: str = 'user'
    password: str = 'password'
    database: str = 'postgres'

# Pydantic configuration class
class RedisClientConfig(BaseModel):
    host: str = 'localhost'
    port: int = 6379

# Dataclass for event
@dataclass
class Event:
    event_type: str
    data: dict

# Initialize clients (in a real scenario, handle connections appropriately)
minio_client = Minio(MinioClientConfig.endpoint, 
                     access_key=MinioClientConfig.access_key, 
                     secret_key=MinioClientConfig.secret_key)

# Initialize clients (in a real scenario, handle connections appropriately)
pg_conn = psycopg2.connect(
    host=PostgresClientConfig.host,
    port=PostgresClientConfig.port,
    user=PostgresClientConfig.user,
    password=PostgresClientConfig.password,
    dbname=PostgresClientConfig.database)

# Initialize clients (in a real scenario, handle connections appropriately)
redis_client = redis.Redis(
    host=RedisClientConfig.host, 
    port=RedisClientConfig.port)

# This is where we start the Flask App and create API endpoints consisting of: @app.route ( TheRoute, TheRoutesMethod)
app = Flask(__name__)

@app.route('/event', methods=['POST'])
def handle_event():
    event_data = request.json
    event = Event(**event_data)
    process_event(event)
    return "Event processed"

def process_event(event: Event):
    # Process event for all services
    # Example: Log event data in PostgreSQL and cache in Redis
    try:
        with pg_conn.cursor() as cur:
            cur.execute("INSERT INTO events (event_type, data) VALUES (%s, %s)", (event.event_type, str(event.data)))
            pg_conn.commit()
        redis_client.set(f"event:{event.data['file_name']}", str(event.data))
    except Exception as e:
        print(f"Error processing event: {e}")

# Simulating an event for demonstration
simulated_event_data = {
    'event_type': 'file_uploaded',
    'data': {
        'file_name': 'example.txt',
        'file_size': '1024',
        'bucket_name': 'test-bucket'
    }
}

# Create an Event instance using simulated data
simulated_event = Event(**simulated_event_data)

# Call the process_event function with the simulated event
process_event(simulated_event)

if __name__ == "__main__":
    app.run(debug=True)
