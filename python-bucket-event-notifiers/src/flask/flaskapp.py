from pydantic import BaseModel, BaseSettings
from dataclasses import dataclass
from flask import Flask, request
from python-dotenv import load_dotenv
import psycopg2
import redis
import os

load_dotenv()

# Pydantic configuration class with default values
class MinioClientConfig(BaseSettings):
    endpoint: str = os.getenv('MINIO_ENDPOINT', 'localhost:9000') # Defaults to demo container hostname
    access_key: str = os.getenv('MINIO_ACCESS_KEY', 'minio')
    secret_key: str = os.getenv('MINIO_SECRET_KEY', 'minio123')

# Pydantic configuration class with default values
class PostgresClientConfig(BaseSettings):
    host: str = os.getenv('POSTGRES_HOST', 'postgres') # Defaults to demo container hostname
    port: int = os.getenv('POSTGRES_PORT', 5432)
    user: str = os.getenv('POSTGRES_USER', 'myuser')
    password: str = os.getenv('POSTGRES_PASSWORD', 'mypassword') 
    database: str = os.getenv('POSTGRES_DB', 'postgres')

# Pydantic configuration class with default values
class RedisClientConfig(BaseSettings):
    host: str = os.getenv('REDIS_HOST', 'redis') # Defaults to demo container hostname
    port: int = os.getenv('REDIS_PORT', 6379)

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
    app.run(debug=True, host='0.0.0.0', port=5000)