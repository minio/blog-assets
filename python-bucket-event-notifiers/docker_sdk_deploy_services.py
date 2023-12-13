import docker
from pydantic import BaseModel

# Define the ServiceConfig model
class ServiceConfig(BaseModel):
    name: str
    docker_image: str
    ports: dict
    environment: dict = {}

# Initialize Docker client
client = docker.from_env()

# Function to deploy a service
def deploy_service(service_config: ServiceConfig):
    print(f"Deploying {service_config.name}")
    client.containers.run(
        image=service_config.docker_image,
        detach=True,
        ports=service_config.ports,
        environment=service_config.environment,
        name=service_config.name
    )

# Service configurations
flask_config = ServiceConfig(
    name="flaskapp",
    docker_image="flaskapp_image",  # Replace with your Flask Docker image
    ports={'5000/tcp': 5000},
    environment={"FLASK_ENV": "development"}
)

minio_config = ServiceConfig(
    name="minio",
    docker_image="minio/minio",
    ports={'9000/tcp': 9000},
    environment={"MINIO_ROOT_USER": "minio", "MINIO_ROOT_PASSWORD": "minio123"}
)

postgres_config = ServiceConfig(
    name="postgres",
    docker_image="postgres",
    ports={'5432/tcp': 5432},
    environment={"POSTGRES_PASSWORD": "yourpassword"}  # Replace with your password
)

redis_config = ServiceConfig(
    name="redis",
    docker_image="redis",
    ports={'6379/tcp': 6379}
)

# Deploy all services
if __name__ == "__main__":
    deploy_service(flask_config)
    deploy_service(minio_config)
    deploy_service(postgres_config)
    deploy_service(redis_config)
