# This will download the original docker compose file from Milvus. It does not use AIStor Free.
wget https://github.com/milvus-io/milvus/releases/download/v2.6.7/milvus-standalone-docker-compose.yml -O docker-compose.yml

# Set up a Python virtual environment and install pymilvus into it.
python -m venv .venv
source .venv/bin/activate
pip install pymilvus

# Start Milvus, MinIO, and Etc containers.
docker compose -f docker-compose-cpu.yml --env-file config.env up -d
docker compose -f docker-compose-gpu.yml --env-file config.env up -d

# Check that containers are up and running.
docker compose -f docker-compose-cpu.yml --env-file config.env ps

# Clean up
docker compose -f docker-compose-cpu.yml --env-file config.env down
sudo rm -rf volumes
