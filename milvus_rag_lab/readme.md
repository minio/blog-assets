# This will download the original docker compose file from Milvus. It does not use AIStor Free.
wget https://github.com/milvus-io/milvus/releases/download/v2.6.7/milvus-standalone-docker-compose.yml -O docker-compose.yml

# Start Milvus, MinIO, and Etc.
docker compose -f docker-compose-cpu.yml --env-file config.env up -d
docker compose -f docker-compose-gpu.yml --env-file config.env up -d

# Check that containers are up and running.
docker compose -f docker-compose-cpu.yml --env-file config.env down

# Clean up
docker compose -f docker-compose-cpu.yml --env-file config.env down
sudo rm -rf volumes

# Getting Python setup.
source .venv/bin/activate
pip install pymilvus
