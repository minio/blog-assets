# Use an argument for the base image
ARG BASE_IMAGE=minio/minio:latest
FROM $BASE_IMAGE

# Add custom configuration or scripts if needed
# COPY your-custom-config /your-config-location
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 9000 9001
ENTRYPOINT ["/entrypoint.sh"]
