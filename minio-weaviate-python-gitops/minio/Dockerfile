FROM minio/minio
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENV MINIO_ROOT_USER=minio \
    MINIO_ROOT_PASSWORD=minio123
EXPOSE 9000 9001
ENTRYPOINT ["/entrypoint.sh"]
