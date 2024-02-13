# minio-weaviate-python-gitops

[Link to the Article](https://blog.min.io/minio-weaviate-python-gitops)

Integrating services by building containers using GitHub Actions and Workflows, while pushing custom images to DockerHub

## In this directory is:

- `README.md` (this file)
- `/minio/` folder for building custom MinIO container image.
- `/app/` folder for building custom Python container image.
- `docker-compose.yaml` for orchestrating the build.
- `docker-workflow.yml` for running GitHub Actions and steps.
