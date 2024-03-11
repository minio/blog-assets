# README for /minio-self-hosted Directory

This directory contains all the necessary configurations, scripts, and documentation to help you deploy MinIO in a self-hosted Docker Swarm environment using GitHub Actions. These assets are designed to facilitate a smooth and efficient setup, aligning with the principles of GitOps and automated CI/CD pipelines.

Directory Contents:

	•	docker-compose.yml: The Docker Compose file used for deploying MinIO as part of a Docker Swarm stack. It is adapted from our previous exploration titled “MinIO Weaviate Python GitOps” and tailored for self-hosted scenarios.
	•	deploy-minio-on-rpi-swarm.yml: This YAML file defines the GitHub Actions workflow responsible for automating the deployment process of MinIO onto a Raspberry Pi (RPI) Docker Swarm cluster. It includes steps for checking out code, loading the Docker Compose configuration, deploying the MinIO stack, and verifying the deployment.
	•	github-runner.service: A Systemd service file template to help you set up a GitHub Actions runner as a persistent service on your Docker Swarm leader node. This ensures that your CI/CD pipelines run reliably.
	•	README.md: The file you’re currently reading, providing an overview of the directory contents and guidance on how to use them.

Getting Started:

1. Prepare Your Environment:
	•	Ensure your Docker Swarm is configured and running.
	•	Have a GitHub repository ready for setting up the GitHub Actions runner.

2. Deploy GitHub Actions Runner:
	•	Follow the instructions within deploy-minio-on-rpi-swarm.yml to set up and configure your self-hosted GitHub Actions runner within your Docker Swarm environment.

3.	MinIO Deployment:
	•	Customize docker-compose.yml as needed for your specific MinIO deployment requirements.
	•	Push the changes to your repository to trigger the GitHub Actions workflow and start the MinIO deployment process.

4.	Service Management:
	•	Use the github-runner.service file to create a Systemd service for your GitHub Actions runner, ensuring it automatically starts and remains active.

Support and Contributions:

For more in-depth discussions, questions, or collaboration proposals, reach out to us on the [MinIO Slack channel](https://minio.slack.com).

Together, let’s enhance the way we deploy and manage MinIO in self-hosted environments, leveraging the power of automation and GitOps practices.

Thank you for exploring the MinIO Self-Hosted Deployment Assets. We’re excited to see how you implement and benefit from these resources in your projects.
