# MinIO Quickstart Guide

[![Slack](https://slack.min.io/slack?type=svg)](https://slack.min.io) [![Docker Pulls](https://img.shields.io/docker/pulls/minio/minio.svg?maxAge=604800)](https://hub.docker.com/r/minio/minio/) [![license](https://img.shields.io/badge/license-AGPL%20V3-blue)](https://github.com/minio/minio/blob/master/LICENSE)

[![MinIO](https://raw.githubusercontent.com/minio/minio/master/.github/logo.svg?sanitize=true)](https://min.io)

MinIO is a High Performance Object Storage released under GNU Affero General Public License v3.0. It is API compatible with Amazon S3 cloud storage service. Use MinIO to build high performance infrastructure for machine learning, analytics and application data workloads.

This README provides quickstart instructions on running MinIO on bare metal hardware, including container-based installations. For Kubernetes environments, use the [MinIO Kubernetes Operator](https://github.com/minio/operator/blob/master/README.md).

## Container Installation

Use the following commands to run a standalone MinIO server as a container.

Standalone MinIO servers are best suited for early development and evaluation. Certain features such as versioning, object locking, and bucket replication
require distributed deploying MinIO with Erasure Coding. For extended development and production, deploy MinIO with Erasure Coding enabled - specifically,
with a *minimum* of 4 drives per MinIO server. See [MinIO Erasure Code Overview](https://min.io/docs/minio/linux/operations/concepts/erasure-coding.html)
for more complete documentation.

