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

### Stable

Run the following command to run the latest stable image of MinIO as a container using an ephemeral data volume:

```sh
podman run -p 9000:9000 -p 9001:9001 \
  quay.io/minio/minio server /data --console-address ":9001"
```

The MinIO deployment starts using default root credentials `minioadmin:minioadmin`. You can test the deployment using the MinIO Console, an embedded
object browser built into MinIO Server. Point a web browser running on the host machine to <http://127.0.0.1:9000> and log in with the
root credentials. You can use the Browser to create buckets, upload objects, and browse the contents of the MinIO server.

You can also connect using any S3-compatible tool, such as the MinIO Client `mc` commandline tool. See
[Test using MinIO Client `mc`](#test-using-minio-client-mc) for more information on using the `mc` commandline tool. For application developers,
see <https://min.io/docs/minio/linux/developers/minio-drivers.html> to view MinIO SDKs for supported languages.

> NOTE: To deploy MinIO on with persistent storage, you must map local persistent directories from the host OS to the container using the `podman -v` option. For example, `-v /mnt/data:/data` maps the host OS drive at `/mnt/data` to `/data` on the container.

## macOS

Use the following commands to run a standalone MinIO server on macOS.

Standalone MinIO servers are best suited for early development and evaluation. Certain features such as versioning, object locking, and bucket replication require distributed deploying MinIO with Erasure Coding. For extended development and production, deploy MinIO with Erasure Coding enabled - specifically, with a *minimum* of 4 drives per MinIO server. See [MinIO Erasure Code Overview](https://min.io/docs/minio/linux/operations/concepts/erasure-coding.html) for more complete documentation.

### Homebrew (recommended)

Run the following command to install the latest stable MinIO package using [Homebrew](https://brew.sh/). Replace ``/data`` with the path to the drive or directory in which you want MinIO to store data.

```sh
brew install minio/stable/minio
minio server /data
```

> NOTE: If you previously installed minio using `brew install minio` then it is recommended that you reinstall minio from `minio/stable/minio` official repo instead.

```sh
brew uninstall minio
brew install minio/stable/minio
```

The MinIO deployment starts using default root credentials `minioadmin:minioadmin`. You can test the deployment using the MinIO Console, an embedded web-based object browser built into MinIO Server. Point a web browser running on the host machine to <http://127.0.0.1:9000> and log in with the root credentials. You can use the Browser to create buckets, upload objects, and browse the contents of the MinIO server.

You can also connect using any S3-compatible tool, such as the MinIO Client `mc` commandline tool. See [Test using MinIO Client `mc`](#test-using-minio-client-mc) for more information on using the `mc` commandline tool. For application developers, see <https://min.io/docs/minio/linux/developers/minio-drivers.html/> to view MinIO SDKs for supported languages.

### Binary Download

Use the following command to download and run a standalone MinIO server on macOS. Replace ``/data`` with the path to the drive or directory in which you want MinIO to store data.

```sh
wget https://dl.min.io/server/minio/release/darwin-amd64/minio
chmod +x minio
./minio server /data
```

The MinIO deployment starts using default root credentials `minioadmin:minioadmin`. You can test the deployment using the MinIO Console, an embedded web-based object browser built into MinIO Server. Point a web browser running on the host machine to <http://127.0.0.1:9000> and log in with the root credentials. You can use the Browser to create buckets, upload objects, and browse the contents of the MinIO server.

You can also connect using any S3-compatible tool, such as the MinIO Client `mc` commandline tool. See [Test using MinIO Client `mc`](#test-using-minio-client-mc) for more information on using the `mc` commandline tool. For application developers, see <https://min.io/docs/minio/linux/developers/minio-drivers.html> to view MinIO SDKs for supported languages.

## GNU/Linux

Use the following command to run a standalone MinIO server on Linux hosts running 64-bit Intel/AMD architectures. Replace ``/data`` with the path to the drive or directory in which you want MinIO to store data.

```sh
wget https://dl.min.io/server/minio/release/linux-amd64/minio
chmod +x minio
./minio server /data
```

The following table lists supported architectures. Replace the `wget` URL with the architecture for your Linux host.

| Architecture                   | URL                                                        |
| --------                       | ------                                                     |
| 64-bit Intel/AMD               | <https://dl.min.io/server/minio/release/linux-amd64/minio>   |
| 64-bit ARM                     | <https://dl.min.io/server/minio/release/linux-arm64/minio>   |
| 64-bit PowerPC LE (ppc64le)    | <https://dl.min.io/server/minio/release/linux-ppc64le/minio> |
| IBM Z-Series (S390X)           | <https://dl.min.io/server/minio/release/linux-s390x/minio>   |

The MinIO deployment starts using default root credentials `minioadmin:minioadmin`. You can test the deployment using the MinIO Console, an embedded web-based object browser built into MinIO Server. Point a web browser running on the host machine to <http://127.0.0.1:9000> and log in with the root credentials. You can use the Browser to create buckets, upload objects, and browse the contents of the MinIO server.

You can also connect using any S3-compatible tool, such as the MinIO Client `mc` commandline tool. See [Test using MinIO Client `mc`](#test-using-minio-client-mc) for more information on using the `mc` commandline tool. For application developers, see <https://min.io/docs/minio/linux/developers/minio-drivers.html> to view MinIO SDKs for supported languages.

> NOTE: Standalone MinIO servers are best suited for early development and evaluation. Certain features such as versioning, object locking, and bucket replication require distributed deploying MinIO with Erasure Coding. For extended development and production, deploy MinIO with Erasure Coding enabled - specifically, with a *minimum* of 4 drives per MinIO server. See [MinIO Erasure Code Overview](https://min.io/docs/minio/linux/operations/concepts/erasure-coding.html#) for more complete documentation.

## Microsoft Windows

To run MinIO on 64-bit Windows hosts, download the MinIO executable from the following URL:

```sh
https://dl.min.io/server/minio/release/windows-amd64/minio.exe
```

Use the following command to run a standalone MinIO server on the Windows host. Replace ``D:\`` with the path to the drive or directory in which you want MinIO to store data. You must change the terminal or powershell directory to the location of the ``minio.exe`` executable, *or* add the path to that directory to the system ``$PATH``:

```sh
minio.exe server D:\
```

The MinIO deployment starts using default root credentials `minioadmin:minioadmin`. You can test the deployment using the MinIO Console, an embedded web-based object browser built into MinIO Server. Point a web browser running on the host machine to <http://127.0.0.1:9000> and log in with the root credentials. You can use the Browser to create buckets, upload objects, and browse the contents of the MinIO server.

You can also connect using any S3-compatible tool, such as the MinIO Client `mc` commandline tool. See [Test using MinIO Client `mc`](#test-using-minio-client-mc) for more information on using the `mc` commandline tool. For application developers, see <https://min.io/docs/minio/linux/developers/minio-drivers.html> to view MinIO SDKs for supported languages.

> NOTE: Standalone MinIO servers are best suited for early development and evaluation. Certain features such as versioning, object locking, and bucket replication require distributed deploying MinIO with Erasure Coding. For extended development and production, deploy MinIO with Erasure Coding enabled - specifically, with a *minimum* of 4 drives per MinIO server. See [MinIO Erasure Code Overview](https://min.io/docs/minio/linux/operations/concepts/erasure-coding.html#) for more complete documentation.

## Install from Source

Use the following commands to compile and run a standalone MinIO server from source. Source installation is only intended for developers and advanced users. If you do not have a working Golang environment, please follow [How to install Golang](https://golang.org/doc/install). Minimum version required is [go1.19](https://golang.org/dl/#stable)

```sh
go install github.com/minio/minio@latest
```

The MinIO deployment starts using default root credentials `minioadmin:minioadmin`. You can test the deployment using the MinIO Console, an embedded web-based object browser built into MinIO Server. Point a web browser running on the host machine to <http://127.0.0.1:9000> and log in with the root credentials. You can use the Browser to create buckets, upload objects, and browse the contents of the MinIO server.

You can also connect using any S3-compatible tool, such as the MinIO Client `mc` commandline tool. See [Test using MinIO Client `mc`](#test-using-minio-client-mc) for more information on using the `mc` commandline tool. For application developers, see <https://min.io/docs/minio/linux/developers/minio-drivers.html> to view MinIO SDKs for supported languages.

> NOTE: Standalone MinIO servers are best suited for early development and evaluation. Certain features such as versioning, object locking, and bucket replication require distributed deploying MinIO with Erasure Coding. For extended development and production, deploy MinIO with Erasure Coding enabled - specifically, with a *minimum* of 4 drives per MinIO server. See [MinIO Erasure Code Overview](https://min.io/docs/minio/linux/operations/concepts/erasure-coding.html) for more complete documentation.

MinIO strongly recommends *against* using compiled-from-source MinIO servers for production environments.

## Deployment Recommendations

### Allow port access for Firewalls

By default MinIO uses the port 9000 to listen for incoming connections. If your platform blocks the port by default, you may need to enable access to the port.

### ufw

For hosts with ufw enabled (Debian based distros), you can use `ufw` command to allow traffic to specific ports. Use below command to allow access to port 9000

```sh
ufw allow 9000
```

Below command enables all incoming traffic to ports ranging from 9000 to 9010.

```sh
ufw allow 9000:9010/tcp
```

### firewall-cmd

For hosts with firewall-cmd enabled (CentOS), you can use `firewall-cmd` command to allow traffic to specific ports. Use below commands to allow access to port 9000

```sh
firewall-cmd --get-active-zones
```

This command gets the active zone(s). Now, apply port rules to the relevant zones returned above. For example if the zone is `public`, use

```sh
firewall-cmd --zone=public --add-port=9000/tcp --permanent
```

Note that `permanent` makes sure the rules are persistent across firewall start, restart or reload. Finally reload the firewall for changes to take effect.

```sh
firewall-cmd --reload
```

### iptables

For hosts with iptables enabled (RHEL, CentOS, etc), you can use `iptables` command to enable all traffic coming to specific ports. Use below command to allow
access to port 9000

```sh
iptables -A INPUT -p tcp --dport 9000 -j ACCEPT
service iptables restart
```

Below command enables all incoming traffic to ports ranging from 9000 to 9010.

```sh
iptables -A INPUT -p tcp --dport 9000:9010 -j ACCEPT
service iptables restart
```

## Test MinIO Connectivity

### Test using MinIO Console

MinIO Server comes with an embedded web based object browser. Point your web browser to <http://127.0.0.1:9000> to ensure your server has started successfully.

> NOTE: MinIO runs console on random port by default, if you wish to choose a specific port use `--console-address` to pick a specific interface and port.

