# Kafka Schema Registry MinIO

In the previous Notebook we saw how to use Kafka Connectors to stream events directly to MinIO but thats the simplest way to stream data which may not be efficient and performant enough for production usecases with large workloads and it is errorprone for instance for a given kafka topic if data chanages like a new column gets added, an existing column gets removed or the data type of a given column gets modified consumer may not be aware of these changes and there is a possibility of corruption.

Kafka Schema Registry is a component in the Apache Kafka ecosystem that provides a centralized schema management service for Kafka producers and consumers. It allows producers to register schemas for the data they produce, and consumers to retrieve and use these schemas for data validation and deserialization. The Schema Registry helps ensure that data exchanged through Kafka is compliant with a predefined schema, enabling data consistency, compatibility, and evolution across different systems and applications.

Here are some key benefits of using Kafka Schema Registry:

* Schema Evolution: As data formats and requirements evolve over time, it is common for producers and consumers to undergo changes to their data schemas. Kafka Schema Registry provides support for schema evolution, allowing producers to register new versions of schemas while maintaining compatibility with existing consumers. Consumers can retrieve the appropriate schema version for deserialization, ensuring that data is processed correctly even when schema changes occur

* Data Validation: Kafka Schema Registry enables data validation by allowing producers to register schemas with predefined data types, field names, and other constraints. Consumers can then retrieve and use these schemas to validate incoming data, ensuring that data conforms to the expected structure and format. This helps prevent data processing errors and improves data quality

* Schema Management: Kafka Schema Registry provides a centralized repository for managing schemas, making it easier to track, version, and manage changes to data schemas. Producers and consumers can register, retrieve, and manage schemas through a simple API, allowing for centralized schema governance and management.

* Interoperability: Kafka Schema Registry promotes interoperability between different producers and consumers by providing a standardized way to define and manage data schemas. Producers and consumers written in different programming languages or using different serialization frameworks can use a common schema registry to ensure data consistency and compatibility across the ecosystem

* Backward and Forward Compatibility: Kafka Schema Registry allows producers to register backward and forward compatible schemas, enabling smooth upgrades and changes to data schemas without disrupting existing producers and consumers. Backward compatibility ensures that older consumers can still process data produced with a newer schema, while forward compatibility allows newer consumers to process data produced with an older schema

Strimzi Operator doesn't come with Schema Registry yet we will use the one availabe in confluent helm repository.

In this Notebook we will do the following

1. Setup Kafka Schema Registry using Helm charts
1. Create and deploy a sample producer that uses an Avro schema and sends events
1. Build a KafkaConnect continer which has Avro dependency
1. Deploy KafkaConnect using the above container
1. Deploy KafkaConector that reads the schema from Schema registry, consumes topic events from the producer and stores data into MinIO in parquet format

## Setup Schema Registry

We will clone the confluent helm repository using the following command

We will clone the confluent helm repository using the following command

```
!git clone https://github.com/confluentinc/cp-helm-charts.git
```

```
#move to schema registry folder
%cd cp-helm-charts/charts/cp-schema-registry
```


Use the blow command to install schema registry using the helm charts, we will need to provide the bootstrap server endpoint of the existing kafka cluster we deployed for the installation to be successful

```
!helm install kafka-schema-registry --set kafka.bootstrapServers="PLAINTEXT://my-kafka-cluster-kafka-bootstrap:9092" . -n kafka
```

You can check if the Schema Registry is up and running successfully by checking the logs as shown below

```
!kubectl -n kafka logs -f --selector=app=cp-schema-registry -c cp-schema-registry-server # stop this shell once you are done
```

```
Apr 12, 2023 4:52:25 PM org.glassfish.jersey.internal.inject.Providers checkProviderRuntime
WARNING: A provider io.confluent.kafka.schemaregistry.rest.resources.SchemasResource registered in SERVER runtime does not implement any provider interfaces applicable in the SERVER runtime. Due to constraint configuration problems the provider io.confluent.kafka.schemaregistry.rest.resources.SchemasResource will be ignored. 
Apr 12, 2023 4:52:25 PM org.glassfish.jersey.internal.inject.Providers checkProviderRuntime
WARNING: A provider io.confluent.kafka.schemaregistry.rest.resources.SubjectVersionsResource registered in SERVER runtime does not implement any provider interfaces applicable in the SERVER runtime. Due to constraint configuration problems the provider io.confluent.kafka.schemaregistry.rest.resources.SubjectVersionsResource will be ignored. 
[2023-04-12 16:52:26,511] INFO HV000001: Hibernate Validator 6.1.2.Final (org.hibernate.validator.internal.util.Version)
[2023-04-12 16:52:28,268] INFO Started o.e.j.s.ServletContextHandler@3241713e{/,null,AVAILABLE} (org.eclipse.jetty.server.handler.ContextHandler)
[2023-04-12 16:52:28,378] INFO Started o.e.j.s.ServletContextHandler@7051777c{/ws,null,AVAILABLE} (org.eclipse.jetty.server.handler.ContextHandler)
[2023-04-12 16:52:28,478] INFO Started NetworkTrafficServerConnector@418c5a9c{HTTP/1.1, (http/1.1)}{0.0.0.0:8081} (org.eclipse.jetty.server.AbstractConnector)
[2023-04-12 16:52:28,479] INFO Started @18811ms (org.eclipse.jetty.server.Server)
[2023-04-12 16:52:28,481] INFO Server started, listening for requests... (io.confluent.kafka.schemaregistry.rest.SchemaRegistryMain)
^C
```

## Create Avro Topic

Create a YAML file for the kafka topic nyc-avro-topic as shown below and apply it.

```
%%writefile deployment/kafka-nyc-avro-topic.yaml
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaTopic
metadata:
  name: nyc-avro-topic
  namespace: kafka
  labels:
    strimzi.io/cluster: my-kafka-cluster
spec:
  partitions: 3
  replicas: 3
```

Overwriting deployment/kafka-nyc-avro-topic.yaml

```
!kubectl apply -f deployment/kafka-nyc-avro-topic.yaml
kafkatopic.kafka.strimzi.io/nyc-avro-topic created
```

```
!kubectl -n kafka get kafkatopic nyc-avro-topic
NAME             CLUSTER            PARTITIONS   REPLICATION FACTOR   READY
nyc-avro-topic   my-kafka-cluster   3            3                    True
```

## Producer with Avro Schema

We will create a simple python producer than register Avro schema with the Kafka Schema Registry and sends kafka topic events. This will be based on the producer that we already had in the previous Notebook

```
%%writefile sample-code/producer/src/avro-producer.py
import logging
import os

import fsspec
import pandas as pd
import s3fs
from avro.schema import make_avsc_object
from confluent_kafka.avro import AvroProducer

logging.basicConfig(level=logging.INFO)

# Avro schema

value_schema_dict = {
    "type": "record",
    "name": "nyc_avro",
    "fields": [
        {
            "name": "VendorID",
            "type": "long"
        },
        {
            "name": "tpep_pickup_datetime",
            "type": "string"
        },
        {
            "name": "tpep_dropoff_datetime",
            "type": "string"
        },
        {
            "name": "passenger_count",
            "type": "double"
        },
        {
            "name": "trip_distance",
            "type": "double"
        },
        {
            "name": "RatecodeID",
            "type": "double"
        },
        {
            "name": "store_and_fwd_flag",
            "type": "string"
        },
        {
            "name": "PULocationID",
            "type": "long"
        },
        {
            "name": "DOLocationID",
            "type": "long"
        },
        {
            "name": "payment_type",
            "type": "long"
        },
        {
            "name": "fare_amount",
            "type": "double"
        },
        {
            "name": "extra",
            "type": "double"
        },
        {
            "name": "mta_tax",
            "type": "double"
        },
        {
            "name": "tip_amount",
            "type": "double"
        },
        {
            "name": "tolls_amount",
            "type": "double"
        },
        {
            "name": "improvement_surcharge",
            "type": "double"
        },
        {
            "name": "total_amount",
            "type": "double"
        },
    ]
}

value_schema = make_avsc_object(value_schema_dict)

producer_config = {
    "bootstrap.servers": "my-kafka-cluster-kafka-bootstrap:9092",
    "schema.registry.url": "http://kafka-schema-registry-cp-schema-registry:8081"
}

producer = AvroProducer(producer_config, default_value_schema=value_schema)

fsspec.config.conf = {
    "s3":
        {
            "key": os.getenv("AWS_ACCESS_KEY_ID", "openlakeuser"),
            "secret": os.getenv("AWS_SECRET_ACCESS_KEY", "openlakeuser"),
            "client_kwargs": {
                "endpoint_url": "https://play.min.io:50000"
            }
        }
}
s3 = s3fs.S3FileSystem()
total_processed = 0
i = 1
for df in pd.read_csv('s3a://openlake/spark/sample-data/taxi-data.csv', chunksize=10000):
    count = 0
    for index, row in df.iterrows():
        producer.produce(topic="nyc-avro-topic", value=row.to_dict())
        count += 1

    total_processed += count
    if total_processed % 10000 * i == 0:
        producer.flush()
        logging.info(f"total processed till now {total_processed} for topic 'nyc-avro-topic'")
        i += 1
```
Overwriting sample-code/producer/src/avro-producer.py

add requirements and Dockerfile based on which we will build the docker image

```
%%writefile sample-code/producer/requirements.txt
pandas==2.0.0
s3fs==2023.4.0
pyarrow==11.0.0
kafka-python==2.0.2
confluent_kafka[avro]==2.1.0
```

Overwriting sample-code/producer/requirements.txt

```
%%writefile sample-code/producer/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY src/avro-producer.py .
CMD ["python3", "-u", "./avro-producer.py"]
```

Overwriting sample-code/producer/Dockerfile


Build and push the docker image for the producer using the above docker file into your docker registry or you can use the one available in openlake [openlake/kafka-demo-avro-producer](https://hub.docker.com/r/openlake/kafka-demo-avro-producer/tags)

Let's create a YAML that deploys our producer in kubernetes cluster as a job

```
%%writefile deployment/avro-producer.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: avro-producer-job
  namespace: kafka
spec:
  template:
    metadata:
      name: avro-producer-job
    spec:
      containers:
      - name: avro-producer-job
        image: openlake/kafka-demo-avro-producer:latest
      restartPolicy: Never
```

Writing deployment/avro-producer.yaml

Deploy the `avro-producer.yaml` file

```
!kubectl apply -f deployment/avro-producer.yaml
job.batch/avro-producer-job created
```

You can check the logs by using the below command

```
!kubectl logs -f job.batch/avro-producer-job -n kafka # stop this shell once you are done
Error from server (NotFound): jobs.batch "avro-producer-job" not found
```

## Build Kafka Connect Image

Lets build a kafka connect image that has S3 and Avro dependencies

```
%%writefile sample-code/connect/Dockerfile
FROM confluentinc/cp-kafka-connect:7.0.9 as cp
RUN confluent-hub install --no-prompt confluentinc/kafka-connect-s3:10.4.2
RUN confluent-hub install --no-prompt confluentinc/kafka-connect-avro-converter:7.3.3
FROM quay.io/strimzi/kafka:0.34.0-kafka-3.4.0
USER root:root
# Add S3 dependency
COPY --from=cp /usr/share/confluent-hub-components/confluentinc-kafka-connect-s3/ /opt/kafka/plugins/kafka-connect-s3/
# Add Avro dependency
COPY --from=cp /usr/share/confluent-hub-components/confluentinc-kafka-connect-avro-converter/ /opt/kafka/plugins/avro/
```

Writing sample-code/connect/Dockerfile

Build and push the docker image for the producer using the above docker file into your docker registry or you can use the one available in openlake [openlake/kafka-connect:0.34.0](https://hub.docker.com/r/openlake/kafka-connect/tags)

Before we deploy the KafkaConnect we first need to create storage topcis if not already present for the KafkaConnect to work as expected.

## Deploy Kafka Connect

Create a YAML file for Kafka Connect that uses the above image and deploy it in k8s. The KafkaConnect will have 1 replica and make use of ths storage topics that we created in the previous Notebook.

NOTE: `spec.template.connectContainer.env` has the creds defiend in order for KafkaConnect to store data in Minio cluster, other details like the `endpoint_url, bucket_name` will be part of `KafkaConnector`. `key.converter` and `value.converter` is pointing to AvroConverter (`io.confluent.connect.avro.AvroConverter`)

```
%%writefile deployment/avro-connect.yaml
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaConnect
metadata:
  name: avro-connect-cluster
  namespace: kafka
  annotations:
    strimzi.io/use-connector-resources: "true"
spec:
  image: openlake/kafka-connect:0.34.0
  version: 3.4.0
  replicas: 1
  bootstrapServers: my-kafka-cluster-kafka-bootstrap:9093
  tls:
    trustedCertificates:
      - secretName: my-kafka-cluster-cluster-ca-cert
        certificate: ca.crt
  config:
    bootstrap.servers: my-kafka-cluster-kafka-bootstrap:9092
    group.id: avro-connect-cluster
    key.converter: io.confluent.connect.avro.AvroConverter
    value.converter: io.confluent.connect.avro.AvroConverter
    internal.key.converter: org.apache.kafka.connect.json.JsonConverter
    internal.value.converter: org.apache.kafka.connect.json.JsonConverter
    key.converter.schemas.enable: false
    value.converter.schemas.enable: false
    offset.storage.topic: connect-offsets
    offset.storage.replication.factor: 1
    config.storage.topic: connect-configs
    config.storage.replication.factor: 1
    status.storage.topic: connect-status
    status.storage.replication.factor: 1
    offset.flush.interval.ms: 10000
    plugin.path: /opt/kafka/plugins
    offset.storage.file.filename: /tmp/connect.offsets
  template:
    connectContainer:
      env:
        - name: AWS_ACCESS_KEY_ID
          value: "openlakeuser"
        - name: AWS_SECRET_ACCESS_KEY
          value: "openlakeuser"
```

Overwriting deployment/avro-connect.yaml

```
!kubectl apply -f deployment/avro-connect.yaml
```

kafkaconnect.kafka.strimzi.io/avro-connect-cluster created

## Deploy Kafka Sink Connector

Now that we have the Kafka Connect up and running next step is to deploy the sink connector that will poll `nyc-avro-topic` and store the data into MinIO bucket `openlake-tmp` in `parquet` format.

`connector.class` - specifies what type of connector the sink connector will use in our case it is `io.confluent.connect.s3.S3SinkConnector`

`store.url` - MinIO endpoint URL where you want to store the data from KafkaConnect

`storage.class` - specifies which storage class to use in our case since we are storing in MinIO `io.confluent.connect.s3.storage.S3Storage` will be used

`format.class` - Format type in which the data will be stored into MinIO, since we would like to store `parquet` we will use `io.confluent.connect.s3.format.parquet.ParquetFormat` implementation

`value.converter` - Since we want to convert the binary data to avro we will use `io.confluent.connect.avro.AvroConverter`

`parquet.codec` - Specifies what type of compression we would like to use for the parquet files, in our case we will use `snappy`

`schema.registry.url` - Specifies the endpoint from which the connector can pull, validate the schema and deserialize the data from the producer

```
%%writefile deployment/avro-connector.yaml
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaConnector
metadata:
  name: "avro-connector"
  namespace: "kafka"
  labels:
    strimzi.io/cluster:
      avro-connect-cluster
spec:
  class: io.confluent.connect.s3.S3SinkConnector
  config:
    connector.class: io.confluent.connect.s3.S3SinkConnector
    task.max: '1'
    topics: nyc-avro-topic
    s3.region: us-east-1
    s3.bucket.name: openlake-tmp
    s3.part.size: '5242880'
    flush.size: '10000'
    topics.dir: nyc-taxis-avro
    timezone: UTC
    store.url: https://play.min.io:50000
    storage.class: io.confluent.connect.s3.storage.S3Storage
    format.class: io.confluent.connect.s3.format.parquet.ParquetFormat
    partitioner.class: io.confluent.connect.storage.partitioner.DefaultPartitioner
    s3.credentials.provider.class: com.amazonaws.auth.DefaultAWSCredentialsProviderChain
    behavior.on.null.values: ignore
    auto.register.schemas: false
    parquet.codec: snappy
    schema.registry.url: http://kafka-schema-registry-cp-schema-registry:8081
    value.converter: io.confluent.connect.avro.AvroConverter
    key.converter: org.apache.kafka.connect.storage.StringConverter
    value.converter.schema.registry.url: http://kafka-schema-registry-cp-schema-registry:8081
```

Overwriting deployment/avro-connector.yaml

```
!kubectl apply -f deployment/avro-connector.yaml
```

kafkaconnector.kafka.strimzi.io/avro-connector created

If all goes well we can see files being added to Minio `openlake-tmp` bucket by executing the below command

```
!mc ls --summarize --recursive play/openlake-tmp/nyc-taxis-avro/nyc-avro-topic/
```

```
]11;?\[2023-04-12 12:45:31 PDT] 167KiB STANDARD partition=0/nyc-avro-topic+0+0000000000.snappy.parquet
[2023-04-12 12:45:33 PDT] 187KiB STANDARD partition=0/nyc-avro-topic+0+0000010000.snappy.parquet
[2023-04-12 12:45:34 PDT] 179KiB STANDARD partition=0/nyc-avro-topic+0+0000020000.snappy.parquet
[2023-04-12 12:45:35 PDT] 167KiB STANDARD partition=0/nyc-avro-topic+0+0000030000.snappy.parquet
[2023-04-12 12:45:36 PDT] 178KiB STANDARD partition=0/nyc-avro-topic+0+0000040000.snappy.parquet
[2023-04-12 12:45:37 PDT] 179KiB STANDARD partition=0/nyc-avro-topic+0+0000050000.snappy.parquet
[2023-04-12 12:45:38 PDT] 165KiB STANDARD partition=0/nyc-avro-topic+0+0000060000.snappy.parquet
[2023-04-12 12:45:40 PDT] 187KiB STANDARD partition=0/nyc-avro-topic+0+0000070000.snappy.parquet
[2023-04-12 12:45:41 PDT] 167KiB STANDARD partition=0/nyc-avro-topic+0+0000080000.snappy.parquet
[2023-04-12 12:45:42 PDT] 188KiB STANDARD partition=0/nyc-avro-topic+0+0000090000.snappy.parquet
[2023-04-12 12:45:43 PDT] 191KiB STANDARD partition=0/nyc-avro-topic+0+0000100000.snappy.parquet

Total Size: 1.9 MiB
Total Objects: 11
```

The current setup that we have is significantly faster and highly storage efficient that the previous basic setup that we had in the previous [Notebook](https://github.com/minio/openlake/blob/af2cbb6a0aef4534016447ec8a995364146503c9/kafka/kafka-minio.ipynb), you can try running both the Producers and connectors and see the peformance and memory differences.

With this we have an end-to-end setup for efficiently for producing data kafka topics using Avro schema and consuming it directly into MinIO in parquet format.

## Experimental: Iceberg

Recently Iceberg connector support has been added to kafka by `getindata` [here](https://github.com/getindata/kafka-connect-iceberg-sink) is the repo. Below we will explore how to store the `nyc-avro-topic` data directly as Iceberg table into MinIO. This is still experimental and not ready for production IMO.

## Iceberg Kafka Connect

Let's create a KafkaConnect that has Iceberg dependencies, make sure to edit the `spec.config.build.output.image` and `spec.config.build.output.pushSecret` to point to your Docker Registry before deploying

```
%%writefile deployment/iceberg-connect.yaml
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaConnect
metadata:
    name: iceberg-connect-cluster
    namespace: "kafka"
    annotations:
        strimzi.io/use-connector-resources: "true"
spec:
    version: 3.4.0
    replicas: 1
    bootstrapServers: my-kafka-cluster-kafka-bootstrap:9093
    tls:
        trustedCertificates:
            -   secretName: my-kafka-cluster-cluster-ca-cert
                certificate: ca.crt
    logging:
        type: inline
        loggers:
            log4j.rootLogger: "ERROR"
            log4j.logger.com.getindata.kafka.connect.iceberg.sink.IcebergSinkTask: "DEBUG"
            log4j.logger.org.apache.hadoop.io.compress.CodecPool: "WARN"
    config:
        group.id: iceberg-connect-cluster
        offset.storage.topic: connect-offsets
        config.storage.topic: connect-configs
        status.storage.topic: connect-status
        config.storage.replication.factor: 1
        offset.storage.replication.factor: 1
        status.storage.replication.factor: 1
        config.providers: file,secret,configmap
        config.providers.file.class: org.apache.kafka.common.config.provider.FileConfigProvider
        config.providers.secret.class: io.strimzi.kafka.KubernetesSecretConfigProvider
        config.providers.configmap.class: io.strimzi.kafka.KubernetesConfigMapConfigProvider
        bootstrap.servers: my-kafka-cluster-kafka-bootstrap:9092
        key.converter: io.confluent.connect.avro.AvroConverter
        value.converter: io.confluent.connect.avro.AvroConverter
        internal.key.converter: org.apache.kafka.connect.json.JsonConverter
        internal.value.converter: org.apache.kafka.connect.json.JsonConverter
        key.converter.schemas.enable: false
        value.converter.schemas.enable: false
        plugin.path: /opt/kafka/plugins
        offset.storage.file.filename: /tmp/connect.offsets
        build:
            output:
                type: docker
                image: <NameOfYourRegistry>
                pushSecret: <RegistrySecret>
            plugins:
                -   name: kafka-avro-converter
                    artifacts:
                        -   type: zip
                            url: https://d1i4a15mxbxib1.cloudfront.net/api/plugins/confluentinc/kafka-connect-avro-converter/versions/7.3.1/confluentinc-kafka-connect-avro-converter-7.3.1.zip
                -   name: iceberg
                    artifacts:
                        -   type: zip
                            url: https://github.com/getindata/kafka-connect-iceberg-sink/releases/download/0.3.1/kafka-connect-iceberg-sink-0.3.1-plugin.zip
    resources:
        requests:
            cpu: "0.1"
            memory: 512Mi
        limits:
            cpu: "5"
            memory: 15Gi
    template:
        pod:
            tmpDirSizeLimit: "1Gi" # this is required for hive/nessie catalogs to work
        connectContainer:
            env:
                # important for using AWS s3 client sdk
                -   name: AWS_REGION
                    value: "us-east-1"
                -   name: AWS_ACCESS_KEY_ID
                    value: "openlakeuser"
                -   name: AWS_SECRET_ACCESS_KEY
                    value: "openlakeuser"
                -   name: S3_ENDPOINT
                    value: "https://play.min.io:50000"
```

Overwriting deployment/iceberg-connect.yaml

```
# Deploy the above KafkaConnect CRD
!kubectl apply -f deployment/iceberg-connect.yaml
```

kafkaconnect.kafka.strimzi.io/iceberg-connect-cluster created

## Deploy Iceberg Sink Connector
Now that we have the Iceberg KafkaConnect deployed lets deploy the KafkaConnector that will store Iceberg table directly into MinIO. There are 3 possible Connectors that you can use as shown below

## Hadoop Iceberg Sink Connector
Below deployment will use the `Hadoop` catalog to create and maintain Iceberg table in MinIO

```
%%writefile deployment/iceberg-hadoop-connector.yaml
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaConnector
metadata:
    name: iceberg-hadoop-sink-connector
    namespace: kafka
    labels:
        strimzi.io/cluster: my-connect-cluster
    annotations:
        strimzi.io/restart: "true"
spec:
    class: com.getindata.kafka.connect.iceberg.sink.IcebergSink
    tasksMax: 1
    config:
        task.max: '1'
        topics: nyc-avro-topic
        timezone: UTC
        schema.compatibility: NONE
        behavior.on.null.values: ignore
        auto.register.schemas: true
        schema.registry.url: http://kafka-schema-registry-cp-schema-registry:8081
        value.converter: io.confluent.connect.avro.AvroConverter
        key.converter: org.apache.kafka.connect.storage.StringConverter
        partitioner.class: io.confluent.connect.storage.partitioner.DefaultPartitioner
        value.converter.schema.registry.url: http://kafka-schema-registry-cp-schema-registry:8081
        table.namespace: "kafka"
        table.prefix: ""
        table.auto-create: true
        table.write-format: "parquet"
        iceberg.catalog.default.type: hive
        iceberg.catalog-impl: "org.apache.iceberg.hadoop.HadoopCatalog"
        iceberg.table-default.write.metadata.delete-after-commit.enabled: true
        iceberg.table-default.write.metadata.previous-versions-max: 10
        iceberg.table-default.write.merge.mode: "merge-on-read"
        iceberg.table-default.write.delete.mode: "merge-on-read"
        iceberg.table-default.commit.manifest.min-count-to-merge: 5
        iceberg.catalog-name: mycatalog
        iceberg.warehouse: "s3a://opentable-tmp/warehouse/nyc"
        iceberg.fs.defaultFS: "s3a://opentable-tmp"
        iceberg.fs.s3a.path.style.access: true
        iceberg.fs.s3a.fast.upload: true
        iceberg.fs.s3a.fast.upload.buffer: "bytebuffer"
        iceberg.fs.s3a.endpoint: https://play.min.io:50000
        iceberg.fs.s3a.impl: "org.apache.hadoop.fs.s3a.S3AFileSystem"
        iceberg.fs.s3a.access.key: 'openlakeuser'
        iceberg.fs.s3a.secret.key: 'openlakeuser'
        iceberg.fs.s3a.connection.ssl.enabled: true
        receive.buffer.bytes: 20485760
        fetch.max.bytes: 52428800
        consumer.override.max.poll.records: 10000
        offset.storage.file.filename: /tmp/connect.offsets
```

Overwriting deployment/iceberg-hadoop-connector.yaml

## Hive Iceberg Sink Connector

Below deployment will use the `Hive catalog` to create and maintain Iceberg table in MinIO

Note: `iceberg.uri`, `iceberg.catalog-impl`, `iceberg.table-default.write.data.path`, `iceberg.table-default.write.metadata.path` are required for Iceberg Hive catalog work

```
%%writefile deployment/iceberg-hive-connector.yaml
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaConnector
metadata:
    name: iceberg-hive-sink-connector
    namespace: kafka
    labels:
        strimzi.io/cluster: my-connect-cluster
    annotations:
        strimzi.io/restart: "true"
spec:
    class: com.getindata.kafka.connect.iceberg.sink.IcebergSink
    tasksMax: 1
    config:
        task.max: '1'
        topics: nyc-avro-topic
        timezone: UTC
        schema.compatibility: NONE
        behavior.on.null.values: ignore
        auto.register.schemas: true
        schema.registry.url: http://kafka-schema-registry-cp-schema-registry:8081
        value.converter: io.confluent.connect.avro.AvroConverter
        key.converter: org.apache.kafka.connect.storage.StringConverter
        partitioner.class: io.confluent.connect.storage.partitioner.DefaultPartitioner
        value.converter.schema.registry.url: http://kafka-schema-registry-cp-schema-registry:8081
        table.namespace: "kafka"
        table.prefix: ""
        table.auto-create: true
        table.write-format: "parquet"
        iceberg.catalog.default.type: hive
        iceberg.uri: thrift://metastore-svc:9083 # required for Hive catalog to work
        iceberg.catalog-impl: "org.apache.iceberg.hive.HiveCatalog"
        iceberg.table-default.write.data.path: "s3a://openlake-tmp/warehouse/nyc/nyc-taxi-data"
        iceberg.table-default.write.metadata.path: "s3a://openlake-tmp/warehouse/nyc/nyc-taxi-data/metadata"
        iceberg.table-default.write.metadata.delete-after-commit.enabled: true
        iceberg.io-impl: "org.apache.iceberg.aws.s3.S3FileIO"
        iceberg.engine.hive.enabled: true
        iceberg.catalog-name: "default"
        iceberg.catalog.default.catalog-impl: org.apache.iceberg.hive.HiveCatalog
        iceberg.warehouse: "s3a://opentable-tmp/warehouse/nyc"
        iceberg.fs.defaultFS: "s3a://opentable-tmp"
        iceberg.fs.s3.path-style-access: true
        iceberg.fs.s3.fast.upload: true
        iceberg.fs.s3.fast.upload.buffer: "bytebuffer"
        iceberg.fs.s3.endpoint: https://play.min.io:50000
        iceberg.fs.s3.access-key-id: 'openlakeuser'
        iceberg.fs.s3.secret-access-key: 'openlakeuser'
        iceberg.fs.s3a.connection.ssl.enabled: true
        receive.buffer.bytes: 20485760
        fetch.max.bytes: 52428800
        consumer.override.max.poll.records: 10000
        offset.storage.file.filename: /tmp/connect.offsets
```

Overwriting deployment/iceberg-hive-connector.yaml

## Nessie Iceberg Sink Connector
Below deployment will use the `Nessie catalog` to create and maintain Iceberg table in MinIO.

Note: `iceberg.uri`, `iceberg.ref`, `iceberg.catalog-impl` are the key things that change to make Iceberg Nessie catalog work

```
%%writefile deployment/iceberg-nessie-connector.yaml
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaConnector
metadata:
    name: iceberg-nessie-sink-connector
    namespace: kafka
    labels:
        strimzi.io/cluster: my-connect-cluster
    annotations:
        strimzi.io/restart: "true"
spec:
    class: com.getindata.kafka.connect.iceberg.sink.IcebergSink
    tasksMax: 1
    config:
        task.max: '1'
        topics: nyc-avro-topic
        timezone: UTC
        schema.compatibility: NONE
        behavior.on.null.values: ignore
        auto.register.schemas: true
        schema.registry.url: http://kafka-schema-registry-cp-schema-registry:8081
        value.converter: io.confluent.connect.avro.AvroConverter
        key.converter: org.apache.kafka.connect.storage.StringConverter
        partitioner.class: io.confluent.connect.storage.partitioner.DefaultPartitioner
        value.converter.schema.registry.url: http://kafka-schema-registry-cp-schema-registry:8081
        table.namespace: "kafka"
        table.prefix: ""
        table.auto-create: true
        table.write-format: "parquet"
        iceberg.uri: http://nessie.nessie-ns.svc:19120/api/v1 # required for Nessie catalog to work
        iceberg.ref: "dev"  # required for Nessie (branch name)
        iceberg.catalog-impl: "org.apache.iceberg.nessie.NessieCatalog"
        iceberg.table-default.write.metadata.delete-after-commit.enabled: true
        iceberg.io-impl: "org.apache.iceberg.aws.s3.S3FileIO"
        iceberg.engine.hive.enabled: true
        iceberg.catalog-name: mycatalog
        iceberg.catalog.default.catalog-impl: org.apache.iceberg.hive.HiveCatalog
        iceberg.warehouse: "s3a://opentable-tmp/warehouse/nyc"
        iceberg.fs.defaultFS: "s3a://opentable-tmp"
        iceberg.fs.s3.path-style-access: true
        iceberg.fs.s3.fast.upload: true
        iceberg.fs.s3.fast.upload.buffer: "bytebuffer"
        iceberg.fs.s3.endpoint: https://play.min.io:50000
        iceberg.fs.s3.access-key-id: 'openlakeuser'
        iceberg.fs.s3.secret-access-key: 'openlakeuser'
        iceberg.fs.s3.connection.ssl.enabled: true
        receive.buffer.bytes: 20485760
        fetch.max.bytes: 52428800
        consumer.override.max.poll.records: 10000
        offset.storage.file.filename: /tmp/connect.offsets
```

Overwriting deployment/iceberg-nessie-connector.yaml


Use any of the following command to deploy the KafkaConnector with the Iceberg Catalog of your choice, by default hadoop catalog has be enabled below

```
!kubectl apply -f deployment/iceberg-hadoop-connector.yaml

# !kubectl apply -f deployment/iceberg-hive-connector.yaml

# !kubectl apply -f deployment/iceberg-nessie-connector.yaml
```

This brings us to the endof this Notebook, with the above steps you should have end-to-end setup to stream data from Kafka to MinIO as Iceberg table directly. As mentioned earlier Iceberg connector for Kafka is experimental based on the initial experiments that I performed and not yet ready for production, this could change soon as there is active development going on. If you have Spark already setup and would like a production ready solution for storing Iceberg tables in MinIO you can explore Spark Streaming