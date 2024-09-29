# Week 6: Kafka Streaming with Python

1. Kafka docker-compose file
```yaml
version: '3.6'
networks:
  default:
    name: kafka-spark-network
    external: true
services:
  broker:
    image: confluentinc/cp-kafka:7.2.0
    hostname: broker
    container_name: broker
    depends_on:
      - zookeeper
    ports:
      - '9092:9092'
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: 'zookeeper:2181'
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_LISTENERS: PLAINTEXT://broker:29092,PLAINTEXT_HOST://broker:9092
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://broker:29092,PLAINTEXT_HOST://localhost:9092
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS: 0
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
  schema-registry:
    image: confluentinc/cp-schema-registry:7.2.0
    hostname: schema-registry
    container_name: schema-registry
    depends_on:
      - zookeeper
      - broker
    ports:
      - "8081:8081"
    environment:
      # SCHEMA_REGISTRY_KAFKASTORE_CONNECTION_URL: "zookeeper:2181" #(depreciated)
      SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS: "broker:29092"
      SCHEMA_REGISTRY_HOST_NAME: "localhost"
      SCHEMA_REGISTRY_LISTENERS: "http://0.0.0.0:8081" #(default: http://0.0.0.0:8081)
  zookeeper:
    image: confluentinc/cp-zookeeper:7.2.0
    hostname: zookeeper
    container_name: zookeeper
    ports:
      - '2181:2181'
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
  control-center:
    image: confluentinc/cp-enterprise-control-center:7.2.0
    hostname: control-center
    container_name: control-center
    depends_on:
      - zookeeper
      - broker
      - schema-registry
    ports:
      - "9021:9021"
    environment:
      CONTROL_CENTER_BOOTSTRAP_SERVERS: 'broker:29092'
      CONTROL_CENTER_ZOOKEEPER_CONNECT: 'zookeeper:2181'
      CONTROL_CENTER_SCHEMA_REGISTRY_URL: "http://localhost:8081"
      CONTROL_CENTER_REPLICATION_FACTOR: 1
      CONTROL_CENTER_INTERNAL_TOPICS_PARTITIONS: 1
      CONTROL_CENTER_MONITORING_INTERCEPTOR_TOPIC_PARTITIONS: 1
      CONFLUENT_METRICS_TOPIC_REPLICATION: 1
      PORT: 9021

  kafka-rest:
    image: confluentinc/cp-kafka-rest:7.2.0
    hostname: kafka-rest
    ports:
      - "8082:8082"
    depends_on:
      - schema-registry
      - broker
    environment:
      KAFKA_REST_BOOTSTRAP_SERVERS: 'broker:29092'
      KAFKA_REST_ZOOKEEPER_CONNECT: 'zookeeper:2181'
      KAFKA_REST_SCHEMA_REGISTRY_URL: 'http://localhost:8081'
      KAFKA_REST_HOST_NAME: localhost
      KAFKA_REST_LISTENERS: 'http://0.0.0.0:8082'
```
- remark: the code in *kafka/docker-compose.yml* didn't use this configuration, but instead, use the similar one from official github *[kafka-stack-docker-compose - conduktor](https://github.com/conduktor/kafka-stack-docker-compose)*
- `broker`, `schema-registry`, `zookeeper`, `control-center` are required to run the kafka
- `kafka-rest` is optional, but helping us debugging via REST API
- `networks` is also optional for kafka alone, but we need it to communicate with spark
- **Some important parameter to config that're worth mentioned and more researched are `KAFKA_LISTENERS` and `KAFKA_ADVERTISED_LISTENERS` used for configuring how kafka communicate within docker and access to the broker outside**


## More resources on Kafka Python
- [Kafka-Python explained in 10 lines of code](https://towardsdatascience.com/kafka-python-explained-in-10-lines-of-code-800e3e07dad1)
- [Apache Kafka ฉบับผู้เริ่มต้น #1: Hello Apache Kafka](https://medium.com/linedevth/apache-kafka-%E0%B8%89%E0%B8%9A%E0%B8%B1%E0%B8%9A%E0%B8%9C%E0%B8%B9%E0%B9%89%E0%B9%80%E0%B8%A3%E0%B8%B4%E0%B9%88%E0%B8%A1%E0%B8%95%E0%B9%89%E0%B8%99-1-hello-apache-kafka-242788d4f3c6)

## Appendix
- [Random User API - Mock up Data Generator](https://randomuser.me/documentation#intro)