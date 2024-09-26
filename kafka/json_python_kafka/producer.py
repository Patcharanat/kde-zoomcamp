"""
Python-Kafka POC Producer
"""

import json

from kafka import KafkaProducer
from kafka.errors import KafkaTimeoutError

from utils import DataGeneratorAPI, ConfigurationReader
from schema import RandomUserSchema

if __name__ == "__main__":
    data: list[RandomUserSchema] = DataGeneratorAPI(results=10).get_data_with_schema()
    read_config: dict[str, dict] = ConfigurationReader(config_path="config.yaml").get_config()

    kafka_producer_config: dict = read_config.get("KAFKA_PRODUCER", {})
    BOOTSTRAP_SERVICES: str = kafka_producer_config.get("BOOTSTRAP_SERVICES", None)
    KAFKA_TOPIC: str = kafka_producer_config.get("KAFKA_TOPIC", None)

    producer_config: dict = {
        "bootstrap_servers": read_config.get("BOOTSTRAP_SERVICES"),
        "key_serializer": lambda key: str(key).encode(),  # TODO: verify if encode method is necessary
        "value_serializer": lambda stream_data: json.dumps(stream_data, default=str).encode("utf-8"),
    }

    # not wrapping KafkaProducer as another OOP object for learning native Kafka-Python API
    producer = KafkaProducer(**producer_config)
    for transaction in data:
        try:
            record = producer.send(
                topic=KAFKA_TOPIC,
                key=transaction.id,
                value=transaction
            )
            print(f"Record: {transaction.id} successfully produced at offset {record.get().offset()}")
        except KafkaTimeoutError as e:
            print(e.__str__())
