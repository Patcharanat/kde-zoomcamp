"""
Python-Kafka POC Consumer
"""

import json

from kafka import KafkaConsumer

from utils import ConfigurationReader
from schema import RandomUserSchema


if __name__ == "__main__":
    read_config: dict = ConfigurationReader(config_path="config.yaml").get_config() # absoulte path to config may required
    consumer_config: dict = read_config.get("KAFKA_CONSUMER", {})

    KAFKA_TOPICS: list[str] = consumer_config.get("KAFKA_TOPIC")
    config = {
        "bootstrap_servers": consumer_config.get("BOOTSTRAP_SERVICES"),
        "auto_offset_reset": consumer_config.get("auto_offset_reset"),
        "enable_auto_commit": consumer_config.get("enable_auto_commit"),
        "key_deserializer": lambda key: str(key.decode("utf-8")),
        "value_deserializer": lambda x: json.loads(x.decode("utf-8"), object_hook=lambda d: RandomUserSchema(data=d)),
        "group_id": "consumer.group.id.example",
    }

    # not wrapping KafkaConsumer as another OOP object for learning native Kafka-Python API
    consumer = KafkaConsumer(**config)
    consumer.subscribe(topics=KAFKA_TOPICS)

    print(f"Start consuming message from kafka: {KAFKA_TOPICS}")
    while True:
        try:
            # SIGINT can't be handled when polling, limit timeout to 1 second.
            message = consumer.poll(1.0)
            if message is None or message == {}:
                continue
            for message_key, message_value in message.items():
                for msg_val in message_value:
                    print(msg_val.key, msg_val.value)
        except KeyboardInterrupt:
            break  # force stop kafka
