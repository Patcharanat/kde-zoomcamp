"""
Utilities used in Python-Kafka POC
1. DataGeneratorAPI
    - Request Random User Data from open API
2. ConfigurationReader
    - used to read configuration file as json/yaml format
"""

import requests
import yaml
import json
from typing import Any

from schema import RandomUserSchema


class DataGeneratorAPI:
    """
    Random User API check the link for more info: https://randomuser.me/documentation#intro
    For simple use case, please check these parameters:

    :param results: number of records returned
    :type results: int
    :param seed: can be string or sequence of characters to generate the same set of users (limit randomness)
    :type seed: str | int
    :param format: returned format [json, csv, yaml, xml] (default json)
    :type format: str

    Example:
        data = DataGenerator(results=10).get_data()
    """

    def __init__(
        self,
        results: int = 10,
        format: str | None = "json",
        seed: int | str | None = None,
    ) -> None:
        self.base_url = "https://randomuser.me/api/"
        self.results = results
        self.format = format
        self.seed = seed

    def formulate_url(self):
        """
        Formulate URL according to query parameters
        """
        if self.seed:
            self.url = (
                self.base_url
                + f"?results={self.results}&format={self.format}&seed={self.seed}"
            )
        else:
            self.url = self.base_url + f"?results={self.results}&format={self.format}"

    @staticmethod
    def request_batch(url: str) -> requests.Response:
        """
        Get response from REST API

        :param url: ready URL for request
        :type url: str

        Return: response

        """
        response = requests.get(url)
        return response

    def get_data(self) -> list[dict[str, Any]]:
        """
        Class Entrypoint, Getting data from API response
        """
        self.formulate_url()
        response = self.request_batch(url=self.url)
        return response.json()["results"]

    def get_data_with_schema(self) -> list[RandomUserSchema]:
        """
        Alternative Class Entrypoint, Getting data from API response with pre-defined schema
        """
        self.formulate_url()
        response: requests.Response = self.request_batch(url=self.url)
        data: list = response.json()["results"]

        records: list = []
        for record in data:
            records.append(RandomUserSchema(data=record))
        return records


class ConfigurationReader:
    """
    Configurations File Reader depends on file extension
    :params config_path: relative path to configuration file
    :type config_path: str
        Example:
            config_path = "./path/to/config_file.extension"

    Class Usage Example:
        config = ConfigurationReader(config_path="...").get_config()
    """

    def __init__(
        self,
        config_path: str,
    ) -> None:
        self.config_path = config_path

    @staticmethod
    def identify_extension(config_path: str) -> str:
        """
        identify extension to choose the right method to read
        Return: str of file extension
        """
        try:
            return config_path.split(".")[-1]
        except:
            raise ValueError("Invalid `config_path` parameter")

    def read_configuration(self) -> dict:
        """
        Congiurations Reader Written in factory pattern to choose reading method
        """
        extension: str = self.identify_extension(self.config_path)

        read_methods: dict = {
            "json": lambda x: json.load(x),
            "yaml": lambda x: yaml.safe_load(x),
            "yml": lambda x: yaml.safe_load(x),
        }

        with open(self.config_path, "r") as config_file:
            content: dict = read_methods.get(extension)(config_file)
            return content

    def get_config(self) -> dict:
        """
        Entrypoint, return configuration file's content as a dict
        """
        return self.read_configuration()
