"""
Schema for Kafka POC (DE Zoomcamp)
"""
import datetime
from typing import Any


class RandomUserSchema:
    
    def __init__(self, data: dict[str, Any]) -> None:
        self.id: str = "-".join([data.get("id").get("name"), data.get("id").get("value")])
        self.name_title: str = data.get("name").get("title")
        self.name_first: str = data.get("name").get("first")
        self.name_last: str = data.get("name").get("last")
        self.gender: str = data.get("gender")
        self.date_of_birth: datetime = data.get("dob").get("date")
        self.user_age_year: int = data.get("dob").get("age")
        self.email: str = data.get("email")
        self.phone: str = data.get("phone")
        self.picture: str = data.get("picture").get("large")
        self.nationality: str = data.get("nat")
        self.location_city: str = data.get("location").get("city")
        self.location_state: str = data.get("location").get("state")
        self.location_country: str = data.get("location").get("country")
        self.location_postcode: str = data.get("location").get("postcode")
        self.location_coordinates_latitude: float = data.get("location").get("coordinates").get("latitude")
        self.location_coordinates_longitude: float = data.get("location").get("coordinates").get("longitude")
        self.login_uuid: str = data.get("login").get("uuid")
        self.login_username: str = data.get("login").get("username")
        self.login_password: str = data.get("login").get("password")
        self.registered_date: datetime = data.get("registered").get("date")
        self.registered_age_year: int = data.get("registered").get("age")

    # @classmethod
    # def from_dict(cls, data: dict[str, Any]) -> None:
    #     return cls(
    #         id = "-".join([data.get("id").get("name"), data.get("id").get("value")]),
    #         name_title = data.get("name").get("title"),
    #         name_first = data.get("name").get("first"),
    #         name_last = data.get("name").get("last"),
    #         gender = data.get("gender"),
    #         date_of_birth = data.get("dob").get("date"),
    #         user_age_year = data.get("dob").get("age"),
    #         email = data.get("email"),
    #         phone = data.get("phone"),
    #         picture = data.get("picture").get("large"),
    #         nationality = data.get("nat"),
    #         location_city = data.get("location").get("city"),
    #         location_state = data.get("location").get("state"),
    #         location_country = data.get("location").get("country"),
    #         location_postcode = data.get("location").get("postcode"),
    #         location_coordinates_latitude = data.get("location").get("coordinates").get("latitude"),
    #         location_coordinates_longitude = data.get("location").get("coordinates").get("longitude"),
    #         login_uuid = data.get("login").get("uuid"),
    #         login_username = data.get("login").get("username"),
    #         login_password = data.get("login").get("password"),
    #         registered_date = data.get("registered").get("date"),
    #         registered_age_year = data.get("registered").get("age")        
    #     )

# schema = {
#     "id": "-".join([data.get("id").get("name"), data.get("id").get("value")]),
#     "name_title": data.get("name").get("title"),
#     "name_first": data.get("name").get("first"),
#     "name_last": data.get("name").get("last"),
#     "gender": data.get("gender"),
#     "date_of_birth": data.get("dob").get("date"),
#     "user_age_year": data.get("dob").get("age"),
#     "email": data.get("email"),
#     "phone": data.get("phone"),
#     "picture": data.get("picture").get("large"),
#     "nationality": data.get("nat"),
#     "location_city": data.get("location").get("city"),
#     "location_state": data.get("location").get("state"),
#     "location_country": data.get("location").get("country"),
#     "location_postcode": data.get("location").get("postcode"),
#     "location_coordinates_latitude": data.get("location").get("coordinates").get("latitude"),
#     "location_coordinates_longitude": data.get("location").get("coordinates").get("longitude"),
#     "login_uuid": data.get("login").get("uuid"),
#     "login_username": data.get("login").get("username"),
#     "login_password": data.get("login").get("password"),
#     "registered_date": data.get("registered").get("date"),
#     "registered_age_year": data.get("registered").get("age")
# }