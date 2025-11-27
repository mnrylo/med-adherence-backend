# app/config.py
import os

from pydantic_settings import BaseSettings
from pydantic import AnyUrl


class Settings(BaseSettings):
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "med_adherence"

    class Config:
        env_file = ".env"


settings = Settings()

