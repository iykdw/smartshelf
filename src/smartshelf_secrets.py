from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    GOOGLE_BOOKS_API_KEY: SecretStr

    model_config = SettingsConfigDict(env_file=".env")
