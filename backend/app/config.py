from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root, regardless of the working directory the app is started from.
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    app_name: str = "Bitcoin Forecasting App"
    environment: str = "development"

    # Optional: only needed if a future data source requires an API key.
    # SecretStr keeps the value out of logs, tracebacks, and repr() output.
    market_data_api_key: SecretStr | None = None

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")


settings = Settings()
