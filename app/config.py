from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672"
    groq_api_key: str = ""
    google_api_key: str = ""
    postgres_url: str = "postgresql://postgres:postgres@localhost:5433/gridtrack_docker"
    mcp_api_key: str = ""
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_database: str = "gridtrack"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
