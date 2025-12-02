from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    # App
    PROJECT_NAME: str = "FastAPI Todo"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    
    # DB
    POSTGRES_HOST: str = "db"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "todo_db"
    POSTGRES_USER: str = "todo_user"
    POSTGRES_PASSWORD: str = "todo_pass"
    DATABASE_URL: str | None = None  

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    @property
    def sqlalchemy_database_uri(self) -> str:
        """
        Construct PostgreSQL connection string for SQLAlchemy.
        
        Returns:
            str: Database connection URI
        """
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


settings = Settings()