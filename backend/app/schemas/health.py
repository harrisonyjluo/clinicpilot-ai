"""Health endpoint response model."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    database: str = Field(examples=["ok"])
    app: str
    version: str
