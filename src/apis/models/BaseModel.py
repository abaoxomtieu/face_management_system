from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from src.utils.logger import get_date_time


class BaseDocument(BaseModel):
    created_at: Optional[datetime] = Field(
        default_factory=lambda: get_date_time().replace(tzinfo=None)
    )
    updated_at: Optional[datetime] = Field(
        default_factory=lambda: get_date_time().replace(tzinfo=None)
    )
    expire_at: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True

