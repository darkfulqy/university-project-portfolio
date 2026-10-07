from datetime import datetime
from pydantic import BaseModel


class UploadOut(BaseModel):
    id: int
    filename: str
    file_hash: str
    size: int
    mime_type: str
    path: str
    created_at: datetime

    model_config = {"from_attributes": True}
