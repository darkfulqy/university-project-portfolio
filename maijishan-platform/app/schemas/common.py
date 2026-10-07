from typing import Generic, Optional, TypeVar
from pydantic import BaseModel


T = TypeVar("T")


class ErrorInfo(BaseModel):
    code: str
    message: str


class ApiResponse(BaseModel, Generic[T]):
    data: Optional[T]
    error: Optional[ErrorInfo]
    request_id: str
