from pydantic import BaseModel


class TagCreate(BaseModel):
    name: str


class TagOut(BaseModel):
    id: int
    name: str

    model_config = {"from_attributes": True}
