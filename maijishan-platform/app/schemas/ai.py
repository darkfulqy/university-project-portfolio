from pydantic import BaseModel, Field


class AiChatRequest(BaseModel):
    user_message: str
    context_ids: list[int] | None = None


class AiSearchRequest(BaseModel):
    query: str
    top_k: int | None = Field(default=None, ge=1, le=10)


class CitationItem(BaseModel):
    source: str
    text: str
    score: float
    images: list[str] | None = None


class AiSearchResponse(BaseModel):
    citations: list[CitationItem] = Field(default_factory=list)
    context_parts: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)


class AiChatResponse(BaseModel):
    assistant_message: str
    citations: list[CitationItem] = Field(default_factory=list)
    context_parts: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    used_tools: bool = True
