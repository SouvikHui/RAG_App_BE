from pydantic import BaseModel
from typing import List,Optional

class URLRequest(BaseModel):
    urls: List[str]

class QueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[str]]

class YT_URL(BaseModel):
    yt_url: str
 
class QueryResponse(BaseModel):
    answer: str
