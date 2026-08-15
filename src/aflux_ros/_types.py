from pydantic import BaseModel, ConfigDict


class TopicInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    topic: str
    message_type: str
    num_messages: int
