from pydantic import BaseModel, ConfigDict


class TopicInfo(BaseModel):
    """Describe a topic in a ROS bag."""

    model_config = ConfigDict(frozen=True)

    topic: str
    message_type: str
    num_messages: int
