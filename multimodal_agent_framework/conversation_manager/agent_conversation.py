"""
This class is meant to hold the conversation context with respect to an agent.
This will allow us to continue the conversation and execution from where we left off.
The main issue in a network based environment is the disconnection of a conversation
and hence in those cases, we need a way to preserve the conversation context.
"""


class AgentConversation:
    def __init__(self, agent_name=None, chat_history=[], metadata=None):
        self._agent_name = agent_name
        self._chat_history = chat_history
        # Consolidated all agent state and other fields into metadata
        self._metadata = metadata or {}

    @property
    def chat_history(self):
        return self._chat_history

    @chat_history.setter
    def chat_history(self, value):
        self._chat_history = value

    @property
    def agent_name(self):
        return self._agent_name

    @agent_name.setter
    def agent_name(self, value):
        self._agent_name = value

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    @classmethod
    def from_json(cls, json_data):
        """
        Creates an instance of the class from a JSON dictionary.

        Args:
            json_data (dict): A dictionary containing the JSON data with keys:
                - "agent_name": The name of the agent.
                - "chat_history": The chat history of the agent.
                - "metadata": All other fields consolidated into metadata.

        Returns:
            An instance of the class with attributes populated from the JSON data.
        """
        # Handle backward compatibility by moving legacy fields to metadata
        metadata = json_data.get("metadata", {})

        return cls(
            agent_name=json_data.get("agent_name"),
            chat_history=json_data.get("chat_history", []),
            metadata=metadata,
        )

    def to_json(self):
        return {
            "agent_name": self._agent_name,
            "chat_history": self._chat_history,
            "metadata": self._metadata,
        }
