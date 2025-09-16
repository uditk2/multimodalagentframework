from typing import Dict, List, Any
from multimodal_agent_framework.conversation_manager.agent_conversation import (
    AgentConversation,
)
from multimodal_agent_framework.conversation_manager.storage.base_storage import (
    BaseStorage,
)


class AgentConversationManager:
    """
    Manager for agent conversations with configurable storage backends.

    Accepts any storage implementation that follows the BaseStorage interface.
    """

    def __init__(self, storage: BaseStorage):
        """
        Initialize the conversation manager with a storage backend.

        Args:
            storage (BaseStorage): Storage implementation (e.g., S3Storage, FileStorage)
        """
        self._storage = storage

    def load_conversation(
        self, user_id: str, agent_name: str, chat_id: str
    ) -> AgentConversation:
        """Load a conversation using the configured storage backend."""
        return self._storage.load_conversation(user_id, agent_name, chat_id)

    def save_conversation(
        self,
        user_id: str,
        agent_name: str,
        agent_conversation: AgentConversation,
        chat_id: str,
    ) -> None:
        """Save a conversation using the configured storage backend."""
        self._storage.save_conversation(
            user_id, agent_name, agent_conversation, chat_id
        )

    def list_conversations(
        self,
        user_id: str,
        agent_name: str,
        chat_id_prefix: str = "",
        sort_by_update_time: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        List all conversation IDs for a given user and agent, optionally filtered by a chat_id prefix.

        Args:
            user_id (str): The user ID
            agent_name (str): The agent name
            chat_id_prefix (str, optional): Optional prefix to filter chat IDs. Defaults to "".
            sort_by_update_time (bool, optional): Whether to sort conversations by last update time. Defaults to False.

        Returns:
            List[Dict[str, Any]]: List of conversation details with chat_id and last_update_time
        """
        return self._storage.list_conversations(
            user_id, agent_name, chat_id_prefix, sort_by_update_time
        )

    def delete_conversation(self, user_id: str, agent_name: str, chat_id: str) -> bool:
        """Delete a conversation using the configured storage backend."""
        return self._storage.delete_conversation(user_id, agent_name, chat_id)
