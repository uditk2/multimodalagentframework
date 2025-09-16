"""
Abstract base class for conversation storage implementations.
This provides a generic interface for different storage backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from multimodal_agent_framework.conversation_manager.agent_conversation import (
    AgentConversation,
)


class BaseStorage(ABC):
    """
    Abstract base class for conversation storage backends.
    """

    @abstractmethod
    def save_conversation(
        self,
        user_id: str,
        agent_name: str,
        agent_conversation: AgentConversation,
        chat_id: str,
    ) -> None:
        """
        Save a conversation to storage.

        Args:
            user_id (str): The user ID
            agent_name (str): The agent name
            agent_conversation (AgentConversation): The conversation object to save
            chat_id (str): The chat ID

        Raises:
            ValueError: If saving fails
        """
        pass

    @abstractmethod
    def load_conversation(
        self, user_id: str, agent_name: str, chat_id: str
    ) -> Optional[AgentConversation]:
        """
        Load a conversation from storage.

        Args:
            user_id (str): The user ID
            agent_name (str): The agent name
            chat_id (str): The chat ID

        Returns:
            AgentConversation: The conversation object, or None if not found

        Raises:
            ValueError: If loading fails
        """
        pass

    @abstractmethod
    def list_conversations(
        self,
        user_id: str,
        agent_name: str,
        chat_id_prefix: str = "",
        sort_by_update_time: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        List all conversation IDs for a given user and agent.

        Args:
            user_id (str): The user ID
            agent_name (str): The agent name
            chat_id_prefix (str, optional): Optional prefix to filter chat IDs
            sort_by_update_time (bool, optional): Whether to sort by last update time

        Returns:
            List[Dict[str, Any]]: List of conversation details with chat_id and last_update_time

        Raises:
            ValueError: If listing fails
        """
        pass

    @abstractmethod
    def delete_conversation(self, user_id: str, agent_name: str, chat_id: str) -> bool:
        """
        Delete a conversation from storage.

        Args:
            user_id (str): The user ID
            agent_name (str): The agent name
            chat_id (str): The chat ID

        Returns:
            bool: True if deleted successfully, False if not found

        Raises:
            ValueError: If deletion fails
        """
        pass
