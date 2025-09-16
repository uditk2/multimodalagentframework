"""
Local file storage implementation for conversation persistence.
"""

import os
import pandas as pd
import json
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from multimodal_agent_framework.conversation_manager.storage.base_storage import (
    BaseStorage,
)
from multimodal_agent_framework.conversation_manager.agent_conversation import (
    AgentConversation,
)
from multimodal_agent_framework.logging_config import get_logger

logger = get_logger()


class FileStorage(BaseStorage):
    """
    Local file storage implementation for conversation persistence.
    """

    def __init__(self, base_path: str = None):
        """
        Initialize file storage.

        Args:
            base_path (str): Base directory path for storing conversations.
                           If None, uses './conversations' in current directory.
        """
        self._base_path = Path(base_path) if base_path else Path("./conversations")

        # Create base directory if it doesn't exist
        self._base_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"File storage initialized at: {self._base_path.absolute()}")

    def _get_file_path(self, user_id: str, agent_name: str, chat_id: str) -> Path:
        """Get the file path for a conversation file."""
        dir_path = self._base_path / agent_name / user_id
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / f"{chat_id}.parquet"

    def save_conversation(
        self,
        user_id: str,
        agent_name: str,
        agent_conversation: AgentConversation,
        chat_id: str,
    ) -> None:
        """
        Save a conversation to local file.
        """
        try:
            logger.debug(
                f"Saving conversation for user {user_id} with agent {agent_name}"
            )
            file_path = self._get_file_path(
                user_id=user_id, agent_name=agent_name, chat_id=chat_id
            )

            # Convert conversation to JSON and prepare for parquet storage
            conversation_dict = agent_conversation.to_json()
            conversation_dict["chat_history"] = json.dumps(
                conversation_dict["chat_history"]
            )
            conversation_dict["metadata"] = json.dumps(conversation_dict["metadata"])

            # Create DataFrame and save to parquet
            conversation_data = pd.DataFrame([conversation_dict])
            conversation_data.to_parquet(file_path, index=False)

        except Exception as e:
            logger.error(f"Error saving conversation: {traceback.format_exc()}")
            raise ValueError("Failed to save conversation. Please try again.")

    def load_conversation(
        self, user_id: str, agent_name: str, chat_id: str
    ) -> Optional[AgentConversation]:
        """
        Load a conversation from local file.
        """
        if chat_id is None:
            raise ValueError("chat_id is required to load conversation")
        if user_id is None:
            raise ValueError("user_id is required to load conversation")

        try:
            file_path = self._get_file_path(
                user_id=user_id, agent_name=agent_name, chat_id=chat_id
            )

            # Check if file exists
            if not file_path.exists():
                return None

            # Read the parquet file
            conversation_data = pd.read_parquet(file_path)
            conversation_dict = conversation_data.iloc[0].to_dict()

            # Convert JSON strings back to objects
            conversation_dict["chat_history"] = json.loads(
                conversation_dict["chat_history"]
            )
            if (
                "metadata" in conversation_dict
                and conversation_dict["metadata"] is not None
            ):
                try:
                    conversation_dict["metadata"] = json.loads(
                        conversation_dict["metadata"]
                    )
                except json.JSONDecodeError:
                    conversation_dict["metadata"] = {}

            # Create and return AgentConversation object
            agent_conversation = AgentConversation.from_json(conversation_dict)
            return agent_conversation

        except Exception as e:
            logger.error(f"Error retrieving chat history: {str(e)}")
            raise ValueError("Failed to load conversation. Please try again.")

    def list_conversations(
        self,
        user_id: str,
        agent_name: str,
        chat_id_prefix: str = "",
        sort_by_update_time: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        List all conversation IDs for a given user and agent.
        """
        try:
            dir_path = self._base_path / agent_name / user_id

            # Check if directory exists
            if not dir_path.exists():
                return []

            conversations = []
            for file_path in dir_path.glob("*.parquet"):
                chat_id = file_path.stem  # filename without extension
                if chat_id.startswith(chat_id_prefix):
                    # Get file modification time
                    stat = file_path.stat()
                    last_update_time = datetime.fromtimestamp(stat.st_mtime)

                    conversations.append(
                        {"chat_id": chat_id, "last_update_time": last_update_time}
                    )

            # Sort by last update time if requested
            if sort_by_update_time:
                conversations.sort(key=lambda x: x["last_update_time"], reverse=True)

            return conversations

        except Exception as e:
            logger.error(f"Error listing conversations: {str(e)}")
            raise ValueError(f"Failed to list conversations: {str(e)}")

    def delete_conversation(self, user_id: str, agent_name: str, chat_id: str) -> bool:
        """
        Delete a conversation from local file system.
        """
        try:
            file_path = self._get_file_path(
                user_id=user_id, agent_name=agent_name, chat_id=chat_id
            )

            # Check if file exists
            if not file_path.exists():
                return False

            # Delete the file
            file_path.unlink()

            # Clean up empty directories
            try:
                # Remove user directory if empty
                user_dir = file_path.parent
                if user_dir.is_dir() and not any(user_dir.iterdir()):
                    user_dir.rmdir()

                    # Remove agent directory if empty
                    agent_dir = user_dir.parent
                    if agent_dir.is_dir() and not any(agent_dir.iterdir()):
                        agent_dir.rmdir()
            except OSError:
                # Directory not empty, which is fine
                pass

            return True

        except Exception as e:
            logger.error(f"Error deleting conversation: {str(e)}")
            raise ValueError("Failed to delete conversation. Please try again.")
