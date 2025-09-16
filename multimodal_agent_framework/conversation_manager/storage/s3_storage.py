"""
AWS S3 storage implementation for conversation persistence.
"""

import boto3
import pandas as pd
import io
import os
import json
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

from multimodal_agent_framework.conversation_manager.storage.base_storage import (
    BaseStorage,
)
from multimodal_agent_framework.conversation_manager.agent_conversation import (
    AgentConversation,
)
from multimodal_agent_framework.logging_config import get_logger

logger = get_logger()
load_dotenv()


class S3Storage(BaseStorage):
    """
    AWS S3 storage implementation for conversation persistence.
    """

    def __init__(self, bucket_name: str = None, conversations_folder: str = None):
        """
        Initialize S3 storage.

        Args:
            bucket_name (str): S3 bucket name. If None, reads from AGENT_CONVERSATIONS_BUCKET env var.
            conversations_folder (str): Folder path in S3. If None, reads from AGENT_CONVERSATIONS_FOLDER env var.
        """
        try:
            self._s3_client = boto3.client(
                "s3",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )

            self._bucket_name = bucket_name or os.getenv("AGENT_CONVERSATIONS_BUCKET")
            if not self._bucket_name:
                raise ValueError(
                    "S3 bucket name not provided and AGENT_CONVERSATIONS_BUCKET not set in environment variables"
                )

            self._agent_conversations_folder = conversations_folder or os.getenv(
                "AGENT_CONVERSATIONS_FOLDER"
            )
            if not self._agent_conversations_folder:
                raise ValueError(
                    "S3 conversations folder not provided and AGENT_CONVERSATIONS_FOLDER not set in environment variables"
                )

            # Verify S3 connection
            self._s3_client.head_bucket(Bucket=self._bucket_name)
            logger.debug("Successfully connected to S3")

        except Exception as e:
            logger.error(f"Failed to initialize S3 connection: {str(e)}")
            raise

    def _get_file_path(self, user_id: str, agent_name: str, chat_id: str) -> str:
        """Get the S3 key path for a conversation file."""
        return f"{self._agent_conversations_folder}/{agent_name}/{user_id}/{chat_id}.parquet"

    def save_conversation(
        self,
        user_id: str,
        agent_name: str,
        agent_conversation: AgentConversation,
        chat_id: str,
    ) -> None:
        """
        Save a conversation to S3.
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
            buffer = io.BytesIO()
            conversation_data.to_parquet(buffer, index=False)
            buffer.seek(0)

            # Upload to S3
            self._s3_client.put_object(
                Bucket=self._bucket_name, Key=file_path, Body=buffer
            )

        except Exception as e:
            logger.error(f"Error saving conversation: {traceback.format_exc()}")
            raise ValueError("Failed to save conversation. Please try again.")

    def load_conversation(
        self, user_id: str, agent_name: str, chat_id: str
    ) -> Optional[AgentConversation]:
        """
        Load a conversation from S3.
        """
        if chat_id is None:
            raise ValueError("chat_id is required to load conversation")
        if user_id is None:
            raise ValueError("user_id is required to load conversation")

        try:
            file_path = self._get_file_path(
                user_id=user_id, agent_name=agent_name, chat_id=chat_id
            )
            response = self._s3_client.get_object(
                Bucket=self._bucket_name, Key=file_path
            )

            # Read the parquet from S3
            conversation_data = pd.read_parquet(io.BytesIO(response["Body"].read()))
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

        except self._s3_client.exceptions.NoSuchKey:
            return None
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
            prefix = f"{self._agent_conversations_folder}/{agent_name}/{user_id}/"

            # List objects in the bucket with the specified prefix
            paginator = self._s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self._bucket_name, Prefix=prefix)

            conversations = []
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        # Extract chat_id from the key
                        key = obj["Key"]
                        file_name = key.split("/")[-1]
                        if file_name.endswith(".parquet"):
                            chat_id = file_name[:-8]  # Remove '.parquet' extension
                            if chat_id.startswith(chat_id_prefix):
                                last_update_time = obj["LastModified"]
                                conversations.append(
                                    {
                                        "chat_id": chat_id,
                                        "last_update_time": last_update_time,
                                    }
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
        Delete a conversation from S3.
        """
        try:
            file_path = self._get_file_path(
                user_id=user_id, agent_name=agent_name, chat_id=chat_id
            )

            # Check if object exists
            try:
                self._s3_client.head_object(Bucket=self._bucket_name, Key=file_path)
            except self._s3_client.exceptions.NoSuchKey:
                return False

            # Delete the object
            self._s3_client.delete_object(Bucket=self._bucket_name, Key=file_path)
            return True

        except Exception as e:
            logger.error(f"Error deleting conversation: {str(e)}")
            raise ValueError("Failed to delete conversation. Please try again.")
