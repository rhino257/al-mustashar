import logging
import json
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointTuple, empty_checkpoint
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer # Changed import
from supabase import AsyncClient, create_client # Using AsyncClient for async operations
from gotrue.types import User

logger = logging.getLogger(__name__)

class SupabaseSaver(BaseCheckpointSaver):
    """
    A LangGraph checkpoint saver that persists the agent's state to Supabase.

    This saver uses two tables: 'chats' for conversation metadata and 'chat_messages'
    for individual messages within a chat.
    """

    def __init__(self, supabase_client: AsyncClient): # Corrected AysncClient to AsyncClient
        super().__init__() # Added super init
        self.serializer = JsonPlusSerializer() # Initialize serializer
        self.supabase_client = supabase_client
        self.chat_table = self.supabase_client.table("chats")
        self.messages_table = self.supabase_client.table("chat_messages")

    async def _get(self, config: Dict[str, Any]) -> Optional[Checkpoint]: # Changed return type
        """
        Retrieves the latest checkpoint for a given thread_id from Supabase.
        """
        thread_id = config["configurable"]["thread_id"]
        user_id = config["configurable"].get("user_id") # Optional, but good for RLS

        logger.info(f"[SupabaseSaver] Attempting to load checkpoint for thread_id: {thread_id}")

        try:
            # 1. Fetch chat metadata to ensure chat_id exists and get latest update time
            chat_response = await self.chat_table.select("id, updated_at").eq("id", thread_id).single().execute()
            if not chat_response.data:
                logger.info(f"[SupabaseSaver] No chat found for thread_id: {thread_id}. Returning empty checkpoint.")
                return None # No existing chat, return empty

            # 2. Fetch messages for the chat_id, ordered by message_order
            messages_response = await self.messages_table.select("*")\
                .eq("chat_id", thread_id)\
                .order("message_order", ascending=True)\
                .execute()

            messages: List[BaseMessage] = []
            if messages_response.data:
                for msg_data in messages_response.data:
                    msg_type = msg_data.get('message_type')
                    content = msg_data.get('content', '')
                    metadata = msg_data.get('metadata', {})

                    if msg_type == 'human':
                        messages.append(HumanMessage(content=content, **metadata))
                    elif msg_type == 'ai':
                        messages.append(AIMessage(content=content, **metadata))
                    elif msg_type == 'system':
                        messages.append(SystemMessage(content=content, **metadata))
                    elif msg_type == 'tool':
                        messages.append(ToolMessage(content=content, **metadata))
                    else:
                        logger.warning(f"Unknown message type '{msg_type}' for message ID {msg_data.get('id')}. Skipping.")
                logger.info(f"[SupabaseSaver] Loaded {len(messages)} messages for thread_id: {thread_id}")
            else:
                logger.info(f"[SupabaseSaver] No messages found for thread_id: {thread_id}.")

            # Construct the checkpoint state.
            checkpoint_data = {
                "v": 1, # Checkpoint version
                "ts": chat_response.data["updated_at"], # Use chat's updated_at as timestamp
                "id": str(UUID(thread_id)), # Checkpoint ID is thread_id
                "channel_values": {
                    "messages": self.serializer.dumps(messages) # Serialize messages
                },
                "channel_versions": {}, # Not strictly needed for simple channels
                "metadata": {},
                "parent_ts": None, # Not tracking parent checkpoints for now
                "parent_id": None,
            }
            
            # The checkpoint object itself is what LangGraph expects
            checkpoint = Checkpoint(
                v=checkpoint_data["v"],
                ts=checkpoint_data["ts"],
                id=checkpoint_data["id"],
                channel_values={k: self.serializer.loads(v) for k, v in checkpoint_data["channel_values"].items()}, # Use serializer
                channel_versions=checkpoint_data["channel_versions"],
                metadata=checkpoint_data["metadata"],
                parent_ts=checkpoint_data["parent_ts"],
                parent_id=checkpoint_data["parent_id"],
            )

            return checkpoint # Return Checkpoint directly

        except Exception as e:
            logger.error(f"[SupabaseSaver] Error loading checkpoint for thread_id {thread_id}: {e}", exc_info=True)
            return None # Return None on error

    async def _put(self, config: Dict[str, Any], checkpoint: Checkpoint) -> None:
        """
        Persists the current checkpoint (state) to Supabase.
        """
        thread_id = config["configurable"]["thread_id"]
        user_id = config["configurable"].get("user_id")

        logger.info(f"[SupabaseSaver] Persisting checkpoint for thread_id: {thread_id}")

        try:
            # 1. Ensure the chat exists or create it
            chat_data = {
                "id": thread_id,
                "user_id": user_id,
                "updated_at": checkpoint.ts 
            }
            await self.chat_table.upsert(chat_data, on_conflict="id").execute()
            logger.info(f"[SupabaseSaver] Upserted chat entry for thread_id: {thread_id}")

            # 2. Extract messages from the checkpoint
            messages_json_bytes = checkpoint.channel_values.get("messages") # This will be bytes
            if messages_json_bytes:
                # Assuming messages_json_bytes is the direct output from serializer.dumps()
                # which should be bytes if it's from JsonPlusSerializer.dumps
                current_messages: List[BaseMessage] = self.serializer.loads(messages_json_bytes)
            else:
                current_messages = []

            # 3. Fetch existing messages to identify new ones
            existing_messages_response = await self.messages_table.select("id, message_order")\
                .eq("chat_id", thread_id)\
                .order("message_order", ascending=True)\
                .execute()
            
            existing_message_orders = {msg['message_order'] for msg in existing_messages_response.data}

            messages_to_insert = []
            for i, msg in enumerate(current_messages):
                if i not in existing_message_orders:
                    msg_data = {
                        "chat_id": thread_id,
                        "user_id": user_id,
                        "message_type": msg.type, 
                        "content": msg.content,
                        "metadata": msg.additional_kwargs, 
                        "timestamp": checkpoint.ts, 
                        "message_order": i, 
                    }
                    messages_to_insert.append(msg_data)
            
            if messages_to_insert:
                await self.messages_table.insert(messages_to_insert).execute()
                logger.info(f"[SupabaseSaver] Inserted {len(messages_to_insert)} new messages for thread_id: {thread_id}")
            else:
                logger.info(f"[SupabaseSaver] No new messages to insert for thread_id: {thread_id}")

        except Exception as e:
            logger.error(f"[SupabaseSaver] Error persisting checkpoint for thread_id {thread_id}: {e}", exc_info=True)

    async def aget(self, config: Dict[str, Any]) -> Optional[Checkpoint]: # Changed return type
        return await self._get(config)

    async def aput(self, config: Dict[str, Any], checkpoint: Checkpoint) -> None:
        await self._put(config, checkpoint)

    async def aget_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """
        Retrieves the checkpoint and constructs a CheckpointTuple.
        This method is called by the Pregel execution engine.
        """
        checkpoint = await self.aget(config) # Call our existing aget
        if checkpoint is None:
            return None
        # Construct and return CheckpointTuple.
        # For parent_checkpoint, we are not currently tracking parent-child relationships
        # in a way that would populate it here from a simple 'aget' call.
        # If the graph requires parent checkpoint info, this might need more sophisticated loading.
        return CheckpointTuple(config=config, checkpoint=checkpoint, parent_checkpoint=None)
