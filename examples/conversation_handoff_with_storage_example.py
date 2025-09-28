"""
Example demonstrating conversation handoff between OpenAI and Claude agents with persistent storage.

This example shows how to:
1. Start a conversation with an OpenAI-powered MultiModalAgent
2. Save the conversation using AgentConversationManager with file storage
3. Load and continue the conversation with a Claude-powered MultiModalAgent
4. Persist the complete conversation history to files
"""

from multimodal_agent_framework import (
    MultiModalAgent,
    OpenAIConnector,
    ClaudeConnector,
    get_openai_client,
    get_claude_client,
)
from multimodal_agent_framework.conversation_manager.agent_conversation_manager import (
    AgentConversationManager,
)
from multimodal_agent_framework.conversation_manager.agent_conversation import (
    AgentConversation,
)
from multimodal_agent_framework.conversation_manager.storage.file_storage import (
    FileStorage,
)
from multimodal_agent_framework.conversation_manager.storage.s3_storage import S3Storage
import uuid
from datetime import datetime


def demonstrate_persistent_conversation_handoff():
    """
    Demonstrates conversation handoff with persistent storage between OpenAI and Claude agents.
    """
    print("=== Persistent Conversation Handoff Example ===\n")

    # Setup conversation manager with file storage
    storage = FileStorage(base_path="./conversation_handoff_storage")
    conversation_manager = AgentConversationManager(storage=storage)

    # Generate unique IDs for this conversation session
    user_id = "user_demo"
    chat_id = f"handoff_{uuid.uuid4().hex[:8]}"

    print(f"Starting conversation session: {chat_id}")
    print(f"Storage location: ./conversation_handoff_storage\n")

    # Initialize OpenAI agent for initial conversation
    openai_agent = MultiModalAgent(
        name="OpenAI_TechAdvisor",
        system_prompt="""You are a senior technical advisor. Provide detailed technical analysis
        and recommendations. Be thorough in your explanations and consider multiple factors.""",
        connector=OpenAIConnector(get_openai_client()),
    )

    # Step 1: Ask OpenAI agent a question and save the conversation
    print("Step 1: Asking OpenAI agent about cloud architecture...")
    question = """
    I'm designing a new SaaS application that needs to handle real-time notifications,
    file uploads, user authentication, and data analytics. What cloud architecture
    would you recommend using AWS services? Please provide specific service recommendations.
    """

    openai_response, chat_history = openai_agent.execute_user_ask(
        user_input=question, model="gpt-5-nano"
    )

    print(f"OpenAI Response:\n{openai_response[:300]}...\n")

    # Create and save the conversation after OpenAI's response
    conversation = AgentConversation(
        agent_name="OpenAI_TechAdvisor",
        chat_history=chat_history,
        metadata={
            "session_start": datetime.now().isoformat(),
            "current_agent": "OpenAI_TechAdvisor",
            "conversation_stage": "initial_consultation",
            "topic": "cloud_architecture",
        },
    )

    conversation_manager.save_conversation(
        user_id, "OpenAI_TechAdvisor", conversation, chat_id
    )
    print(
        f"✅ Saved conversation after OpenAI response ({len(chat_history)} messages)\n"
    )

    # Step 2: Add a follow-up question and update the stored conversation
    print("Step 2: Follow-up question to OpenAI...")
    followup = "What about cost optimization strategies for this architecture? Any specific tips for a startup budget?"

    openai_response2, updated_chat_history = openai_agent.execute_user_ask(
        user_input=followup, chat_history=chat_history, model="gpt-4o"
    )

    print(f"OpenAI Follow-up Response:\n{openai_response2[:300]}...\n")

    # Update the conversation with the new chat history
    conversation.chat_history = updated_chat_history
    conversation.metadata["last_openai_response"] = datetime.now().isoformat()
    conversation.metadata["conversation_stage"] = "detailed_discussion"

    conversation_manager.save_conversation(
        user_id, "OpenAI_TechAdvisor", conversation, chat_id
    )
    print(
        f"✅ Updated conversation after follow-up ({len(updated_chat_history)} messages)\n"
    )

    # Step 3: Load the conversation and hand it off to Claude
    print("Step 3: Loading conversation and handing off to Claude agent...")

    # Load the saved conversation
    loaded_conversation = conversation_manager.load_conversation(
        user_id, "OpenAI_TechAdvisor", chat_id
    )

    if loaded_conversation:
        print(
            f"✅ Loaded conversation with {len(loaded_conversation.chat_history)} messages"
        )
        print(f"   Original agent: {loaded_conversation.agent_name}")
        print(f"   Topic: {loaded_conversation.metadata.get('topic')}\n")
    else:
        print("❌ Failed to load conversation")
        return

    # Initialize Claude agent for review
    claude_agent = MultiModalAgent(
        name="Claude_Reviewer",
        system_prompt="""You are a critical architecture reviewer and security expert.
        Review technical recommendations with focus on security, scalability, and best practices.
        Provide constructive feedback and alternative suggestions when appropriate.""",
        connector=ClaudeConnector(get_claude_client()),
    )

    # Claude reviews the OpenAI conversation
    claude_review_prompt = """
    Please review the previous technical consultation about AWS cloud architecture.

    Provide your assessment on:
    1. The architectural recommendations - are they sound and scalable?
    2. Security considerations that may have been missed
    3. Alternative approaches or services that might be better
    4. Any red flags or concerns about the proposed solution
    5. Additional best practices for a SaaS application

    Be specific and provide actionable feedback.
    """

    claude_response, final_chat_history = claude_agent.execute_user_ask(
        user_input=claude_review_prompt,
        chat_history=loaded_conversation.chat_history,  # Continue from loaded history
        model="claude-3-5-sonnet-20241022",
    )

    print(f"Claude's Review:\n{claude_response[:400]}...\n")

    # Step 4: Save the complete conversation with Claude's review
    # Update the conversation to reflect Claude's involvement
    loaded_conversation.chat_history = final_chat_history
    loaded_conversation.agent_name = (
        "Multi_Agent_Session"  # Indicate multiple agents involved
    )
    loaded_conversation.metadata.update(
        {
            "claude_review_time": datetime.now().isoformat(),
            "conversation_stage": "expert_review_complete",
            "agents_involved": ["OpenAI_TechAdvisor", "Claude_Reviewer"],
            "final_message_count": len(final_chat_history),
        }
    )

    # Save the final conversation
    conversation_manager.save_conversation(
        user_id, "Multi_Agent_Session", loaded_conversation, chat_id
    )
    print(
        f"✅ Saved complete conversation with Claude's review ({len(final_chat_history)} messages)\n"
    )

    # Step 5: Demonstrate listing and retrieving conversations
    print("Step 5: Listing all conversations for this user...")

    # List conversations for OpenAI agent
    openai_conversations = conversation_manager.list_conversations(
        user_id, "OpenAI_TechAdvisor", sort_by_update_time=True
    )
    print(f"OpenAI agent conversations: {len(openai_conversations)}")

    # List conversations for multi-agent sessions
    multi_agent_conversations = conversation_manager.list_conversations(
        user_id, "Multi_Agent_Session", sort_by_update_time=True
    )
    print(f"Multi-agent conversations: {len(multi_agent_conversations)}")

    for conv in multi_agent_conversations:
        print(f"  - {conv['chat_id']} (updated: {conv['last_update_time']})")

    # Step 6: Show conversation summary
    print(f"\n=== Conversation Summary ===")
    print(f"Session ID: {chat_id}")
    print(f"Total messages: {len(final_chat_history)}")
    print(f"Agents involved: {loaded_conversation.metadata.get('agents_involved')}")
    print(f"Topic: {loaded_conversation.metadata.get('topic')}")

    # Show message breakdown by role
    role_counts = {}
    for msg in final_chat_history:
        role = msg.get("role", "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1

    print(f"Message breakdown: {role_counts}")

    return chat_id, final_chat_history


def demonstrate_conversation_continuation():
    """
    Demonstrate loading a saved conversation from file storage and continuing it with S3 storage.
    This shows migration between storage backends.
    """
    print("\n\n=== Conversation Continuation with Storage Migration Example ===\n")

    # Step 1: Setup file storage manager to load existing conversation
    file_storage = FileStorage(base_path="./conversation_handoff_storage")
    file_manager = AgentConversationManager(storage=file_storage)

    user_id = "user_demo"

    # List existing conversations from file storage
    print("Step 1: Loading conversation from file storage...")
    conversations = file_manager.list_conversations(
        user_id, "Multi_Agent_Session", sort_by_update_time=True
    )

    if not conversations:
        print("No existing conversations found. Run the handoff example first.")
        return

    # Load the most recent conversation from file storage
    latest_chat_id = conversations[0]["chat_id"]
    print(f"Loading conversation from file: {latest_chat_id}")

    loaded_conversation = file_manager.load_conversation(
        user_id, "Multi_Agent_Session", latest_chat_id
    )

    if not loaded_conversation:
        print("Failed to load conversation from file storage")
        return

    print(
        f"✅ Loaded conversation from file with {len(loaded_conversation.chat_history)} messages"
    )
    print(f"   Agents involved: {loaded_conversation.metadata.get('agents_involved')}")
    print(f"   Last stage: {loaded_conversation.metadata.get('conversation_stage')}\n")

    # Step 2: Setup S3 storage manager for continuation
    print("Step 2: Setting up S3 storage for continuation...")
    try:
        s3_storage = S3Storage()  # Uses environment variables for AWS credentials
        s3_manager = AgentConversationManager(storage=s3_storage)
        print("✅ S3 storage manager initialized")
    except Exception as e:
        print(f"❌ Failed to initialize S3 storage: {e}")
        print("Falling back to file storage for continuation...")
        s3_manager = file_manager

    # Step 3: Continue the conversation with a new question
    claude_agent = MultiModalAgent(
        name="Claude_Reviewer",
        system_prompt="""You are continuing a technical architecture review.
        Refer to the previous discussion and provide additional insights.""",
        connector=ClaudeConnector(get_claude_client()),
    )

    continuation_question = """
    Based on our previous discussion about the AWS architecture, could you provide
    a specific implementation roadmap? What should be the first 3 services to implement
    and in what order? Also, what are the estimated costs for each phase?
    """

    print("Step 3: Continuing conversation with new question...")
    response, updated_history = claude_agent.execute_user_ask(
        user_input=continuation_question,
        chat_history=loaded_conversation.chat_history,
        model="claude-3-5-sonnet-20241022",
    )

    print(f"Continuation Response:\n{response[:400]}...\n")

    # Step 4: Update metadata and save to S3 (or file if S3 unavailable)
    loaded_conversation.chat_history = updated_history
    loaded_conversation.metadata.update(
        {
            "continued_at": datetime.now().isoformat(),
            "conversation_stage": "implementation_roadmap",
            "storage_migration": (
                "file_to_s3" if s3_manager != file_manager else "file_only"
            ),
            "total_continuations": loaded_conversation.metadata.get(
                "total_continuations", 0
            )
            + 1,
            "continuation_topics": ["implementation_roadmap", "cost_estimation"],
        }
    )

    # Generate new chat_id for the continued conversation in S3
    continued_chat_id = f"continued_{latest_chat_id}_{uuid.uuid4().hex[:6]}"

    print("Step 4: Saving continued conversation...")
    if s3_manager != file_manager:
        print("   Saving to S3 storage...")
        s3_manager.save_conversation(
            user_id, "Multi_Agent_Session", loaded_conversation, continued_chat_id
        )
        print(
            f"✅ Saved continued conversation to S3 ({len(updated_history)} messages)"
        )

        # Also save to file storage as backup
        print("   Creating backup in file storage...")
        file_manager.save_conversation(
            user_id,
            "Multi_Agent_Session_Backup",
            loaded_conversation,
            continued_chat_id,
        )
        print("✅ Backup saved to file storage")
    else:
        print("   Saving to file storage...")
        file_manager.save_conversation(
            user_id, "Multi_Agent_Session", loaded_conversation, continued_chat_id
        )
        print(
            f"✅ Saved continued conversation to file ({len(updated_history)} messages)"
        )

    # Step 5: Verify conversation exists in target storage
    print("\nStep 5: Verifying saved conversation...")
    verification_conversation = s3_manager.load_conversation(
        user_id, "Multi_Agent_Session", continued_chat_id
    )

    if verification_conversation:
        print(
            f"✅ Verification successful - conversation has {len(verification_conversation.chat_history)} messages"
        )
        print(
            f"   Storage migration status: {verification_conversation.metadata.get('storage_migration')}"
        )
        print(
            f"   Total continuations: {verification_conversation.metadata.get('total_continuations')}"
        )
    else:
        print("❌ Verification failed - could not load continued conversation")

    # Step 6: List conversations in both storage systems for comparison
    print("\nStep 6: Storage comparison...")

    file_conversations = file_manager.list_conversations(user_id, "Multi_Agent_Session")
    print(f"File storage conversations: {len(file_conversations)}")
    for conv in file_conversations[-2:]:  # Show last 2
        print(f"  - {conv['chat_id']} (file)")

    if s3_manager != file_manager:
        try:
            s3_conversations = s3_manager.list_conversations(
                user_id, "Multi_Agent_Session"
            )
            print(f"S3 storage conversations: {len(s3_conversations)}")
            for conv in s3_conversations[-2:]:  # Show last 2
                print(f"  - {conv['chat_id']} (S3)")
        except Exception as e:
            print(f"Could not list S3 conversations: {e}")

    return continued_chat_id


if __name__ == "__main__":
    try:
        # Run the persistent conversation handoff example
        chat_id, chat_history = demonstrate_persistent_conversation_handoff()

        # Demonstrate continuing a saved conversation
        continued_chat_id = demonstrate_conversation_continuation()

        print(f"\n=== Final Summary ===")
        print(f"✅ Successfully demonstrated persistent conversation handoff")
        print(f"✅ Conversations saved to ./conversation_handoff_storage")
        print(f"✅ Latest session: {continued_chat_id or chat_id}")
        print("✅ All conversation history preserved and retrievable!")

    except Exception as e:
        print(f"Error running example: {e}")
        print("Make sure you have set your API keys in environment variables:")
        print("- OPENAI_API_KEY")
        print("- ANTHROPIC_API_KEY")
