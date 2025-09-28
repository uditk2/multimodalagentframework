"""
Example demonstrating conversation handoff between OpenAI and Claude agents.

This example shows how to:
1. Start a conversation with an OpenAI-powered MultiModalAgent
2. Take the chat history from that conversation
3. Pass it to a Claude-powered MultiModalAgent for review/opinion
"""

from multimodal_agent_framework import (
    MultiModalAgent,
    OpenAIConnector,
    ClaudeConnector,
    get_openai_client,
    get_claude_client,
)


def demonstrate_conversation_handoff():
    """
    Demonstrates passing a conversation from OpenAI agent to Claude agent.
    """
    print("=== Conversation Handoff Example ===\n")

    # Initialize OpenAI agent for initial conversation
    openai_agent = MultiModalAgent(
        name="OpenAI_Analyst",
        system_prompt="""You are a technical analyst. Analyze problems methodically and provide
        detailed technical explanations. Be thorough in your reasoning.""",
        connector=OpenAIConnector(get_openai_client()),
    )

    # Initialize Claude agent for review/opinion
    claude_agent = MultiModalAgent(
        name="Claude_Reviewer",
        system_prompt="""You are a critical reviewer and second opinion provider. Review the
        previous conversation and provide your perspective, critique, or alternative viewpoint.
        Be constructive but don't hesitate to disagree if you have a different opinion.""",
        connector=ClaudeConnector(get_claude_client()),
    )

    # Step 1: Ask OpenAI agent a technical question
    print("Step 1: Asking OpenAI agent about microservices vs monolith architecture...")
    question = """
    I'm building a new e-commerce platform and trying to decide between microservices
    and monolithic architecture. The team has 5 developers, we expect moderate traffic
    initially but hope to scale significantly. What would you recommend and why?
    """

    openai_response, openai_chat_history = openai_agent.execute_user_ask(
        user_input=question, model="gpt-5-nano"
    )

    print(f"OpenAI Response:\n{openai_response}\n")
    print(f"Chat history length: {len(openai_chat_history)} messages\n")

    # Step 2: Continue the conversation with OpenAI
    print("Step 2: Follow-up question to OpenAI agent...")
    followup = "What about the deployment and monitoring complexity differences?"

    openai_response2, updated_chat_history = openai_agent.execute_user_ask(
        user_input=followup, chat_history=openai_chat_history, model="gpt-5-nano"
    )

    print(f"OpenAI Follow-up Response:\n{openai_response2}\n")
    print(f"Updated chat history length: {len(updated_chat_history)} messages\n")

    # Step 3: Pass the entire conversation to Claude for review
    print("Step 3: Passing conversation to Claude for review...")
    claude_review_prompt = f"""
    Please review the previous conversation about microservices vs monolithic architecture.
    The conversation was between a user asking for architectural advice and an AI analyst.

    Please provide your opinion on:
    1. The quality and accuracy of the technical advice given
    2. Any important considerations that might have been missed
    3. Whether you agree or disagree with the recommendations and why
    4. Any additional insights you'd like to add

    Be honest in your assessment and provide your own perspective.
    """

    claude_response, final_chat_history = claude_agent.execute_user_ask(
        user_input=claude_review_prompt,
        chat_history=updated_chat_history,  # Pass the OpenAI conversation history
        model="claude-3-5-sonnet-20241022",
    )

    print(f"Claude's Review:\n{claude_response}\n")
    print(f"Final chat history length: {len(final_chat_history)} messages")

    # Step 4: Show the complete conversation flow
    print("\n=== Complete Conversation Flow ===")
    for i, message in enumerate(final_chat_history):
        role = message.get("role", "unknown")
        content = message.get("content", "")
        if isinstance(content, list):
            # Handle multimodal content
            text_content = ""
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_content += item.get("text", "")
            content = text_content

        print(f"\n[{i+1}] {role.upper()}:")
        print(f"{content[:200]}{'...' if len(content) > 200 else ''}")

    return final_chat_history


def demonstrate_multi_agent_discussion():
    """
    Extended example: Multiple rounds of discussion between OpenAI and Claude agents.
    """
    print("\n\n=== Multi-Agent Discussion Example ===\n")

    # Create agents with distinct personalities
    openai_agent = MultiModalAgent(
        name="OpenAI_Optimist",
        system_prompt="""You are an optimistic technology enthusiast. You tend to focus on
        the benefits and potential of new technologies. Be encouraging and highlight opportunities.""",
        connector=OpenAIConnector(get_openai_client()),
    )

    claude_agent = MultiModalAgent(
        name="Claude_Skeptic",
        system_prompt="""You are a thoughtful skeptic. You consider risks, challenges, and
        potential downsides. You're not negative, but you ensure all perspectives are considered.""",
        connector=ClaudeConnector(get_claude_client()),
    )

    # Start discussion
    topic = "Should our startup adopt AI-driven code generation tools for our development process?"

    print(f"Discussion Topic: {topic}\n")

    # Round 1: OpenAI's perspective
    openai_response, chat_history = openai_agent.execute_user_ask(
        user_input=topic, model="gpt-4o"
    )
    print(f"OpenAI (Optimist) says:\n{openai_response}\n")

    # Round 1: Claude's counter-perspective
    claude_prompt = (
        "Please respond to the previous argument. What are your thoughts and concerns?"
    )
    claude_response, chat_history = claude_agent.execute_user_ask(
        user_input=claude_prompt,
        chat_history=chat_history,
        model="claude-3-5-sonnet-20241022",
    )
    print(f"Claude (Skeptic) responds:\n{claude_response}\n")

    # Round 2: OpenAI addresses Claude's concerns
    openai_counter = "Please address the concerns raised and provide counter-arguments."
    openai_response2, chat_history = openai_agent.execute_user_ask(
        user_input=openai_counter, chat_history=chat_history, model="gpt-5-nano"
    )
    print(f"OpenAI counters:\n{openai_response2}\n")

    # Final synthesis
    synthesis_prompt = """Based on this discussion, please provide a balanced final
    recommendation that takes both perspectives into account."""

    final_response, final_history = claude_agent.execute_user_ask(
        user_input=synthesis_prompt,
        chat_history=chat_history,
        model="claude-3-5-sonnet-latest",
    )
    print(f"Final Synthesis:\n{final_response}\n")

    return final_history


if __name__ == "__main__":
    try:
        # Basic conversation handoff example
        chat_history1 = demonstrate_conversation_handoff()

        # Multi-agent discussion example
        chat_history2 = demonstrate_multi_agent_discussion()

        print(f"\n=== Summary ===")
        print(f"First example generated {len(chat_history1)} total messages")
        print(f"Second example generated {len(chat_history2)} total messages")
        print(
            "Both examples demonstrate successful conversation handoff between different AI providers!"
        )

    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set your API keys in environment variables:")
        print("- OPENAI_API_KEY")
        print("- ANTHROPIC_API_KEY")
