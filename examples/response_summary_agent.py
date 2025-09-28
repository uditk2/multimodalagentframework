from multimodal_agent_framework import (
    MultiModalAgent,
    OpenAIConnector,
    get_openai_client,
)


class ResponseSummaryAgent:
    SYSTEM_PROMPT = """
    Please generate a summary of the provided information. 
    Try to be succinct and capture the essence of the information.
    For code blocks, only provide a point wise description of the code block.
    Always think step by step and provide a summary.

    """

    def __init__(self):
        self.agent = MultiModalAgent(
            name="ResponseSummary",
            system_prompt=self.SYSTEM_PROMPT,
            reviewer=None,
            connector=OpenAIConnector(get_openai_client()),
        )

    def generate_summary(self, response=None, chat_history=None):
        response, _ = self.agent.execute_user_ask(
            user_input=response, chat_history=chat_history, model="gpt-4o-mini"
        )
        return response


if __name__ == "__main__":
    agent = ResponseSummaryAgent()

    test_text = """
    This is a sample text that needs to be summarized. It contains multiple sentences
    and discusses various topics including technology, artificial intelligence, and
    software development. The text is meant to test the summarization capabilities
    of the agent and ensure that it can extract the key points effectively.
    """

    print("Testing Response Summary Agent...")
    print("Input text:", test_text)
    print("\nGenerating summary...")

    summary = agent.generate_summary(test_text)
    print("\nSummary:", summary)
