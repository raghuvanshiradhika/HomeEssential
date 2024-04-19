import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
# Load environment variables from .env file
load_dotenv()


class OpenAIChat:
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-3.5-turbo",
            temperature=0,
        )

    def submit_prompt(self, question) -> str:
        chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "You are a helpful assistant that re-writes the user's sentence with correct grammar"
                    )
                ),
                HumanMessagePromptTemplate.from_template("{text}"),
            ]
        )
        messages = chat_template.format_messages(text={question})
        customer_response = self.llm.invoke(messages)

        return customer_response.content
