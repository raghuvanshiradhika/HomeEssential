import gradio as gr
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import Tool
from app.retriever import build_retriever
from openai_chat import OpenAIChat
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import AgentExecutor

# Initialize retriever, LLM, and chain
retriever = build_retriever("data/Product_Catalogue.docx")
openai_chat = OpenAIChat()
search = GoogleSerperAPIWrapper()
home_tool = create_retriever_tool(
    retriever,
    name="home_essentials_database",
    description="Useful when you need to find information about home essentials."
)
google_tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run,
)
tools = [home_tool, google_tool]


def chatbot(input, history=[]):
    # chain = RetrievalQA.from_chain_type(llm=openai_chat.llm,
    # chain_type="stuff",
    #  retriever=retriever.as_retriever())
    # output = chain(input)
    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_tool_calling_agent(openai_chat.llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True)
    output = agent_executor.invoke({"input": input})
    history.append((input, str(output["output"])))
    return history, history


gr.Interface(fn=chatbot,
             inputs=["text", "state"],
             outputs=["chatbot", "state"]).launch(debug=True)
