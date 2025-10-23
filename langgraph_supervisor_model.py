### Sample code for a Supervisor Model using LangGraph
###

##Step 1: Import and setup

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import MessagesState, END, StateGraph, START
from langgraph.types import Command
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.prebuilt import create_react_agent
from langchain_community.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults

##Step2: initialize the model
# Llama3.2 model
ollama_model = ChatOllama(model="llama3.2")

## Define tools 
## A search tool and a grammar checking tool
duckduckgo_tool = DuckDuckGoSearchResults()
@tool
def grammar_check_tool(text: Annotated[str, "The text to review for grammar corrections"]):
    """Suggest grammar improvements to the given text."""
    return f"(Example response) Reviewed grammar for: {text}"

## Routing logic and state
##Router is used to parse the supervisor's structured response.
##State tracks current messages and which agent is next.

members = ["content_writer", "proofreader"]
options = members + ["FINISH"]
class Router(TypedDict):
    """Worker to route to next agent.If no worker needed , route to finish"""
    next: Literal["content_writer", "proofreader", "FINISH"]
class State(MessagesState):
    next: str

## Step 5: Supervisor Agent

system_prompt = f"""
You are a supervisor, tasked with managing a conversation between the following workers: {members}. 
Given the user's request, respond with the worker to act next. 
Each worker will perform a task and respond with results and status. 
When finished, respond with FINISH.
"""
def supervisor_node(state: State) -> Command[Literal["content_writer", "proofreader", "__end__"]]:
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = ollama_model.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END
    return Command(goto=goto, update={"next": goto})

## Step 6: Worker Agents

#Content Writer Agent
# Uses ReAct-based reasoning with the search tool.
#Sends the result back to the supervisor.

def content_writer_node(state: State) -> Command[Literal["supervisor"]]:
    content_writer = create_react_agent(ollama_model, tools=[duckduckgo_tool])
    result = content_writer.invoke(state)
    return Command(
        update={"messages": [
            HumanMessage(content=result["messages"][-1].content, name="content_writer")
        ]},
        goto="supervisor"
    )

# Proofreader Agent
#Uses the grammar-check tool to suggest improvements.
#Then loops back to the supervisor.

def proofreader_node(state: State) -> Command[Literal["supervisor"]]:
    proofreader = create_react_agent(ollama_model, tools=[grammar_check_tool])
    result = proofreader.invoke(state)
    return Command(
        update={"messages": [
            HumanMessage(content=result["messages"][-1].content, name="proofreader")
        ]},
        goto="supervisor"
    )

## Step 7: Connect and Compile the Graph
#Nodes are added. Workflow starts from supervisor. Final app is compiled.

graph = StateGraph(State)
graph.add_node("supervisor", supervisor_node)
graph.add_node("content_writer", content_writer_node)
graph.add_node("proofreader", proofreader_node)
graph.add_edge(START, "supervisor")
app = graph.compile()

#Draw the graph
from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))

##Step 8: Run the application

for s in app.stream({"messages": [("user", "Write an article on the benefits of AI in education.")]}, subgraphs=True):
    print(s)
    print("----")