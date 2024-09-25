import os, sys
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.utils.config import cfg
from src.utils.helper import get_llm

chat_memory = MemorySaver()

def get_chat_chain(model="GPT-4o-Mini", temperature=0.0):
    model = get_llm(model, temperature)
    
    def filter_messages(messages: list):
        return messages[-5:]
    
    # Define the function that calls the model
    def call_model(state: MessagesState):
        messages = filter_messages(state["messages"])
        response = model.invoke(messages)
        return {"messages": response}
    
    # Define a new graph
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
    app = workflow.compile(checkpointer=chat_memory)
    return app
