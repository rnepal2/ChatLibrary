import os, sys
from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.utils.config import cfg
from src.utils.helper import get_llm
from src.vectordb.retriever import VectorDBRetriever
from src.rag.tools import question_router, retriever_grader, answer_grader
from src.rag.tools import hallucination_grader, question_rewriter

VERBOSE = cfg.VERBOSE

def retrieve(state, verbose=VERBOSE):
    if verbose:
        print("RETRIEVE DOCUMENTS")
    question = state["question"]
    collections = state['collections']
  
    retriever = VectorDBRetriever(collection_list=collections, n_results=5)
    results = retriever.retrieve_documents(question)
    
    documents, metadatas = [], []
    for item in results:
        documents.extend(item['documents'][0])
        metadatas.extend(item['metadatas'][0])
    documents = [Document(page_content=doc, metadata=data) for doc, data in zip(documents, metadatas)]
    return {"documents": documents, "question": question}


def generate(state, verbose=VERBOSE):
    if verbose:
        print("GENERATE ANSWER")
    question = state["question"]
    documents = state["documents"]
    _documents = [d.page_content for d in documents]

    system_prompt = """
                    ROLE: You are an assistant that helps to answer user question based on retrieved documents.
                    
                    TASK: Your task is to carefully analyze user question and provided documents and answer the 
                    question based on the retrieved documents. If you cannot answer the question based on the 
                    documents, just reply that the question can not be answered based on provided documents.

                    INPUT: The user question is:
                    {question}

                    And, list of relevant provided documents are:
                    {documents}.

                    NOTE: Your answer should be concise, professional and valuable (only based on the documents) 
                    for the user who is usually subject matter expert in their domain. 
                """
    prompt = PromptTemplate(
        template=system_prompt,
        input_variables=["question", "documents"],
    )
    chain = prompt | get_llm(state['model']) | StrOutputParser()
    generation = chain.invoke({"question": question, "documents": _documents})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state, verbose=VERBOSE):
    if verbose:
        print("CHECK DOCUMENT RELEVANCE")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retriever_grader(state['model']).invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            if verbose:
                print("DOCUMENT -> RELEVANT")
            filtered_docs.append(d)
        else:
            if verbose:
                print("DOCUMENT -> NOT RELEVANT")
            continue
    return {"documents": filtered_docs, "question": question}


def transform_query(state, verbose=VERBOSE):
    if verbose:
        print("TRANSFORM QUERY")
    question = state["question"]
    documents = state["documents"]
    better_question = question_rewriter(state['model']).invoke({"question": question})
    return {"documents": documents, "question": better_question}


def web_search(state, verbose=VERBOSE):
    if verbose:
        print("DEFAULT ANSWER")
    question = state["question"]
    web_results = "No relevant documents present to answer the question."
    web_results = Document(page_content=web_results)
    return {"documents": [web_results], "question": question}

# Edges
def route_question(state, verbose=VERBOSE):
    if verbose:
        print("ROUTE QUESTION")
    question = state["question"]
    source = question_router(state['model']).invoke({"question": question})
    if source.datasource == "web_search":
        if verbose:
            print("ROUTE QUESTION -> DEFAULT ANSWER")
        return "web_search"
    elif source.datasource == "vectorstore":
        if verbose:
            print("ROUTE QUESTION -> RAG")
        return "vectorstore"


def decide_to_generate(state, verbose=VERBOSE):
    if verbose:
        print("ASSESS GRADED DOCUMENTS")
    filtered_documents = state["documents"]

    if not filtered_documents:
        if verbose:
            print("RETRIEVEED DOCUMENTS NOT RELEVANT -> TRANSFORM QUERY")
        return "transform_query"
    else:
        if verbose:
            print("DECISION -> GENERATE")
        return "generate"


def grade_generation_v_documents_and_question(state, verbose=VERBOSE):
    if verbose:
        print("CHECK HALLUCINATIONS")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    grader = hallucination_grader(state['model'])
    score = grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    if grade == "yes":
        if verbose:
            print("GENERATION -> NO HALLUCIATIONS")
        score = answer_grader().invoke(
            {"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            if verbose:
                print("GENERATION -> ANSWERS THE QUESTION")
            return "useful"
        else:
            if verbose:
                print("GENERATION: DOESNOT ANSWER THE QUESTION")
            return "not useful"
    else:
        if verbose:
            print("GENERATION -> HALLUCIATIONS -> RETRY")
        return "not supported"


class GraphState(TypedDict):
    """
        Represents the state of our graph.
    """
    model: str = 'GPT-4o-Mini'
    question: str
    generation: str
    documents: List[Document]
    collections: List[str] = ['Library_1', 'Library_2']


def build_graph():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("web_search", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    # Build graph
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "web_search": "web_search",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app
