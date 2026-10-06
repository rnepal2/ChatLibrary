"""
Graders for corrective RAG.

Each builder takes any chat model that supports structured output
(e.g. an AzureChatOpenAI, ChatOpenAI, or ChatAnthropic instance) and
returns a LangChain runnable you can `.invoke(...)` directly.

Extracted from the ChatLibrary corrective-RAG pipeline (Sept 2024);
kept free of Azure/ChromaDB/Streamlit so it can be reused anywhere.
"""

from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


def build_question_router(llm, vectorstore_description: str = "the vectorstore"):
    """LLM router: send the question to the vectorstore or fall back to a default path."""
    structured = llm.with_structured_output(RouteQuery)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at routing a user question to a vectorstore or a default answer.\n"
                f"The vectorstore contains documents related to: {vectorstore_description}.\n"
                "Use the vectorstore for questions on these topics. Otherwise, use web_search "
                "(i.e. the question cannot be answered from the library).",
            ),
            ("human", "{question}"),
        ]
    )
    return prompt | structured


class GradeDocuments(BaseModel):
    """Binary score for relevance check on a retrieved document."""

    binary_score: Literal["yes", "no"] = Field(
        description="Document is relevant to the question, 'yes' or 'no'"
    )


def build_relevance_grader(llm):
    """LLM grader: is this retrieved document relevant to the question?"""
    structured = llm.with_structured_output(GradeDocuments)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grader assessing relevance of a retrieved document to a user question. \n\n"
                "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
                "It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n"
                "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.",
            ),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    return prompt | structured


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in a generated answer."""

    binary_score: Literal["yes", "no"] = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


def build_hallucination_grader(llm):
    """LLM grader: is the generation grounded in the retrieved facts? ('yes' = grounded)"""
    structured = llm.with_structured_output(GradeHallucinations)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grader assessing whether an LLM generation is grounded in or \n"
                "supported by a set of retrieved facts. \n\n"
                "Give a binary score 'yes' or 'no'. 'yes' means that the answer is grounded in\n"
                "or supported by the provided set of facts, 'no' means the opposite.",
            ),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )
    return prompt | structured


class GradeAnswer(BaseModel):
    """Binary score to assess whether an answer addresses the question."""

    binary_score: Literal["yes", "no"] = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


def build_answer_grader(llm):
    """LLM grader: does the generation actually answer the user's question?"""
    structured = llm.with_structured_output(GradeAnswer)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grader assessing whether an answer addresses / resolves a question \n\n"
                "Give a binary score 'yes' or 'no'. 'yes' means that the answer resolves the question.",
            ),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )
    return prompt | structured


def build_question_rewriter(llm):
    """LLM chain: rewrite a question to be better optimized for vector retrieval."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a question re-writer that converts an input question to a better version that is optimized \n\n"
                "for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.",
            ),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()
