"""Corrective-RAG loop: retrieve -> grade -> generate -> verify, with retries.

The loop is deliberately framework-agnostic: you supply

- ``retrieve(question) -> list`` of documents (each with ``page_content`` and ``metadata``),
- ``generate(question, documents) -> str`` that produces an answer,
- a chat model supporting structured output (used to build the graders),

and the loop orchestrates relevance grading, query rewriting, and
hallucination checks. It returns a ``RAGResult`` with the answer, the
documents that backed it, and an audit trail of what happened.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from .graders import (
    build_answer_grader,
    build_hallucination_grader,
    build_question_rewriter,
    build_relevance_grader,
)


@dataclass
class RAGResult:
    """Outcome of one corrective-RAG run."""

    answer: str
    documents: List[Any] = field(default_factory=list)
    backed_by_library: bool = False
    retries: int = 0
    audit: List[str] = field(default_factory=list)


class CorrectiveRAG:
    def __init__(
        self,
        llm,
        retrieve: Callable[[str], List[Any]],
        generate: Callable[[str, List[Any]], str],
        *,
        default_answer: Optional[Callable[[str], str]] = None,
        max_rewrites: int = 2,
        max_regenerations: int = 1,
    ):
        """
        Args:
            llm: chat model with structured output; used for all graders.
            retrieve: fn mapping a question to candidate documents.
            generate: fn mapping (question, documents) to an answer string.
            default_answer: fn used when the question cannot be answered from
                the library. Defaults to a short "cannot answer" message.
            max_rewrites: how many query-rewrite/re-retrieve cycles to attempt.
            max_regenerations: how many times to regenerate after a
                hallucination is detected before giving up on the attempt.
        """
        self.llm = llm
        self.retrieve = retrieve
        self.generate = generate
        self.default_answer = (
            default_answer
            or (lambda q: "This question cannot be answered based on the provided documents.")
        )
        self.max_rewrites = max_rewrites
        self.max_regenerations = max_regenerations

        self.relevance_grader = build_relevance_grader(llm)
        self.hallucination_grader = build_hallucination_grader(llm)
        self.answer_grader = build_answer_grader(llm)
        self.rewriter = build_question_rewriter(llm)

    @staticmethod
    def _doc_text(doc: Any) -> str:
        return getattr(doc, "page_content", str(doc))

    def _grade_relevance(self, question: str, docs: List[Any]) -> List[Any]:
        relevant = []
        for doc in docs:
            score = self.relevance_grader.invoke(
                {"question": question, "document": self._doc_text(doc)}
            )
            if score.binary_score == "yes":
                relevant.append(doc)
        return relevant

    def run(self, question: str) -> RAGResult:
        audit: List[str] = []
        retries = 0
        current = question

        for rewrite in range(self.max_rewrites + 1):
            candidates = self.retrieve(current)
            audit.append(f"retrieved {len(candidates)} docs for query {rewrite + 1}")
            docs = self._grade_relevance(current, candidates)
            audit.append(f"{len(docs)} docs passed relevance grading")

            if not docs:
                if rewrite < self.max_rewrites:
                    current = self.rewriter.invoke({"question": current})
                    retries += 1
                    audit.append(f"no relevant docs; rewrote query to: {current!r}")
                    continue
                audit.append("no relevant docs after rewrites; falling back to default answer")
                return RAGResult(
                    answer=self.default_answer(question),
                    documents=[],
                    backed_by_library=False,
                    retries=retries,
                    audit=audit,
                )

            for regen in range(self.max_regenerations + 1):
                answer = self.generate(current, docs)
                facts = [self._doc_text(d) for d in docs]
                grounded = self.hallucination_grader.invoke(
                    {"documents": facts, "generation": answer}
                )
                if grounded.binary_score != "yes":
                    retries += 1
                    audit.append(f"generation {regen + 1} not grounded; regenerating")
                    continue
                useful = self.answer_grader.invoke(
                    {"question": question, "generation": answer}
                )
                if useful.binary_score == "yes":
                    audit.append("answer grounded and addresses the question")
                    return RAGResult(
                        answer=answer,
                        documents=docs,
                        backed_by_library=True,
                        retries=retries,
                        audit=audit,
                    )
                audit.append("answer does not address the question")
                break  # stop regenerating; try a rewritten query instead

            if rewrite < self.max_rewrites:
                current = self.rewriter.invoke({"question": current})
                retries += 1
                audit.append(f"rewrote query to: {current!r}")

        audit.append("exhausted retries; falling back to default answer")
        return RAGResult(
            answer=self.default_answer(question),
            documents=[],
            backed_by_library=False,
            retries=retries,
            audit=audit,
        )
