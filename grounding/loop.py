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
        self.llm = llm
        self.retrieve = retrieve
        self.generate = generate
        self.default_answer = default_answer or (
            lambda question: "This question cannot be answered based on the provided documents."
        )
        self.max_rewrites = max_rewrites
        self.max_regenerations = max_regenerations

        self.relevance_grader = build_relevance_grader(llm)
        self.hallucination_grader = build_hallucination_grader(llm)
        self.answer_grader = build_answer_grader(llm)
        self.rewriter = build_question_rewriter(llm)

    @staticmethod
    def _text(document: Any) -> str:
        return getattr(document, "page_content", str(document))

    def _relevant(self, question: str, documents: List[Any]) -> List[Any]:
        kept = []
        for document in documents:
            score = self.relevance_grader.invoke(
                {"question": question, "document": self._text(document)}
            )
            if score.binary_score == "yes":
                kept.append(document)
        return kept

    def run(self, question: str) -> RAGResult:
        audit: List[str] = []
        retries = 0
        current = question

        for rewrite in range(self.max_rewrites + 1):
            candidates = self.retrieve(current)
            audit.append(f"retrieved {len(candidates)} docs for query {rewrite + 1}")
            documents = self._relevant(current, candidates)
            audit.append(f"{len(documents)} docs passed relevance grading")

            if not documents:
                if rewrite < self.max_rewrites:
                    current = self.rewriter.invoke({"question": current})
                    retries += 1
                    audit.append(f"no relevant docs; rewrote query to: {current!r}")
                    continue
                audit.append("no relevant docs after rewrites; falling back to default answer")
                return RAGResult(
                    answer=self.default_answer(question),
                    retries=retries,
                    audit=audit,
                )

            for regeneration in range(self.max_regenerations + 1):
                answer = self.generate(current, documents)
                facts = [self._text(d) for d in documents]
                grounded = self.hallucination_grader.invoke(
                    {"documents": facts, "generation": answer}
                )
                if grounded.binary_score != "yes":
                    retries += 1
                    audit.append(f"generation {regeneration + 1} not grounded; regenerating")
                    continue
                useful = self.answer_grader.invoke(
                    {"question": question, "generation": answer}
                )
                if useful.binary_score == "yes":
                    audit.append("answer grounded and addresses the question")
                    return RAGResult(
                        answer=answer,
                        documents=documents,
                        backed_by_library=True,
                        retries=retries,
                        audit=audit,
                    )
                audit.append("answer does not address the question")
                break

            if rewrite < self.max_rewrites:
                current = self.rewriter.invoke({"question": current})
                retries += 1
                audit.append(f"rewrote query to: {current!r}")

        audit.append("exhausted retries; falling back to default answer")
        return RAGResult(
            answer=self.default_answer(question),
            retries=retries,
            audit=audit,
        )
