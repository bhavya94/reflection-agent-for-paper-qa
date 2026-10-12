"""BM25 retrieval over the paragraphs of one paper."""

import bm25s
import Stemmer

from reflective_qa.data.qasper import QasperSplit
from reflective_qa.data.records import Paper, Paragraph, Question


def _tokenize(texts: list[str]) -> list[list[str]]:
    # A new stemmer per call: PyStemmer instances must not be shared across threads.
    return bm25s.tokenize(texts, stopwords="en", stemmer=Stemmer.Stemmer("english"), return_ids=False,
                          show_progress=False)


def passage_text(paragraph: Paragraph) -> str:
    """A paragraph as it is indexed and shown to the answer model: section titles first, because they
    carry topic words the paragraph often doesn't repeat."""
    return f"{paragraph.section.replace(' ::: ', '. ')}. {paragraph.text}"


class PaperRetriever:
    def __init__(self, paper: Paper):
        self.paragraphs = paper.paragraphs
        self._bm25 = bm25s.BM25()
        self._bm25.index(_tokenize([passage_text(p) for p in self.paragraphs]), show_progress=False)

    def retrieve(self, question: str, k: int) -> list[Paragraph]:
        """Top-k paragraphs by BM25 score; every paragraph if the paper has k or fewer.
        Paragraphs with equal scores (often all zero near the bottom of the top-k) come in arbitrary order."""
        k = min(k, len(self.paragraphs))
        if k == 0:
            return []
        ranked, _ = self._bm25.retrieve(_tokenize([question]), k=k, show_progress=False)
        return [self.paragraphs[i] for i in ranked[0]]


class Retriever:
    """BM25 over a whole split. Each paper is indexed the first time a question asks about it, then reused."""

    def __init__(self, split: QasperSplit):
        self._split = split
        self._indexes: dict[str, PaperRetriever] = {}

    def index(self, paper_id: str) -> PaperRetriever:
        if paper_id not in self._indexes:
            self._indexes[paper_id] = PaperRetriever(self._split.paper(paper_id))
        return self._indexes[paper_id]

    def retrieve(self, question: Question, k: int) -> list[Paragraph]:
        """The top-k paragraphs of the question's paper; none for k = 0 (closed-book), without building an index."""
        return self.index(question.paper_id).retrieve(question.text, k) if k else []
