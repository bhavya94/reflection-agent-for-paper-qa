"""BM25 retrieval over the paragraphs of one paper."""

import bm25s
import Stemmer

from reflective_qa.data import Paper, Paragraph

_STEMMER = Stemmer.Stemmer("english")


def _tokenize(texts: list[str]) -> list[list[str]]:
    return bm25s.tokenize(texts, stopwords="en", stemmer=_STEMMER, return_ids=False, show_progress=False)


def _indexed_text(paragraph: Paragraph) -> str:
    """Section titles carry topic words a paragraph often doesn't repeat."""
    return f"{paragraph.section.replace(' ::: ', '. ')}. {paragraph.text}"


class PaperRetriever:
    def __init__(self, paper: Paper):
        self.paragraphs = paper.paragraphs
        self._bm25 = bm25s.BM25()
        self._bm25.index(_tokenize([_indexed_text(p) for p in self.paragraphs]), show_progress=False)

    def retrieve(self, question: str, k: int) -> list[Paragraph]:
        """Top-k paragraphs by BM25 score; every paragraph if the paper has k or fewer."""
        k = min(k, len(self.paragraphs))
        if k == 0:
            return []
        ranked, _ = self._bm25.retrieve(_tokenize([question]), k=k, show_progress=False)
        return [self.paragraphs[i] for i in ranked[0]]
