"""
RAG Pipeline for OpenAI Research Papers
Uses arXiv API to fetch papers, embeds them, and enables Q&A with failure analysis
"""

import os
import json
import time
import hashlib
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from openai import OpenAI

# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    published: str
    url: str
    chunks: List[str] = field(default_factory=list)

@dataclass
class RetrievedChunk:
    text: str
    paper_title: str
    arxiv_id: str
    similarity_score: float
    chunk_index: int

@dataclass
class RAGResult:
    query: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    latency_ms: float
    failure_flags: List[str]
    failure_details: Dict
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ─────────────────────────────────────────────
# ArXiv Fetcher
# ─────────────────────────────────────────────

class ArxivFetcher:
    BASE_URL = "http://export.arxiv.org/api/query"

    def fetch_papers(self, query: str = "large language models", max_results: int = 30) -> List[Paper]:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        return self._parse_feed(response.text)

    def _parse_feed(self, xml_text: str) -> List[Paper]:
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_text)
        papers = []
        for entry in root.findall("atom:entry", ns):
            arxiv_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
            title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
            abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
            published = entry.find("atom:published", ns).text[:10]
            url = f"https://arxiv.org/abs/{arxiv_id}"
            papers.append(Paper(arxiv_id, title, abstract, authors, published, url))
        return papers


# ─────────────────────────────────────────────
# Text Chunker
# ─────────────────────────────────────────────

class TextChunker:
    def __init__(self, chunk_size: int = 400, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_paper(self, paper: Paper) -> List[str]:
        """Chunk paper title + abstract into overlapping windows."""
        text = f"Title: {paper.title}\n\nAuthors: {', '.join(paper.authors[:3])}\n\nAbstract: {paper.abstract}"
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += self.chunk_size - self.overlap
        return chunks


# ─────────────────────────────────────────────
# Vector Store (FAISS-free, pure numpy)
# ─────────────────────────────────────────────

class VectorStore:
    def __init__(self):
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict] = []

    def add(self, embedding: np.ndarray, metadata: Dict):
        self.embeddings.append(embedding)
        self.metadata.append(metadata)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict]]:
        if not self.embeddings:
            return []
        matrix = np.array(self.embeddings)
        # Cosine similarity
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_embedding)
        norms = np.where(norms == 0, 1e-10, norms)
        scores = matrix @ query_embedding / norms
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(float(scores[i]), self.metadata[i]) for i in top_indices]

    def size(self) -> int:
        return len(self.embeddings)


# ─────────────────────────────────────────────
# Failure Analyzer
# ─────────────────────────────────────────────

class FailureAnalyzer:
    LOW_SIMILARITY_THRESHOLD = 0.70
    HIGH_SIMILARITY_THRESHOLD = 0.90
    MIN_ANSWER_LENGTH = 50
    HALLUCINATION_PHRASES = [
        "i don't have information",
        "i cannot find",
        "not mentioned in",
        "the context does not",
        "no information provided",
    ]
    UNCERTAINTY_PHRASES = [
        "i'm not sure", "i am not sure", "it's unclear",
        "it is unclear", "might be", "possibly", "perhaps",
        "i believe", "i think", "i'm uncertain"
    ]

    def analyze(
        self,
        query: str,
        answer: str,
        chunks: List[RetrievedChunk]
    ) -> Tuple[List[str], Dict]:
        flags = []
        details = {}

        # 1. Low retrieval confidence
        if chunks:
            top_score = chunks[0].similarity_score
            avg_score = np.mean([c.similarity_score for c in chunks])
            details["top_similarity"] = round(top_score, 4)
            details["avg_similarity"] = round(float(avg_score), 4)
            if top_score < self.LOW_SIMILARITY_THRESHOLD:
                flags.append("LOW_RETRIEVAL_CONFIDENCE")
                details["low_confidence_reason"] = f"Top similarity {top_score:.3f} < {self.LOW_SIMILARITY_THRESHOLD}"
        else:
            flags.append("NO_CHUNKS_RETRIEVED")

        # 2. Potential hallucination (model flagging itself)
        answer_lower = answer.lower()
        triggered = [p for p in self.HALLUCINATION_PHRASES if p in answer_lower]
        if triggered:
            flags.append("POSSIBLE_HALLUCINATION")
            details["hallucination_triggers"] = triggered

        # 3. High uncertainty language
        uncertain = [p for p in self.UNCERTAINTY_PHRASES if p in answer_lower]
        if uncertain:
            flags.append("HIGH_UNCERTAINTY")
            details["uncertainty_phrases"] = uncertain

        # 4. Very short answer
        if len(answer.split()) < 20:
            flags.append("SUSPICIOUSLY_SHORT_ANSWER")
            details["answer_word_count"] = len(answer.split())

        # 5. Context diversity (all chunks from same paper = narrow retrieval)
        if chunks:
            unique_papers = len(set(c.arxiv_id for c in chunks))
            details["unique_papers_retrieved"] = unique_papers
            if unique_papers == 1 and len(chunks) >= 3:
                flags.append("LOW_SOURCE_DIVERSITY")

        # 6. Query-answer topic drift (basic keyword overlap)
        query_words = set(query.lower().split()) - {"what", "how", "why", "is", "are", "the", "a", "an", "in", "of"}
        answer_words = set(answer.lower().split())
        overlap = len(query_words & answer_words) / max(len(query_words), 1)
        details["query_answer_overlap"] = round(overlap, 3)
        if overlap < 0.1:
            flags.append("POSSIBLE_TOPIC_DRIFT")

        details["total_flags"] = len(flags)
        details["health_score"] = max(0, 100 - len(flags) * 20)

        return flags, details


# ─────────────────────────────────────────────
# RAG System
# ─────────────────────────────────────────────

class RAGSystem:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.vector_store = VectorStore()
        self.chunker = TextChunker()
        self.failure_analyzer = FailureAnalyzer()
        self.papers: List[Paper] = []
        self.results_log: List[RAGResult] = []
        self.embed_model = "text-embedding-3-small"
        self.chat_model = "gpt-4o-mini"

    def ingest_papers(self, papers: List[Paper], progress_callback=None):
        """Embed and store all paper chunks."""
        self.papers = papers
        total_chunks = 0
        for i, paper in enumerate(papers):
            chunks = self.chunker.chunk_paper(paper)
            paper.chunks = chunks
            for j, chunk in enumerate(chunks):
                embedding = self._embed(chunk)
                self.vector_store.add(embedding, {
                    "text": chunk,
                    "paper_title": paper.title,
                    "arxiv_id": paper.arxiv_id,
                    "chunk_index": j,
                    "url": paper.url
                })
                total_chunks += 1
            if progress_callback:
                progress_callback(i + 1, len(papers))
            time.sleep(0.1)  # Rate limit buffer
        return total_chunks

    def query(self, question: str, top_k: int = 5) -> RAGResult:
        start = time.time()

        # Retrieve
        query_embedding = self._embed(question)
        raw_results = self.vector_store.search(query_embedding, top_k=top_k)

        chunks = [
            RetrievedChunk(
                text=meta["text"],
                paper_title=meta["paper_title"],
                arxiv_id=meta["arxiv_id"],
                similarity_score=score,
                chunk_index=meta["chunk_index"]
            )
            for score, meta in raw_results
        ]

        # Generate
        context = "\n\n---\n\n".join([
            f"[Paper: {c.paper_title}]\n{c.text}" for c in chunks
        ])
        answer = self._generate(question, context)

        latency = (time.time() - start) * 1000

        # Analyze failures
        flags, details = self.failure_analyzer.analyze(question, answer, chunks)

        result = RAGResult(
            query=question,
            answer=answer,
            retrieved_chunks=chunks,
            latency_ms=round(latency, 2),
            failure_flags=flags,
            failure_details=details
        )
        self.results_log.append(result)
        return result

    def _embed(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(input=text, model=self.embed_model)
        return np.array(response.data[0].embedding)

    def _generate(self, question: str, context: str) -> str:
        system_prompt = """You are a research assistant specializing in AI and machine learning papers.
Answer the user's question based ONLY on the provided context from arXiv papers.
If the context doesn't contain enough information, say so clearly.
Be precise, cite paper titles when possible, and avoid speculation."""

        user_prompt = f"""Context from research papers:
{context}

Question: {question}

Answer based on the above context:"""

        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=600
        )
        return response.choices[0].message.content

    def get_eval_summary(self) -> Dict:
        """Summarize failure patterns across all queries."""
        if not self.results_log:
            return {}

        all_flags = []
        for r in self.results_log:
            all_flags.extend(r.failure_flags)

        flag_counts = {}
        for f in all_flags:
            flag_counts[f] = flag_counts.get(f, 0) + 1

        avg_latency = np.mean([r.latency_ms for r in self.results_log])
        avg_health = np.mean([r.failure_details.get("health_score", 100) for r in self.results_log])
        clean_queries = sum(1 for r in self.results_log if not r.failure_flags)

        return {
            "total_queries": len(self.results_log),
            "clean_queries": clean_queries,
            "flagged_queries": len(self.results_log) - clean_queries,
            "flag_distribution": flag_counts,
            "avg_latency_ms": round(float(avg_latency), 2),
            "avg_health_score": round(float(avg_health), 1),
        }
