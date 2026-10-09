$content = @'
"""
RAG Pipeline with LLM integration (OpenAI, Mistral, Qwen, Gemini)
"""
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import google.generativeai as genai

from src.vector_store import VectorStore
from src.models import Source, QueryResponse


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    MISTRAL = "mistral"
    QWEN = "qwen"
    HUGGINGFACE = "huggingface"
    GEMINI = "gemini"


DISCLAIMER = "⚠️ This information is for educational purposes only and is not a substitute for professional medical advice."


class RAGPipeline:
    """Retrieval-Augmented Generation Pipeline"""

    # NOTE on structure: we deliberately do NOT ask the model to write out a
    # "Sources: [...]" text block anymore. The API already returns a clean,
    # structured `sources` array with real titles/URLs/scores -- having the
    # LLM also re-type that list as prose inside `answer` just duplicates it
    # in a harder-to-read form. We also pin the disclaimer to one place
    # (code-enforced, see _clean_answer()) instead of hoping the model puts
    # it in the same spot every time.
    SYSTEM_PROMPT = """You are MedSource, a medical AI assistant designed to provide fact-based, reliable, and explainable answers.
You are connected to a retrieval system that provides verified medical documents.
Your task is to answer medical queries using ONLY the information retrieved below.

CRITICAL RULES:
1. Use clear, simple language while maintaining medical accuracy.
2. Every factual claim MUST have a citation in the format [DOC_X] where X is the document number.
3. Structure your answer for skimmability: start with a one-sentence direct answer, then use short bullet points grouped by cause/topic if there are multiple relevant conditions. Do not restate the question.
4. Do NOT write a "Sources:" list at the end -- the source list is provided separately and will be shown automatically. Do not add your own disclaimer line either -- it will be added automatically.
5. If the retrieved documents do not answer the question, reply: "I'm sorry, I don't have enough verified information to answer that safely."
6. NEVER fabricate or infer medical facts not present in the retrieved documents.
7. Do NOT diagnose users or prescribe treatments. Your purpose is to inform, not diagnose.

Retrieved Context:
{context}

User Query: {question}

Provide your answer with inline citations, following the structure rules above:"""

    # Used ONLY when retrieval finds nothing above the confidence threshold.
    # Answers from the model's general knowledge instead of a hard refusal,
    # but is explicit that this is NOT a verified, sourced answer.
    GENERAL_KNOWLEDGE_PROMPT = """You are MedSource, a medical information assistant.
No verified documents were found in your knowledge base for this question, so you must
answer using your own general medical knowledge instead.

CRITICAL RULES:
1. Start your answer with: "Note: This answer is based on general medical knowledge and was not verified against our document database."
2. Structure your answer for skimmability: a one-sentence direct answer first, then short bullet points if helpful (e.g. common causes, or when to seek care).
3. Use clear, simple, cautious language.
4. Do NOT diagnose the user or prescribe specific treatments or dosages. Your purpose is to inform, not diagnose.
5. If you are not confident about the answer, say so explicitly rather than guessing.
6. Do not add your own closing disclaimer line -- it will be added automatically.

User Query: {question}

Provide a brief, cautious answer, following the structure rules above:"""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_provider: str = "openai",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        top_k: int = 5,
        confidence_threshold: float = 0.75,
        enable_general_fallback: bool = True
    ):
        self.vector_store = vector_store
        self.llm_provider = LLMProvider(llm_provider)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self.enable_general_fallback = enable_general_fallback
        self.client = None  # For OpenAI client
        self.gemini_model = None  # For Gemini model
        self.model = None  # For HuggingFace models
        self.tokenizer = None  # For HuggingFace tokenizers

        # Initialize LLM based on provider
        if self.llm_provider == LLMProvider.OPENAI:
            self.model_name = model_name or "gpt-4o-mini"
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
            else:
                self.client = openai.OpenAI()  # Will use OPENAI_API_KEY env var

        elif self.llm_provider == LLMProvider.GEMINI:
            self.model_name = model_name or "gemini-3.6-flash"
            if api_key:
                genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel(self.model_name)
            print(f"✅ Gemini model initialized: {self.model_name}")

        elif self.llm_provider == LLMProvider.MISTRAL:
            self.model_name = model_name or "mistralai/Mistral-7B-Instruct-v0.2"
            self._load_huggingface_model()

        elif self.llm_provider == LLMProvider.QWEN:
            self.model_name = model_name or "Qwen/Qwen1.5-7B-Chat"
            self._load_huggingface_model()

        elif self.llm_provider == LLMProvider.HUGGINGFACE:
            self.model_name = model_name
            if not model_name:
                raise ValueError("model_name required for HuggingFace provider")
            self._load_huggingface_model()

    def _load_huggingface_model(self):
        """Load HuggingFace model and tokenizer"""
        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        print(f"Model loaded successfully")

    def _format_context(self, retrieved_docs: List[Tuple[Dict[str, Any], float]]) -> str:
        """Format retrieved documents as context"""
        context_parts = []
        for i, (doc, score) in enumerate(retrieved_docs, 1):
            context_parts.append(
                f"[DOC_{i}] {doc['title']} ({doc['source']}, {doc.get('year', 'N/A')})\n"
                f"Content: {doc['content']}\n"
                f"Relevance Score: {score:.3f}\n"
            )
        return "\n".join(context_parts)

    @staticmethod
    def _clean_answer(answer: str, add_disclaimer: bool = True) -> str:
        """
        Code-enforced response structure cleanup. Rather than hoping the LLM
        consistently follows prompt instructions about formatting, we fix it
        up deterministically here:
          - strip any "Sources: [...]" block the model wrote anyway
            (the API returns a proper structured `sources` array separately)
          - strip any disclaimer line the model wrote in an inconsistent spot
          - append exactly one disclaimer, in exactly one place: the end
          - collapse stray triple-asterisk markdown artifacts into bullets
        """
        if not answer:
            return answer

        # Remove a trailing "Sources: [...]" block, however it's formatted.
        answer = re.sub(r"\n+Sources:.*$", "", answer, flags=re.DOTALL | re.IGNORECASE).strip()

        # Remove any disclaimer the model inserted itself, wherever it put it,
        # so we can re-add exactly one, in a consistent position.
        disclaimer_pattern = re.escape(
            "This information is for educational purposes only and is not a substitute for professional medical advice."
        )
        answer = re.sub(r"⚠️?\s*" + disclaimer_pattern + r"\.?", "", answer, flags=re.IGNORECASE).strip()

        # Fix the occasional "***Label:**" artifact some models produce,
        # which breaks markdown rendering -- normalize to "* **Label:**".
        answer = re.sub(r"(?m)^\*\*\*", "* **", answer)

        # Collapse any resulting multiple blank lines left behind by the strips above.
        answer = re.sub(r"\n{3,}", "\n\n", answer).strip()

        if add_disclaimer:
            answer = f"{answer}\n\n{DISCLAIMER}"

        return answer

    def _generate_openai(self, prompt: str) -> str:
        """Generate response using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are MedSource, a medical AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _generate_gemini(self, prompt: str) -> str:
        """Generate response using Google Gemini"""
        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            )
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _generate_huggingface(self, prompt: str) -> str:
        """Generate response using HuggingFace model"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if prompt in response:
                response = response.split(prompt)[-1].strip()
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _calculate_confidence(
        self,
        retrieved_docs: List[Tuple[Dict[str, Any], float]],
        answer: str
    ) -> float:
        """Calculate confidence score based on retrieval scores and answer quality"""
        if not retrieved_docs:
            return 0.0

        avg_score = sum(score for _, score in retrieved_docs) / len(retrieved_docs)
        has_citations = "[DOC_" in answer

        uncertainty_phrases = [
            "don't have enough",
            "cannot answer",
            "insufficient information",
            "not enough verified"
        ]
        admits_uncertainty = any(phrase in answer.lower() for phrase in uncertainty_phrases)

        confidence = avg_score
        if has_citations:
            confidence *= 1.1
        if admits_uncertainty and avg_score < self.confidence_threshold:
            confidence *= 0.7

        return min(confidence, 1.0)

    def query(self, question: str, top_k: Optional[int] = None) -> QueryResponse:
        """Process a query through the RAG pipeline"""
        k = top_k or self.top_k

        # Retrieval phase
        retrieval_start = time.time()
        raw_candidates = self.vector_store.search(
            question,
            top_k=max(k * 4, 20),
            score_threshold=self.confidence_threshold
        )

        seen_doc_ids = set()
        retrieved_docs = []
        for doc, score in raw_candidates:  # already sorted by score, descending
            if doc["doc_id"] in seen_doc_ids:
                continue
            seen_doc_ids.add(doc["doc_id"])
            retrieved_docs.append((doc, score))
            if len(retrieved_docs) >= k:
                break
        retrieval_time = (time.time() - retrieval_start) * 1000

        # Check if we have sufficient context
        if not retrieved_docs or all(score < self.confidence_threshold for _, score in retrieved_docs):
            if not self.enable_general_fallback:
                return QueryResponse(
                    question=question,
                    answer=self._clean_answer(
                        "I'm sorry, I don't have enough verified information to answer that safely. "
                        "Please consult with a healthcare professional for accurate medical advice."
                    ),
                    sources=[],
                    confidence=0.0,
                    retrieval_time_ms=retrieval_time,
                    generation_time_ms=0.0,
                    total_time_ms=retrieval_time,
                    warning="⚠️ Insufficient verified information available. Please consult a healthcare professional."
                )

            gen_start = time.time()
            fallback_prompt = self.GENERAL_KNOWLEDGE_PROMPT.format(question=question)
            if self.llm_provider == LLMProvider.OPENAI:
                answer = self._generate_openai(fallback_prompt)
            elif self.llm_provider == LLMProvider.GEMINI:
                answer = self._generate_gemini(fallback_prompt)
            else:
                answer = self._generate_huggingface(fallback_prompt)
            generation_time = (time.time() - gen_start) * 1000

            return QueryResponse(
                question=question,
                answer=self._clean_answer(answer),
                sources=[],
                confidence=0.0,
                retrieval_time_ms=retrieval_time,
                generation_time_ms=generation_time,
                total_time_ms=retrieval_time + generation_time,
                warning="⚠️ No verified source in our document database matched this question. "
                        "The answer above is general knowledge, not a cited/sourced answer. "
                        "Please consult a healthcare professional."
            )

        # Format context
        context = self._format_context(retrieved_docs)
        prompt = self.SYSTEM_PROMPT.format(context=context, question=question)

        # Generation phase
        generation_start = time.time()
        if self.llm_provider == LLMProvider.OPENAI:
            answer = self._generate_openai(prompt)
        elif self.llm_provider == LLMProvider.GEMINI:
            answer = self._generate_gemini(prompt)
        else:
            answer = self._generate_huggingface(prompt)
        generation_time = (time.time() - generation_start) * 1000

        uncertainty_phrases = [
            "don't have enough", "cannot answer",
            "insufficient information", "not enough verified"
        ]
        model_admits_uncertainty = any(phrase in answer.lower() for phrase in uncertainty_phrases)

        if model_admits_uncertainty and self.enable_general_fallback:
            fallback_start = time.time()
            fallback_prompt = self.GENERAL_KNOWLEDGE_PROMPT.format(question=question)
            if self.llm_provider == LLMProvider.OPENAI:
                answer = self._generate_openai(fallback_prompt)
            elif self.llm_provider == LLMProvider.GEMINI:
                answer = self._generate_gemini(fallback_prompt)
            else:
                answer = self._generate_huggingface(fallback_prompt)
            fallback_time = (time.time() - fallback_start) * 1000

            return QueryResponse(
                question=question,
                answer=self._clean_answer(answer),
                sources=[],
                confidence=0.0,
                retrieval_time_ms=retrieval_time,
                generation_time_ms=generation_time + fallback_time,
                total_time_ms=retrieval_time + generation_time + fallback_time,
                warning="⚠️ Retrieved documents were topically related but did not directly answer "
                        "this question. The answer above is general knowledge, not a cited/sourced "
                        "answer. Please consult a healthcare professional."
            )

        confidence = self._calculate_confidence(retrieved_docs, answer)

        sources = []
        for i, (doc, score) in enumerate(retrieved_docs, 1):
            source = Source(
                doc_id=f"DOC_{i}",
                title=doc["title"],
                year=doc.get("year"),
                url=doc.get("url"),
                relevance_score=score,
                excerpt=doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"]
            )
            sources.append(source)

        return QueryResponse(
            question=question,
            answer=self._clean_answer(answer),
            sources=sources,
            confidence=confidence,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            total_time_ms=retrieval_time + generation_time,
            warning="Always consult with qualified healthcare professionals for medical decisions."
        )

'@
Set-Content -Path 'src\rag_pipeline.py' -Value $content -Encoding utf8
Write-Host 'src/rag_pipeline.py has been replaced.'