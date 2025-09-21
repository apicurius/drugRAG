#!/usr/bin/env python3
"""
Token Management Utility for RAG Context Truncation
Prevents vLLM token limit errors by intelligently managing prompt length
"""

import tiktoken
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class TokenManager:
    """Manages token counting and context truncation for RAG systems"""

    def __init__(self, model_name: str = "gpt-3.5-turbo", max_context_tokens: int = 3500):
        """
        Initialize token manager

        Args:
            model_name: Model name for tiktoken encoding (default works for most models)
            max_context_tokens: Maximum tokens to allow in context (leave room for prompt + response)
        """
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base encoding if model not found
            self.encoding = tiktoken.get_encoding("cl100k_base")

        self.max_context_tokens = max_context_tokens

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, using character estimate")
            return len(text) // 4  # Rough estimate: 4 chars per token

    def truncate_context_pairs(self, pairs: List[str], base_prompt: str) -> Tuple[str, int]:
        """
        Truncate drug-effect pairs to fit within token limits
        Prioritizes pairs by order (assuming they're relevance-sorted)

        Args:
            pairs: List of formatted pairs like "• drug → side_effect"
            base_prompt: The base prompt without context

        Returns:
            (truncated_context, num_pairs_included)
        """
        base_tokens = self.count_tokens(base_prompt)
        available_tokens = self.max_context_tokens - base_tokens - 100  # Safety margin

        if available_tokens <= 0:
            logger.warning("Base prompt too long, using minimal context")
            return "", 0

        included_pairs = []
        current_tokens = 0

        for pair in pairs:
            pair_tokens = self.count_tokens(pair + "\n")

            if current_tokens + pair_tokens <= available_tokens:
                included_pairs.append(pair)
                current_tokens += pair_tokens
            else:
                break

        context = "\n".join(included_pairs) if included_pairs else "No relevant pairs found within token limit"
        return context, len(included_pairs)

    def truncate_context_documents(self, documents: List[str], base_prompt: str) -> Tuple[str, int]:
        """
        Truncate drug documents to fit within token limits
        Prioritizes documents by order and truncates individual documents if needed

        Args:
            documents: List of drug documents
            base_prompt: The base prompt without context

        Returns:
            (truncated_context, num_documents_included)
        """
        base_tokens = self.count_tokens(base_prompt)
        available_tokens = self.max_context_tokens - base_tokens - 100  # Safety margin

        if available_tokens <= 0:
            logger.warning("Base prompt too long, using minimal context")
            return "", 0

        included_docs = []
        current_tokens = 0

        for i, doc in enumerate(documents):
            doc_tokens = self.count_tokens(doc)

            if current_tokens + doc_tokens <= available_tokens:
                # Document fits completely
                included_docs.append(doc)
                current_tokens += doc_tokens
            elif current_tokens == 0:
                # First document is too long, truncate it
                truncated_doc = self.truncate_single_document(doc, available_tokens)
                included_docs.append(truncated_doc)
                current_tokens = available_tokens
                break
            else:
                # No more room for additional documents
                break

        context = "\n\n".join(included_docs) if included_docs else "No relevant documents found within token limit"
        return context, len(included_docs)

    def truncate_single_document(self, document: str, max_tokens: int) -> str:
        """
        Truncate a single document to fit within token limit
        Tries to preserve beginning and end, with indication of truncation
        """
        doc_tokens = self.count_tokens(document)
        if doc_tokens <= max_tokens:
            return document

        # Try to keep beginning and end portions
        sentences = document.split('. ')
        if len(sentences) <= 2:
            # Short document, just truncate
            words = document.split()
            truncated_words = []
            current_tokens = 0

            for word in words:
                word_tokens = self.count_tokens(' ' + word)
                if current_tokens + word_tokens < max_tokens - 50:  # Leave room for truncation notice
                    truncated_words.append(word)
                    current_tokens += word_tokens
                else:
                    break

            return ' '.join(truncated_words) + "... [TRUNCATED DUE TO LENGTH]"

        # For longer documents, keep beginning and end
        beginning_tokens = max_tokens // 3
        ending_tokens = max_tokens // 3

        # Get beginning
        beginning_words = []
        current_tokens = 0
        for word in document.split():
            word_tokens = self.count_tokens(' ' + word)
            if current_tokens + word_tokens < beginning_tokens:
                beginning_words.append(word)
                current_tokens += word_tokens
            else:
                break

        # Get ending
        ending_words = []
        current_tokens = 0
        for word in reversed(document.split()):
            word_tokens = self.count_tokens(' ' + word)
            if current_tokens + word_tokens < ending_tokens:
                ending_words.insert(0, word)
                current_tokens += word_tokens
            else:
                break

        beginning_text = ' '.join(beginning_words)
        ending_text = ' '.join(ending_words)

        return f"{beginning_text}... [MIDDLE SECTION TRUNCATED] ...{ending_text}"

    def check_prompt_length(self, prompt: str) -> Dict[str, Any]:
        """
        Check if a prompt exceeds token limits

        Returns:
            Dictionary with token count, status, and recommendations
        """
        token_count = self.count_tokens(prompt)

        status = {
            'token_count': token_count,
            'max_allowed': self.max_context_tokens,
            'within_limit': token_count <= self.max_context_tokens,
            'usage_percentage': (token_count / self.max_context_tokens) * 100
        }

        if token_count > self.max_context_tokens:
            status['recommendation'] = f"Reduce context by {token_count - self.max_context_tokens} tokens"
        elif token_count > self.max_context_tokens * 0.9:
            status['recommendation'] = "Context is near limit, consider reducing for safety"
        else:
            status['recommendation'] = "Prompt length is acceptable"

        return status

# Factory function for easy instantiation
def create_token_manager(model_type: str = "qwen", max_tokens: int = None) -> TokenManager:
    """
    Create token manager with appropriate settings for different models

    Args:
        model_type: "qwen" or "llama3" (determines context window size)
        max_tokens: Override default token limit
    """
    # Set conservative defaults based on model type
    if max_tokens is None:
        if model_type == "qwen":
            max_tokens = 3500  # Conservative for Qwen models
        elif model_type == "llama3":
            max_tokens = 3500  # Conservative for LLAMA3 models
        else:
            max_tokens = 3000  # Very conservative default

    return TokenManager(max_context_tokens=max_tokens)