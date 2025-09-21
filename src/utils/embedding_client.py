#!/usr/bin/env python3
"""
Robust OpenAI Embedding Client

This module provides a robust wrapper around OpenAI's embedding API
to handle common 400 errors and issues:
- Input validation and sanitization
- Token limit handling and truncation
- Rate limiting and retry logic
- Proper error handling and logging
"""

import re
import time
import logging
import tiktoken
from typing import List, Optional
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

# Suppress HTTP request logs to keep terminal clean during evaluations
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class RobustEmbeddingClient:
    """Robust OpenAI embedding client that handles common API errors"""

    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        """
        Initialize the robust embedding client.

        Args:
            api_key: OpenAI API key
            model: Embedding model to use (default: text-embedding-ada-002)
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

        # Token limits for different models
        self.token_limits = {
            "text-embedding-ada-002": 8192,
            "text-embedding-3-small": 8192,
            "text-embedding-3-large": 8192
        }

        # Initialize tokenizer for token counting
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Close approximation
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")  # Fallback

        logger.info(f"✅ Robust embedding client initialized with model: {model}")

    def sanitize_text(self, text: str) -> str:
        """
        Sanitize input text to prevent API errors.

        Args:
            text: Input text to sanitize

        Returns:
            str: Sanitized text
        """
        if not text:
            return ""

        # Convert to string if not already
        text = str(text)

        # Remove or replace problematic characters
        # Keep only printable ASCII and common unicode characters
        text = re.sub(r'[^\x20-\x7E\u00A0-\u024F\u1E00-\u1EFF]', ' ', text)

        # Replace multiple whitespaces with single space
        text = re.sub(r'\s+', ' ', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for

        Returns:
            int: Number of tokens
        """
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}. Using character approximation.")
            # Fallback: rough approximation (1 token ≈ 4 characters)
            return len(text) // 4

    def truncate_text(self, text: str, max_tokens: Optional[int] = None) -> str:
        """
        Truncate text to fit within token limits.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed (uses model limit if None)

        Returns:
            str: Truncated text
        """
        if not text:
            return ""

        if max_tokens is None:
            max_tokens = self.token_limits.get(self.model, 8192)

        # Leave some buffer for safety
        max_tokens = int(max_tokens * 0.95)

        current_tokens = self.count_tokens(text)

        if current_tokens <= max_tokens:
            return text

        logger.warning(f"Text has {current_tokens} tokens, truncating to {max_tokens}")

        # Binary search for optimal truncation point
        words = text.split()
        left, right = 0, len(words)

        while left < right:
            mid = (left + right + 1) // 2
            truncated = " ".join(words[:mid])

            if self.count_tokens(truncated) <= max_tokens:
                left = mid
            else:
                right = mid - 1

        truncated_text = " ".join(words[:left])
        logger.info(f"Truncated from {current_tokens} to {self.count_tokens(truncated_text)} tokens")

        return truncated_text

    def validate_input(self, text: str) -> bool:
        """
        Validate input text for embedding API.

        Args:
            text: Text to validate

        Returns:
            bool: True if valid, False otherwise
        """
        if not text:
            logger.error("Empty or None text provided")
            return False

        if not isinstance(text, str):
            logger.error(f"Invalid input type: {type(text)}")
            return False

        if len(text.strip()) == 0:
            logger.error("Text contains only whitespace")
            return False

        return True

    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=lambda retry_state: logger.info(f"Retrying embedding request (attempt {retry_state.attempt_number})")
    )
    def _make_embedding_request(self, text: str) -> List[float]:
        """
        Make the actual embedding request with retry logic.

        Args:
            text: Preprocessed text to embed

        Returns:
            List[float]: Embedding vector

        Raises:
            Exception: If all retries fail
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding

        except openai.BadRequestError as e:
            logger.error(f"OpenAI Bad Request (400): {e}")
            # Log additional details for debugging
            logger.error(f"Input text length: {len(text)} chars, {self.count_tokens(text)} tokens")
            logger.error(f"Input preview: {repr(text[:100])}")
            raise

        except openai.RateLimitError as e:
            logger.warning(f"Rate limit hit: {e}")
            raise  # Will be retried by tenacity

        except openai.APIError as e:
            logger.error(f"OpenAI API Error: {e}")
            raise  # Will be retried by tenacity

        except Exception as e:
            logger.error(f"Unexpected embedding error: {e}")
            raise

    def get_embedding(self, text: str, max_tokens: Optional[int] = None) -> Optional[List[float]]:
        """
        Get embedding for text with robust error handling.

        Args:
            text: Input text to embed
            max_tokens: Maximum tokens allowed (uses model limit if None)

        Returns:
            Optional[List[float]]: Embedding vector or None if failed
        """
        # Input validation
        if not self.validate_input(text):
            return None

        # Sanitize the input
        sanitized_text = self.sanitize_text(text)
        if not sanitized_text:
            logger.error("Text became empty after sanitization")
            return None

        # Truncate if necessary
        truncated_text = self.truncate_text(sanitized_text, max_tokens)
        if not truncated_text:
            logger.error("Text became empty after truncation")
            return None

        # Make the embedding request
        try:
            embedding = self._make_embedding_request(truncated_text)
            logger.debug(f"Successfully generated embedding (dimension: {len(embedding)})")
            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding after all retries: {e}")
            return None

    def get_embeddings_batch(self, texts: List[str], max_tokens: Optional[int] = None, batch_size: int = 50) -> List[Optional[List[float]]]:
        """
        Get embeddings for multiple texts using RATE-LIMITED BATCH PROCESSING.

        Args:
            texts: List of texts to embed
            max_tokens: Maximum tokens allowed per text
            batch_size: Number of texts to process in each batch (conservative default: 50)

        Returns:
            List[Optional[List[float]]]: List of embedding vectors (None for failed ones)
        """
        if not texts:
            return []

        # Use smaller batch size for large datasets to avoid rate limits
        if len(texts) > 1000:
            batch_size = min(batch_size, 20)  # Very conservative for large datasets
            delay_between_batches = 2.0  # 2 second delays
        elif len(texts) > 500:
            batch_size = min(batch_size, 30)
            delay_between_batches = 1.0  # 1 second delays
        else:
            delay_between_batches = 0.5  # 0.5 second delays

        logger.info(f"🚀 BATCH EMBEDDING: {len(texts)} embeddings in batches of {batch_size} (delay: {delay_between_batches}s)")

        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        # Process in batches with clean progress bar
        with tqdm(total=len(texts), desc="🔗 Embeddings", unit="emb", ncols=100) as pbar:
            for i in range(0, len(texts), batch_size):
                batch_num = i // batch_size + 1
                batch = texts[i:i + batch_size]

                # Preprocess all texts in batch
                processed_batch = []
                valid_indices = []

                for j, text in enumerate(batch):
                    if not self.validate_input(text):
                        processed_batch.append(None)
                        continue

                    sanitized = self.sanitize_text(text)
                    if not sanitized:
                        processed_batch.append(None)
                        continue

                    truncated = self.truncate_text(sanitized, max_tokens)
                    if not truncated:
                        processed_batch.append(None)
                        continue

                    processed_batch.append(truncated)
                    valid_indices.append(j)

                # Get valid texts for API call
                valid_texts = [text for text in processed_batch if text is not None]

                if not valid_texts:
                    # All texts in batch failed validation
                    all_embeddings.extend([None] * len(batch))
                    pbar.update(len(batch))
                    continue

                # Make SINGLE API call for entire batch with retry logic
                try:
                    batch_embeddings = self._make_batch_embedding_request(valid_texts)

                    # Map results back to original positions
                    embedding_iter = iter(batch_embeddings)
                    batch_results = []

                    for processed_text in processed_batch:
                        if processed_text is None:
                            batch_results.append(None)
                        else:
                            batch_results.append(next(embedding_iter))

                    all_embeddings.extend(batch_results)

                except Exception as e:
                    logger.error(f"Batch {batch_num} failed: {e}. Using individual fallback.")
                    # Fallback to individual processing for this batch
                    batch_results = []
                    for text in batch:
                        embedding = self.get_embedding(text, max_tokens)
                        batch_results.append(embedding)
                        time.sleep(0.1)  # Small delay for individual fallback
                    all_embeddings.extend(batch_results)

                # Update progress bar
                pbar.update(len(batch))
                pbar.set_postfix({'batch': f'{batch_num}/{total_batches}', 'success': f'{len([e for e in all_embeddings if e is not None])}'})

                # Rate limiting: delay between batches (except last batch)
                if batch_num < total_batches:
                    time.sleep(delay_between_batches)

        success_count = sum(1 for e in all_embeddings if e is not None)
        logger.info(f"✅ Batch embedding complete: {success_count}/{len(all_embeddings)} successful ({success_count/len(all_embeddings)*100:.1f}%)")
        return all_embeddings

    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=lambda retry_state: logger.info(f"Retrying batch embedding request (attempt {retry_state.attempt_number})")
    )
    def _make_batch_embedding_request(self, texts: List[str]) -> List[List[float]]:
        """
        Make a SINGLE API call for multiple texts.

        Args:
            texts: List of preprocessed texts to embed

        Returns:
            List[List[float]]: List of embedding vectors

        Raises:
            Exception: If the request fails
        """
        try:
            logger.debug(f"Making batch API call for {len(texts)} texts")
            response = self.client.embeddings.create(
                input=texts,  # Pass entire list in single request
                model=self.model
            )

            embeddings = [data.embedding for data in response.data]
            logger.debug(f"Batch API call successful: {len(embeddings)} embeddings returned")
            return embeddings

        except openai.BadRequestError as e:
            logger.error(f"OpenAI Batch Bad Request (400): {e}")
            logger.error(f"Batch size: {len(texts)}")
            logger.error(f"Sample text lengths: {[len(t) for t in texts[:3]]}")
            raise

        except openai.RateLimitError as e:
            logger.warning(f"Rate limit hit for batch: {e}")
            raise  # Will be retried by tenacity

        except openai.APIError as e:
            logger.error(f"OpenAI Batch API Error: {e}")
            raise  # Will be retried by tenacity

        except Exception as e:
            logger.error(f"Unexpected batch embedding error: {e}")
            raise


def create_embedding_client(api_key: str, model: str = "text-embedding-ada-002") -> RobustEmbeddingClient:
    """
    Factory function to create a robust embedding client.

    Args:
        api_key: OpenAI API key
        model: Embedding model to use

    Returns:
        RobustEmbeddingClient: Configured embedding client
    """
    return RobustEmbeddingClient(api_key, model)


# Example usage and testing
if __name__ == "__main__":
    # This is for testing - requires API key
    import os

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Set OPENAI_API_KEY environment variable to test")
        exit(1)

    # Test the robust client
    client = create_embedding_client(api_key)

    # Test normal case
    embedding = client.get_embedding("This is a test drug side effect query")
    print(f"Normal case: {'✅ Success' if embedding else '❌ Failed'}")

    # Test edge cases
    test_cases = [
        "",  # Empty string
        None,  # None input
        "   ",  # Whitespace only
        "A" * 10000,  # Very long string
        "Special chars: 🔬💊⚠️",  # Unicode characters
        "Control chars: \x00\x01\x02",  # Control characters
    ]

    for i, test_text in enumerate(test_cases):
        embedding = client.get_embedding(str(test_text) if test_text is not None else None)
        result = "✅ Handled" if embedding is not None else "⚠️ Handled gracefully"
        print(f"Edge case {i+1}: {result}")