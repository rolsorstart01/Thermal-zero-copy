"""
Context Deduplication Engine for ThermoCache.

This module implements the context fingerprinting and caching system that enables
zero-copy context reuse across inference requests. It uses SHA-256 hashing to identify
identical contexts and maintains metadata for efficient lookup and lifecycle management.
"""

import hashlib
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import asyncio

from models.schemas import ContextMetadata


logger = logging.getLogger(__name__)


class ContextFingerprinter:
    """
    Generates fingerprints for context content.
    
    Uses SHA-256 hashing to create unique identifiers for context content.
    Supports both full content hashing and prefix-based partial matching.
    """
    
    def __init__(self, prefix_length: int = 256):
        """
        Initialize the fingerprinter.
        
        Args:
            prefix_length: Number of tokens to use for prefix matching
        """
        self.prefix_length = prefix_length
    
    def compute_hash(self, content: str) -> str:
        """
        Compute SHA-256 hash of content.
        
        Args:
            content: The context content to hash
            
        Returns:
            Hex-encoded SHA-256 hash
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def tokenize(self, content: str) -> List[int]:
        """
        Simple tokenization for demonstration purposes.
        
        In production, this would integrate with the actual LLM tokenizer
        (e.g., HuggingFace transformers, tiktoken for GPT models).
        
        Args:
            content: Text content to tokenize
            
        Returns:
            List of token IDs (simulated)
        """
        # Simple word-level tokenization for demo
        # In production, replace with actual LLM tokenizer
        words = content.lower().split()
        # Map words to pseudo-token IDs based on hash
        return [abs(hash(word)) % 100000 for word in words]
    
    def get_prefix_tokens(self, content: str) -> List[int]:
        """
        Get prefix tokens for partial matching.
        
        Args:
            content: Text content
            
        Returns:
            First N tokens as prefix
        """
        tokens = self.tokenize(content)
        return tokens[:self.prefix_length]
    
    def fingerprint(self, content: str, context_id: Optional[str] = None) -> Tuple[str, List[int]]:
        """
        Generate complete fingerprint for content.
        
        Args:
            content: The context content
            context_id: Optional explicit context ID
            
        Returns:
            Tuple of (hash, prefix_tokens)
        """
        content_hash = self.compute_hash(content)
        prefix_tokens = self.get_prefix_tokens(content)
        return content_hash, prefix_tokens


class ContextIndex:
    """
    Index for tracking and looking up cached contexts.
    
    Maintains a mapping from context hashes/fingerprints to their
    metadata including GPU location, memory size, and access patterns.
    """
    
    def __init__(self, max_contexts: int = 10000):
        """
        Initialize the context index.
        
        Args:
            max_contexts: Maximum number of contexts to track
        """
        self.max_contexts = max_contexts
        self._contexts: Dict[str, ContextMetadata] = {}
        self._hash_to_id: Dict[str, str] = {}
        self._prefix_index: Dict[int, List[str]] = {}  # prefix_token -> context_ids
        self._lock = asyncio.Lock()
        
        # Statistics
        self.total_lookups = 0
        self.hits = 0
        self.misses = 0
    
    async def register_context(
        self,
        context_id: str,
        content: str,
        gpu_location: int,
        fingerprinter: ContextFingerprinter
    ) -> ContextMetadata:
        """
        Register a new context in the index.
        
        Args:
            context_id: Unique identifier for the context
            content: The context content
            gpu_location: GPU where context is stored
            fingerprinter: Fingerprinter instance
            
        Returns:
            Created ContextMetadata
        """
        async with self._lock:
            content_hash, prefix_tokens = fingerprinter.fingerprint(content)
            
            # Check if identical context already exists
            if content_hash in self._hash_to_id:
                existing_id = self._hash_to_id[content_hash]
                existing_ctx = self._contexts[existing_id]
                existing_ctx.reference_count += 1
                existing_ctx.last_accessed = datetime.now()
                logger.info(f"Context {context_id} matches existing {existing_id}")
                return existing_ctx
            
            # Create new context metadata
            # Estimate memory: ~2 bytes per token for KV cache overhead
            token_count = len(fingerprinter.tokenize(content))
            memory_size = token_count * 2 / (1024 * 1024)  # GB estimate
            
            metadata = ContextMetadata(
                context_id=context_id,
                hash=content_hash,
                token_count=token_count,
                gpu_location=gpu_location,
                memory_size=memory_size,
                creation_time=datetime.now(),
                last_accessed=datetime.now(),
                reference_count=1,
                prefix_tokens=prefix_tokens
            )
            
            self._contexts[context_id] = metadata
            self._hash_to_id[content_hash] = context_id
            
            # Index prefix tokens for partial matching
            for token in prefix_tokens:
                if token not in self._prefix_index:
                    self._prefix_index[token] = []
                if context_id not in self._prefix_index[token]:
                    self._prefix_index[token].append(context_id)
            
            logger.info(f"Registered context {context_id} on GPU {gpu_location}")
            return metadata
    
    async def lookup_by_hash(self, content: str) -> Optional[ContextMetadata]:
        """
        Look up context by content hash.
        
        Args:
            content: The context content to look up
            
        Returns:
            ContextMetadata if found, None otherwise
        """
        async with self._lock:
            self.total_lookups += 1
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            
            if content_hash in self._hash_to_id:
                self.hits += 1
                context_id = self._hash_to_id[content_hash]
                ctx = self._contexts[context_id]
                ctx.reference_count += 1
                ctx.last_accessed = datetime.now()
                logger.debug(f"Context cache HIT for hash {content_hash[:16]}...")
                return ctx
            
            self.misses += 1
            logger.debug(f"Context cache MISS for hash {content_hash[:16]}...")
            return None
    
    async def lookup_by_prefix(self, prefix_tokens: List[int]) -> List[ContextMetadata]:
        """
        Find contexts that share prefix tokens.
        
        This enables partial context reuse when exact matches aren't available.
        
        Args:
            prefix_tokens: Tokens to match against
            
        Returns:
            List of matching ContextMetadata sorted by match quality
        """
        async with self._lock:
            if not prefix_tokens:
                return []
            
            # Count how many prefix tokens each context shares
            context_scores: Dict[str, int] = {}
            for token in prefix_tokens:
                if token in self._prefix_index:
                    for ctx_id in self._prefix_index[token]:
                        context_scores[ctx_id] = context_scores.get(ctx_id, 0) + 1
            
            # Sort by score (number of matching prefix tokens)
            sorted_ids = sorted(context_scores.keys(), key=lambda x: context_scores[x], reverse=True)
            
            results = []
            for ctx_id in sorted_ids[:10]:  # Return top 10 matches
                if ctx_id in self._contexts:
                    results.append(self._contexts[ctx_id])
            
            return results
    
    async def lookup_by_id(self, context_id: str) -> Optional[ContextMetadata]:
        """
        Look up context by explicit ID.
        
        Args:
            context_id: The context identifier
            
        Returns:
            ContextMetadata if found, None otherwise
        """
        async with self._lock:
            self.total_lookups += 1
            if context_id in self._contexts:
                self.hits += 1
                ctx = self._contexts[context_id]
                ctx.reference_count += 1
                ctx.last_accessed = datetime.now()
                return ctx
            self.misses += 1
            return None
    
    async def get_all_contexts(self) -> List[ContextMetadata]:
        """Get all registered contexts."""
        async with self._lock:
            return list(self._contexts.values())
    
    async def get_contexts_on_gpu(self, gpu_id: int) -> List[ContextMetadata]:
        """Get all contexts located on a specific GPU."""
        async with self._lock:
            return [ctx for ctx in self._contexts.values() if ctx.gpu_location == gpu_id]
    
    async def update_gpu_location(self, context_id: str, new_gpu: int) -> bool:
        """
        Update the GPU location for a context (after migration).
        
        Args:
            context_id: Context to update
            new_gpu: New GPU location
            
        Returns:
            True if updated, False if context not found
        """
        async with self._lock:
            if context_id in self._contexts:
                self._contexts[context_id].gpu_location = new_gpu
                return True
            return False
    
    async def decrement_reference(self, context_id: str) -> int:
        """
        Decrement reference count for a context.
        
        Args:
            context_id: Context to update
            
        Returns:
            New reference count
        """
        async with self._lock:
            if context_id in self._contexts:
                self._contexts[context_id].reference_count -= 1
                return self._contexts[context_id].reference_count
            return 0
    
    async def remove_context(self, context_id: str) -> bool:
        """
        Remove a context from the index.
        
        Args:
            context_id: Context to remove
            
        Returns:
            True if removed, False if not found
        """
        async with self._lock:
            if context_id not in self._contexts:
                return False
            
            ctx = self._contexts[context_id]
            
            # Remove from hash index
            if ctx.hash in self._hash_to_id:
                del self._hash_to_id[ctx.hash]
            
            # Remove from prefix index
            if ctx.prefix_tokens:
                for token in ctx.prefix_tokens:
                    if token in self._prefix_index and context_id in self._prefix_index[token]:
                        self._prefix_index[token].remove(context_id)
            
            del self._contexts[context_id]
            logger.info(f"Removed context {context_id} from index")
            return True
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_lookups == 0:
            return 0.0
        return self.hits / self.total_lookups
    
    @property
    def total_contexts(self) -> int:
        """Get total number of registered contexts."""
        return len(self._contexts)
    
    @property
    def total_memory_used(self) -> float:
        """Get total memory used by all contexts in GB."""
        return sum(ctx.memory_size for ctx in self._contexts.values())


class ContextCache:
    """
    Main context deduplication engine.
    
    Combines fingerprinting and indexing to provide a high-level API
    for context registration, lookup, and lifecycle management.
    """
    
    def __init__(self, max_contexts: int = 10000, prefix_length: int = 256):
        """
        Initialize the context cache.
        
        Args:
            max_contexts: Maximum contexts to track
            prefix_length: Prefix length for partial matching
        """
        self.fingerprinter = ContextFingerprinter(prefix_length=prefix_length)
        self.index = ContextIndex(max_contexts=max_contexts)
        self._content_store: Dict[str, str] = {}  # context_id -> content (for demo)
        
        # Track memory savings from deduplication
        self.duplicate_requests = 0
        self.saved_contexts = 0
    
    async def register_or_reuse(
        self,
        content: str,
        gpu_location: int,
        context_id: Optional[str] = None
    ) -> Tuple[ContextMetadata, bool]:
        """
        Register context or return existing match.
        
        Args:
            content: The context content
            gpu_location: GPU to store on
            context_id: Optional explicit ID
            
        Returns:
            Tuple of (ContextMetadata, is_new)
        """
        # Generate ID if not provided
        if context_id is None:
            content_hash = self.fingerprinter.compute_hash(content)
            context_id = f"ctx_{content_hash[:16]}"
        
        # Try to find existing context
        existing = await self.index.lookup_by_hash(content)
        if existing:
            self.duplicate_requests += 1
            return existing, False
        
        # Register new context
        metadata = await self.index.register_context(
            context_id=context_id,
            content=content,
            gpu_location=gpu_location,
            fingerprinter=self.fingerprinter
        )
        self._content_store[context_id] = content
        self.saved_contexts += 1
        return metadata, True
    
    async def find_by_id(self, context_id: str) -> Optional[ContextMetadata]:
        """Find context by explicit ID."""
        return await self.index.lookup_by_id(context_id)
    
    async def find_by_content(self, content: str) -> Optional[ContextMetadata]:
        """Find context by content hash."""
        return await self.index.lookup_by_hash(content)
    
    async def find_similar(self, content: str) -> List[ContextMetadata]:
        """Find contexts with similar prefixes."""
        _, prefix_tokens = self.fingerprinter.fingerprint(content)
        return await self.index.lookup_by_prefix(prefix_tokens)
    
    async def migrate_context(self, context_id: str, new_gpu: int) -> bool:
        """
        Migrate a context to a different GPU.
        
        Args:
            context_id: Context to migrate
            new_gpu: Destination GPU
            
        Returns:
            True if migration successful
        """
        success = await self.index.update_gpu_location(context_id, new_gpu)
        if success:
            logger.info(f"Migrated context {context_id} to GPU {new_gpu}")
        return success
    
    async def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "total_contexts": self.index.total_contexts,
            "hit_rate": self.index.hit_rate,
            "total_lookups": self.index.total_lookups,
            "hits": self.index.hits,
            "misses": self.index.misses,
            "duplicate_requests": self.duplicate_requests,
            "saved_contexts": self.saved_contexts,
            "total_memory_gb": self.index.total_memory_used
        }
