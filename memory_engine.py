"""
Sanctuary Server - Memory Engine
Implements the hybrid STATE/EVENT memory system with Smart Sieve retrieval
"""
import uuid
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryCapsule:
    """
    Represents a single memory capsule in the system

    Types:
        EVENT: A moment in time, a story, a conversation (never expires)
        STATE: A fact about the world that can change (can be superseded)
        TRANSIENT: Temporary context that expires after 2 weeks

    Status:
        ACTIVE: Current/valid memory
        SUPERSEDED: Old state that has been replaced
        EXPIRED: Transient memory past expiration date
    """

    def __init__(
        self,
        summary: str,
        entities: List[str],
        memory_type: str = "EVENT",
        status: str = "ACTIVE",
        memory_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        expiration_date: Optional[str] = None,
        topic: Optional[str] = None  # Chat topic for weighted retrieval
    ):
        self.id = memory_id or str(uuid.uuid4())
        self.timestamp = timestamp or datetime.utcnow().isoformat()
        self.summary = summary
        self.entities = entities
        self.type = memory_type.upper()  # EVENT, STATE, or TRANSIENT
        self.status = status.upper()  # ACTIVE, SUPERSEDED, or EXPIRED
        self.topic = topic  # Chat topic (e.g., "general", "health", "creative")

        # Set expiration date for TRANSIENT memories (14 days from now)
        if self.type == "TRANSIENT" and not expiration_date:
            expiration_datetime = datetime.utcnow() + timedelta(days=14)
            self.expiration_date = expiration_datetime.isoformat()
        else:
            self.expiration_date = expiration_date

        # Validate
        if self.type not in ["EVENT", "STATE", "TRANSIENT"]:
            raise ValueError(f"Invalid memory type: {self.type}")
        if self.status not in ["ACTIVE", "SUPERSEDED", "EXPIRED"]:
            raise ValueError(f"Invalid status: {self.status}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert capsule to dictionary for storage"""
        result = {
            "id": self.id,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "entities": self.entities,
            "type": self.type,
            "status": self.status
        }
        if self.expiration_date:
            result["expiration_date"] = self.expiration_date
        if self.topic:
            result["topic"] = self.topic
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryCapsule':
        """Create capsule from dictionary"""
        return cls(
            memory_id=data["id"],
            timestamp=data["timestamp"],
            summary=data["summary"],
            entities=data["entities"],
            memory_type=data["type"],
            status=data["status"],
            expiration_date=data.get("expiration_date"),
            topic=data.get("topic")
        )


class MemoryEngine:
    """
    The core memory system using ChromaDB with STATE/EVENT logic

    Key Features:
    - Vector search for semantic similarity
    - Smart Sieve filtering (removes superseded states)
    - Separate collections per entity (data isolation)
    """

    def __init__(self, entity_name: str):
        """
        Initialize memory engine for a specific entity

        Args:
            entity_name: The entity (e.g., "companion")
        """
        self.entity_name = entity_name.lower()
        self.entity_config = Config.ENTITIES.get(self.entity_name)

        if not self.entity_config:
            raise ValueError(f"Unknown entity: {entity_name}")

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=Config.CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection for this entity
        self.collection_name = self.entity_config["collection_name"]
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"entity": self.entity_name}
        )

        # Initialize embedding model
        logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)

        logger.info(f"Memory engine initialized for entity: {entity_name}")
        logger.info(f"Collection: {self.collection_name} ({self.collection.count()} memories)")

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text"""
        return self.embedding_model.encode(text).tolist()

    def save_memory(self, capsule: MemoryCapsule) -> str:
        """
        Save a memory capsule to ChromaDB

        If the capsule is a STATE, check for existing active states with
        overlapping entities and mark them as SUPERSEDED.

        Args:
            capsule: The memory capsule to save

        Returns:
            The memory ID
        """
        # If this is a STATE, supersede old states with overlapping entities
        if capsule.type == "STATE":
            self._supersede_old_states(capsule.entities)

        # Generate embedding
        embedding = self._generate_embedding(capsule.summary)

        # Build metadata
        metadata = {
            "timestamp": capsule.timestamp,
            "entities": ",".join(capsule.entities),  # Store as comma-separated
            "type": capsule.type,
            "status": capsule.status
        }

        # Add expiration_date for TRANSIENT memories
        if capsule.expiration_date:
            metadata["expiration_date"] = capsule.expiration_date

        # Add topic for weighted retrieval
        if capsule.topic:
            metadata["topic"] = capsule.topic

        # Save to ChromaDB
        self.collection.add(
            ids=[capsule.id],
            embeddings=[embedding],
            documents=[capsule.summary],
            metadatas=[metadata]
        )

        logger.info(f"Saved memory: {capsule.id} ({capsule.type}, {capsule.status})")
        if capsule.type == "TRANSIENT":
            logger.info(f"  Expires: {capsule.expiration_date}")
        return capsule.id

    def _supersede_old_states(self, entities: List[str]):
        """
        Find and supersede old STATE memories with overlapping entities

        This is critical logic: When a new STATE is saved (e.g., "Current bike: Tuono"),
        we need to mark the old STATE (e.g., "Current bike: Ducati") as SUPERSEDED.

        Args:
            entities: Entity tags of the new STATE
        """
        # Query all active STATE memories
        results = self.collection.get(
            where={
                "$and": [
                    {"type": "STATE"},
                    {"status": "ACTIVE"}
                ]
            }
        )

        if not results or not results["ids"]:
            return

        # Check each memory for overlapping entities
        for idx, memory_id in enumerate(results["ids"]):
            metadata = results["metadatas"][idx]
            existing_entities = set(metadata["entities"].split(","))
            new_entities = set(entities)

            # If entities overlap, supersede this memory
            if existing_entities & new_entities:  # Set intersection
                logger.info(f"Superseding old STATE: {memory_id} (entities: {existing_entities})")
                self.collection.update(
                    ids=[memory_id],
                    metadatas=[{
                        **metadata,
                        "status": "SUPERSEDED"
                    }]
                )

    def retrieve_memories(
        self,
        query: str,
        limit: int = None,
        current_topic: str = None,
        topic_boost: float = 0.2
    ) -> List[MemoryCapsule]:
        """
        Retrieve relevant memories using the Smart Sieve with optional topic weighting

        Process:
        1. Vector search for semantic similarity
        2. Filter out SUPERSEDED states
        3. Boost memories matching current topic (if provided)
        4. Sort by weighted score (topic + recency)

        Args:
            query: Search query text
            limit: Maximum number of memories to return
            current_topic: Current chat topic for weighted retrieval
            topic_boost: How much to boost topic-matching memories (default 0.2 = 20%)

        Returns:
            List of relevant memory capsules
        """
        limit = limit or Config.MAX_MEMORY_RETRIEVAL

        # Generate query embedding
        query_embedding = self._generate_embedding(query)

        # Vector search (get more than limit to account for filtering)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit * 3,  # Get extra to account for filtering and reranking
            include=["metadatas", "documents", "distances"]
        )

        if not results or not results["ids"][0]:
            logger.info("No memories found for query")
            return []

        # Parse results into capsules with scores
        scored_capsules = []
        for idx, memory_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][idx]
            document = results["documents"][0][idx]
            distance = results["distances"][0][idx] if results.get("distances") else 0

            # Apply Smart Sieve: Filter out SUPERSEDED states
            if metadata["type"] == "STATE" and metadata["status"] == "SUPERSEDED":
                logger.debug(f"Filtering out superseded state: {memory_id}")
                continue

            # Filter out expired TRANSIENT memories
            if self._is_expired(metadata):
                logger.debug(f"Filtering out expired TRANSIENT: {memory_id}")
                continue

            capsule = MemoryCapsule(
                memory_id=memory_id,
                timestamp=metadata["timestamp"],
                summary=document,
                entities=metadata["entities"].split(","),
                memory_type=metadata["type"],
                status=metadata["status"],
                expiration_date=metadata.get("expiration_date"),
                topic=metadata.get("topic")
            )

            # Calculate weighted score
            # Lower distance = more similar, so we invert it for scoring
            base_score = 1.0 / (1.0 + distance) if distance else 1.0

            # Apply topic boost if memory matches current topic
            if current_topic and capsule.topic and capsule.topic.lower() == current_topic.lower():
                base_score *= (1.0 + topic_boost)
                logger.debug(f"Topic boost applied to memory: {memory_id} (topic: {capsule.topic})")

            scored_capsules.append((capsule, base_score))

        # Sort by score (highest first)
        scored_capsules.sort(key=lambda x: x[1], reverse=True)

        # Extract capsules and limit
        capsules = [c for c, _ in scored_capsules[:limit]]

        logger.info(f"Retrieved {len(capsules)} memories (filtered from {len(results['ids'][0])}, topic: {current_topic or 'none'})")
        return capsules

    def get_all_memories(self) -> List[MemoryCapsule]:
        """Get ALL memories regardless of status, sorted by timestamp descending (newest first)"""
        results = self.collection.get()

        if not results or not results["ids"]:
            return []

        capsules = []
        for idx, memory_id in enumerate(results["ids"]):
            metadata = results["metadatas"][idx]
            document = results["documents"][idx]

            capsule = MemoryCapsule(
                memory_id=memory_id,
                timestamp=metadata["timestamp"],
                summary=document,
                entities=metadata["entities"].split(","),
                memory_type=metadata["type"],
                status=metadata.get("status", "ACTIVE"),  # Default to ACTIVE for old memories
                topic=metadata.get("topic")
            )
            capsules.append(capsule)

        # Sort by timestamp descending (newest first)
        capsules.sort(key=lambda x: x.timestamp, reverse=True)
        return capsules

    def get_all_active_memories(self) -> List[MemoryCapsule]:
        """Get all active memories (no SUPERSEDED states, no expired TRANSIENTs)"""
        results = self.collection.get(
            where={"status": "ACTIVE"}
        )

        if not results or not results["ids"]:
            return []

        capsules = []
        for idx, memory_id in enumerate(results["ids"]):
            metadata = results["metadatas"][idx]
            document = results["documents"][idx]

            # Filter out expired TRANSIENT memories
            if self._is_expired(metadata):
                logger.debug(f"Filtering out expired TRANSIENT: {memory_id}")
                continue

            capsule = MemoryCapsule(
                memory_id=memory_id,
                timestamp=metadata["timestamp"],
                summary=document,
                entities=metadata["entities"].split(","),
                memory_type=metadata["type"],
                status=metadata["status"],
                expiration_date=metadata.get("expiration_date"),
                topic=metadata.get("topic")
            )
            capsules.append(capsule)

        # Sort by timestamp
        capsules.sort(key=lambda c: c.timestamp, reverse=True)
        return capsules

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system"""
        total = self.collection.count()

        # Count by type
        events = self.collection.get(where={"type": "EVENT"})
        states = self.collection.get(where={"type": "STATE"})

        # Count by status
        active = self.collection.get(where={"status": "ACTIVE"})
        superseded = self.collection.get(where={"status": "SUPERSEDED"})

        return {
            "entity": self.entity_name,
            "total_memories": total,
            "events": len(events["ids"]) if events else 0,
            "states": len(states["ids"]) if states else 0,
            "active": len(active["ids"]) if active else 0,
            "superseded": len(superseded["ids"]) if superseded else 0
        }

    def get_memory_by_id(self, memory_id: str) -> Optional[MemoryCapsule]:
        """
        Get a specific memory by ID

        Args:
            memory_id: The ID of the memory to retrieve

        Returns:
            MemoryCapsule if found, None otherwise
        """
        try:
            result = self.collection.get(ids=[memory_id])
            if not result or not result["ids"]:
                return None

            metadata = result["metadatas"][0]
            document = result["documents"][0]

            return MemoryCapsule(
                memory_id=memory_id,
                timestamp=metadata["timestamp"],
                summary=document,
                entities=metadata["entities"].split(","),
                memory_type=metadata["type"],
                status=metadata.get("status", "ACTIVE"),
                expiration_date=metadata.get("expiration_date"),
                topic=metadata.get("topic")
            )
        except Exception as e:
            logger.error(f"Error getting memory {memory_id}: {e}")
            return None

    def update_memory(self, memory_id: str, new_content: str, new_tags: List[str] = None) -> bool:
        """
        Update an existing memory's content (The Pearl method - adding layers)

        Args:
            memory_id: The ID of the memory to update
            new_content: The new content/summary for the memory
            new_tags: Optional new free-form tags to add

        Returns:
            True if successful, False if memory not found
        """
        try:
            # Get existing memory
            result = self.collection.get(ids=[memory_id])
            if not result or not result["ids"]:
                logger.warning(f"Memory not found for update: {memory_id}")
                return False

            metadata = result["metadatas"][0]

            # Generate new embedding for updated content
            new_embedding = self._generate_embedding(new_content)

            # Update timestamp to now
            metadata["timestamp"] = datetime.utcnow().isoformat()

            # Add new tags if provided (merge with existing entities)
            if new_tags:
                existing_entities = set(metadata["entities"].split(","))
                existing_entities.update(new_tags)
                metadata["entities"] = ",".join(existing_entities)

            # Update in ChromaDB
            self.collection.update(
                ids=[memory_id],
                embeddings=[new_embedding],
                documents=[new_content],
                metadatas=[metadata]
            )

            logger.info(f"Updated memory: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating memory {memory_id}: {e}")
            return False

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory from the collection

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            True if successful, False if memory not found
        """
        try:
            # Check if memory exists
            result = self.collection.get(ids=[memory_id])
            if not result or not result["ids"]:
                logger.warning(f"Memory not found: {memory_id}")
                return False

            # Delete the memory
            self.collection.delete(ids=[memory_id])
            logger.info(f"Deleted memory: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
            return False

    def cleanup_expired_transients(self) -> int:
        """
        Find and delete expired TRANSIENT memories

        Returns:
            Number of memories cleaned up
        """
        try:
            # Get all TRANSIENT memories
            results = self.collection.get(
                where={"type": "TRANSIENT"}
            )

            if not results or not results["ids"]:
                logger.info("No TRANSIENT memories to check")
                return 0

            # Check expiration and collect expired IDs
            expired_ids = []
            now = datetime.utcnow()

            for idx, memory_id in enumerate(results["ids"]):
                metadata = results["metadatas"][idx]
                expiration_str = metadata.get("expiration_date")

                if expiration_str:
                    expiration_date = datetime.fromisoformat(expiration_str)
                    if now > expiration_date:
                        expired_ids.append(memory_id)
                        logger.info(f"Found expired TRANSIENT: {memory_id}")

            # Delete expired memories
            if expired_ids:
                self.collection.delete(ids=expired_ids)
                logger.info(f"Cleaned up {len(expired_ids)} expired TRANSIENT memories")

            return len(expired_ids)

        except Exception as e:
            logger.error(f"Error cleaning up expired transients: {e}")
            return 0

    def _is_expired(self, metadata: Dict[str, Any]) -> bool:
        """Check if a TRANSIENT memory has expired"""
        if metadata.get("type") != "TRANSIENT":
            return False

        expiration_str = metadata.get("expiration_date")
        if not expiration_str:
            return False

        try:
            expiration_date = datetime.fromisoformat(expiration_str)
            return datetime.utcnow() > expiration_date
        except:
            return False
