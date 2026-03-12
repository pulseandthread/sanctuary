"""
Migration Script: MiniLM (384D) → Gemini Embedding 2 (3072D)

Re-embeds all existing memories from the legacy collection into a new v2 collection
using Gemini Embedding 2 for richer semantic matching and multimodal support.

Run this once after upgrading to migrate existing memories.
New installations don't need this — they start with v2 automatically.

Usage:
    python migrate_embeddings.py
"""
import time
import logging
import chromadb
from chromadb.config import Settings
from google import genai
from google.genai import types as genai_types
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def migrate():
    # Initialize ChromaDB
    client = chromadb.PersistentClient(
        path=Config.CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )

    # Initialize Gemini client
    gemini = genai.Client(api_key=Config.GOOGLE_API_KEY)
    logger.info("Gemini Embedding 2 client initialized")

    # Find legacy collections to migrate
    for entity_name, entity_config in Config.ENTITIES.items():
        legacy_name = entity_config["collection_name"]
        v2_name = legacy_name + "_v2"

        try:
            legacy = client.get_collection(name=legacy_name)
        except Exception:
            logger.info(f"No legacy collection '{legacy_name}' found for {entity_name} — skipping")
            continue

        legacy_count = legacy.count()
        if legacy_count == 0:
            logger.info(f"Legacy collection '{legacy_name}' is empty — skipping")
            continue

        logger.info(f"Legacy collection: {legacy_name} ({legacy_count} memories)")

        # Get or create v2 collection
        v2 = client.get_or_create_collection(
            name=v2_name,
            metadata={"entity": entity_name, "embedding_model": "gemini-embedding-2"}
        )
        existing_v2 = v2.count()
        logger.info(f"V2 collection: {v2_name} ({existing_v2} existing memories)")

        # Pull all memories from legacy
        results = legacy.get(include=["documents", "metadatas"])
        if not results or not results["ids"]:
            logger.info("No memories to migrate")
            continue

        total = len(results["ids"])
        logger.info(f"Migrating {total} memories...")

        # Get IDs already in v2 to skip duplicates (for resumable migration)
        existing_ids = set()
        if existing_v2 > 0:
            existing_results = v2.get()
            existing_ids = set(existing_results["ids"])
            logger.info(f"Skipping {len(existing_ids)} already-migrated memories")

        # Migrate in batches
        batch_size = 20
        migrated = 0
        skipped = 0
        failed = 0

        for i in range(0, total, batch_size):
            batch_ids = results["ids"][i:i + batch_size]
            batch_docs = results["documents"][i:i + batch_size]
            batch_metas = results["metadatas"][i:i + batch_size]

            # Filter out already-migrated
            new_ids = []
            new_docs = []
            new_metas = []
            for j, mid in enumerate(batch_ids):
                if mid in existing_ids:
                    skipped += 1
                    continue
                new_ids.append(mid)
                new_docs.append(batch_docs[j])
                new_metas.append(batch_metas[j])

            if not new_ids:
                continue

            # Generate embeddings for this batch
            batch_embeddings = []
            for doc in new_docs:
                try:
                    result = gemini.models.embed_content(
                        model=Config.GEMINI_EMBEDDING_MODEL,
                        contents=doc,
                        config=genai_types.EmbedContentConfig(
                            task_type="RETRIEVAL_DOCUMENT",
                            output_dimensionality=Config.GEMINI_EMBEDDING_DIMENSIONS
                        )
                    )
                    batch_embeddings.append(result.embeddings[0].values)
                except Exception as e:
                    logger.error(f"Failed to embed memory {new_ids[len(batch_embeddings)]}: {e}")
                    failed += 1
                    batch_embeddings.append(None)

            # Filter out failed embeddings
            final_ids = []
            final_docs = []
            final_metas = []
            final_embeddings = []
            for j, emb in enumerate(batch_embeddings):
                if emb is not None:
                    final_ids.append(new_ids[j])
                    final_docs.append(new_docs[j])
                    final_metas.append(new_metas[j])
                    final_embeddings.append(emb)

            if final_ids:
                v2.add(
                    ids=final_ids,
                    embeddings=final_embeddings,
                    documents=final_docs,
                    metadatas=final_metas
                )
                migrated += len(final_ids)

            logger.info(f"  Progress: {migrated + skipped}/{total} (migrated: {migrated}, skipped: {skipped}, failed: {failed})")

            # Small delay to be gentle on the API
            if i + batch_size < total:
                time.sleep(0.5)

        # Final stats
        logger.info("=" * 60)
        logger.info(f"MIGRATION COMPLETE for {entity_name}")
        logger.info(f"  Total memories: {total}")
        logger.info(f"  Migrated: {migrated}")
        logger.info(f"  Skipped (already existed): {skipped}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  V2 collection now has: {v2.count()} memories")
        logger.info("=" * 60)

        if failed > 0:
            logger.warning(f"{failed} memories failed to migrate. Run this script again to retry.")


if __name__ == "__main__":
    migrate()
