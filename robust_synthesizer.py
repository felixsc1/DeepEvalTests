"""
Robust synthesizer with batch processing, resume functionality, and retry logic
"""
import os
import json
import time
import asyncio
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import shutil
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import logging

from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import EvolutionConfig, ContextConstructionConfig, FiltrationConfig
from deepeval.models import GPTModel, OpenAIEmbeddingModel
from deepeval.dataset import Golden

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SynthesisProgress:
    """Tracks synthesis progress for resume functionality"""
    processed_paths: List[str]
    completed_goldens: List[Dict]
    failed_paths: List[str]
    last_batch_index: int
    total_paths: int
    start_time: float
    last_save_time: float

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'SynthesisProgress':
        return cls(**data)

class RobustSynthesizer:
    """
    A robust synthesizer that processes documents in batches with:
    - Incremental saving
    - Resume functionality
    - Exponential backoff retry
    - Better error handling
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",  # Changed from gpt-5 to gpt-4o-mini for reliability
        embedding_model: str = "text-embedding-3-small",
        batch_size: int = 3,  # Process only 3 documents at a time
        max_concurrent: int = 4,  # Very conservative concurrency
        progress_file: str = "synthesis_progress.json",
        output_file: str = "synthetic_dataset_with_urls.json",
        max_retries: int = 5,
        base_timeout: int = 60,  # Shorter base timeout
    ):
        self.batch_size = batch_size
        self.progress_file = progress_file
        self.output_file = output_file
        self.max_retries = max_retries
        self.base_timeout = base_timeout

        # Initialize models with conservative settings
        self.llm = GPTModel(
            model=llm_model,
            generation_kwargs={
                "timeout": base_timeout,
                # Note: max_retries is handled at the synthesizer level, not in model kwargs
            }
        )
        self.embedding_model = OpenAIEmbeddingModel(
            model=embedding_model,
            timeout=base_timeout,
        )

        # Initialize synthesizer with conservative settings
        evolution_config = EvolutionConfig(num_evolutions=0)

        context_cfg = ContextConstructionConfig(
            critic_model=self.llm,
            embedder=self.embedding_model,
            max_contexts_per_document=1,  # Reduced from 2 to 1
            context_quality_threshold=0.7,  # Increased threshold for better quality
            context_similarity_threshold=0.4,  # Increased to reduce duplicates
            max_retries=3,
        )

        filtration_cfg = FiltrationConfig(
            critic_model=self.llm,
            synthetic_input_quality_threshold=0.6,  # Increased threshold
            max_quality_retries=2,  # Reduced retries
        )

        self.synthesizer = Synthesizer(
            model=self.llm,
            async_mode=False,  # Use sync mode for better control
            evolution_config=evolution_config,
            filtration_config=filtration_cfg,
            max_concurrent=max_concurrent,
        )

        self.context_cfg = context_cfg

    def load_progress(self) -> Optional[SynthesisProgress]:
        """Load synthesis progress from file"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                progress = SynthesisProgress.from_dict(data)
                logger.info(f"Loaded progress: {len(progress.processed_paths)}/{progress.total_paths} paths processed")
                return progress
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}")
        return None

    def save_progress(self, progress: SynthesisProgress):
        """Save synthesis progress to file"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress.to_dict(), f, indent=2)
            progress.last_save_time = time.time()
            logger.info(f"Progress saved: {len(progress.processed_paths)}/{progress.total_paths} paths processed")
        except Exception as e:
            logger.error(f"Could not save progress: {e}")

    def save_goldens(self, goldens: List[Golden], path_to_url: Dict[str, str]):
        """Save goldens to output file"""
        try:
            # Load existing goldens if file exists
            existing_goldens = []
            if os.path.exists(self.output_file):
                with open(self.output_file, 'r') as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        existing_goldens = existing_data

            # Process new goldens
            for golden in goldens:
                if golden.source_file:
                    url = path_to_url.get(golden.source_file, 'unknown')
                    if not hasattr(golden, 'additional_metadata') or golden.additional_metadata is None:
                        golden.additional_metadata = {}
                    golden.additional_metadata['source_url'] = url

            # Combine and save
            all_goldens = existing_goldens + [g.__dict__ for g in goldens]
            with open(self.output_file, 'w') as f:
                json.dump(all_goldens, f, indent=2)

            logger.info(f"Saved {len(goldens)} goldens to {self.output_file} (total: {len(all_goldens)})")
        except Exception as e:
            logger.error(f"Could not save goldens: {e}")

    def exponential_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay"""
        return min(2 ** attempt, 60)  # Max 60 seconds

    async def process_batch_with_retry(
        self,
        batch_paths: List[str],
        attempt: int = 0
    ) -> Tuple[List[Golden], List[str]]:
        """Process a batch of paths with retry logic"""
        try:
            logger.info(f"Processing batch of {len(batch_paths)} paths (attempt {attempt + 1})")

            # Use sync mode but with timeout wrapper
            goldens = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.synthesizer.generate_goldens_from_docs(
                        document_paths=batch_paths,
                        max_goldens_per_context=1,  # Reduced from 2 to 1
                        include_expected_output=True,
                        context_construction_config=self.context_cfg
                    )
                ),
                timeout=self.base_timeout * 3  # 3x the base timeout
            )

            logger.info(f"Successfully processed batch: {len(goldens)} goldens generated")
            return goldens, []

        except asyncio.TimeoutError:
            if attempt < self.max_retries:
                delay = self.exponential_backoff(attempt)
                logger.warning(f"Batch timed out, retrying in {delay}s (attempt {attempt + 1}/{self.max_retries + 1})")
                await asyncio.sleep(delay)
                return await self.process_batch_with_retry(batch_paths, attempt + 1)
            else:
                logger.error(f"Batch failed after {self.max_retries + 1} attempts")
                return [], batch_paths

        except Exception as e:
            if attempt < self.max_retries:
                delay = self.exponential_backoff(attempt)
                logger.warning(f"Batch failed with error: {e}, retrying in {delay}s (attempt {attempt + 1}/{self.max_retries + 1})")
                await asyncio.sleep(delay)
                return await self.process_batch_with_retry(batch_paths, attempt + 1)
            else:
                logger.error(f"Batch failed permanently after {self.max_retries + 1} attempts: {e}")
                return [], batch_paths

    async def process_all_documents(
        self,
        all_paths: List[str],
        path_to_url: Dict[str, str]
    ) -> List[Golden]:
        """Process all documents with batching and resume functionality"""
        # Load existing progress
        progress = self.load_progress()
        if progress is None:
            progress = SynthesisProgress(
                processed_paths=[],
                completed_goldens=[],
                failed_paths=[],
                last_batch_index=0,
                total_paths=len(all_paths),
                start_time=time.time(),
                last_save_time=time.time()
            )

        # Filter out already processed paths
        remaining_paths = [p for p in all_paths if p not in progress.processed_paths]
        logger.info(f"Resuming synthesis: {len(remaining_paths)} paths remaining out of {len(all_paths)}")

        all_goldens = []

        # Process in batches
        for i in range(0, len(remaining_paths), self.batch_size):
            batch_paths = remaining_paths[i:i + self.batch_size]
            batch_index = progress.last_batch_index + (i // self.batch_size) + 1

            logger.info(f"Processing batch {batch_index}: {len(batch_paths)} paths")

            # Process batch with retry
            batch_goldens, failed_paths = await self.process_batch_with_retry(batch_paths)

            # Update progress
            progress.processed_paths.extend([p for p in batch_paths if p not in failed_paths])
            progress.failed_paths.extend(failed_paths)
            progress.last_batch_index = batch_index
            all_goldens.extend(batch_goldens)

            # Save progress and results periodically
            if len(all_goldens) > 0 or len(failed_paths) > 0:
                self.save_goldens(all_goldens, path_to_url)
                self.save_progress(progress)
                all_goldens = []  # Clear to avoid duplicate saving

            # Small delay between batches to prevent overwhelming the API
            await asyncio.sleep(1)

        # Save any remaining goldens
        if all_goldens:
            self.save_goldens(all_goldens, path_to_url)
            self.save_progress(progress)

        # Final summary
        total_time = time.time() - progress.start_time
        logger.info(".2f")
        logger.info(f"Failed paths: {len(progress.failed_paths)}")

        return progress.completed_goldens

def cleanup_vector_db():
    """Clean up any corrupted vector database files"""
    db_paths = ['.vector_db', '.chroma_db', 'chroma.sqlite3']
    for db_path in db_paths:
        if os.path.exists(db_path):
            try:
                if os.path.isdir(db_path):
                    shutil.rmtree(db_path)
                else:
                    os.remove(db_path)
                logger.info(f"Cleaned up {db_path}")
            except Exception as e:
                logger.warning(f"Could not clean up {db_path}: {e}")

async def run_robust_synthesis(
    document_paths: List[str],
    path_to_url: Dict[str, str],
    batch_size: int = 3,
    max_concurrent: int = 4
):
    """Main function to run robust synthesis"""
    logger.info("Starting robust synthesis...")

    # Clean up any corrupted databases
    cleanup_vector_db()

    # Initialize synthesizer
    synthesizer = RobustSynthesizer(
        batch_size=batch_size,
        max_concurrent=max_concurrent,
    )

    # Run synthesis
    await synthesizer.process_all_documents(document_paths, path_to_url)

    logger.info("Synthesis completed!")

if __name__ == "__main__":
    # Example usage
    import json

    # Load document paths
    with open('filtered_document_paths.json', 'r') as f:
        data = json.load(f)
    document_paths = data['filtered_document_paths']

    # Create path_to_url mapping (you'll need to implement this based on your data)
    path_to_url = {}  # Implement this based on your source_map.json files

    # Run synthesis
    asyncio.run(run_robust_synthesis(document_paths, path_to_url))
