"""
chuncker
--------
Splits extracted Markdown documents into retrieval-ready chunks.
Applies a header-aware splitter to preserve clause structure and a
recursive character splitter for large sections; produces
``DocumentChunk`` objects carrying full metadata (company, product,
document type, header path) used by the vector index and filters.
"""

import structlog
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from tqdm.auto import tqdm

from sources.config import (
    ChunkingConfig,
    DocumentChunk,
    DocumentMetadata,
    DocumentType,
    config as app_config,
    generate_chunk_id,
)

logger = structlog.get_logger(__name__)


class Chuncker:
    """Split Markdown documents into retrieval-ready DocumentChunk objects.

    Applies a header-aware splitter to preserve clause structure and a
    recursive character splitter for large sections.
    """

    def __init__(self) -> None:
        """Initialize the chunker with the module-level application configuration."""
        self._app_config = app_config

    def chunk_file(
        self,
        file_path: Path,
        document_type: DocumentType,
    ) -> list[DocumentChunk]:
        """Chunk one Markdown file into DocumentChunk objects.

        Args:
            file_path (Path): Path to the Markdown file.
            document_type (DocumentType): Document type used to select chunking parameters.

        Returns:
            list[DocumentChunk]: Chunks produced from the file, each carrying full
                metadata derived from the file path and document type.

        Raises:
            FileNotFoundError: If file_path does not exist.
        """
        file_path = Path(file_path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")

        logger.info(
            "chunking_file", file=file_path.name, document_type=document_type.value
        )

        text = file_path.read_text(encoding="utf-8")
        chunking_config = self._app_config.get_chunking_config(document_type)
        base_metadata = self._parse_metadata_from_path(file_path, document_type)

        header_sections = self._split_by_headers(text, chunking_config)
        splitter = self._build_recursive_splitter(
            chunk_size=chunking_config.chunk_size,
            chunk_overlap=chunking_config.chunk_overlap,
        )

        all_chunks: list[DocumentChunk] = []

        for section in header_sections:
            section_metadata = base_metadata.model_copy(
                update={
                    "header_1": section.metadata.get("header_1"),
                    "header_2": section.metadata.get("header_2"),
                    "header_3": section.metadata.get("header_3"),
                }
            )

            section_chunks = self._split_and_filter(
                splitter=splitter,
                text=section.page_content,
                min_chunk_size=chunking_config.min_chunk_size,
            )

            for chunk_idx, chunk_text in enumerate(section_chunks):
                chunk_id = generate_chunk_id(
                    chunk_text,
                    section_metadata,
                    f"chunk_{chunk_idx}",
                )

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    metadata=section_metadata,
                )
                all_chunks.append(chunk)

        logger.info(
            "chunks_generated",
            total=len(all_chunks),
            file=file_path.name,
        )
        return all_chunks

    def chunk_folder(
        self,
        folder_path: Path,
        document_type: DocumentType,
    ) -> list[DocumentChunk]:
        """Chunk all Markdown files in a folder using a process pool.

        Args:
            folder_path (Path): Path to the folder to search recursively for .md files.
            document_type (DocumentType): Document type applied to all files in the folder.

        Returns:
            list[DocumentChunk]: All chunks produced across all Markdown files found.

        Raises:
            FileNotFoundError: If folder_path does not exist.
        """
        folder_path = Path(folder_path).resolve()

        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        md_files = sorted(folder_path.rglob("*.md"))

        if not md_files:
            logger.info("no_markdown_files", folder=str(folder_path))
            return []

        logger.info(
            "markdown_files_found", count=len(md_files), folder=str(folder_path)
        )

        all_chunks: list[DocumentChunk] = []
        max_workers = self._app_config.concurrency.max_workers

        # ProcessPoolExecutor rather than ThreadPoolExecutor: text splitting is
        # CPU-bound (no I/O), so separate processes avoid the GIL bottleneck.
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.chunk_file, md_file, document_type): md_file
                for md_file in md_files
            }

            for future in tqdm(
                as_completed(future_to_file),
                total=len(md_files),
                desc="Chunking files",
                unit="file",
                **self._app_config.tqdm.to_kwargs(),
            ):
                md_file = future_to_file[future]
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                except Exception as exc:
                    logger.warning("chunk_failed", file=md_file.name, error=str(exc))

        logger.info("total_chunks_generated", total=len(all_chunks))
        return all_chunks

    @staticmethod
    def _split_by_headers(
        text: str,
        chunking_config: ChunkingConfig,
    ) -> list[Any]:
        """Split Markdown text into sections delimited by configured header levels.

        Args:
            text (str): Markdown text to split.
            chunking_config (ChunkingConfig): Chunking parameters including the
                header levels to split on.

        Returns:
            list[Any]: LangChain Document objects, one per header-delimited section.
        """
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=chunking_config.headers_to_split_on,
            # strip_headers=True removes the # / ## / ### tokens from the chunk text
            # so they're not duplicated in the indexed content; they're stored
            # separately in metadata (header_1, header_2, header_3).
            strip_headers=True,
        )
        return splitter.split_text(text)

    @staticmethod
    def _build_recursive_splitter(
        chunk_size: int,
        chunk_overlap: int,
    ) -> RecursiveCharacterTextSplitter:
        """Build a RecursiveCharacterTextSplitter with the given size parameters.

        Args:
            chunk_size (int): Maximum number of characters per chunk.
            chunk_overlap (int): Number of characters to overlap between consecutive chunks.

        Returns:
            RecursiveCharacterTextSplitter: Configured splitter instance.
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # keep_separator="start" places the separator at the beginning of the
            # next chunk so clause-opening connectives stay with the clause they
            # introduce rather than being orphaned at the end of the previous chunk.
            keep_separator="start",
        )

    @staticmethod
    def _split_and_filter(
        splitter: RecursiveCharacterTextSplitter,
        text: str,
        min_chunk_size: int,
    ) -> list[str]:
        """Split text and discard chunks below the minimum length threshold.

        Args:
            splitter (RecursiveCharacterTextSplitter): Splitter to apply.
            text (str): Text to split.
            min_chunk_size (int): Minimum number of characters required to retain a chunk.

        Returns:
            list[str]: Filtered list of chunk strings.
        """
        return [
            chunk for chunk in splitter.split_text(text) if len(chunk) >= min_chunk_size
        ]

    @staticmethod
    def _parse_metadata_from_path(
        file_path: Path,
        document_type: DocumentType,
    ) -> DocumentMetadata:
        """Extract company, product, and category metadata from a Markdown file path.

        The file stem must follow the convention ``{company}_{product}``. The product
        category is inferred from the immediate parent directory name.

        Args:
            file_path (Path): Absolute path to the Markdown file.
            document_type (DocumentType): Document type to store in the metadata.

        Returns:
            DocumentMetadata: Metadata populated from the file path structure.
        """
        filename_stem = file_path.stem
        parts = filename_stem.split("_", maxsplit=1)

        if len(parts) < 2:
            logger.warning(
                "invalid_filename_convention",
                file=file_path.name,
                hint="Expected '{company}_{product}.md'",
            )
            company_name = "unknown"
            product_name = filename_stem
        else:
            company_name = parts[0]
            product_name = parts[1]

            if not company_name or not product_name:
                logger.warning(
                    "empty_company_or_product",
                    file=file_path.name,
                )
                company_name = "unknown" if not company_name else company_name
                product_name = filename_stem if not product_name else product_name

        parent_name = file_path.parent.name if file_path.parent else ""

        # These directory names are structural folders in the data tree, not
        # product category labels; anything else is treated as the category.
        invalid_categories = {"extracted_documents", "data", "ipid", "owu", ""}
        if parent_name in invalid_categories:
            logger.warning(
                "invalid_product_category",
                file=file_path.name,
                folder=parent_name,
            )
            product_category = "unknown"
        else:
            product_category = parent_name

        return DocumentMetadata(
            company_name=company_name,
            product_name=product_name,
            product_category=product_category,
            document_type=document_type.value,
            source_file=file_path.name,
        )
