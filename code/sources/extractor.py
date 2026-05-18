"""
extractor
---------
Converts insurance PDFs (OWU and IPID) into clean Markdown using a
vision-capable LLM. Each page is rasterised, sent to the model with a
lightweight cross-page context window, and the resulting Markdown is
concatenated and saved next to the source tree.
"""

import structlog
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from threading import Semaphore
import base64

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import pypdfium2 as pdfium
from PIL import Image
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.auto import tqdm

from sources.config import EXTRACTION_PROMPT, config as app_config

logger = structlog.get_logger(__name__)


class Extractor:
    """Convert PDF pages into Markdown using a vision-capable LLM."""

    def __init__(self) -> None:
        """Initialize the extractor with the module-level application configuration."""
        self._app_config = app_config
        self._llm = ChatOpenAI(
            model=self._app_config.llm.model_name,
            temperature=self._app_config.llm.temperature,
            max_retries=0,
        )
        self._api_semaphore = Semaphore(self._app_config.llm.max_concurrent_api_calls)

    def extract_file(self, file_path: Path) -> Path:
        """Extract one PDF file to Markdown and write the output to disk.

        Each page is rendered as an image and processed by the LLM. An optional
        cross-page context window from the previous page is included when configured.

        Args:
            file_path (Path): Path to the PDF file to extract.

        Returns:
            Path: Path to the written Markdown output file.

        Raises:
            FileNotFoundError: If file_path does not exist.
            ValueError: If file_path does not have a .pdf extension.
        """
        file_path = Path(file_path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {file_path.suffix}")

        logger.info("extracting_file", file=file_path.name)

        output_path = self._get_output_path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        markdown_pages: list[str] = []
        pdf = pdfium.PdfDocument(str(file_path))

        for page_num in tqdm(
            range(len(pdf)),
            desc=f"Extracting {file_path.name}",
            unit="page",
            **self._app_config.tqdm.to_kwargs(),
        ):
            logger.info(
                "processing_page",
                page=page_num + 1,
                total=len(pdf),
                file=file_path.name,
            )
            page_image = self._render_page(pdf, page_num)

            previous_context = None
            if (
                self._app_config.llm.enable_cross_page_context
                and page_num > 0
                and markdown_pages
            ):
                previous_page = markdown_pages[-1]
                context_chars = self._app_config.llm.cross_page_context_chars
                # Take the tail of the previous page: document text flows from the
                # bottom of one page to the top of the next, so the final chars carry
                # the most relevant continuity signal for the LLM.
                previous_context = (
                    previous_page[-context_chars:]
                    if len(previous_page) > context_chars
                    else previous_page
                )

            page_markdown = self._extract_page(page_image, previous_context)
            markdown_pages.append(page_markdown)

        pdf.close()
        full_markdown = "\n\n".join(markdown_pages)
        output_path.write_text(full_markdown, encoding="utf-8")

        logger.info("markdown_saved", path=str(output_path))
        return output_path

    def extract_folder(self, folder_path: Path) -> list[Path]:
        """Extract all PDF files found recursively in a folder.

        Args:
            folder_path (Path): Path to the folder to search for .pdf files.

        Returns:
            list[Path]: Sorted list of paths to written Markdown output files.

        Raises:
            FileNotFoundError: If folder_path does not exist.
        """
        folder_path = Path(folder_path).resolve()

        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        pdf_files = sorted(folder_path.rglob("*.pdf"))

        if not pdf_files:
            logger.info("no_pdf_files", folder=str(folder_path))
            return []

        logger.info("pdf_files_found", count=len(pdf_files), folder=str(folder_path))

        return self.extract_files(pdf_files)

    def extract_files(self, file_paths: list[Path]) -> list[Path]:
        """Extract a list of PDF files concurrently using a thread pool.

        Args:
            file_paths (list[Path]): PDF files to extract.

        Returns:
            list[Path]: Sorted list of paths to written Markdown output files.
                Files that fail extraction are omitted and logged as warnings.
        """
        pdf_files = sorted(Path(file_path).resolve() for file_path in file_paths)

        if not pdf_files:
            logger.info("no_pdf_files_to_extract")
            return []

        results: list[Path] = []
        max_workers = min(self._app_config.concurrency.max_workers, len(pdf_files))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.extract_file, pdf_file): pdf_file
                for pdf_file in pdf_files
            }

            for future in tqdm(
                as_completed(future_to_file),
                total=len(pdf_files),
                desc="Extracting PDFs",
                unit="file",
                **self._app_config.tqdm.to_kwargs(),
            ):
                pdf_file = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    logger.warning(
                        "extraction_failed", file=pdf_file.name, error=str(exc)
                    )

        logger.info("extraction_complete", extracted=len(results), total=len(pdf_files))
        return sorted(results)

    def _render_page(self, pdf: pdfium.PdfDocument, page_num: int) -> Image.Image:
        """Render one PDF page as a PIL image at 2x scale.

        Args:
            pdf (pdfium.PdfDocument): Open PDF document.
            page_num (int): Zero-based index of the page to render.

        Returns:
            Image.Image: Rendered page as a PIL image.
        """
        page = pdf[page_num]
        bitmap = page.render(scale=2)
        pil_image = bitmap.to_pil()
        return pil_image

    def _extract_page(
        self, page_image: Image.Image, previous_page_context: str | None = None
    ) -> str:
        """Extract Markdown from one page image using the LLM.

        Encodes the image as base64 PNG and sends it together with the extraction
        prompt. Retries on failure according to the configured retry policy.

        Args:
            page_image (Image.Image): Rendered page image to process.
            previous_page_context (str | None): Trailing text from the previous page
                used as a continuity hint, or None if not applicable.

        Returns:
            str: Extracted Markdown text, or a comment placeholder on failure.
        """
        if previous_page_context:
            context_prompt = (
                f"Context from previous page (for continuity):\n"
                f"```\n{previous_page_context}\n```\n\n"
                f"{EXTRACTION_PROMPT}"
            )
        else:
            context_prompt = EXTRACTION_PROMPT

        buffered = BytesIO()
        page_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        @retry(
            retry=retry_if_exception_type((Exception,)),
            stop=stop_after_attempt(self._app_config.llm.retry_max_attempts),
            wait=wait_exponential(
                min=self._app_config.llm.retry_min_wait,
                max=self._app_config.llm.retry_max_wait,
            ),
            reraise=True,
        )
        def _call_api_with_retry() -> str:
            """Invoke the LLM under the concurrency semaphore with retry."""
            with self._api_semaphore:
                logger.info("api_call_started")
                response = self._llm.invoke(
                    [
                        HumanMessage(
                            content=[
                                {"type": "text", "text": context_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_base64}"
                                    },
                                },
                            ]
                        )
                    ]
                )
                content = self._get_response_text(response)
                return content if content else "<!-- empty page -->"

        try:
            return _call_api_with_retry()
        except Exception as exc:
            logger.warning("page_extraction_failed", error=str(exc))
            return "<!-- extraction failed -->"

    @staticmethod
    def _get_response_text(response: object) -> str:
        """Read text content from a LangChain response object.

        Tries the .text attribute first, then .content as a string, then
        .content as a list of typed text blocks.

        Args:
            response (object): LangChain response object from an LLM invocation.

        Returns:
            str: Extracted and stripped text, or an empty string if none is found.
        """
        response_text = getattr(response, "text", None)
        if isinstance(response_text, str):
            stripped = response_text.strip()
            if stripped:
                return stripped

        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            blocks: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in {"text", "output_text"} and isinstance(
                    block.get("text"), str
                ):
                    text = block["text"].strip()
                    if text:
                        blocks.append(text)
            return "\n".join(blocks).strip()

        return ""

    def _get_output_path(self, input_path: Path) -> Path:
        """Build the output Markdown path mirroring the source tree structure.

        Replaces the raw documents root with the extracted documents root and
        changes the file extension to .md.

        Args:
            input_path (Path): Absolute path to the source PDF file.

        Returns:
            Path: Destination path for the Markdown output file.
        """
        raw_dir = self._app_config.paths.raw_documents_dir.resolve()
        extracted_dir = self._app_config.paths.extracted_documents_dir.resolve()

        try:
            relative = input_path.relative_to(raw_dir)
        except ValueError:
            relative = Path(input_path.name)

        return extracted_dir / relative.with_suffix(".md")
