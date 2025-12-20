import pdfplumber
import hashlib
from collections import Counter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import os


class DocumentProcessor:
    def __init__(self):
        self.chunks = []
        self.starter_questions = []

    def process(self, file_path: str):
        """
        Main pipeline: Read -> Table Extract -> Frequency Analysis -> Clean -> Chunk.
        Returns: (full_text, chunks_with_metadata)
        """
        all_text = ""
        pages_content = []  # List of text per page

        # Calculate Hash
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # 1. Extract Tables with structured column:value format
                tables = page.extract_tables()
                table_text = ""
                if tables:
                    for table in tables:
                        # Check if table has a header row (first row with text)
                        if len(table) < 2:
                            continue

                        # First row is assumed to be headers
                        headers = [
                            str(cell).strip() if cell else "" for cell in table[0]
                        ]

                        # Skip if no valid headers
                        if not any(headers):
                            # Fallback to simple markdown
                            clean_table = [
                                [str(cell) if cell else "" for cell in row]
                                for row in table
                            ]
                            table_text += (
                                "\n"
                                + "\n".join(
                                    [
                                        "| " + " | ".join(row) + " |"
                                        for row in clean_table
                                    ]
                                )
                                + "\n"
                            )
                            continue

                        # Process each data row with column:value format
                        for row in table[1:]:  # Skip header row
                            cells = [str(cell).strip() if cell else "" for cell in row]
                            if not any(cells):  # Skip empty rows
                                continue

                            # Build structured string: "Col1: Val1 | Col2: Val2 | ..."
                            parts = []
                            for header, cell in zip(headers, cells):
                                if header and cell:
                                    parts.append(f"{header}: {cell}")

                            if parts:
                                table_text += "\n" + " | ".join(parts) + "\n"

                # 2. Extract Text
                raw_text = page.extract_text() or ""

                # Combine
                page_combined = raw_text + table_text
                pages_content.append(page_combined)

        # 3. Frequency Analysis (Dynamic Cleaning)
        bad_lines = self._analyze_frequencies(pages_content)

        clean_pages = []
        for i, page_text in enumerate(pages_content):
            clean_text = self._clean_text(page_text, bad_lines)
            clean_pages.append(clean_text)
            all_text += clean_text + "\n"

        # 4. Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.chunks = splitter.create_documents(
            clean_pages
        )  # This loses page numbers if we aren't careful

        # Let's do manual chunking to keep metadata?
        # Actually langchain's create_documents accepts metadatas list.
        # But for simplicity in this MVP, let's just use the splitter on the full text
        # OR split per page to keep page numbers.

        docs = []
        for i, page_text in enumerate(clean_pages):
            page_docs = splitter.create_documents(
                [page_text],
                metadatas=[
                    {
                        "page": i + 1,
                        "source": os.path.basename(file_path),
                        "file_hash": file_hash,
                    }
                ],
            )
            docs.extend(page_docs)

        self.chunks = docs

        # 5. Smart Onboarding (Sample chunks)
        # We don't call this automatically here to allow mocking in tests easier,
        # but typically it would be part of process or called after.

        return all_text, docs

    def _analyze_frequencies(self, pages: list[str]) -> set[str]:
        """
        Detect lines that appear on > 80% of pages.
        """
        if len(pages) < 3:
            return set()

        line_counts = Counter()
        for page in pages:
            lines = set(page.split("\n"))  # Unique lines per page
            line_counts.update(lines)

        threshold = len(pages) * 0.8
        bad_lines = {
            line
            for line, count in line_counts.items()
            if count > threshold and len(line.strip()) > 5
        }
        return bad_lines

    def _clean_text(self, text: str, bad_lines: set[str]) -> str:
        lines = text.split("\n")
        good_lines = [line for line in lines if line not in bad_lines]
        return "\n".join(good_lines)

    def generate_starter_questions(self):
        """
        Generate 3 questions based on random chunks.
        """
        if not self.chunks:
            return []

        # Context is first 3 chunks to get an overview
        context = "\n".join([d.page_content for d in self.chunks[:3]])

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
        prompt = f"""
        Based on the following document excerpt, generate 3 short, specific questions a user might ask.
        Return ONLY a JSON list of strings, e.g. ["Question 1", "Question 2"].
        
        Excerpt:
        {context}
        """

        try:
            response = llm.invoke(prompt)
            import json

            # Heuristic cleanup if model returns markdown code block
            content = response.content.strip()
            if "```" in content:
                content = content.split("```")[1].replace("json", "").strip()
            return json.loads(content)
        except Exception:
            return [
                "What is this document about?",
                "Summarize the key points.",
                "List specific details.",
            ]
