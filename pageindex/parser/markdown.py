import re
from pathlib import Path
from .protocol import ContentNode, ParsedDocument
from ..index.utils import count_tokens

# Patterns
_ATX_HEADER = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#+\s*)?$")
_SETEXT_H1 = re.compile(r"^={3,}\s*$")
_SETEXT_H2 = re.compile(r"^-{3,}\s*$")
_FENCE_OPEN = re.compile(r"^(`{3,}|~{3,})")
_FRONTMATTER_FENCE = re.compile(r"^---\s*$")


class MarkdownParser:
    def supported_extensions(self) -> list[str]:
        return [".md", ".markdown"]

    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        path = Path(file_path)
        model = kwargs.get("model")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        lines, metadata = self._strip_frontmatter(lines)
        headers = self._extract_headers(lines)
        nodes = self._build_nodes(headers, lines, model)

        return ParsedDocument(
            doc_name=path.stem,
            nodes=nodes,
            metadata=metadata,
        )

    @staticmethod
    def _strip_frontmatter(lines: list[str]) -> tuple[list[str], dict | None]:
        """Strip YAML frontmatter (--- delimited) from the beginning of the file.

        Returns the remaining lines and raw frontmatter as metadata.
        """
        if not lines or not _FRONTMATTER_FENCE.match(lines[0]):
            return lines, None

        for i in range(1, len(lines)):
            if _FRONTMATTER_FENCE.match(lines[i]):
                raw = "\n".join(lines[1:i])
                remaining = lines[i + 1:]
                return remaining, {"frontmatter": raw}

        # No closing fence found — not valid frontmatter, return as-is
        return lines, None

    def _extract_headers(self, lines: list[str]) -> list[dict]:
        """Extract all ATX and setext headers, respecting fenced code blocks."""
        headers = []
        in_fence = False
        fence_pattern = None  # tracks the char and min length to close

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            # Track fenced code blocks
            fence_match = _FENCE_OPEN.match(stripped)
            if fence_match:
                marker = fence_match.group(1)
                if not in_fence:
                    in_fence = True
                    fence_char = marker[0]
                    fence_len = len(marker)
                    fence_pattern = (fence_char, fence_len)
                elif stripped[0] == fence_pattern[0] and len(marker) >= fence_pattern[1]:
                    # Only close if same char and at least as many
                    in_fence = False
                    fence_pattern = None
                continue

            if in_fence:
                continue

            # ATX headers: # Title
            atx = _ATX_HEADER.match(stripped)
            if atx:
                headers.append({
                    "title": atx.group(2).strip(),
                    "level": len(atx.group(1)),
                    "line_num": line_num,
                })
                continue

            # Setext headers: underline on next line detected by looking back
            # We check if *this* line is an underline and the previous line is text
            if line_num >= 2:
                prev = lines[line_num - 2].strip()  # previous line (0-indexed)
                if prev and not _FENCE_OPEN.match(prev) and not _ATX_HEADER.match(prev):
                    if _SETEXT_H1.match(stripped):
                        headers.append({
                            "title": prev,
                            "level": 1,
                            "line_num": line_num - 1,
                        })
                        continue
                    if _SETEXT_H2.match(stripped):
                        headers.append({
                            "title": prev,
                            "level": 2,
                            "line_num": line_num - 1,
                        })
                        continue

        return headers

    def _build_nodes(
        self, headers: list[dict], lines: list[str], model: str | None
    ) -> list[ContentNode]:
        if not headers:
            # No headers — entire content becomes a single node
            text = "\n".join(lines).strip()
            if not text:
                return []
            tokens = count_tokens(text, model=model)
            return [ContentNode(content=text, tokens=tokens, index=1)]

        nodes = []

        # Content before the first header → preamble node
        first_header_line = headers[0]["line_num"]
        if first_header_line > 1:
            preamble = "\n".join(lines[: first_header_line - 1]).strip()
            if preamble:
                tokens = count_tokens(preamble, model=model)
                nodes.append(ContentNode(content=preamble, tokens=tokens, index=1))

        # One node per header section
        for i, header in enumerate(headers):
            start = header["line_num"] - 1
            end = headers[i + 1]["line_num"] - 1 if i + 1 < len(headers) else len(lines)
            text = "\n".join(lines[start:end]).strip()
            tokens = count_tokens(text, model=model)
            nodes.append(
                ContentNode(
                    content=text,
                    tokens=tokens,
                    title=header["title"],
                    index=header["line_num"],
                    level=header["level"],
                )
            )

        return nodes
