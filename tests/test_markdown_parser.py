import pytest
from pathlib import Path
from pageindex.parser.markdown import MarkdownParser
from pageindex.parser.protocol import ContentNode, ParsedDocument

@pytest.fixture
def sample_md(tmp_path):
    md = tmp_path / "test.md"
    md.write_text("""# Chapter 1
Some intro text.

## Section 1.1
Details here.

## Section 1.2
More details.

# Chapter 2
Another chapter.
""")
    return str(md)

def test_supported_extensions():
    parser = MarkdownParser()
    exts = parser.supported_extensions()
    assert ".md" in exts
    assert ".markdown" in exts

def test_parse_returns_parsed_document(sample_md):
    parser = MarkdownParser()
    result = parser.parse(sample_md)
    assert isinstance(result, ParsedDocument)
    assert result.doc_name == "test"

def test_parse_nodes_have_level(sample_md):
    parser = MarkdownParser()
    result = parser.parse(sample_md)
    assert len(result.nodes) == 4
    assert result.nodes[0].level == 1
    assert result.nodes[0].title == "Chapter 1"
    assert result.nodes[1].level == 2
    assert result.nodes[1].title == "Section 1.1"
    assert result.nodes[3].level == 1

def test_parse_nodes_have_content(sample_md):
    parser = MarkdownParser()
    result = parser.parse(sample_md)
    assert "Some intro text" in result.nodes[0].content
    assert "Details here" in result.nodes[1].content

def test_parse_nodes_have_index(sample_md):
    parser = MarkdownParser()
    result = parser.parse(sample_md)
    for node in result.nodes:
        assert node.index is not None


# --- New tests for improved parser ---

def test_preamble_before_first_header(tmp_path):
    """Content before the first header should become a preamble node."""
    md = tmp_path / "preamble.md"
    md.write_text("""This is a preamble paragraph.

# First Header
Body text.
""")
    parser = MarkdownParser()
    result = parser.parse(str(md))
    assert len(result.nodes) == 2
    # Preamble node has no level/title
    assert result.nodes[0].level is None
    assert result.nodes[0].title is None
    assert "preamble paragraph" in result.nodes[0].content
    # Header node is normal
    assert result.nodes[1].level == 1
    assert result.nodes[1].title == "First Header"


def test_headerless_file(tmp_path):
    """A file with no headers should produce a single node."""
    md = tmp_path / "plain.md"
    md.write_text("Just some plain text\nwith multiple lines.\n")
    parser = MarkdownParser()
    result = parser.parse(str(md))
    assert len(result.nodes) == 1
    assert result.nodes[0].level is None
    assert "plain text" in result.nodes[0].content


def test_empty_file(tmp_path):
    """An empty file should produce zero nodes."""
    md = tmp_path / "empty.md"
    md.write_text("")
    parser = MarkdownParser()
    result = parser.parse(str(md))
    assert len(result.nodes) == 0


def test_yaml_frontmatter_stripped(tmp_path):
    """YAML frontmatter should be stripped and stored as metadata."""
    md = tmp_path / "front.md"
    md.write_text("""---
title: My Doc
author: Alice
---

# Introduction
Hello world.
""")
    parser = MarkdownParser()
    result = parser.parse(str(md))
    assert result.metadata is not None
    assert "title: My Doc" in result.metadata["frontmatter"]
    # Frontmatter should not appear in node content
    for node in result.nodes:
        assert "title: My Doc" not in node.content


def test_yaml_frontmatter_preserves_original_line_numbers(tmp_path):
    """Node indexes should use original file line numbers after frontmatter."""
    md = tmp_path / "front_lines.md"
    md.write_text("""---
title: My Doc
---

# Introduction
Hello world.
""")
    parser = MarkdownParser()
    result = parser.parse(str(md))
    assert result.nodes[0].title == "Introduction"
    assert result.nodes[0].index == 5


def test_thematic_break_at_start_not_stripped_as_frontmatter(tmp_path):
    """Markdown that starts with thematic breaks should not be stripped."""
    md = tmp_path / "thematic.md"
    md.write_text("""---

# Introduction
Hello world.

---

# Second
More text.
""")
    parser = MarkdownParser()
    result = parser.parse(str(md))
    assert result.metadata is None
    assert result.nodes[0].level is None
    assert result.nodes[0].content == "---"
    assert result.nodes[1].title == "Introduction"
    assert result.nodes[1].index == 3
    assert result.nodes[2].title == "Second"


def test_setext_h1(tmp_path):
    """Setext-style H1 (=== underline) should be recognized."""
    md = tmp_path / "setext.md"
    md.write_text("""Main Title
==========

Some content here.

Sub Title
---------

More content.
""")
    parser = MarkdownParser()
    result = parser.parse(str(md))
    assert len(result.nodes) == 2
    assert result.nodes[0].title == "Main Title"
    assert result.nodes[0].level == 1
    assert result.nodes[1].title == "Sub Title"
    assert result.nodes[1].level == 2


@pytest.mark.parametrize(
    ("underline", "level"),
    [
        ("=", 1),
        ("-", 2),
    ],
)
def test_setext_single_character_underline(tmp_path, underline, level):
    """Setext underlines may be a single marker character."""
    md = tmp_path / "setext_single.md"
    md.write_text(f"""Title
{underline}

Body text.
""")
    parser = MarkdownParser()
    result = parser.parse(str(md))
    assert len(result.nodes) == 1
    assert result.nodes[0].title == "Title"
    assert result.nodes[0].level == level


@pytest.mark.parametrize("prefix", ["- item", "1. item", "> quote", "| A | B |"])
def test_setext_requires_paragraph_previous_line(tmp_path, prefix):
    """Setext underline should not turn list/quote/table lines into headers."""
    md = tmp_path / "not_setext.md"
    md.write_text(f"""{prefix}
---

# Real Header
Content.
""")
    parser = MarkdownParser()
    result = parser.parse(str(md))
    titles = [n.title for n in result.nodes if n.title]
    assert prefix not in titles
    assert "Real Header" in titles
    assert result.nodes[0].level is None
    assert prefix in result.nodes[0].content


def test_headers_inside_code_blocks_ignored(tmp_path):
    """Headers inside fenced code blocks should not be detected."""
    md = tmp_path / "code.md"
    md.write_text("""# Real Header

```
# Not a header
## Also not a header
```

# Another Real Header
More text.
""")
    parser = MarkdownParser()
    result = parser.parse(str(md))
    titles = [n.title for n in result.nodes if n.title]
    assert "Real Header" in titles
    assert "Another Real Header" in titles
    assert "Not a header" not in titles
    assert "Also not a header" not in titles


def test_tilde_code_fences(tmp_path):
    """Tilde fences (~~~) should also be respected."""
    md = tmp_path / "tilde.md"
    md.write_text("""# Header

~~~
# Fake header inside tilde fence
~~~

# Real Header After
Text.
""")
    parser = MarkdownParser()
    result = parser.parse(str(md))
    titles = [n.title for n in result.nodes if n.title]
    assert "Header" in titles
    assert "Real Header After" in titles
    assert "Fake header inside tilde fence" not in titles


def test_long_fence_requires_matching_close(tmp_path):
    """A ```` (4-backtick) opening fence should not close with ``` (3-backtick)."""
    md = tmp_path / "longfence.md"
    md.write_text("""# Before

````
# Inside fence
```
# Still inside — 3 backticks can't close 4-backtick fence
```
````

# After
Done.
""")
    parser = MarkdownParser()
    result = parser.parse(str(md))
    titles = [n.title for n in result.nodes if n.title]
    assert "Before" in titles
    assert "After" in titles
    assert "Inside fence" not in titles
    assert "Still inside" not in titles


def test_fence_close_requires_only_marker_and_spaces(tmp_path):
    """Fence-like content with trailing text should not close the code block."""
    md = tmp_path / "fence_close.md"
    md.write_text("""# Before

```
```not a closing fence
# Still code
```

# After
Done.
""")
    parser = MarkdownParser()
    result = parser.parse(str(md))
    titles = [n.title for n in result.nodes if n.title]
    assert "Before" in titles
    assert "After" in titles
    assert "Still code" not in titles


def test_atx_closing_hashes(tmp_path):
    """ATX headers with closing hashes like '## Title ##' should parse cleanly."""
    md = tmp_path / "closing.md"
    md.write_text("""## Title ##
Content.
""")
    parser = MarkdownParser()
    result = parser.parse(str(md))
    assert result.nodes[0].title == "Title"
    assert result.nodes[0].level == 2
