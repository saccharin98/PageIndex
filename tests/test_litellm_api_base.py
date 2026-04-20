import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


def _make_response(content="ok"):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice])


def test_index_utils_completion_passes_openai_base_url(monkeypatch):
    from pageindex.index import utils

    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.test/v1")

    with patch.object(utils.litellm, "completion", return_value=_make_response()) as mock_completion:
        assert utils.llm_completion("gpt-4o", "hello") == "ok"

    assert mock_completion.call_args.kwargs["api_base"] == "http://example.test/v1"


def test_index_utils_completion_passes_openai_base_url_for_openai_prefix(monkeypatch):
    from pageindex.index import utils

    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.test/v1")

    with patch.object(utils.litellm, "completion", return_value=_make_response()) as mock_completion:
        assert utils.llm_completion("openai/gpt-4o", "hello") == "ok"

    assert mock_completion.call_args.kwargs["api_base"] == "http://example.test/v1"


@pytest.mark.parametrize("model", [
    "ollama/llama3.1",
    "lm_studio/llama3.1",
    "hosted_vllm/llama3.1",
    "litellm/ollama/llama3.1",
])
def test_index_utils_completion_passes_openai_base_url_for_compatible_provider_prefixes(
    monkeypatch,
    model,
):
    from pageindex.index import utils

    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.test/v1")

    with patch.object(utils.litellm, "completion", return_value=_make_response()) as mock_completion:
        assert utils.llm_completion(model, "hello") == "ok"

    assert mock_completion.call_args.kwargs["api_base"] == "http://example.test/v1"


def test_index_utils_completion_omits_openai_base_url_for_anthropic(monkeypatch):
    from pageindex.index import utils

    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.test/v1")

    with patch.object(utils.litellm, "completion", return_value=_make_response()) as mock_completion:
        assert utils.llm_completion("anthropic/claude-3-5-sonnet-20241022", "hello") == "ok"

    assert "api_base" not in mock_completion.call_args.kwargs


def test_index_utils_completion_omits_openai_base_url_for_gemini(monkeypatch):
    from pageindex.index import utils

    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.test/v1")

    with patch.object(utils.litellm, "completion", return_value=_make_response()) as mock_completion:
        assert utils.llm_completion("gemini/gemini-1.5-pro", "hello") == "ok"

    assert "api_base" not in mock_completion.call_args.kwargs


def test_index_utils_completion_omits_api_base_by_default(monkeypatch):
    from pageindex.index import utils

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("CHATGPT_API_BASE", raising=False)

    with patch.object(utils.litellm, "completion", return_value=_make_response()) as mock_completion:
        assert utils.llm_completion("gpt-4o", "hello") == "ok"

    assert "api_base" not in mock_completion.call_args.kwargs


def test_index_utils_acompletion_passes_openai_api_base(monkeypatch):
    from pageindex.index import utils

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_BASE", "http://api-base.example/v1")

    with patch.object(utils.litellm, "acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _make_response("async ok")
        result = asyncio.run(utils.llm_acompletion("gpt-4o", "hello"))

    assert result == "async ok"
    assert mock_acompletion.call_args.kwargs["api_base"] == "http://api-base.example/v1"


def test_legacy_utils_completion_passes_chatgpt_api_base(monkeypatch):
    import pageindex.utils as utils

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv("CHATGPT_API_BASE", "http://legacy.example/v1")

    with patch.object(utils.litellm, "completion", return_value=_make_response()) as mock_completion:
        assert utils.llm_completion("gpt-4o", "hello") == "ok"

    assert mock_completion.call_args.kwargs["api_base"] == "http://legacy.example/v1"


def test_legacy_utils_completion_passes_openai_base_url_for_lm_studio(monkeypatch):
    import pageindex.utils as utils

    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.test/v1")

    with patch.object(utils.litellm, "completion", return_value=_make_response()) as mock_completion:
        assert utils.llm_completion("lm_studio/llama3.1", "hello") == "ok"

    assert mock_completion.call_args.kwargs["api_base"] == "http://example.test/v1"


def test_legacy_utils_completion_omits_openai_base_url_for_anthropic(monkeypatch):
    import pageindex.utils as utils

    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.test/v1")

    with patch.object(utils.litellm, "completion", return_value=_make_response()) as mock_completion:
        assert utils.llm_completion("anthropic/claude-3-5-sonnet-20241022", "hello") == "ok"

    assert "api_base" not in mock_completion.call_args.kwargs
