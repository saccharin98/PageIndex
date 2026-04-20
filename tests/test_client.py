# tests/sdk/test_client.py
import os

import pytest
from pageindex.client import PageIndexClient, LocalClient, CloudClient


def test_local_client_is_pageindex_client(tmp_path):
    client = LocalClient(model="gpt-4o", storage_path=str(tmp_path / "pi"))
    assert isinstance(client, PageIndexClient)


def test_cloud_client_is_pageindex_client():
    client = CloudClient(api_key="pi-test")
    assert isinstance(client, PageIndexClient)


def test_collection_default_name(tmp_path):
    client = LocalClient(model="gpt-4o", storage_path=str(tmp_path / "pi"))
    col = client.collection()
    assert col.name == "default"


def test_collection_custom_name(tmp_path):
    client = LocalClient(model="gpt-4o", storage_path=str(tmp_path / "pi"))
    col = client.collection("papers")
    assert col.name == "papers"


def test_list_collections_empty(tmp_path):
    client = LocalClient(model="gpt-4o", storage_path=str(tmp_path / "pi"))
    assert client.list_collections() == []


def test_list_collections_after_create(tmp_path):
    client = LocalClient(model="gpt-4o", storage_path=str(tmp_path / "pi"))
    client.collection("papers")
    assert "papers" in client.list_collections()


def test_delete_collection(tmp_path):
    client = LocalClient(model="gpt-4o", storage_path=str(tmp_path / "pi"))
    client.collection("papers")
    client.delete_collection("papers")
    assert "papers" not in client.list_collections()


def test_register_parser(tmp_path):
    client = LocalClient(model="gpt-4o", storage_path=str(tmp_path / "pi"))
    class FakeParser:
        def supported_extensions(self): return [".txt"]
        def parse(self, file_path, **kwargs): pass
    client.register_parser(FakeParser())


def test_pageindex_client_base_url_configures_local_openai_compatible_backend(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    client = PageIndexClient(
        model="ollama/llama3.1",
        base_url="http://example.test/v1",
        storage_path=str(tmp_path / "pi"),
    )

    assert isinstance(client, PageIndexClient)
    assert client._backend._model == "ollama/llama3.1"
    assert os.environ["OPENAI_BASE_URL"] == "http://example.test/v1"


def test_local_client_accepts_base_url(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    client = LocalClient(
        model="ollama/llama3.1",
        base_url="http://example.test/v1",
        storage_path=str(tmp_path / "pi"),
    )

    assert isinstance(client, PageIndexClient)
    assert os.environ["OPENAI_BASE_URL"] == "http://example.test/v1"


def test_pageindex_client_accepts_openai_api_base_env_for_local_compatible_backend(
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_BASE", "http://api-base.example/v1")

    client = PageIndexClient(
        model="ollama/llama3.1",
        storage_path=str(tmp_path / "pi"),
    )

    assert isinstance(client, PageIndexClient)
    assert client._backend._model == "ollama/llama3.1"
