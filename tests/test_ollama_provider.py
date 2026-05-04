"""Sanity checks for OllamaProvider construction and pre-flight logic.

These tests are purely unit-level — no Ollama daemon needed.
Network-calling paths are tested via mocking.
"""
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from unittest.mock import AsyncMock, MagicMock, patch
# Stub out openai so OllamaProvider (which delegates to OpenAICompatProvider)
# can be imported without the real SDK present in the test sandbox.
import sys as _sys
from unittest.mock import MagicMock
if "openai" not in _sys.modules:
    _sys.modules["openai"] = MagicMock()
    _sys.modules["openai"].AsyncOpenAI = MagicMock()
from llm.ollama import OllamaProvider
from llm.base import LLMError


# ─────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────

def test_strip_trailing_v1():
    p = OllamaProvider(model="llama3.2", base_url="http://localhost:11434/v1")
    assert p._native_base == "http://localhost:11434"
    assert "/v1" in p._inner._base_url  # inner compat provider still gets /v1


def test_strip_trailing_slash():
    p = OllamaProvider(model="llama3.2", base_url="http://localhost:11434/")
    assert p._native_base == "http://localhost:11434"


def test_default_values():
    p = OllamaProvider(model="mistral")
    assert p.model == "mistral"
    assert p._keep_alive == "5m"
    assert p.supports_vision is False
    assert p._verified is False


def test_vision_flag_passed_through():
    p = OllamaProvider(model="llama3.2-vision", supports_vision=True)
    assert p.supports_vision is True


# ─────────────────────────────────────────────────────────────────────
# _verify_model
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_verify_model_success():
    """Model in tags list → no error, _verified becomes True."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [{"name": "llama3.2:latest"}, {"name": "mistral:latest"}]
    }

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client

        p = OllamaProvider(model="llama3.2")
        await p._verify_model()
        assert p._verified is True


@pytest.mark.asyncio
async def test_verify_model_bare_name_match():
    """'llama3.2' matches tag 'llama3.2:latest' via bare-name extraction."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"models": [{"name": "llama3.2:latest"}]}

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client

        p = OllamaProvider(model="llama3.2")
        await p._verify_model()
        assert p._verified is True


@pytest.mark.asyncio
async def test_verify_model_not_pulled_raises():
    """Model missing from tags → LLMError with actionable message."""
    import httpx
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"models": [{"name": "mistral:latest"}]}

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client

        p = OllamaProvider(model="llama3.2")
        with pytest.raises(LLMError) as exc_info:
            await p._verify_model()
        assert "ollama pull llama3.2" in str(exc_info.value)
        assert "not pulled" in str(exc_info.value)


@pytest.mark.asyncio
async def test_verify_model_daemon_unreachable():
    """ConnectError → LLMError with 'ollama serve' hint."""
    import httpx

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        mock_client_cls.return_value = mock_client

        p = OllamaProvider(model="llama3.2")
        with pytest.raises(LLMError) as exc_info:
            await p._verify_model()
        assert "ollama serve" in str(exc_info.value)


# ─────────────────────────────────────────────────────────────────────
# Factory integration
# ─────────────────────────────────────────────────────────────────────

def test_factory_builds_ollama():
    from llm.factory import build_provider
    from config import LLMConfig, Secrets

    cfg = LLMConfig(
        provider="ollama",
        model="mistral",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_keep_alive="10m",
    )
    secrets = Secrets()
    provider = build_provider(cfg, secrets)
    assert isinstance(provider, OllamaProvider)
    assert provider.model == "mistral"
    assert provider._keep_alive == "10m"
    assert provider._native_base == "http://127.0.0.1:11434"
