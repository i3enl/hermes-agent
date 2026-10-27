"""Discord attachment decode path against real Brotli (brotlicffi >= 1.2.1.0).

brotlicffi 1.2.1.0 (libbrotli 1.2.1, CVE-2026-111497) raises on bytes that
trail a complete Brotli stream. The adapter deliberately *rejects* such
attachments with ``MalformedAttachmentEncodingError`` (no silent tolerance,
no re-fetch through the URL fallback). Normal ``br`` content must decode.

All HTTP traffic goes to an aiohttp server bound to 127.0.0.1.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import pytest_asyncio

aiohttp = pytest.importorskip("aiohttp")
brotlicffi = pytest.importorskip("brotlicffi")
from aiohttp import web  # noqa: E402

import tests.gateway.test_discord_attachment_download  # noqa: E402,F401  (installs discord mock if needed)
from gateway.config import PlatformConfig  # noqa: E402
from plugins.platforms.discord.adapter import (  # noqa: E402
    DiscordAdapter,
    MalformedAttachmentEncodingError,
    _is_content_decoding_error,
)

_PLAIN = b"# notes\n" + b"hello from discord\n" * 50
_BR = brotlicffi.compress(_PLAIN)


def test_brotlicffi_rejects_trailing_bytes():
    """Guards the pin: the patched decoder must refuse trailing data."""
    assert brotlicffi.decompress(_BR) == _PLAIN
    with pytest.raises(brotlicffi.error):
        brotlicffi.decompress(_BR + b"JUNK")


@pytest_asyncio.fixture
async def br_server():
    async def handler(request):
        body = _BR + (b"JUNK" if request.query.get("trailing") else b"")
        return web.Response(
            body=body,
            headers={"Content-Encoding": "br", "Content-Type": "text/plain"},
        )

    app = web.Application()
    app.router.add_get("/file.md", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/file.md"
    finally:
        await runner.cleanup()


def _att_with_read(url):
    """Mimics discord.Attachment.read(): an aiohttp GET + resp.read()."""

    async def read():
        async with aiohttp.ClientSession() as s:
            async with s.get(url) as resp:
                return await resp.read()

    return SimpleNamespace(url=url, filename="file.md", size=None, read=read)


def _adapter():
    return DiscordAdapter(PlatformConfig(enabled=True, token="***"))


@pytest.mark.asyncio
async def test_read_path_decodes_brotli(br_server):
    assert await _adapter()._read_attachment_bytes(_att_with_read(br_server)) == _PLAIN


@pytest.mark.asyncio
async def test_read_path_rejects_trailing_bytes(br_server):
    att = _att_with_read(br_server + "?trailing=1")
    with patch(
        "plugins.platforms.discord.adapter.is_safe_url",
        side_effect=AssertionError("must not fall back to URL re-fetch"),
    ):
        with pytest.raises(MalformedAttachmentEncodingError):
            await _adapter()._cache_discord_document(att, ".md")


@pytest.mark.asyncio
async def test_url_fallback_decodes_brotli(br_server):
    att = SimpleNamespace(url=br_server, filename="file.md", size=None)
    with patch("plugins.platforms.discord.adapter.is_safe_url", return_value=True):
        assert await _adapter()._cache_discord_document(att, ".md") == _PLAIN


@pytest.mark.asyncio
async def test_url_fallback_rejects_trailing_bytes(br_server):
    att = SimpleNamespace(url=br_server + "?trailing=1", filename="file.md", size=None)
    with patch("plugins.platforms.discord.adapter.is_safe_url", return_value=True):
        with pytest.raises(MalformedAttachmentEncodingError) as ei:
            await _adapter()._cache_discord_document(att, ".md")
    assert isinstance(ei.value, ValueError)


def test_non_encoding_errors_are_not_classified():
    assert not _is_content_decoding_error(RuntimeError("403 Forbidden"))
    assert not _is_content_decoding_error(aiohttp.ClientPayloadError("truncated"))
