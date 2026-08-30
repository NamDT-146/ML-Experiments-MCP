from __future__ import annotations


def test_official_mcp_package_importable():
    import mcp
    from mcp.server.mcpserver import MCPServer

    from aiaun_mcp.server import app

    assert isinstance(app, MCPServer)
    assert app.name == "aiaun"
    assert getattr(mcp, "__file__", None)
