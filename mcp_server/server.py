from mcp.server.fastmcp import FastMCP

from mcp_server.tool_registry import get_registered_tools

mcp = FastMCP("gift-llm-tools")


def register_tools() -> None:
    for tool_name, tool_func in get_registered_tools().items():
        mcp.tool(name=tool_name)(tool_func)


register_tools()


if __name__ == "__main__":
    mcp.run()
