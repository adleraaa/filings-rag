"""End-to-end check of the MCP server over real stdio, against the built index.

Usage: python scripts/mcp_smoke.py   (needs data/index from `filings-rag ingest`)
Starts `filings-rag serve-mcp` as a subprocess, lists tools, runs one search
and fetches the top page, the same way an MCP client such as Claude Desktop would.
"""

import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "filings_rag.cli", "serve-mcp", "--method", "dense"]
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        print("tools:", [t.name for t in tools.tools])
        result = await session.call_tool(
            "search_filings", {"query": "capital expenditures 2018", "company": "3M", "k": 3}
        )
        hits = result.structuredContent["result"]
        for h in hits:
            print(f"{h['doc']} p.{h['page']} score={h['score']}: {h['text'][:80]}...")
        page = await session.call_tool("get_page", {"doc": hits[0]["doc"], "page": hits[0]["page"]})
        print("get_page chars:", len(json.loads(page.content[0].text)["text"]))


if __name__ == "__main__":
    asyncio.run(main())
