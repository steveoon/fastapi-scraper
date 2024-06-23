from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from scrapegraphai.graphs import SmartScraperMultiGraph
import asyncio

app = FastAPI()


@app.get("/api/smart-scraper")
async def scrape(urls: str):
    urls = urls.split(",")
    if not urls:
        raise HTTPException(status_code=400, detail="No URLs provided")

    graph_config = {
        "llm": {
            "model": "ollama/qwen2",
            "temperature": 0,
            "format": "json",
            "base_url": "https://api13b.bitewise.cc",
        },
        "embeddings": {
            "model": "ollama/nomic-embed-text",
            "base_url": "https://api13b.bitewise.cc",
        },
        "verbose": True,
    }

    try:
        smart_scraper_graph = SmartScraperMultiGraph(
            prompt="List me all the projects with their descriptions",
            source=urls,
            config=graph_config,
        )

        # 使用当前时间循环运行异步函数
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, smart_scraper_graph.run)

        if not isinstance(result, dict):
            raise ValueError("Result is not a valid JSON object")

        print("🧚‍♀️ Scraping result:", result)

        return JSONResponse(content=result)
    except Exception as e:
        print("🧚‍♀️ Error:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)
