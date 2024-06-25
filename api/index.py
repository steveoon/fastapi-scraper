from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from scrapegraphai.graphs import SmartScraperMultiGraph
import asyncio
import json
from aiomultiprocess import Pool
from dotenv import load_dotenv
import os

load_dotenv()
openai_key = os.getenv("OPENAI_APIKEY")

app = FastAPI()

prompt = """I need to scrape the content from the provided URLs and organize the information into a structured JSON 
format. The JSON should include the following fields:

1. **Title**: The main title or headline of the page.
2. **Description**: A brief summary or description of the content.
3. **Date**: The date when the content was published or last updated.
4. **Author**: The name of the author or organization that published the content.
5. **Content**: The main body of the content, organized into paragraphs or sections.
6. **Tags**: Relevant tags or categories that describe the content.
7. **URL**: The URL from which the content was scraped.

Make sure to accurately categorize the information and ensure that the content is valuable and relevant. If any field 
is not available, return an empty string for that field. The output should be a list of objects in JSON format, 
where each object corresponds to one of the provided URLs.

Here is an example of the expected JSON format:

[
    {
        "title": "Example Title",
        "description": "This is a brief description of the content.",
        "date": "2024-06-24",
        "author": "John Doe",
        "content": "This is the main body of the content, organized into paragraphs or sections.",
        "tags": ["tag1", "tag2"],
        "url": "https://example.com/project1"
    },
    ...
]

Please make sure the extracted information is accurate and well-organized.
"""


# ************************************************
# Define the configuration for the graph
# ************************************************
class Project(BaseModel):
    title: str = Field(description="The title of the project")
    description: str = Field(description="The description of the project")
    date: Optional[str] = Field(description="The date when the content was published or last updated")
    author: Optional[str] = Field(description="The name of the author or organization that published the content")
    content: str = Field(description="The main body of the content, organized into paragraphs or sections")
    tags: List[str] = Field(description="Relevant tags or categories that describe the content")
    url: str = Field(description="The URL from which the content was scraped")


class Projects(BaseModel):
    projects: List[Project]


def preprocess_result(result, urls):
    preprocessed_projects = []
    for project in result['projects']:
        preprocessed_project = {
            'title': project.get('title', ""),
            'description': project.get('description', ""),
            'date': project.get('date', ""),
            'author': project.get('author', ""),
            'content': project.get('content', ""),
            'tags': project.get('tags', []),
            'url': project.get('url', "")
        }
        preprocessed_projects.append(preprocessed_project)
    return {'projects': preprocessed_projects}


def run_smart_scraper_graph(prompt, urls, graph_config):
    smart_scraper_graph = SmartScraperMultiGraph(
        prompt=prompt,
        source=urls,
        schema=Projects,  # 这里传递的是Projects类
        config=graph_config,
    )
    result = smart_scraper_graph.run()
    return result


async def async_run_smart_scraper_graph(prompt, urls, graph_config):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_smart_scraper_graph, prompt, urls, graph_config)
    return result


@app.get("/api/smart-scraper")
async def scrape(urls: str):
    urls = urls.split(",")
    if not urls:
        raise HTTPException(status_code=400, detail="No URLs provided")

    # graph_config = {
    #     "llm": {
    #         "model": "ollama/mistral",
    #         "temperature": 0,
    #         "format": "json",
    #         "model_tokens": 2000,
    #         "base_url": "http://localhost:11434",
    #     },
    #     "embeddings": {
    #         "model": "ollama/nomic-embed-text",
    #         "temperature": 0,
    #         "base_url": "http://localhost:11434",
    #     },
    #     "verbose": True,
    #     "headless": False
    # }
    graph_config = {
        "llm": {
            "api_key": openai_key,
            "model": "gpt-3.5-turbo",
            "temperature": 0,
            "max_tokens": 1000,
            "base_url": "https://api.ohmygpt.com/v1",
        },
        "verbose": True,
        "headless": False,
    }

    try:
        async with Pool() as pool:
            results = await pool.starmap(
                async_run_smart_scraper_graph, [(prompt, urls, graph_config)]
            )

        # 因为starmap返回的是列表，所以需要取第一个元素
        result = results[0]

        # 打印结果进行调试
        print("Scraper raw result:", result)

        # 解析result，确保它是一个字典
        if isinstance(result, str):
            result = json.loads(result)

        # 预处理结果以确保所有字段有效
        preprocessed_result = preprocess_result(result, urls)

        # Validate the result with the defined schema
        validated_result = Projects(**preprocessed_result)

        print("🧚‍Scraping result:", validated_result)

        return JSONResponse(content=validated_result.dict())
    except Exception as e:
        print("🧚‍Error:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)
