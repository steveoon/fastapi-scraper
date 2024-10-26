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

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_APIKEY")
openai_real_key = os.getenv("OPENAI_REAL_APIKEY")

app = FastAPI()

prompt = """请从给定的URL中抓取内容，并组织成结构化的JSON格式。JSON应包含以下字段：

1. **Title**: 页面的主标题或头条。
2. **Description**: 内容的简要摘要或描述。
3. **Date**: 内容发布或最后更新的日期。
4. **Author**: 发布内容的作者或组织名称。
5. **Content**: 主要内容，按段落或部分组织。
6. **Tags**: 描述内容的相关标签或类别。
7. **URL**: 抓取内容的URL。

请确保信息准确分类，内容有价值且相关。如某字段缺失，返回空字符串。输出应为JSON格式对象列表，每个对象对应的URL应是其爬取内容页面的URL。

请尽量精简回复内容，控制字数在1000以内，确保每个字段的内容简明扼要。

示例JSON格式：

[
    {
        "title": "示例标题",
        "description": "这是内容的简要描述。",
        "date": "2024-06-24",
        "author": "张三",
        "content": "这是主要内容，按段落或部分组织。",
        "tags": ["标签1", "标签2"],
        "url": "https://example.com/project1"
    },
    ...
]

请确保提取的信息准确且组织良好。
"""


class Project(BaseModel):
    title: str = Field(description="The title of the project")
    description: str = Field(description="The description of the project")
    date: Optional[str] = Field(
        description="The date when the content was published or last updated"
    )
    author: Optional[str] = Field(
        description="The name of the author or organization that published the content"
    )
    content: str = Field(
        description="The main body of the content, organized into paragraphs or sections"
    )
    tags: List[str] = Field(
        description="Relevant tags or categories that describe the content"
    )
    url: str = Field(description="The URL from which the content was scraped")


class Projects(BaseModel):
    projects: List[Project]


def preprocess_result(result, urls):
    preprocessed_projects = []
    for project in result["projects"]:
        preprocessed_project = {
            "title": project.get("title", ""),
            "description": project.get("description", ""),
            "date": project.get("date", ""),
            "author": project.get("author", ""),
            "content": project.get("content", ""),
            "tags": project.get("tags", []),
            "url": project.get("url", ""),
        }
        preprocessed_projects.append(preprocessed_project)
    return {"projects": preprocessed_projects}


def run_smart_scraper_graph(prompt, url, graph_config):
    smart_scraper_graph = SmartScraperMultiGraph(
        prompt=prompt,
        source=[url],
        schema=Projects,  # 这里传递的是Projects类
        config=graph_config,
    )
    result = smart_scraper_graph.run()
    return result


async def async_run_smart_scraper_graph(prompt, url, graph_config):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, run_smart_scraper_graph, prompt, url, graph_config
    )
    return result


async def scrape_single_url(url, prompt, graph_config):
    try:
        result = await async_run_smart_scraper_graph(prompt, url, graph_config)
        if isinstance(result, str):
            result = json.loads(result)
        return result
    except Exception as e:
        print(f"Skipping URL due to error: {url}, Error: {str(e)}")
        return None


@app.get("/api/smart-scraper")
async def scrape(urls: str):
    urls = urls.split(",")
    if not urls:
        raise HTTPException(status_code=400, detail="No URLs provided")

    graph_config = {
        "llm": {
            "api_key": openai_key,
            "model": "gpt-4o",
            "max_tokens": 2000,
            "base_url": "https://api.gptsapi.net/v1",
        },
        "verbose": True,
        "headless": False,
        "embeddings": {
            "model": "ollama/mxbai-embed-large",
            "temperature": 0,
            "base_url": "http://localhost:11434",
        },
        "max_depth": 2,
        "max_nodes": 50,
        "verbose": True,
        "headless": False,
    }
    # graph_config = {
    #     "llm": {
    #         "api_key": openai_key,
    #         "model": "gpt-4o",
    #         "temperature": 0,
    #         "max_tokens": 2000,
    #         "base_url": "https://api.ohmygpt.com/v1",
    #     },
    #     "embeddings": {
    #         "api_key": openai_real_key,
    #         "model": "openai-text-embedding-3-small",
    #         "temperature": 0,
    #     },
    #     "verbose": True,
    #     "headless": False,
    # }

    try:
        async with Pool() as pool:
            tasks = [
                pool.apply(scrape_single_url, args=(url, prompt, graph_config))
                for url in urls
            ]
            results = await asyncio.gather(*tasks)

        # 过滤掉 None 的结果
        valid_results = [result for result in results if result is not None]

        # 合并所有结果
        combined_results = {"projects": []}
        for result in valid_results:
            combined_results["projects"].extend(result["projects"])

        # 预处理结果以确保所有字段有效
        preprocessed_result = preprocess_result(combined_results, urls)

        # Validate the result with the defined schema
        validated_result = Projects(**preprocessed_result)

        print("🧚‍Scraping result:", validated_result)

        return JSONResponse(content=validated_result.dict())
    except Exception as e:
        print("🧚‍Error:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)
