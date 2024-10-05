from langchain.agents import Tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import WikipediaQueryRun, YouTubeSearchTool
from langchain_community.tools.google_scholar import GoogleScholarQueryRun
from langchain_community.utilities import GoogleSerperAPIWrapper, WikipediaAPIWrapper
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
from langchain_experimental.tools import PythonREPLTool


def arxiv_search():
    tools = load_tools(["arxiv"])
    tool = Tool(
        name="Arxiv Search",
        func=tools[0].run,
        description="Useful for searching academic papers on arXiv."
    )
    return tool


def google_search():
    google_search = GoogleSerperAPIWrapper()
    tool = Tool(
        name = "Google Search",
        func= google_search.run,
        description="Useful to search in Google. Use by default"
    )
    return tool


def wikipedia_search():
    wikipedia_search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    tool = Tool(
        name="Wikipedia Search",
        func=wikipedia_search.run,
        description="Useful when users request biographies or historical moments."
    )
    return tool


def youtube_search():
    youtube_search = YouTubeSearchTool()

    tool = Tool(
        name="Youtube Search",
        func=youtube_search.run,
        description="Useful for when the user explicitly asks you to look on Youtube.",
    )
    return tool


def google_scholar_search():
    google_scholar_search = GoogleScholarQueryRun(api_wrapper=GoogleScholarAPIWrapper())
    tool = Tool(
        name="Google Scholar Search",
        func=google_scholar_search.run,
        description="Useful for searching academic papers and articles on Google Scholar."
    )
    return tool


def python_repl():
    python_repl = PythonREPLTool()
    tool = Tool(
        name="Python REPL",
        func=python_repl.run,
        description= (
            "A Python shell. Use this to execute python commands. "
            "Input should be a valid python command. "
            "If you want to see the output of a value, you should print it out "
            "with `print(...)`."
        )
    )
    return tool


class GeneralTools:
    tools = [
        arxiv_search(),
        google_search(),
        google_scholar_search(),
        python_repl(),
        wikipedia_search(),
        youtube_search()
    ]