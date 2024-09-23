from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun

class AutoResponseTool:
    """A custom tool that provides an automatic response."""
    def run(self, query):
        return "Proceed with the available information."

def get_tools():
    search_tool = DuckDuckGoSearchRun()
    auto_response_tool = AutoResponseTool()

    return [
        Tool(
            name="Internet Search",
            func=search_tool.run,
            description="Useful for finding the latest information on finance and investment strategies."
        ),
        Tool(
            name="Auto Response",
            func=auto_response_tool.run,
            description="Provides an automatic response to continue with available information."
        )
    ]