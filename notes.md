Use the llamaindex library to create a new application for a university dissertation supervisor called "ESI: ESI Scholarly Instructor". The interface for this will use streamlit

This will be a conversational ai and will have the following tools:
1. duckduckgo tool duckduckgo_instant_search
2. tavily tool TavilyToolSpec
3. a wikipedia tool  wikipediaToolSpec
4. a scholar search tool using SemanticScholarReader
5. a rag tool for retrieving content from a RAG implementation using chromadb. The database for this is located in a directory called ragdb


The webscraper can be used when the user provides a url in the chat, or can be used by the agent to retrieve material from websited found through the duckduckgo-search tool, the tavily-search tool or the wikipedia retriever

The main agent should be able to invoke a code execution agent (CodeAct)  to solve mathematical or statistical problems, and create downloadable plots.  


use gemini-2.5-flash-preview-04-17 as the llm 
use "models/text-embedding-004" (from google import genai)


organize the code in the following files: 
app.py for the main application
stui.py for the interface
agent.py for the agent functionality
tools.py for the tools functionality

Using the llm, provide 4 concise suggested prompts for the user to select from. They should be provided as buttons and the user should have the option to select one of them or write in the chat. These should be provided from the first interaction - before the user says anything, and should be updated after every response of the LLM using the context 



#7. data analysis tool which you need to implement using create_pandas_dataframe_agent  from from langchain_experimental (with access to scipy, numpy, matplotlib, pymc, statmodels,



convert the code to use the llamaindex workflows approach to implement an orchestrator agent that will be the one responding to queries. There will be a number of other ageners: one agennt that will do agentic RAG, another search agent that will be conducting searches using the duckduckgo, tavily, and wikipedia tools, a lit.reviewer that will find literature using semantic scholar, a scraper who will scrape the papers and material found by the search agent and the lit.reviewer, a coder agent that can solve problems by writing and executing code. The orchestrator will communicate will all the agents to ask queries and collect all the information it needs to respond to the chat and provide help.     