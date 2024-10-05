import re
import os
import sys
sys.path.append(os.environ["SAVING_DIRECTORY"] + "/matrag_framework/recall_layer")

from langchain.agents import initialize_agent
from langchain_openai import ChatOpenAI

from tool_hub.general_tools import GeneralTools
from tool_hub.material_science_tools import MaterialScienceTools 


TOOL_ASSESSOR_PROMPT = """
You are a tool selection agent. Your task is to select the most relevant tools for answering questions in the material science domain. Here are the list of tools available:

{material_science_tools}

You must select at least 5 tools!!!!

Given the question: "{question}", select the most relevant tools. 

# OUTPUT FORMAT
The output should be a **Python list** containing the names of the selected tools!!!!
"""


class ToolAssessor:
    def __init__(self, llm):
        self.llm = llm

    def get_tool_names(self, tools):
        return [tool.name for tool in tools]
    
    def extract_tools_list(self, response):
        # Regular expression to find the list in the response string, including within Python code blocks
        list_pattern = re.compile(r'\[\s*(.*?)\s*\]', re.DOTALL)
        match = list_pattern.search(response)
        
        if match:
            # Extract the list content
            list_content = match.group(1)
            # Split the content by commas, strip any extra spaces, and quotes
            tools_list = [item.strip().strip("'").strip('"') for item in list_content.split(',')]
            return tools_list
        else:
            raise ValueError("No valid list found in the response")
    
    def choose_tools(self, question):
        prompt = TOOL_ASSESSOR_PROMPT.format(
            material_science_tools=self.get_tool_names(MaterialScienceTools.tools),
            question=question
        )

        response = self.extract_tools_list(self.llm.invoke(prompt).content)
        try:

            tools_list = response
            
            if isinstance(tools_list, list):
                return tools_list
            else:
                raise ValueError("Invalid output format.")
            
        except Exception as e:
            raise ValueError(f"Failed to parse tools list: {e}")


class ToolExecutor:
    def __init__(self, model_name="gpt-3.5-turbo"):

        self.llm = ChatOpenAI(model_name=model_name, temperature=0.0)

        self.tool_assessor = ToolAssessor(self.llm)
        
        self.general_tools = GeneralTools.tools

        self.material_science_tools = MaterialScienceTools.tools

    def get_material_science_tools(self, question):
        try:
            selected_material_science_tool_names = self.tool_assessor.choose_tools(question=question)
            
            selected_tools = self.general_tools.copy()                                                        
            
            selected_tools.extend(tool for tool in self.material_science_tools if tool.name in selected_material_science_tool_names)
            
            return selected_tools
        
        except Exception as e:
            print(f"An error occurred with tool assessor: {e}")
            
            selected_tools = self.general_tools.copy()                                                        
            
            return selected_tools

    def initialize_agent(self, question, verbose):
        tools = self.get_material_science_tools(question)
        tool_agent = initialize_agent(
            tools, 
            llm=self.llm,
            agent="zero-shot-react-description",
            verbose=verbose,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

        self.tool_agent = tool_agent

    def answer_question(self, question, max_retries=3, verbose=None):
            self.initialize_agent(question, verbose)
            attempt = 0
            while attempt < max_retries:
                try:
                    tool_response = self.tool_agent.invoke(question)
                    return tool_response  # Return the response if successful
                except ValueError:
                    attempt += 1
                    print(f"Attempt {attempt} failed with ValueError. Retrying...")
            raise Exception(f"Failed after {max_retries} attempts due to repeated ValueErrors.")

