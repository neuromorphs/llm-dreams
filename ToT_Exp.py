
from langchain_huggingface import HuggingFaceEndpoint

import warnings
warnings.filterwarnings("ignore")
# # Set the url to your Inference Endpoint below
#https://llama3-8b.k8s.aip.mitre.org - notworks
#https://mixtral-8x22b.k8s.aip.mitre.org - works
#https://llama3-70b.k8s.aip.mitre.org - works
your_endpoint_url = "https://llama3-70b.k8s.aip.mitre.org"
llm = HuggingFaceEndpoint(
    endpoint_url=f"{your_endpoint_url}",
    max_new_tokens=2048,
    top_k=20,
    top_p=0.95,
    typical_p=0.99,
    temperature=0.99,
    repetition_penalty=1.03,
)
#######

# from langchain_community.llms.openai import OpenAI
# import os
# openai_api_key = os.environ["OPENAI_API_KEY"]
# llm = OpenAI(model=f"{your_endpoint_url}/v1", openai_api_key=openai_api_key)

import os
from uuid import uuid4

unique_id = uuid4().hex[0:8]

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = f"Agent Tot"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxx"
# os.environ['OPENAI_API_KEY'] = str("xxxxxxxxxxxxxxxxxxxxxxxx")

#######

#from langchain.llms import OpenAI
#llm = OpenAI(temperature=1, max_tokens=512, model="text-davinci-003")

#######
#sudoku_puzzle =   "3,*,*,2|1,*,3,*|*,1,*,3|4,*,*,1"
sudoku_puzzle =   "3,4,1,2|1,*,3,4|2,1,*,3|4,3,*,1"
sudoku_solution = "3,4,1,2|1,2,3,4|2,1,4,3|4,3,2,1"
problem_description = f"""
{sudoku_puzzle}

- This is a 4x4 Sudoku puzzle.
- The * represents a cell to be filled.
- The | character separates rows.
- At each step, replace one or more * with digits 1-4.
- There must be no duplicate digits in any row, column or 2x2 subgrid.
- Keep the known digits from previous valid thoughts in place.
- Each thought can be a partial or the final solution.
- Each thought should be different from the previous thought
""".strip()
print(problem_description)

#######
# The following code implement a simple rule based checker for 
# a specific 4x4 sudoku puzzle.
#######

from typing import Tuple
from langchain_experimental.tot.checker import ToTChecker
from langchain_experimental.tot.thought import ThoughtValidity
import re


all_thoughts = []
class MyChecker(ToTChecker):
    def evaluate(self, problem_description: str, thoughts: Tuple[str, ...] = ()) -> ThoughtValidity:
        last_thought = thoughts[-1]
        clean_solution = last_thought.replace(" ", "").replace('"', "")
        regex_solution = clean_solution.replace("*", ".").replace("|", "\\|")
        if sudoku_solution in clean_solution:
            return ThoughtValidity.VALID_FINAL
        elif re.search(regex_solution, sudoku_solution):
            return ThoughtValidity.VALID_INTERMEDIATE
        else:
            return ThoughtValidity.INVALID

#######
# Testing the MyChecker class above:
#######
checker = MyChecker()
assert checker.evaluate("", ("3,*,*,2|1,*,3,*|*,1,*,3|4,*,*,1",)) == ThoughtValidity.VALID_INTERMEDIATE
assert checker.evaluate("", ("3,4,1,2|1,2,3,4|2,1,4,3|4,3,2,1",)) == ThoughtValidity.VALID_FINAL
assert checker.evaluate("", ("3,4,1,2|1,2,3,4|2,1,4,3|4,3,*,1",)) == ThoughtValidity.VALID_INTERMEDIATE
assert checker.evaluate("", ("3,4,1,2|1,2,3,4|2,1,4,3|4,*,3,1",)) == ThoughtValidity.INVALID

#######
# Initialize and run the ToT chain, 
# with maximum number of interactions k set to 30 and 
# the maximum number child thoughts c set to 8.
#######

from langchain_experimental.tot.base import ToTChain

tot_chain = ToTChain(llm=llm, checker=MyChecker(), k=60, c=10, verbose=True, verbose_llm=False)
tot_chain.run(problem_description=problem_description)
print('Yaya we did it!')

# import pygraphviz as pgv
# def add_nodes_edges(graph, level):
#     if len(tot_chain.tot_memory.stack[level].children) == 0:
#         return
#     node_data = tot_chain.tot_memory.stack[level]
#     label = node_data.get('label', node_data.text)
#     graph.add_node(level, label=label, shape='box', style='filled', fillcolor='lightyellow', color='black')
#     children = node_data.children
#     for child, child_data in children.items():
#         add_nodes_edges(graph, child)
#         graph.add_edge(level, child)

# # Create a new graph
# G = pgv.AGraph(directed=True)

# # Add the root node with a textbox style
# root = 0  # Assuming the root is at level 0
# root_data = tot_chain.tot_memory.stack[root]
# root_label = root_data.text
# G.add_node(root, label=root_label, shape='box', style='filled', fillcolor='lightyellow', color='black')
# add_nodes_edges(G, root)

# # Layout and render the graph
# G.layout(prog='dot')
# G.draw('tree.png')
# #######