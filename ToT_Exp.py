
from langchain_huggingface import HuggingFaceEndpoint

# Set the url to your Inference Endpoint below
your_endpoint_url = "https://mixtral-8x22b.k8s.aip.mitre.org"
llm = HuggingFaceEndpoint(
    endpoint_url=f"{your_endpoint_url}",
    max_new_tokens=2048,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
)
#######

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

sudoku_puzzle =   "3,*,*,2|1,*,3,*|*,1,*,3|4,*,*,1"
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

tot_chain = ToTChain(llm=llm, checker=MyChecker(), k=30, c=5, verbose=True, verbose_llm=False)
tot_chain.run(problem_description=problem_description)

#######