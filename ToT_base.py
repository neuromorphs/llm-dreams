from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama


template ="""
Step1 :
 
I have a problem related to the following words: {input}. Could you brainstorm three distinct solutions? Please consider a variety of factors such as {perfect_factors}. These words end up in groups of 4 linked by an unknown concept, group them and try to discover the linking concept.
A:
"""

prompt = PromptTemplate(
    input_variables=["input","perfect_factors"],
    template = template                      
)

chain1 = LLMChain(
    llm=ChatOllama(temperature=0,model="phi3:mini"),
    prompt=prompt,
    output_key="solutions",
    verbose=True,
)

template ="""
Step 2:

For each of the three proposed solutions, evaluate their potential. Consider their pros and cons, and potential confusions. Assign a probability of success and a confidence level to each option based on these factors

{solutions}

A:"""

prompt = PromptTemplate(
    input_variables=["solutions"],
    template = template                      
)

chain2 = LLMChain(
    llm=ChatOllama(temperature=0,model="phi3:mini"),
    prompt=prompt,
    output_key="review"
)

template ="""
Step 3:

For each solution, deepen the thought process. Generate potential scenarios, analogies, and references that may be useful. 
{review}

A:"""

prompt = PromptTemplate(
    input_variables=["review"],
    template = template                      
)

chain3 = LLMChain(
    llm=ChatOllama(temperature=0,model="phi3:mini"),
    prompt=prompt,
    output_key="deepen_thought_process"
)

template ="""
Step 4:

Based on the evaluations and scenarios, rank the solutions in order of promise. Provide a justification for each ranking and offer any final thoughts or considerations for each solution
{deepen_thought_process}

A:"""

prompt = PromptTemplate(
    input_variables=["deepen_thought_process"],
    template = template                      
)

chain4 = LLMChain(
    llm=ChatOllama(temperature=0,model="phi3:mini"),
    prompt=prompt,
    output_key="ranked_solutions"
)

from langchain.chains import SequentialChain

overall_chain = SequentialChain(
    chains=[chain1, chain2, chain3, chain4],
    input_variables=["input", "perfect_factors"],
    output_variables=["ranked_solutions"],
    verbose=True
)

print(overall_chain({"input":"MOBILE, FOLLOWERS, SHOVELS, BUFFALO, LIKES, INSULTS, SHARES, SHEEP, APARTMENT, BILLINGS, PUPPETS, OPTIONS, EQUITY, PHOENIX, STOCKS, LEMMINGS.", "perfect_factors":"The potential relations of the words"}))

