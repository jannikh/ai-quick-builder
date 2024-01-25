import os
import re
import openai
import json
import sys
from copy import deepcopy
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema import BaseOutputParser, OutputParserException
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough, RunnableParallel, RunnableBranch
from pydantic import BaseModel, Field, validator, ValidationError
from typing import List


# an AI class that takes in the following optional parameters:
# As soon as there is a prompt, the AI will be invoked and the result will be returned
# - prompt:str
# The following details can be given for invocation, as well as the settings written below
# - params:dict
# - batch:callable
# - input_list:list
# - output:type
# 
# Settings
# - model:str
# - provider:str
# - temperature:float
# - metadata:dict
# - tags:list
# - caching:bool
# - cache_location:str

NUMERICALS_DEFAULT_WITHOUT_FLOAT = False # Enabling this would prefer whole numbers by default
AUTO_INVOKE_ON_CREATION = False # Enabling this would deactivate the smart type casting it's doing
AUTO_CAST_TO_PREFERRED_TYPE = True # Forces results to be the type you specified in the output parameter

VALID_PROVIDERS = ["openai"]
VALID_CACHE_LOCATIONS = [
	"none", # No caching at all, even if caching is True
	"instance", # Caches only per class instance, which usually means per prompt
	"session", # Caches for a session, so for the module import (global variable in this module)
	"project", # Caches for a project, to be stored in the folder where the script is executed
	"module", # Caches for this module in the python cache folder
] # Anything else will be interpreted as a filesystem path
# You may point a file system path to the cloud so you can share the cache between multiple instances

class AI:
	def __init__(self,
		prompt:str=None,
		params:dict={},
		param_default:str|None=None,
		batch:callable=None,
		input_list:list=None,
		output:type=None,
		model:str="gpt-3.5-turbo",
		provider="openai",
		temperature=0.75,
		metadata={},
		tags:list=["AI() call"],
		caching=True,
		cache_location="session",
		append_history=False,
		history_names=("Question", "Answer"),
		numericals_default_to_int=NUMERICALS_DEFAULT_WITHOUT_FLOAT,
	):
		# Any settings relevant for invoking this AI instance
		self.variables = {
			'prompt': prompt, # The prompt to be used for the AI invocation, as soon as one is given, the AI will be called
			'params': params, # Dict of additional parameters that can be placed in the prompt {placeholders}
			'param_default': param_default, # Default value for parameters that are not given, defaults to raising an error
			'batch': batch, # For batch calling on a list of inputs, this should be yet another AI() instance with {lambda} as prompt placeholder
			'input_list': input_list, # A list of inputs for batch calling
			'output': output, # Desired output type of the AI invocation, defaults to str
			'model': model, # The model to be used for the AI invocation, defaults to gpt-3.5-turbo as of now (2024)
			'provider': provider, # The provider to be used for the AI invocation, currently only openai implemented
			'temperature': temperature, # Temperature of the model during invocation, defaults to 0.75
			'metadata': metadata, # Any metadata added to the AI call for better langsmith tracing
			'tags': tags, # Any tags you want to add to the metadata for better langsmith tracing
			'caching': caching, # Whether to use caching for the AI invocation (assumes determinism and will give the same result for the same inputs), defaults to True
			'cache_location': cache_location, # What cache behavior to use, defines the permanency (session, project-folder, python-module, or other disk
			'append_history': append_history, # Automatically formats and appends the full chat history to the prompt
			'history_names': history_names, # Names to be used for the chatbot history for pretty formatting. Given as tuple: (ai, user)
			'numericals_default_to_int': numericals_default_to_int, # Whether to assume that numerical invocations should be cast to int instead of float (so whether the AI has the option to add decimal points to numerical requests)
			# Notice that this is just the default behavior for unspecified output types. You can always specifc the desired output type when creating an instance or invoking a prompt.
		}
		self.results = [] # Structure: [({input_params}, "output"), ({input_params}, "output"), ...]
		self.chat = []

		# Checking providers
		if provider.lower() not in VALID_PROVIDERS:
			raise NotImplementedError(f"Provider {provider} is not available, please use {' or '.join(VALID_PROVIDERS)} for now.")

		# Setting up cache
		self.cache_init()

		if AUTO_INVOKE_ON_CREATION and kwargs['prompt']:
			self.invoke(kwargs['prompt'], **kwargs)

	def __call__(self, prompt:str|None=None, **kwargs):
		if prompt is not None:
			kwargs['prompt'] = prompt
		# if self.variables['append_history'] and self.variables['prompt'] and self.variables['prompt'] != kwargs['prompt']:
		# 	kwargs['prompt'] = self.variables['prompt'] + "\n" + f"{self.variables['history_names'][1]}: " + kwargs['prompt']
		self.variables |= kwargs
		return self

	def __repr__(self):
		parameters = ", ".join([f"{key}={self.variables[key]}" for key in self.variables])
		return f"AI({parameters})"

	# Invoking and executing the actual AI call. That's what we're here for, that's the heavy lifting.
	def invoke(self, prompt:str=None, **kwargs):
		# The invoke method accepts keyword arguments and ensures all keys in self.variables have a value, using default values from self.variables if not provided.
		for key in self.variables:
			if key not in kwargs or kwargs[key] is None:
				kwargs[key] = self.variables[key]

		output = kwargs['output']
		if output is None:
			output = str
		self.variables['output'] = output

		if prompt is None:
			prompt = kwargs.get('prompt', None)
		if not prompt:
			raise ValueError("No prompt was given for the AI invocation.")

		
		# Preparing parameters and placing them into the prompt
		prompt_args = deepcopy(kwargs['params'])
		if isinstance(prompt_args, AI):
			prompt_args = prompt_args.todict()
		prompt = replace_placeholders(prompt, prompt_args, kwargs['param_default'])

		if kwargs['append_history']:
			chat_history = ""
			for i in range(len(self.chat)):
				chat_history += f"""
{kwargs['history_names'][0]}: {self.chat[i][0]['prompt']}
{kwargs['history_names'][1]}: {self.chat[i][1]}"""
			chat_history = chat_history.strip()
			full_prompt = chat_history + "\n" + f"{kwargs['history_names'][0]}: " + prompt
			full_prompt += "\n" + f"{kwargs['history_names'][1]}: "
		else:
			full_prompt = prompt

		# Caching
		if kwargs['caching']:
			cached_result = self.cache_get(kwargs | {'full_prompt': full_prompt})
			if cached_result:
				return cached_result

		# Setting up llm with desired temperature and model
		llm = ChatOpenAI(temperature=kwargs['temperature'], model=kwargs['model'])

		ai_chain = ChatPromptTemplate.from_template(full_prompt)

		if output is str:
			ai_chain |= llm
			ai_chain |= StrOutputParser()
		else:
			response_description = ''
			# response_description = prompt # Disabled because it would just appear twice
			if output == int:
				response_format = function_int("") # Your response as an integer number")
			elif output == float:
				response_format = function_float("")
			elif output == bool:
				response_format = function_bool("")
			elif output == list:
				response_format = function_array("", function_str(""))
			elif output == dict:
				response_description = "An object that defines its own list of properties."
				response_format = function_array("", function_object({
					"key": function_str(""),
					"value": function_str(""),
				}, ["key", "value"]))

			output_structure = create_function_call(
				"Response",
				response_description,
				function_object({
					"response": response_format,
				}, ["response"]),
			)
			ai_chain |= llm_with_function_call(llm, output_structure)
			ai_chain |= JsonOutputFunctionsParser()
			ai_chain |= (lambda x: x['response'])

			if output == dict:
				ai_chain |= (lambda dict_list: {item['key']: item['value'] for item in dict_list})
		
		# Use this anywhere in the chains to easily view the current state of the variables being passed through at that point
		def debug_print(x):
			print(f"\n{x}\n")
			return(x)
		
		result = ai_chain.invoke(prompt_args, {'tags': kwargs['tags'], 'metadata': kwargs['metadata']})

		if 'cast_to_preffered_type' in kwargs and kwargs['cast_to_preffered_type']:
			result = output(result)
		if not 'cast_to_preffered_type' in kwargs and AUTO_CAST_TO_PREFERRED_TYPE:
			result = output(result)

		# Store in instance-history and in cache
		self.results.append((deepcopy(kwargs) | {'full_prompt': full_prompt}, result))
		self.chat.append((deepcopy(kwargs) | {'prompt': prompt}, result))
		if kwargs['caching']:
			self.cache_set(kwargs | {'full_prompt': full_prompt}, result)
		return(result)




	# run is a synonym for invoke
	def run(self, **kwargs):
		if type(self) is str:
			self = AI(**kwargs | {'prompt': self})
		self.invoke(prompt, **kwargs)


	def result(self, preferred_type:type=None, cast_to_preferred_type=True):
		if preferred_type is None:
			preferred_type = str
			cast_to_preferred_type = False
		if self.results and self.results[-1][0] == self.variables:
			result = self.results[-1][1]
		elif not self.variables['prompt']:
				raise ValueError("No prompt was given for the AI invocation. Cannot get the result of AI() without prompt")
		elif self.variables['output'] is None:
			result = self.invoke(output=preferred_type)
		else:
			result = self.invoke()
		if cast_to_preferred_type and preferred_type is not None:
			result = preferred_type(result)
		return result
	
	# Casting
	def __str__(self): return self.result(str)
	def __int__(self): return self.result(int)
	def __float__(self): return self.result(float)
	def __bool__(self): return self.result(bool)
	def __list__(self): return self.result(list)
	def __nonzero__(self): return self.result(bool)

	def __num__(self):
		if self.variables['numericals_default_to_int']:
			return self.__int__()
		else:
			return self.__float__()

	def __iter__(self):
		return self.result(list).__iter__()

	def tostr(self, **kwargs):
		if type(self) is str:
			self = AI(**kwargs | {'prompt': self})
		self.variables |= kwargs
		return self.__str__()

	def toint(self, **kwargs):
		if type(self) is str:
			self = AI(**kwargs | {'prompt': self})
		self.variables |= kwargs
		return self.__int__()

	def tofloat(self, **kwargs):
		if type(self) is str:
			self = AI(**kwargs | {'prompt': self})
		self.variables |= kwargs
		return self.__float__()

	def tobool(self, **kwargs):
		if type(self) is str:
			self = AI(**kwargs | {'prompt': self})
		self.variables |= kwargs
		return self.__bool__()

	def check(self, **kwargs):
		if type(self) is str:
			self = AI(**kwargs | {'prompt': self})
		self.variables |= kwargs
		return self.__bool__()

	def tolist(self, **kwargs):
		if type(self) is str:
			self = AI(**kwargs | {'prompt': self})
		self.variables |= kwargs
		return self.__list__()

	def todict(self, **kwargs):
		if type(self) is str:
			self = AI(**kwargs | {'prompt': self})
		self.variables |= kwargs
		return self.result(dict)

	def tonum(self):
		if self.variables['numericals_default_to_int']:
			return self.toint()
		else:
			return self.tofloat()
	
	def make_numeric(anything):
		if isinstance(anything, AI):
			return anything.tonum()
		else:
			return anything

	def __add__(self, other): return AI.make_numeric(self) + AI.make_numeric(other)
	def __sub__(self, other): return AI.make_numeric(self) - AI.make_numeric(other)
	def __mul__(self, other): return AI.make_numeric(self) * AI.make_numeric(other)
	def __floordiv__(self, other): return AI.make_numeric(self) // AI.make_numeric(other)
	def __truediv__(self, other): return AI.make_numeric(self) / AI.make_numeric(other)
	def __mod__(self, other): return AI.make_numeric(self) % AI.make_numeric(other)
	def __pow__(self, other): return AI.make_numeric(self) ** AI.make_numeric(other)
	def __lt__(self, other): return AI.make_numeric(self) < AI.make_numeric(other)
	def __le__(self, other): return AI.make_numeric(self) <= AI.make_numeric(other)
	def __eq__(self, other): return AI.make_numeric(self) == AI.make_numeric(other)
	def __ne__(self, other): return AI.make_numeric(self) != AI.make_numeric(other)
	def __ge__(self, other): return AI.make_numeric(self) >= AI.make_numeric(other)
	def __gt__(self, other): return AI.make_numeric(self) > AI.make_numeric(other)

	
	# This will enable LANGCHAIN_TRACING_V2 and LANGCHAIN_ENDPOINT if not set differently in environment variables
	def set_langsmith_project(project_name:str):
		# Setting up langsmith project
		if project_name:
			if not os.getenv('LANGCHAIN_TRACING_V2'):
				os.environ['LANGCHAIN_TRACING_V2'] = 'true'
			if not os.getenv('LANGCHAIN_ENDPOINT'):
				os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
			os.environ['LANGCHAIN_PROJECT'] = project_name
		else:
			os.environ['LANGCHAIN_TRACING_V2'] = ""
			os.environ['LANGCHAIN_ENDPOINT'] = ""
			os.environ['LANGCHAIN_PROJECT'] = ""

	def cache_init(self):
		if not self.variables['caching'] or self.variables['cache_location'].lower() == 'none':
			return
		elif self.variables['cache_location'].lower() == 'instance':
			if not hasattr(self, 'cache'):
				self.cache = {}
		elif self.variables['cache_location'].lower() == 'session':
			global cache
			if not 'cache' in globals():
				cache = {}
		else:
			raise NotImplementedError(f"Cache location {self.variables['cache_location']} is not implemented yet.")

	def cache_get(self, key):
		if not self.variables['caching'] or self.variables['cache_location'].lower() == 'none':
			return False
		elif self.variables['cache_location'].lower() == 'instance':
			return self.cache.get(str(key), False)
		elif self.variables['cache_location'].lower() == 'session':
			global cache
			return cache.get(str(key), False)
		else:
			raise NotImplementedError(f"Cache location {self.variables['cache_location']} is not implemented yet.")

	def is_cached(self, key):
		if not self.variables['caching'] or self.variables['cache_location'].lower() == 'none':
			return False
		elif self.variables['cache_location'].lower() == 'instance':
			return str(key) in self.cache
		elif self.variables['cache_location'].lower() == 'session':
			global cache
			return str(key) in cache
		else:
			raise NotImplementedError(f"Cache location {self.variables['cache_location']} is not implemented yet.")
	
	def cache_set(self, key, value):
		if not self.variables['caching'] or self.variables['cache_location'].lower() == 'none':
			return False
		elif self.variables['cache_location'].lower() == 'instance':
			self.cache[str(key)] = value
		elif self.variables['cache_location'].lower() == 'session':
			global cache
			cache[str(key)] = value
		else:
			raise NotImplementedError(f"Cache location {self.variables['cache_location']} is not implemented yet.")

	def retrieve_whole_cache(self):
		if not self.variables['caching'] or self.variables['cache_location'].lower() == 'none':
			return False
		elif self.variables['cache_location'].lower() == 'instance':
			return self.cache
		elif self.variables['cache_location'].lower() == 'session':
			global cache
			return cache
		else:
			raise NotImplementedError(f"Cache location {self.variables['cache_location']} is not implemented yet.")

# Helper functions to easily create OpenAI function call definitions
def llm_with_function_call(llm, function_structure: dict):
	return llm.bind(function_call = {'name': function_structure['name']}, functions = [function_structure])

def create_function_call(name:str, description:str, parameters:dict):
	return_dict = {'name': name}
	if description:
		return_dict |= {'description': description}
	return_dict |= {'parameters': parameters}
	return return_dict

def function_object(properties:dict, required: list | None):
	return_dict = {'type': 'object'}
	return_dict |= {'properties': properties}
	if required is not None:
		return_dict |= {'required': required}
	return return_dict

def function_type(type_name:str, description:str):
	return_dict = {'type': type_name}
	if description:
		return_dict |= {'description': description}
	return return_dict

def function_array(description:str, items: dict):
	return function_type('array', description) | {'items': items}

def function_str(description:str):
	return function_type('string', description)

def function_int(description:str):
	return function_type('integer', description)

def function_float(description:str):
	return function_type('number', description)

def function_bool(description:str):
	return function_type('boolean', description)

def replace_placeholders(text, replacements, default=None):
	"""
	Replace placeholders in the text with corresponding values from replacements dictionary.

	:param text: A string containing placeholders in curly braces.
	:param replacements: A dictionary where keys are placeholder names and values are replacement values.
	:return: Text with placeholders replaced by their corresponding values.
	"""
	# Find all placeholders in the text
	placeholders = re.findall(r'\{(.*?)\}', text)

	# Replace each placeholder with the corresponding value from the replacements dictionary
	for placeholder in placeholders:
		if placeholder in replacements:
			text = text.replace('{' + placeholder + '}', replacements[placeholder])
		elif default is not None:
			text = text.replace('{' + placeholder + '}', default)
		else:
			raise KeyError(f"Placeholder {placeholder} was not found in replacements dictionary.")

	return text
