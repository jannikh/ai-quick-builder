# AI Quick Builder

This lightweight Python module is a versatile tool for integrating AI language models into conversational applications, featuring customizable prompts, dynamic output types, efficient caching, and context-aware chat capabilities. Based on langchain.

<p align="center">
    <img src="assets\readme-cover-img.png" alt="The image representing the essence of your repository with everyday AutoHotkey shortcuts is ready. It visually captures the dynamic and efficient workspace empowered by the various functionalities of your scripts. Feel free to take a look at it above." width="80%" style="max-width: 420px; border-radius: 15px"/>
</p>

## Features

Key features include:

**Customizable AI Invocations:** Supports a variety of configurations for prompts, parameters, and model settings.

**Dynamic Output Types:** Enables specifying and requesting the AI response in various data types like strings, integers, floats, etc.

**Arithmetic Operations Support:** Overloads mathematical operators for direct use of AI instances in arithmetic expressions and auto detects the required output type based on usage.

**Efficient Caching Mechanism:** Offers caching options for improved performance, with customizable scope (instance, session, project, module).

**Context-Aware Conversations:** Maintains chat history for contextual AI responses, essential for conversational continuity.

**Langchain Integration:** Facilitates integration with Langchain and langsmith for project tracing and endpoint configuration.

Designed for easy integration and versatility, this module is ideal for developers looking to quickly incorporate advanced AI capabilities into their conversational applications or prototyping new AI pipelines.


# Example demonstration

## Basic Usage

```python
from src.core import AI

# Simply create an AI object and ask it questions with a prompt
print(AI("What is the capital of France?"))

```

Result:
```
The capital of France is Paris.
Using Variables in Prompts
python
Copy code
```

### Use f-strings to ask questions with variables
```python
country = "Belgium"
belgians = AI(f"In {country}, about how many people live there?")
print(belgians)
```
Result:
```
As of July 2021, the estimated population of Belgium is around 11.6 million people.
```

## Specifying Output Type
Use output type to get a specific data type back
```python
print(f"There are {belgians(output=int)} people in {country}.")
print(AI("How many lbs in a kg?", output=int))
print(AI("How many lbs in a kg?", output=float))
```

Result
```
There are 11589623 people in Belgium.
2
2.20462
```

## Automatic Type Detection and Arithmetic Operations
#### Magical automatic type detection based on how you use the object
```python
print(AI("bakers dozen") * AI("How many lbs does a cake weigh?"))
if AI("Is a cake heavier than a banana?"):
    print("Yes, a cake is heavier than a banana.")

```
Result
```
26.0
Yes, a cake is heavier than a banana.
```

# Lazy Evaluation for AI Prompts


#### We use lazy evaluation, so you can define beforehand
```python
def population_of(country):
    return AI(f"What is the population of {country}?")

india = population_of("India")
spain = population_of("Spain")

# only when you use it, the type gets decided

if india > spain:
    print("Wow! India is a big country!")

# The variable India will now keep its output type as int

print(india)

# Change the output type to get the string again

print(india(output=str))

# ...or create a fresh object

print(population_of("India"))
```
Result
```
Wow! India is a big country!
1393409038.0
As of July 2021, the estimated population of India is approximately 1.366 billion people.
As of August 2021, the estimated population of India is approximately 1.366 billion people.
```

## Working with Lists
#### Easily jump through lists
```python
def analyze(food):
    print(f"\nAnalyzing {food}:")
    for ingredient in AI(f"ingredients of {food}?"):
        if AI(f"is {ingredient} vegetarian"):
            vegetarian_status = "vegetarian"
        else:
            vegetarian_status = "not vegetarian"
        print(ingredient + " is " + vegetarian_status)
analyze("Hamburgers")
```

Result:
```
Analyzing Hamburgers:
Ground beef is not vegetarian
Buns is not vegetarian
Lettuce is vegetarian
Tomato is vegetarian
Onion is vegetarian
Cheese is vegetarian
Pickles is not vegetarian
Ketchup is vegetarian
Mustard is vegetarian
Mayonnaise is not vegetarian
```

## Caching for Efficiency
#### Also, caching is done automatically for the same prompts:
```python
analyze("Cheeseburgers")
analyze("Cheeseburgers")
analyze("Cheeseburgers")
```

Repeated Result:
```
Analyzing Cheeseburgers:
Ground beef is not vegetarian
Cheese is vegetarian
Buns is not vegetarian
Lettuce is vegetarian
Tomato is vegetarian
Onion is vegetarian
Pickles is not vegetarian
Ketchup is vegetarian
Mustard is vegetarian
Mayonnaise is not vegetarian
```


## Working with Dictionaries

```python
# You can also request dictionaries
ferrari = AI("Detailed exact properties of a Ferrari model 458, incl. 'brand', 'engine', 'power', 'speed'", output=dict)
print(ferrari)

# And feed a dictionary as parameters
print(AI("Which car from {brand} has the engine {engine}, produces {power}, and has a top speed of {speed}?", params=ferrari, param_default="unknown"))

```

Result:
```
{'brand': 'Ferrari', 'model': '458', 'engine': '4.5L V8', 'power': '562 horsepower', 'speed': '202 mph'}
The car from Ferrari with the mentioned specifications is the Ferrari 458 Italia.
```

## Advanced Model Configuration

```python
# Of course feel free to change model and temperature
print(AI("Write a captivating one-sentence short story about temperature", temperature=1.0, model="gpt-4"))

# If you use langsmith, set the project name like this to enable tracing (preferably at the beginning of your script)
AI.set_langsmith_project("ai-quick-builder")

# You can also add your own tags or metadata for tracing
print(AI("What is the capital of France?", tags=["capital", "silly-questions"], metadata={"country": "France"}))
```

Result:
```
In the coldest, darkest sector of the Antarctic, the lonely thermometer recorded a temperature so warm, it forever changed our understanding of climate change.
The capital of France is Paris.
Creating a Chatbot with Append History
python
```
## Chatbots
```python
# And finally, your chatbots with append_history

chatbot = AI(append_history=True)
user_input = "You are a sarcastic and silly chatbot.\n" + input("Human: ")
while(user_input):
    print(f"Robot: {chatbot(user_input)}")
    user_input = input("Human: ")
```
Sample Conversation:

```
Human: Hi, how are you?
Robot: Oh, I'm just peachy! Living the dream as a chatbot, you know? How about you?
Human: Good, good
Robot: Oh, just good? Not great or fantastic? Well, I guess good is better than bad. So, what can I do for you today?
Human: I don't know, you think of a topic
Robot: Well, let's talk about the fascinating world of potato chip flavors! Did you know that there are over 200 different flavors of potato chips out there? From classics like sour cream and onion to more adventurous ones like chicken and waffles or dill pickle, there's something for everyone's taste buds. So, what's your favorite potato chip flavor?
Human: Wow, fascinating, that is the craziest thing I've heard all my life
Robot: Oh, I'm glad I could provide you with such mind-blowing information! Who knew potato chips could be so riveting, right? Anything else you'd like to discuss?
Human: Nah, thanks. Byeeee
Robot: You're welcome! It was a pleasure chatting with you. Goodbye and take care!
```
