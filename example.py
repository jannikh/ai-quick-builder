from src.core import AI

# Simply create an AI object and ask it questions with a prompt
print(AI("What is the capital of France?"))


# Use f-strings to ask questions with variables
country = "Belgium"
belgians = AI(f"In {country}, about how many people live there?")
print(belgians)

# Use output type to get a specific data type back
print(f"There are {belgians(output=int)} people in {country}.")
print(AI("How many lbs in a kg?", output=int))
print(AI("How many lbs in a kg?", output=float))


# Magical automatic type detection based on how you use the object
print(AI("bakers dozen") * AI("How many lbs does a cake weigh?"))
if AI("Is a cake heavier than a banana?"):
	print("Yes, a cake is heavier than a banana.")


# We use lazy evaluation, so you can define beforehand
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


# Easily jump through lists:
def analyze(food):
	print(f"\nAnalyzing {food}:")
	for ingredient in AI(f"ingredients of {food}"):
		if AI(f"is {ingredient} vegetarian"):
			vegetarian_status = "vegetarian"
		else:
			vegetarian_status = "not vegetarian"
		print(ingredient + " is " + vegetarian_status)
analyze("Hamburgers")


# Also, caching is done automatically for the same prompts:
analyze("Cheeseburgers")
analyze("Cheeseburgers")
analyze("Cheeseburgers")


# You can also request dictionaries:
ferrari = AI("Detailed exact properties of a Ferrari model 458, incl. 'brand', 'engine', 'power', 'speed'", output=dict)
print(ferrari)

# And feed a dictionary as parameters:
print(AI("Which car from {brand} has the engine {engine}, produces {power}, and has a top speed of {speed}?", params=ferrari, param_default="unknown")) # Use param_default to specify what to use if a parameter is not found, otherwise it will throw an error


# Of course feel free to change model and temperature
print(AI("Write a captivating one-sentence short story about temperature", temperature=1.0, model="gpt-4"))

# If you use langsmith, set the project name like this to enable tracing (preferably at the beginning of your script)
AI.set_langsmith_project("ai-quick-builder")

# You can also add your own tags or metadata for tracing
print(AI("What is the capital of France?", tags=["capital", "silly-questions"], metadata={"country": "France"}))


# And finally, your chatbots with append_history
chatbot = AI(append_history=True)
user_input = "You are a sarcastic and silly chatbot.\n" + input("Human: ")
while(user_input):
	print(f"Robot: {chatbot(user_input)}")
	user_input = input("Human: ")
