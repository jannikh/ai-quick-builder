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

# You can also generate images. It will return a URL to the image, that you can open or download
print(AI("a very tasty cake shaped like the eiffel tower", output='image'))
# This prompt automatically gets improved by ChatGPT before generating an image using DALL-E 3.
# The improved image description prompt was:
# Create a visually enticing image of a delectable cake artfully designed in the shape of the iconic Eiffel Tower. Render the cake with luscious layers of moist sponge, adorned with smooth buttercream frosting, delicately sculpted to mimic the intricate lattice of the tower's ironwork. Enhance the cake's visual appeal by incorporating edible embellishments, such as shimmering fondant accents, metallic edible gold paint, and miniature macarons adorning the cake's "observation decks." Surround the cake with a picturesque scene, showcasing a charming Parisian backdrop, with the tower standing tall against a romantic sunset sky. Capture the essence of this mouthwatering creation by incorporating rich, vibrant colors, emphasizing the tower's signature bronze hue, and highlighting the cake's irresistible textures.
# To disable this improvement, use output="image raw"


# Magical automatic type detection based on how you use the object
print(AI("bakers dozen") * AI("How many lbs does a cake weigh?")) # number detection
if AI("Is a cake heavier than a banana?"): # asks dicrectly for a boolean value
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
