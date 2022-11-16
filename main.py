##############################################################################################################################
### IMPORTS ##################################################################################################################
##############################################################################################################################
import re
import openai
import threading
from inputs import random_inputs
##############################################################################################################################
### REQUESTS #################################################################################################################
##############################################################################################################################

openai.api_key = "secret-key-you'll-need-your-own-2-test-this"


importance_inputs_and_outputs = []
memory_number = 0
short_term_memory_list = []
long_term_memory_list = []
vision_outputs = []
hearing_outputs = []
decision_outputs = []
cleaned_decicions_list = []

def GPT3_Request_IBTQ(user_input, vision_or_hearing_or_decision, keyword):
    short = short_context()
    long = long_context()
    print("--------------------- gpt3 request ---------------------")
    
    # context creation
    #importance = ("This is the context: 'Short term memory context: " + short + ". Long term memory context: " + long + "'. This is the end of the context." + user_input)
    importance = user_input
    print("importance: ", importance)

    # request creation
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=importance,
        temperature=1,
        max_tokens=4000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
      )
    cleaning_requests(response,vision_or_hearing_or_decision,keyword)
    print(" --------------------- gpt3 end of request ---------------------")
    #return response.choices[0].text

def cleaning_requests(response, vision_or_hearing_or_decision, keyword):
    # cleaning the GPT-3's response
    dirty_response = response.choices[0].text
    try:
        cleaned_response = int(re.sub("[^0-9]", "", dirty_response))
    except:
        cleaned_response = 0
    print("How important GPT-3 Classifies this: ", cleaned_response)

    # Is is a decision?
    possible_outputs_vision_hearing = [0,1,2,3,4,5,6,7,8,9,10]
    possible_outputs_vision_decision = [0,1,2,3,4]
    if cleaned_response in possible_outputs_vision_hearing and vision_or_hearing_or_decision != "decision":
        pass
    elif cleaned_response in possible_outputs_vision_decision and vision_or_hearing_or_decision == "decision":
        pass
    else:
        cleaned_response = 0

    # Is it a vision or hearing or decision?
    if vision_or_hearing_or_decision == "vision":
        vision_outputs.append(cleaned_response)
    elif vision_or_hearing_or_decision == "hearing":
        hearing_outputs.append(cleaned_response)
    elif vision_or_hearing_or_decision == "decision":
        decision_outputs.append(cleaned_response)

    # appending the importance inputs to a list
    importance_inputs_and_outputs.append([keyword, cleaned_response, vision_or_hearing_or_decision]) 

    # Cleaning decision outputs
    for x in decision_outputs:
        if x == 0:
            cleaned_decicions_list.append("You decided do nothing.")
        elif x == 1:
            cleaned_decicions_list.append("You decided to look around.")
        elif x == 2:
            cleaned_decicions_list.append("You decided to interact.")
        elif x == 3:
            cleaned_decicions_list.append("You decided to move.")

    # prining outputs
    print("vision outputs: ", vision_outputs)
    print("hearing outputs: ", hearing_outputs)
    print("decision outputs: ", decision_outputs)

##############################################################################################################################
### MEMORY ###################################################################################################################
##############################################################################################################################

def short_context():
    print("--------------------- short context ---------------------")
    final_context = ""
    if len(short_term_memory_list) == 0:
        #print("short term memory: ", short_term_memory_list)
        return "No short term memory"
    else:
        for x in short_term_memory_list:
            current_context = re.sub('[^A-Za-z0-9]+', ' ', str(x))
            final_context = str(current_context + " " + final_context)
        #print("complete_context:\n", complete_context)
        return final_context

def long_context():
    print("--------------------- long context ---------------------")
    final_context = ""
    if len(long_term_memory_list) == 0:
        print("long term memory: ", long_term_memory_list)
        return "No long term memory"
    else:
        #print("long term memory: ", long_term_memory_list)
        #return "Long term memory: " + str(long_term_memory_list)
        #print("long term memory: ", long_term_memory_list)
        for x in long_term_memory_list:
            current_context = re.sub('[^A-Za-z0-9]+', ' ', str(x))
            final_context = str(current_context + " " + final_context)
        #print("complete_context:\n", complete_context)
        return final_context

def cleaning_memory():
    print("--------------------- cleaning memory ---------------------")
    # ------------------ memory cleaning ------------------
    if len(short_term_memory_list) > 2: # amount of memories, not characters
        short_term_memory_list.pop(0)
        # ------------------ main loop's logic ------------------


    if len(long_term_memory_list) > 2: # amount of memories, not characters
        # lowest_valued_memory = min(long_term_memory_list, key=lambda x: x[1])
        # long_term_memory_list.pop(lowest_valued_memory)
        long_term_memory_list.pop(0)
        # ------------------ main loop's logic ------------------

def creating_vision_memory(counter, vision_input):
    print("--------------------- creating vision memory ---------------------")
    if vision_outputs[counter] >= 0 and vision_outputs[counter] < 8:
        short_term_memory_list.append([vision_input, vision_outputs[counter], "vision"])

    elif vision_outputs[counter] > 8:
        long_term_memory_list.append([vision_input, vision_outputs[counter], "vision"])
    
def creating_hearing_memory(counter, hearing_input):
    print("--------------------- creating hearing memory ---------------------")
    if hearing_outputs[counter] >= 0 and hearing_outputs[counter] < 8:
        short_term_memory_list.append([hearing_input, hearing_outputs[counter], "hearing"])

    elif hearing_outputs[counter] > 8:
        long_term_memory_list.append([hearing_input, hearing_outputs[counter], "hearing"])


def creating_decision_memory(counter):
    print("--------------------- creating decision memory ---------------------")
    if decision_outputs[counter] >= 0 and decision_outputs[counter] < 8:
        short_term_memory_list.append([str(counter), decision_outputs[counter], "decision"])

    elif decision_outputs[counter] > 8:
        long_term_memory_list.append([str(counter), decision_outputs[counter], "decision"])

##############################################################################################################################
### Multithreaded Priority Queues ############################################################################################
##############################################################################################################################

def thread_function():
    print("--------------thread function--------------")
    the_inputs = random_inputs() # receive inputs
    cleaned_inputs = [x for x in the_inputs] # clean inputs
    print("cleaned_inputs: ",cleaned_inputs)

    # Start Multithreaded Priority Queues loops
    counter = 0
    
    for counter in range(len(cleaned_inputs)):  # loops through the inputs

        # ------------------ request header ------------------
        print("counter: ", counter)
        print("cleaned_inputs:", cleaned_inputs)
        vision_input =  cleaned_inputs[counter]
        print("vision_input: ", vision_input)
        hearing_input = cleaned_inputs[counter]
        print("hearing_input: ", hearing_input)
        print("short_term_memory_list: ", short_term_memory_list)
        print("long_term_memory_list: ", long_term_memory_list)
            
        complement_vision = "You see a " + vision_input + " right in front of you. From 0 to 10, how important is that? (RESPOND IN NUMBERS ONLY, PLEASE) \n"
        complement_hearing = "You hear a " + hearing_input + " right in front of you. From 0 to 10, how important is that? (RESPOND IN NUMBERS ONLY, PLEASE) \n"
        complement_decision = "Based on the context given, what do you want to do?\n Option 0: Do nothing.\nOption 1: Look around.\nOption 2: Interact.\nOption 3: Move.\n(Your responde (PLEASE RESPOND A NUMBER FROM 0 TO 3)) \n"

        # ------------------ multi-threading ------------------
        vision = threading.Thread(target=GPT3_Request_IBTQ(complement_vision, "vision", cleaned_inputs[counter]))    
        hearing = threading.Thread(target=GPT3_Request_IBTQ(complement_hearing, "hearing", cleaned_inputs[counter]))
        decision = threading.Thread(target=GPT3_Request_IBTQ(complement_decision, "decision", cleaned_inputs[counter]))
        # --- thread that won last time starts first --- #
        if vision_outputs[counter] > hearing_outputs[counter]:
            # --- start sequence of threads --- #
            print("VISION WINS")
            vision.start() # vision starts first
            vision.join()
            hearing.start()
            hearing.join()
            decision.start()
            decision.join()
            creating_vision_memory(counter, vision_input)
            creating_hearing_memory(counter, hearing_input)
            cleaning_memory()  
            #creating_decision_memory(counter)
            
        # --- thread that won last time starts first --- #
        elif vision_outputs[counter] < hearing_outputs[counter]:
            # --- start sequence of threads --- #
            print("HEARING WINS")
            hearing.start() # hearing starts first
            hearing.join()
            vision.start()
            vision.join()
            decision.start()
            decision.join()
            creating_vision_memory(counter, vision_input)
            creating_hearing_memory(counter, hearing_input)
            cleaning_memory()  
            #creating_decision_memory(counter, counter)
            
        # --- thread that won last time starts first --- #
        elif vision_outputs[counter] == hearing_outputs[counter]:
            # --- start sequence of threads --- #
            print("TIE")
            vision.start()
            vision.join()
            hearing.start()
            hearing.join()
            decision.start()
            decision.join()
            creating_vision_memory(counter, vision_input)
            creating_hearing_memory(counter, hearing_input)
            cleaning_memory()  
            #creating_decision_memory(counter, counter)

        cleaning_memory()    
        counter += 1
            # --- make a memory --- #
    
thread_function()

##############################################################################################################################
### REPORTS ##################################################################################################################
##############################################################################################################################

print("")
print("REPORT OF ALL INPUTS AND OUTPUTS:")
for x in importance_inputs_and_outputs:
    print(x)

print("\nshort term memory:", short_term_memory_list)
print("long term memory:", long_term_memory_list)
print("len short term: ", len(short_term_memory_list))
print("len long term: ", len(long_term_memory_list))
print("-------- choices list --------")
print("cleaned_decicions_list:\n", cleaned_decicions_list)






"""
openai.error.InvalidRequestError: This model's maximum context length is 4097 tokens, however you requested 4120 tokens (120 in your prompt; 4000 for the completion). Please reduce your prompt; or completion length.
"""

"""
PINNED NOTES

TOKENS ARE EXPENSIVE.
14,000 characters is equal to 25 cents USD which is 1.50 reals.

allocation of tokens:
    total: 14 000 characters
    minimum size for next set of IBTQ: 500 characters
    maximum size of short term memory: 6750 characters
    maximum size of long term memory: 6750 characters


# seems like a good idea to add energy consumption and "sleep" to organize the memory
# dreams can be evolutionary algorithm'd and the memory can be evolved
# self preferences and self image needs to be added to memory 
"""

# for next time
# add actions
# add a test room
# learn how to use evolutionary algorithms on all of this


"""
notes to contuinue this:

Add queues based on importance [ ok ]
make an experimental room
add other senses
see how well the importance based system works

"""


"""
notes 2

fix queue, printing werid


short term and long term memory
needs cleaning to fit gpt-3 buffer size
buffer limit: 4000 characters in total 
needs to experiment with 50% short term and 50% long term memory
needs to add takable actions



"""