import openai
import textwrap

def llm_init():
    openai.api_key = 'Please input your API key here.'

    global prompt_system, prompt_history, query_count

    prompt_system = '''
    You are the controller of a quadrupedal robot (A1 robot) with 10 Hz.
    Please inference the output.
    
    The robot's state is represented by a 33-dimensional input space.
    The first 3 dimensions correspond to the robot's linear velocity.
    The next 3 dimensions denote the robot's angular velocity.
    The following 3 dimensions represent the gravity vector.
    The subsequent 12 dimensions represent the joint positions.
    The final 12 dimensions indicate the velocity of each joint.

    The output space is 12-dimension, which is the joint position. 
    
    The order of the joints is [FRH, FRT, FRC, FLH, FLT, FLC, RRH, RRT, RRC, RLH, RLT, RLC].

    After we have the output, we will use 200 Hz PD controller to track it.

    The following are past and consecutive inputs and outputs.
    All numbers are normalized to non-negative integers by our special rule. 
    The output would be impacted by the previous inputs.
    The trend of the outputs should be smooth.

    Your output is only one line and starts with "Output:", please do not output other redundant words.
    
    '''
    prompt_system = textwrap.dedent(prompt_system)
    prompt_system = prompt_system.split('\n', 1)[1]

    prompt_history = ''

    query_count = 0

def llm_query(msg, call_api=True):
    global prompt_system, prompt_history, query_count

    prompt_history = prompt_history + msg + '\n'

    if call_api:
        completion = openai.ChatCompletion.create(
            model="gpt-4-0613", # "gpt-3.5-turbo-16k"
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_history}
            ],
            temperature=0.0
        )

        res = completion.choices[0].message.content
        res = res.split('\n', 1)[0]

    query_count += 1

    if query_count > 50:
        prompt_history = prompt_history.split('\n', 1)[1]
        prompt_history = prompt_history.split('\n', 1)[1]

    if call_api:
        return res
    else:
        return None