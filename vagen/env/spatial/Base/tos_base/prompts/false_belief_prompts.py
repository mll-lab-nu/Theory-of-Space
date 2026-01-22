FALSE_BELIEF_INSTRUCTION = """\
You have returned to the initial position and face north.
There are {n_changes} objects in the room that have been changed (position or orientation).
Note one object is either moved or rotated, not both.
Goal: Explore the room again and identify which objects have been changed and how with minimum costs.
Use the same action set as the exploration phase.
You must use the Term(changes="...") action to submit your answer and terminate.
Format: Term(changes="object1: change_type, object2: change_type")
Example: Term(changes="apple: position, chair: orientation")
You have a maximum of {max_steps} steps.
"""
