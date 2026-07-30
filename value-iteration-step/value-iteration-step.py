def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    states = len(values)
    actions = len(transitions[0]) # in state 0, what options / actions canmwetake? *symmetrical*
    result = []
    for state in range(states):
        action_values = []
        for action in range(actions):
            expected_value = 0
            for state_prime in range(states):    
                expected_value += transitions[state][action][state_prime]*values[state_prime]
            action_values.append(rewards[state][action] + gamma * expected_value)
        result.append(max(action_values))
    return result
        