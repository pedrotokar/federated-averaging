import torch

def server_aggregate(global_model_state, client_model_states, client_lens):
    total_data = sum(client_lens)
    new_state = {}

    for name in global_model_state:
        new_state[name] = torch.zeros_like(global_model_state[name])
    
    for i, state in enumerate(client_model_states):
        client_weight = client_lens[i] / total_data
        for name in new_state:
            new_state[name] += client_weight * state[name]
    
    return new_state