import pickle


def load_params(name: str):
    with open(name) as f:
        params = pickle.load(f)
    return params
