import numpy as np
from ..models import *

def get_label_swap_potentials(predictor, X, Y_pred):

    '''
    Returns a matrix that represents the Softmax output of the weights of each possible label
    for each node in the graph.
    '''

    model = predictor.model
    if not type(model) in [ChainCRF, GraphCRF, GridCRF]:
        raise TypeError("Model must be a GraphCRF type.")

    weight_vector = predictor.w
    n_states = model.n_states
    directed = model.directed
    matrix_list = []

    for s, x in enumerate(X):
        prediction = Y_pred[s]
        label_swap_potentials = np.zeros((x.shape[0], n_states), dtype=np.float)
        for i, node in enumerate(x):
            for state in range(n_states):
                # identify relevant indices of would-be flattened pairwise matrix
                pw_indices = []
                if i > 0:
                    # pw_from_index
                    prev_label = prediction[i-1]
                    pw_indices.append((n_states * (prev_label - 1)) + state)
                if i < len(x):
                    # pw_to_index
                    next_label = prediction[i+1]
                    pw_indices.append((n_states * (next_label - 1)) + state)