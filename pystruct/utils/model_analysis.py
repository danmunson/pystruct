import numpy as np
from ..models import *

def get_label_scores(predictor, X, Y_pred):

    '''
    Returns a matrix that represents the Softmax output of the weights of each possible label
    for each node in the graph.
    '''

    model = predictor.model
    if not type(model) != ChainCRF:
        raise TypeError("Model must be a ChainCRF.")

    weight_vector = predictor.w
    n_states = model.n_states
    state_combos = ((n_states ** 2) + n_states) / 2
    directed = model.directed
    matrix_list = []

    '''
    Return value of unary-only contribution:
    
        np.dot( feature vector corresponding to Node X , weight sub-vector corresponding to Label Y)

    In referencing graph_crf.py, the (flattened) `unaries_acc` matrix represents the subset of the 
    joint-feature vector corresponding to the unary potentials (i.e. feature vector without pairwise 
    relations) - note that each entry of this matrix has a corresponding weight in the prediction function.
    It just so happens that for this matrix, rows correspond to labels and columns to features.

    Therefore, we can determine how much the label Y of node X contributes to the total score of the sequence 
    by taking the dot product of the feature vector for node X and the subset of the weight vector that 
    corresponds to the given label.
    '''
    def unary_contributions()

    '''
    Return the flat-array index corresponding to the coordinates of
    a given matrix (assumed to be asymmetrical)
    '''
    def directed_pw_indices(x_i, y_i):
        return (x_i * n_states) + y_i

    '''
    Return the flat-array index corresponding to the coordinates of 
    a symmetric matrix, assumming that we only flatten elements of 
    its upper diagonal (e.g. in `compress_sym`)
    '''
    def undirected_pw_indices(x_i, y_i):
        mx, mn = (x_i, y_i) if x_i > y_i else (y_i, x_i)
        diff = n_states - mn
        diff_z = ((diff ** 2) + diff) / 2
        return int((state_combos - diff_z) + (mx - mn))


    pw_index_function = undirected_pw_indices
    if directed:
        pw_index_function = directed_pw_indices

    for s, x in enumerate(X):
        prediction = Y_pred[s]
        label_scores = np.zeros((x.shape[0], n_states), dtype=np.float)



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