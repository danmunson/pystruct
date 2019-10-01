import numpy as np
from ..models import *

def get_label_scores(predictor, X, Y, distf=None):

    '''
    Returns a matrix that represents the Softmax-variant output of the contribution 
    of each possible label for each node in the graph.
    '''

    dist_func = np.exp
    if callable(distf):
        dist_func = distf

    model = predictor.model
    if not type(model) == ChainCRF:
        raise TypeError("Model must be a ChainCRF.")

    weight_vector = predictor.w
    directed = model.directed
    n_states = model.n_states
    n_features = X.shape[1]

    state_combos = ((n_states ** 2) + n_states) / 2
    unary_weight_subset = weight_vector[0 : (n_states * n_features)]
    unary_weight_matrix = np.reshape(unary_weight_subset, (n_states, n_features))
    
    '''
    Return value of unary-only contribution for each Label-Node combination:
    
        np.dot( feature vector corresponding to Node X , weight sub-vector corresponding to Label Y)

    In referencing graph_crf.py, the (flattened) `unaries_acc` matrix represents the subset of the 
    joint-feature vector corresponding to the unary potentials (i.e. feature vector without pairwise 
    relations) - note that each entry of this matrix has a corresponding weight in the prediction function.
    It just so happens that for this matrix, rows correspond to labels and columns to features.

    Therefore, we can determine how much the label Y of node X contributes to the total score of the sequence 
    by taking the dot product of the feature vector for node X and the subset of the weight vector that 
    corresponds to the given label.
    '''
    def unary_contributions(feature_matrix):
        # returns contribution matrix: (row i, column j) --> contribution of Label[i] x Node[j]
        return np.dot(unary_weight_matrix, feature_matrix.T)

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
    
    '''
    Return value of pairwise contribution of each Label-Node combination,
    given argmax prediction vector.
    '''
    pw_index_function = directed_pw_indices if directed else undirected_pw_indices
    pw_weight_subset = weight_vector[(n_states ** 2):] if directed else weight_vector[(-1 * state_combos):]

    def pairwise_contributions(label_pairs):
        pw_vector = np.zeros((pw_weight_subset.shape[0]), dtype=np.float)
        for x, y in label_pairs:
            pw_vector[pw_index_function(x,y)] += 1.0
        return np.dot(pw_weight_subset.T, pw_vector)
    
    '''
    Softmax-variant function (for use on each column of total_scores)
    '''
    def apply_softmax(label_score_matrix):
        softmax_matrix = np.zeros((score_matrix.shape[0], score_matrix.shape[1]), dtype=np.float)
        for col in range(score_matrix.shape[1]):
            dist = sum(np.array(list(map(dist_func, score_matrix[:,col]))))
            norm = lambda x : dist_func(x) / dist
            softmax_matrix[:,col] = np.array(list(map(norm, score_matrix[:,col])))
        return softmax_matrix

    # main
    unary_scores = unary_contributions(X)
    pairwise_scores = np.zeros((n_states, X.shape[0]), dtype=np.float)
    for y_i in range(Y.shape[0]):
        for swap_label in range(n_states):
            pw_score = 0
            if y_i == 0:
                pw_score = pairwise_contributions( (swap_label, Y[y_i+1]) )
            elif y_i == len(Y) - 1:
                pw_score = pairwise_contributions( (Y[y_i-1], swap_label) )
            else:
                pw_score = pairwise_contributions( ((Y[y_i-1], swap_label) , (swap_label, Y[y_i+1])) )
        pairwise_scores[swap_label, y_i] = pw_score
    
    total_scores = unary_scores + pairwise_scores
 
    return apply_softmax(total_scores)


