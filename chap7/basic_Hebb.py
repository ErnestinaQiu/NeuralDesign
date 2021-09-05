""" examples 7.1 """
import numpy as np

def Hebb_rule(input_list, target_list):
    """ acquire weights matrix @input_list(consist of input vectors, n*m), @target_list(target value correspond with input, n*d) 
    input_list^T * target_list
    """
    in_ndim = get_ndim(input_list)
    in_arr = np.zeros((len(input_list), in_ndim))
    for i in range(len(input_list)):
        in_arr[i, :len(input_list[i])] = np.array(input_list[i][:])
        
    out_ndim = get_ndim(target_list)
    out_arr = np.zeros((len(target_list), out_ndim))
    for j in range(len(target_list)):
        out_arr[j, :len(target_list[j])] = np.array(target_list[j][:])
        
    weights_matrix = np.dot(np.transpose(target_list), in_arr)
    return weights_matrix

def hebb_predict(input_vector, weights_matrix):
    target = np.dot(weights_matrix, np.transpose(input_vector))
    return target

def get_ndim(vectors_list):
    ndim = 0
    for i in range(len(vectors_list)):
        vector = vectors_list[i]
        if len(vector) > ndim:
            ndim = len(vector)
    return ndim
    
def pseudoinverse(input_list, target_list):
    """ @input_list([vector(1*m), ...]) """
    in_ndim = get_ndim(input_list)
    in_arr = np.zeros((len(input_list), in_ndim))
    for i in range(len(input_list)):
        in_arr[i, :len(input_list[i])] = np.array(input_list[i][:])
    
    out_ndim = get_ndim(target_list)      
    out_arr = np.zeros((len(target_list), out_ndim))
    for j in range(len(target_list)):
        out_arr[j, :len(target_list[j])] = np.array(target_list[j][:])
    
    p_plus = np.dot(np.linalg.inv(np.dot(in_arr, np.transpose(in_arr))), in_arr)  # if ndim=1, the shape is (1, m), m features, 
    weights_matrix = np.dot(np.transpose(out_arr), p_plus) # 
    
    return weights_matrix

def pseudoinverse_predict(input_vector, weights_matrix):
    if not isinstance(input_vector, np.ndarray):
        input_vector = np.array(input_vector)
    print('weights_matrix.shape=={}, input_vector.shape=={}'.format(weights_matrix.shape, input_vector.shape))
    # target = np.dot(weights_matrix, input_vector)
    target = np.dot(weights_matrix, np.transpose(input_vector))
    return target

def _7_1():
    # example 7.1
    p1 = np.array([1, -1, 1, -1])
    p2 = np.array([1, 1, -1, -1])
    t1 = np.array([1, -1])
    t2 = np.array([1, 1])
    input_list = [p1, p2]
    target_list = [t1, t2]
    
    weights_matrix = Hebb_rule(input_list, target_list)
    print(weights_matrix)
    target = hebb_predict(p1, weights_matrix)
    print(target)
    
    # pseudoinverse
    weights_matrix_2 = pseudoinverse(input_list, target_list)
    print('weights_matrix_2: {}'.format(weights_matrix_2))
    target_2 = pseudoinverse_predict(p1, weights_matrix_2)
    print('pseudoinverse prediction: {}'.format(target_2))
    
def self_associate_saver():
    """ 7.2 """
    # 1st model
    p1 = np.array([1, 1, 1, 0, 0, 0])
    t1 = np.array([1, 0, 0])
    p2 = np.array([0, 1, 1, 1, 1, 0])
    t2 = np.array([0, 1, 0])
    p3 = np.array([1, 1, 1, 1, 1, 0])
    t3 = np.array([0, 0, 1])

    # check whether those vectors are orthogonal
    def orthogonal_check(vectors):
        orthogonal = True
        for i in range(len(vectors)):
            vec_1 = vectors[i]
            for j in range(i, len(vectors)):
                vec_2 = vectors[j]
                if np.dot(vec_1, np.transpose(vec_2)) == 0:
                    continue
                else:
                    orthogonal = False
                    break
            if orthogonal == False:
                break
        return orthogonal
    
    input_list = [p1, p2]
    orthogonal_check_answer = orthogonal_check(input_list)
    print('i: {}'.format(orthogonal_check_answer))
    
    output_list = [t1, t2, t3]
    weights_matrix = Hebb_rule(input_list, input_list)
    print('weights_matrix: {}'.format(weights_matrix))
    target = hebb_predict(p3, weights_matrix)
    print('target: {}'.format(target))

    # 2nd model, whose vectorization is different to the 1st one, but because the value is different, which made the first and the second vectors are orthoganal, this performs better than 1st
    p1 = np.array([1, 1, 1, -1, -1, -1])
    t1 = np.array([1, 0, 0])
    p2 = np.array([-1, 1, 1, 1, 1, -1])
    t2 = np.array([0, 1, 0])
    p3 = np.array([1, 1, 1, 1, 1, -1])
    t3 = np.array([0, 0, 1])
    
    input_list = [p1, p2]
    orthogonal_check_answer = orthogonal_check(input_list)
    print('i: {}'.format(orthogonal_check_answer))
    
    output_list = [t1, t2]
    weights_matrix = Hebb_rule(input_list, input_list)
    print('weights_matrix: {}'.format(weights_matrix))
    target = hebb_predict(p3, weights_matrix)
    print('target: {}'.format(target))

    # 3rd, use the pseudoinverse
    weights_matrix_2 = pseudoinverse(input_list, input_list)
    print('weights_matrix_2: {}'.format(weights_matrix_2))
    
    target_2 = pseudoinverse_predict(input_list, weights_matrix_2)
    print('target_2: {}'.format(target_2))
    

        





if __name__ == '__main__':
    # _7_1()
    self_associate_saver()
    