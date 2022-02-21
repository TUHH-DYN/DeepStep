import numpy as np

def align_data(dataset, input_size, output_size, verbose=False):
    total_size = input_size + output_size
    num_elements = int(dataset.shape[0] - total_size + 1)

    inputs = np.zeros(shape=(num_elements, dataset.shape[1], dataset.shape[2], input_size), dtype=dataset.dtype)
    outputs = np.zeros(shape=(num_elements, dataset.shape[1], dataset.shape[2], output_size), dtype=dataset.dtype)
    
    if verbose:
        print('Number of Elements: {}'.format(num_elements))
        print('Input Shape: {}'.format(inputs.shape))
        print('Outputs Shape: {}'.format(outputs.shape))

    for i in range(num_elements):
        inputs[i, :, :, :] = np.transpose(dataset[i:(i+input_size), :, :], (1,2,0))
        outputs[i, :, :, :] = np.transpose(dataset[(i+input_size):(i+input_size+output_size), :, :], (1,2,0))

    return [inputs, outputs]

def randomize_order(inputs, outputs):
    permutation = np.random.permutation(outputs.shape[0])
    inputs_perm = []
    for i in range(len(inputs)):
        inputs_perm.append(inputs[i][permutation])

    outputs_perm = outputs[permutation]
    return inputs_perm, outputs_perm    

def has_nan(inputs, outputs):
    has_nan = False
    for i in range(len(inputs)):
        if np.isnan(inputs[i]).any():
            has_nan = True
    if np.isnan(inputs[i]).any():
        has_nan = True
    return has_nan