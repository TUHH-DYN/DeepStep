import numpy as np
import random
import glob
import enum

import tensorflow as tf

import modules.preprocess as preprocess

class SIZE_UNIT(enum.Enum):
   BYTES = 1
   KB = 2
   MB = 3
   GB = 4

def convert_unit(size_in_bytes, unit):
   """ Convert the size from bytes to other units like KB, MB or GB"""
   if unit == SIZE_UNIT.KB:
       return size_in_bytes/1024
   elif unit == SIZE_UNIT.MB:
       return size_in_bytes/(1024*1024)
   elif unit == SIZE_UNIT.GB:
       return size_in_bytes/(1024*1024*1024)
   else:
       return size_in_bytes

def get_files_batch(files, start, num_files):
    num_inputs = len(files)

    files_batch = []
    for i in range(num_inputs):
        files_batch.append([])

    for i in range(start, start+num_files):
        if i >= len(files[0]):
            break
        for j in range(num_inputs):
            files_batch[j].append(files[j][i]) 

    return files_batch

def load_data(files, input_sizes, output_size, reference_input = 0, augmentation = b'none', randomize_order = True):

    encoding = 'utf-8'
    augmentation = augmentation.decode(encoding)

    num_inputs = len(files)
    outputs = []
    inputs = []
    for i in range(num_inputs):
        inputs.append([])

    for ii in range(len(files[0])):
        for i in range(num_inputs):
            if i == reference_input:
                dataset = np.load(files[i][ii])  
                dataset = dataset.astype(np.float32)
                dataset = dataset[::, ...]
                [inputs_single, labels_single] = preprocess.align_data(dataset, input_sizes[i], output_size)
                inputs[i].append(inputs_single)
                outputs.append(labels_single)
            elif input_sizes[i] == 1:
                dataset = np.load(files[i][ii])
                dataset = dataset.astype(np.float32)
                dataset = dataset[np.newaxis, ...]
                inputs[i].append(dataset)                    
            else:
                assert 0
                    
        for i in range(num_inputs):
            if i == reference_input:
                continue
            n_rep = inputs[reference_input][-1].shape[0]/inputs[i][-1].shape[0]
            inputs[i][-1] = np.transpose(np.repeat(inputs[i][-1], n_rep, axis=0), (0, 1, 2))

    del dataset
    del inputs_single
    del labels_single

    num_datasets = len(inputs[0])
    assert num_datasets == len(outputs)
    for i in range(1, num_inputs):
        assert num_datasets == len(inputs[i])

    num_datasets = len(inputs[0])
    
    if (augmentation == 'mirror') or (augmentation == 'all') or (augmentation == 'mirror_and_flip') or (augmentation == 'all_and_flip'):

        for j in range(num_datasets):
            outputs.append( np.flip(outputs[j], 1))
            for i in range(num_inputs):
                inputs[i].append( np.flip(inputs[i][j], 1))

        for j in range(num_datasets):
            outputs.append( np.flip(outputs[j], 2))
            for i in range(num_inputs):
                inputs[i].append( np.flip(inputs[i][j], 2))


    if (augmentation == 'rotate') or (augmentation == 'all') or (augmentation == 'rotate_and_flip') or (augmentation == 'all_and_flip'):

        for j in range(num_datasets):
            outputs.append( np.rot90(outputs[j], 1, (1,2)) )
            outputs.append( np.rot90(outputs[j], 2, (1,2)) )
            outputs.append( np.rot90(outputs[j], 3, (1,2)) )
            for i in range(num_inputs):
                inputs[i].append( np.rot90(inputs[i][j], 1, (1,2)) )
                inputs[i].append( np.rot90(inputs[i][j], 2, (1,2)) )
                inputs[i].append( np.rot90(inputs[i][j], 3, (1,2)) )

    if (augmentation == 'mirror_and_flip') or (augmentation == 'rotate_and_flip') or (augmentation == 'all_and_flip') or (augmentation == 'flip'):

        num_augmented_outputs = len(outputs)
        for j in range(num_augmented_outputs):
            outputs.append( outputs[j]*(-1.0) )

        for i in range(num_inputs):
            num_augmented_inputs = len(inputs[i])
            for j in range(num_augmented_inputs):
                if i == reference_input:
                    inputs[i].append( inputs[i][j]*(-1.0) )
                else:
                    inputs[i].append(inputs[i][j])

    num_datasets = len(inputs[0])

    outputs = np.concatenate(outputs, axis = 0)          
    for i in range(num_inputs):
        inputs[i] = np.concatenate(inputs[i], axis = 0)
        if i != reference_input:
            inputs[i] = inputs[i][..., np.newaxis]

    if randomize_order:
        inputs, outputs = preprocess.randomize_order(inputs, outputs)

    has_nan = preprocess.has_nan(inputs, outputs)
    assert has_nan == False

    return inputs, outputs

def data_generator_u_n_net_file_batches_single(files, input_sizes, output_size, file_batch_size = 2, reference_input = 0, output_crop = 0, augmentation = b'none', randomize_files = False, randomize_order = True):

    num_inputs = len(input_sizes)
    num_files = len(files[0])

    if randomize_files:
        files = shuffle_src_files(files)

    files_batch = get_files_batch(files, 0, file_batch_size)
    inputs, outputs = load_data(files_batch, input_sizes, output_size, reference_input = reference_input, augmentation = augmentation, randomize_order = randomize_order)

    num_samples = inputs[0].shape[0]
    i = 0   
    file_i = 0 
    run = True

    while run:
        if i >= num_samples:
            i = 0
            file_i += file_batch_size
            if file_i < num_files:
                files_batch = get_files_batch(files, file_i, file_batch_size)
                inputs, outputs = load_data(files_batch, input_sizes, output_size, reference_input = reference_input, augmentation = augmentation, randomize_order = randomize_order)
                num_samples = inputs[0].shape[0]
            else:
                run = False 
        else:
            yield_dict = {}
            for j in range(num_inputs):
                yield_dict['input_'+str(j)] = inputs[j][i]
            
            if output_crop <= 0:
                yield_dict['label'] = outputs[i]
                yield yield_dict, outputs[i]
            else:
                yield_dict['label'] = outputs[i][output_crop:-output_crop,output_crop:-output_crop]
                yield yield_dict, outputs[i][output_crop:-output_crop,output_crop:-output_crop]

            i += 1

def get_src_files(data_path, input_file_prefixes):
    
    num_inputs = len(input_file_prefixes)
    files_src = []
    for i in range(num_inputs):
        files_src.append(sorted(glob.glob(data_path+input_file_prefixes[i]+'*.npy')))
    
    num_datasets = len(files_src[0])
    for i in range(1, num_inputs):
        assert num_datasets == len(files_src[i]) , "num_datasets="+str(num_datasets)+ " != len(files_src["+str(i)+"])="+str(len(files_src[i]) )

    return files_src

def shuffle_src_files(files_src):
    c = list(zip(*files_src))
    random.shuffle(c)
    files_src = list(zip(*c))
    return files_src

def get_files(data_path_train, data_path_test, input_file_prefixes):
    files_train = get_src_files(data_path_train, input_file_prefixes)
    files_test = get_src_files(data_path_test, input_file_prefixes)
    print('num_datasets_train: {}'.format(len(files_train[0])))
    print('num_datasets_test: {}'.format(len(files_test[0])))
    return [files_train, files_test]

def get_dataset_from_generators(files_train, files_test, input_sizes, output_size, batch_size, file_batch_size, reference_input, output_crop = 0, augmentation = 'none', randomize_files = True, randomize_order = True):
        
    num_inputs = len(input_sizes)
    assert len(files_train) == num_inputs
    assert len(files_test) == num_inputs

    print('file_batch_size: {}'.format(file_batch_size))
    print('batch_size: {}'.format(batch_size))
    print('num_inputs: {}'.format(num_inputs))

    output_types = {}
    output_shapes = {}
    for i in range(num_inputs):
        input_name = 'input_'+str(i)
        output_types[input_name] = tf.float32
        output_shapes[input_name] = (None, None, None, input_sizes[i])
    output_types['label'] = tf.float32
    output_shapes['label'] = (None, None, None, output_size)


    dataset_train = tf.data.Dataset.from_generator(data_generator_u_n_net_file_batches_single, args=[files_train, input_sizes, output_size, file_batch_size, reference_input, output_crop, augmentation, randomize_files, randomize_order],
                                                output_types=(output_types, tf.float32))

    dataset_test = tf.data.Dataset.from_generator(data_generator_u_n_net_file_batches_single, args=[files_test, input_sizes, output_size, 1, reference_input, output_crop, 'none', False, False],
                                                output_types=(output_types, tf.float32))

    dataset_train = dataset_train.batch(batch_size).prefetch(1)
    dataset_test = dataset_test.batch(batch_size).prefetch(1)

    return [dataset_train, dataset_test]


