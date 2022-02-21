import tensorflow as tf
from tensorflow import keras

import numpy as np

import modules.loss as loss

def get_u_p_net_model(input_shapes=[(None, None, 3), (None, None, 1), (None, None, 1)],
                      inputs_dangling=[0,0,0],
                      label_shape_for_loss=(None, None, 1),
                      input_ids_last_timestep = [0,2], # [0,2] == input[0][2] holds field at last timestep
                      num_output_frames = 1,
                      num_input_filters = [[96,96,96],[96,96,96],[96,96,96]],
                      conv_kernel_sizes = [3,3,3],
                      pooling_size = 2,
                      depth = 3,
                      concatenate_inputs = False, # or True
                      padding = 'valid', # or 'same'
                      skip_connection_handling = 'concatenate', # or 'add' or 'no_skip_connections'
                      pooling_handling = 'max_pooling', # or 'transpose_convolution'
                      normalization_handling = None, # or 'custom_sample_norm' or 'batch_norm'
                      domain_injection = True, # True or  False
                      domain_injection_ids = [1,0], # [1,0] == input[1][0] holds domain bitmask
                      activation_function = 'relu',
                      activation_function_out = 'relu',
                      dropout_rate = 0.0):
    
    # some asserts to check the model parametrization
    assert padding == 'valid' or padding == 'same'
    assert (skip_connection_handling == 'add' and pooling_size == 2) or skip_connection_handling == 'concatenate' or skip_connection_handling == 'no_skip_connections'
    assert pooling_handling == 'max_pooling' or pooling_handling == 'transpose_convolution'
    assert normalization_handling == None or normalization_handling == 'batch_norm' or normalization_handling == 'custom_sample_norm' or normalization_handling == 'custom_sample_norm_zero_symm'

    # dangling input layers are not connected to the output
    # but can be used to compute custom loss functions
    # for example to apply a domain mask to the loss
    total_dangling = np.sum(inputs_dangling)

    # assign the same pooling size for each layer
    pooling_sizes = []
    for i in range(depth):
        pooling_sizes.append(pooling_size)

    # define number of branches (Inputs)
    num_inputs=len(input_shapes)
    if concatenate_inputs:
        num_branches = 1
    else:
        num_branches=len(input_shapes)-total_dangling

    print(f"num_branches = {num_branches}")

    # Validate Input
    assert (pooling_size == 1) or (pooling_size == 2)
    assert num_branches > 0
    assert num_branches == len(num_input_filters)
    assert depth == len(conv_kernel_sizes)


    # Compute resulting cropping action. If 'valid' convolutions are utillized, 
    # the model output will be of different size than the input. Depending on the 
    # U-Net architecture (depth, pooling definition), the amount of cropping 
    # changes. 

    # compute paddings (for crop)
    kernel_paddings = []
    for i in range(depth):
        # per layer (depth), we will lose some pixels at the boundary
        kernel_paddings.append(int((conv_kernel_sizes[i]-1)/2))
    
    num_conv_sequ = 2
    paddings = []
    for i in range(depth):
        p = 0
        f = 1
        for ii in range(i+1, depth):
            f *= pooling_size
            if ii == depth-1:
                p += f*num_conv_sequ*kernel_paddings[ii]
            else:
                p += 2*f*num_conv_sequ*kernel_paddings[ii]
        paddings.append(p)

    output_crop = 0
    if padding == 'valid':
        output_crop = paddings[0] + 2*2*kernel_paddings[0]

    
    
    # Build the U^p-Net (automatic allocation of the input paths)
    
    # 1. generate input layers
    branch_to_input_map = []
    inputs = []
    for b in range(num_inputs):
        name_str = "input_"+str(b)
        inputs.append(keras.layers.Input(shape=input_shapes[b], name=name_str))

        if inputs_dangling[b] == 0:
            branch_to_input_map.append(b)
    
    # the 'label' layer is also a 'dangling' input layer,
    # not connected to the output layer, but used in custom loss
    # functions

    inputs.append(keras.layers.Input(shape=label_shape_for_loss, name="label"))


    # (optional) sample normalization
    if normalization_handling == 'custom_sample_norm' or normalization_handling == 'custom_sample_norm_zero_symm':

        def max_val_func(inp_x):
            return tf.reduce_max(tf.math.abs(inp_x), axis=(1,2,3))
        
        if normalization_handling == 'custom_sample_norm':
            print ('Using model norm: custom_sample_norm')
            def norm_func(inp_x):
                max_val = inp_x[0]
                inp_x_norm = tf.transpose(inp_x[1], [3, 1, 2, 0])
                inp_x_norm = tf.math.add(tf.constant(0.5), tf.math.divide(inp_x_norm, tf.math.multiply(tf.constant(2.0), max_val)))
                return tf.transpose(inp_x_norm, [3, 1, 2, 0])
        elif normalization_handling == 'custom_sample_norm_zero_symm':
            print ('Using model norm: custom_sample_norm_zero_symm')
            def norm_func(inp_x):
                max_val = inp_x[0]
                inp_x_norm = tf.transpose(inp_x[1], [3, 1, 2, 0])
                inp_x_norm = tf.math.divide(inp_x_norm, tf.math.multiply(tf.constant(2.0), max_val))
                return tf.transpose(inp_x_norm, [3, 1, 2, 0])

        max_val = keras.layers.Lambda(max_val_func)(inputs[input_ids_last_timestep[0]]) 
        
        normalized_input = keras.layers.Lambda(norm_func)([max_val, inputs[input_ids_last_timestep[0]]]) 
        normalized_label = keras.layers.Lambda(norm_func)([max_val, inputs[-1]]) 
        label_layer_for_loss = normalized_label

        preproc_inputs = []
        for i in range(len(inputs)):
            if i == input_ids_last_timestep[0]:
                preproc_inputs.append(normalized_input)
            elif i == len(inputs)-1:
                preproc_inputs.append(normalized_label)
            else:
                preproc_inputs.append(inputs[i])
    else:
        preproc_inputs = inputs
        label_layer_for_loss = inputs[-1]

    if output_crop > 0 and domain_injection == True:
        label_layer_for_loss = label_layer_for_loss[:, output_crop:-output_crop,output_crop:-output_crop,:]

    # 2. first layer 
    x = []
    if concatenate_inputs == True:
        input_list = []
        for b in range(num_inputs):
            if inputs_dangling[b] == 0:
                input_list.append(preproc_inputs[b])
        x.append(keras.layers.Concatenate()(input_list))
    else:
        for b in range(num_branches):
            x.append(preproc_inputs[branch_to_input_map[b]]) 

    # skip connections
    s = [] 
        
    for i in range(depth):
        s.append([])
        for b in range(num_branches):
            if i != 0:
                x[b] = s[i-1][b]
                if pooling_size > 1:
                    if pooling_handling == 'max_pooling':
                        x[b] = keras.layers.MaxPool2D(pooling_size)(x[b])
                    elif pooling_handling == 'transpose_convolution':
                        x[b] = keras.layers.Conv2D(filters=num_input_filters[b][i-1], kernel_size=2, strides=2, padding='valid')(x[b])
                    else:
                        assert False                
                if dropout_rate > 0:
                    x[b] = keras.layers.SpatialDropout2D(dropout_rate)(x[b])

            x[b] = keras.layers.Conv2D(num_input_filters[b][i], conv_kernel_sizes[i], padding=padding, use_bias=True)(x[b])
            if normalization_handling == 'batch_norm':
                x[b] = keras.layers.BatchNormalization()(x[b])
            
            if activation_function[:4] == 'bipo':
                x_b_0, x_b_1 = tf.split(x[b], num_or_size_splits=2, axis=3)
                x_b_0 = keras.layers.Activation(activation_function[4:])(x_b_0)
                x_b_1 = -keras.layers.Activation(activation_function[4:])(-x_b_1)
                x[b] = keras.layers.Concatenate()([x_b_0, x_b_1]) 
            else:
                x[b] = keras.layers.Activation(activation_function)(x[b])

            x[b] = keras.layers.Conv2D(num_input_filters[b][i], conv_kernel_sizes[i], padding=padding)(x[b])
            if normalization_handling == 'batch_norm':
                x[b] = keras.layers.BatchNormalization()(x[b])
            
            if activation_function[:4] == 'bipo':
                x_b_0, x_b_1 = tf.split(x[b], num_or_size_splits=2, axis=3)
                x_b_0 = keras.layers.Activation(activation_function[4:])(x_b_0)
                x_b_1 = -keras.layers.Activation(activation_function[4:])(-x_b_1)
                s[i].append(keras.layers.Concatenate(name="concatenate_activation_i_"+str(i)+"_b_"+str(b))([x_b_0, x_b_1]))
            else:
                s[i].append(keras.layers.Activation(activation_function)(x[b]))

    print('len(s): {}'.format(len(s)))
    for i in range(depth):
        print('len(s[{}]): {}'.format(i, len(s[i])))

    if padding == 'valid':
        for b in range(num_branches):
            for i in range(0,depth-1):
                s[i][b] = s[i][b][:,paddings[i]:-paddings[i],paddings[i]:-paddings[i],:]

    if len(s[depth-1]) == 1:
        m = s[depth-1][0]
    else:
        if skip_connection_handling == 'concatenate':
            m = keras.layers.Concatenate()(s[depth-1]) 
        elif skip_connection_handling == 'add':
            m = keras.layers.Add()(s[depth-1]) 
        elif len(s[depth-1]) == 1:
            m = s[depth-1][0]
        else:   
            m = keras.layers.Concatenate(name="concatenate_s")(s[depth-1])     

    if pooling_size > 1:
        u = keras.layers.Conv2DTranspose(filters=num_input_filters[0][depth-2], kernel_size=2, strides=2, padding='valid')(m)
    else:
        u = m

    if normalization_handling == 'batch_norm':
        u = keras.layers.BatchNormalization()(u)

    if activation_function[:4] == 'bipo':
        u_0, u_1 = tf.split(u, num_or_size_splits=2, axis=3)
        u_0 = keras.layers.Activation(activation_function[4:])(u_0)
        u_1 = -keras.layers.Activation(activation_function[4:])(-u_1)
        u = keras.layers.Concatenate(name="concatenate_activation_u")([u_0, u_1]) 
    else:
        u = keras.layers.Activation(activation_function)(u)


    for i in range(depth-2, -1, -1): 
        
        if skip_connection_handling == 'no_skip_connections':
            tmp_list = [u]
        else:
            tmp_list = s[i].copy()
            tmp_list.append(u)

        if activation_function[:4] == 'bipo' and len(tmp_list) > 1:
            tmp_list_0 = []
            tmp_list_1 = []
            for tmp_list_item in tmp_list:
                tmp_list_item_0, tmp_list_item_1 = tf.split(tmp_list_item, num_or_size_splits=2, axis=3)
                tmp_list_0.append(tmp_list_item_0)
                tmp_list_1.append(tmp_list_item_1)
            tmp_list = tmp_list_0 + tmp_list_1

        if skip_connection_handling == 'concatenate':
            u = keras.layers.Concatenate(name="concatenate_tmp_list_i_"+str(i))(tmp_list) 
        elif skip_connection_handling == 'add':
            u = keras.layers.Add()(tmp_list)  
        elif len(tmp_list) == 1:
            u = tmp_list[0]
        else:
            u = keras.layers.Concatenate(name="concatenate_tmp_list_i_"+str(i))(tmp_list) 

        u = keras.layers.Conv2D(num_input_filters[0][i], conv_kernel_sizes[i], padding=padding)(u)
        if normalization_handling == 'batch_norm':
            u = keras.layers.BatchNormalization()(u)

        if activation_function[:4] == 'bipo':
            u_0, u_1 = tf.split(u, num_or_size_splits=2, axis=3)
            u_0 = keras.layers.Activation(activation_function[4:])(u_0)
            u_1 = -keras.layers.Activation(activation_function[4:])(-u_1)
            u = keras.layers.Concatenate(name="concatenate_activation_u_1_i_"+str(i))([u_0, u_1]) 
        else:
            u = keras.layers.Activation(activation_function)(u)

        u = keras.layers.Conv2D(num_input_filters[0][i], conv_kernel_sizes[i], padding=padding)(u)
        if normalization_handling == 'batch_norm':
            u = keras.layers.BatchNormalization()(u)
        
        if activation_function[:4] == 'bipo':
            u_0, u_1 = tf.split(u, num_or_size_splits=2, axis=3)
            u_0 = keras.layers.Activation(activation_function[4:])(u_0)
            u_1 = -keras.layers.Activation(activation_function[4:])(-u_1)
            u = keras.layers.Concatenate(name="concatenate_activation_u_2_i_"+str(i))([u_0, u_1]) 
        else:
            u = keras.layers.Activation(activation_function)(u)

        if i > 0:
            if pooling_size > 1:
                u = keras.layers.Conv2DTranspose(filters=num_input_filters[0][i], kernel_size=2, strides=2, padding='valid')(u)
            if normalization_handling == 'batch_norm':
                u = keras.layers.BatchNormalization()(u)
        
            if activation_function[:4] == 'bipo':
                u_0, u_1 = tf.split(u, num_or_size_splits=2, axis=3)
                u_0 = keras.layers.Activation(activation_function[4:])(u_0)
                u_1 = -keras.layers.Activation(activation_function[4:])(-u_1)
                u = keras.layers.Concatenate(name="concatenate_activation_u_3_i_"+str(i))([u_0, u_1]) 
            else:
                u = keras.layers.Activation(activation_function)(u)

    # output layer definition
    u = keras.layers.Conv2D(num_output_frames, 1, padding=padding, activation=None, name="last_conv_2d")(u)
    if activation_function_out[:4] == 'bipo':
        u_0, u_1 = tf.split(u, num_or_size_splits=2, axis=3)
        u_0 = keras.layers.Activation(activation_function_out[4:])(u_0)
        u_1 = -keras.layers.Activation(activation_function_out[4:])(-u_1)
        output_u_p_net = keras.layers.Concatenate(name="concatenate_activation_out")([u_0, u_1]) 
    else:
        output_u_p_net = keras.layers.Activation(activation_function_out)(u)

    # (optional) sample normalization (inverse)
    if normalization_handling == 'custom_sample_norm' or normalization_handling == 'custom_sample_norm_zero_symm':

        if normalization_handling == 'custom_sample_norm':

            print ('Using inverse model norm: custom_sample_norm')

            def inv_norm_func(inp_x):
                max_vall = inp_x[0]
                inp_x_norm = tf.transpose(inp_x[1], [3, 1, 2, 0])
                inp_x_norm = tf.math.multiply(tf.math.subtract(inp_x_norm, tf.constant(0.5)), tf.math.multiply(tf.constant(2.0), max_vall))
                return tf.transpose(inp_x_norm, [3, 1, 2, 0])

        elif normalization_handling == 'custom_sample_norm_zero_symm':

            print ('Using inverse model norm: custom_sample_norm_zero_symm')

            def inv_norm_func(inp_x):
                max_vall = inp_x[0]
                inp_x_norm = tf.transpose(inp_x[1], [3, 1, 2, 0])
                inp_x_norm = tf.math.multiply(inp_x_norm, tf.math.multiply(tf.constant(2.0), max_vall))
                return tf.transpose(inp_x_norm, [3, 1, 2, 0])

        output_postproc = keras.layers.Lambda(inv_norm_func)([max_val, output_u_p_net]) 
    else:
        output_postproc = output_u_p_net

    if output_crop > 0 and domain_injection == True:

        output_postproc_padded = tf.keras.layers.ZeroPadding2D(padding=output_crop)(output_postproc)
        
        input_for_output = inputs[input_ids_last_timestep[0]][:,:,:,input_ids_last_timestep[1]]
        input_domain = inputs[domain_injection_ids[0]][:,:,:,domain_injection_ids[1]]
        
        def expand_dims(exp):
            return tf.expand_dims(exp, -1)

        input_for_output = keras.layers.Lambda(expand_dims)(input_for_output)
        input_domain = keras.layers.Lambda(expand_dims)(input_domain)

        if num_output_frames > 1:
            input_list_for_output = [input_for_output] * num_output_frames
            input_list_domain = [input_domain] * num_output_frames
            input_reference_for_output = keras.layers.Concatenate()(input_list_for_output) 
            domain = keras.layers.Concatenate()(input_list_domain) 
        else:
            input_reference_for_output = input_for_output
            domain = input_domain

        output = output_postproc_padded

        def inject_domain(x):
            domain = x[2]
            no_domain = tf.math.subtract(tf.constant(1.0), domain)
            output_postpr = x[0]
            reference = x[1]
            return output_postpr*domain + reference*no_domain

        output = keras.layers.Lambda(inject_domain)([output_postproc_padded, input_reference_for_output, domain])

    else:
        output = output_postproc

    model = keras.Model(inputs=inputs, outputs=[output])

    # just to keep this somewhere:
    # for a custom model / stepping (helpful for debugging) see: https://keras.io/guides/customizing_what_happens_in_fit/

    return [model, output_crop, output_u_p_net, label_layer_for_loss]

