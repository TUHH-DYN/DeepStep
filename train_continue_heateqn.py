import sys, os
import json

from modules.training import process_training

##
## Continue the training of an u^p net configuration for another 50 epochs
## with a learningrate decreasing from 1E-5 to 1E-7
##

def main():
    
    print("Total arguments passed:", len(sys.argv))

    filter_size = [int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])]

    num_input_filters = {
        'u_p_net': [[filter_size[0],filter_size[1],filter_size[2]],[filter_size[0],filter_size[1],filter_size[2]],[filter_size[0],filter_size[1],filter_size[2]]],
        'u_net': [[filter_size[0],filter_size[1],filter_size[2]]],
        'u_net_noskip': [[filter_size[0],filter_size[1],filter_size[2]]],
    }

    concatenate_inputs = {
        'u_p_net': False,
        'u_net': True,
        'u_net_noskip': True,
    }

    name = str(sys.argv[1])
    if name in num_input_filters:
        num_input_filters = num_input_filters[name]
    else:
        num_input_filters = num_input_filters['u_p_net']
    
    if name in concatenate_inputs:
        concatenate_inputs = concatenate_inputs[name]
    else:
        concatenate_inputs = concatenate_inputs['u_p_net']

    if filter_size[0] == filter_size[1]/2 and filter_size[1] == filter_size[2]/2:
        name += '_x2'
    elif filter_size[0] == filter_size[1] and filter_size[1] == filter_size[2]:
        name += '_'+str(filter_size[0])
    else:
        name += '_'+str(filter_size[0])+'_'+str(filter_size[1])+'_'+str(filter_size[2])

    if len(sys.argv) >= 6:
        gpu = int(sys.argv[5])
    else:       
        gpu = 0

    print(f'name: {name}')
    print(f'num_input_filters: {num_input_filters}')
    print(f'concatenate_inputs: {concatenate_inputs}')
    print(f'gpu: {gpu}')

    # exit()

    path_name = os.path.dirname(sys.argv[0])   
    abs_path = os.path.abspath(path_name) 

    # load config

    model_folder = 'data/model_heatequation'

    model_load_folder = 'model_'+name+'_after_300epochs'
    # model_load_folder = 'model_'+name+'_after_1epochs'

    model_output_folder = 'model_'+name+'_after_350epochs'
    # model_output_folder = 'model_'+name+'_after_2epochs'

    model_filename_base = 'model_'+name

    with open(abs_path+'/'+model_folder + '/' + model_load_folder + '/' + model_filename_base + '_config.json') as json_file:
        config = json.load(json_file)

    config['model']['weights_from_file'] = True
    config['model']['history_from_file'] = True
    config['training']['epochs'] = 50
    # config['training']['epochs'] = 1
    config['training']['verbose'] = 2   # 2 = just epoch evaluations | 1 = progress bar for each epoch
    config['training']['gpu'] = gpu
    config['training']['backup'] = True
    config['training']['output_folder'] = model_folder
    config['training']['learningrate'] = 1E-5

    config['training']['learningrate_scheduler'] = 'exp_decay'
    config['training']['learningrate_scheduler_verbose'] = True
    config['training']['learningrate_k'] = 0.092103403719762

    config['training']['output_folder'] = model_folder + '/' + model_output_folder
    config['training']['load_folder'] = model_folder + '/' + model_load_folder

    [min_loss, min_val_loss, min_loss_epoch, min_val_loss_epoch] = process_training(abs_path, config)

    result = {}
    result['min_loss'] = min_loss
    result['min_val_loss'] = min_val_loss
    result['min_loss_epoch'] = int(min_loss_epoch)
    result['min_val_loss_epoch'] = int(min_val_loss_epoch)
    result['filename_base'] = config['training']['filename_base']
    result['iteration'] = 0
    result['padding'] = config['model']['padding'] 
    result['pooling_size'] = config['model']['pooling_size'] 
    result['depth'] = config['model']['depth'] 
    result['num_input_filters'] = config['model']['num_input_filters'][0]
    result['conv_kernel_sizes'] = config['model']['conv_kernel_sizes']
    result['loss_function'] = config['training']['loss']
    result['learningrate'] = config['training']['learningrate']

    json.dump(result, open(abs_path+'/'+config['training']['output_folder']+'/'+config['training']['filename_base']+'_result.json', 'w'))

if __name__ == "__main__":
    main()