# DeepStep: PDE Time Stepping through Deep Learning

## Software Requirements

Install Anaconda and prepare two conda environments (see below). You might be able to use one environment for all packages, but in our experience the handling of the FEniCS and Tensorflow packages is easier if they are used in seperate environments.

### Tensorflow environment (for deep learning related tasks)

    conda create --name tf
    conda activate tf
    conda install -c conda-forge tensorflow-gpu=2.2.0 notebook pydot ipywidgets scikit-learn plotly opencv python-kaleido pandas bokeh tabulate matplotlib tikzplotlib


### FEniCS environment (for data generation)

    conda create --name fenics
    conda activate fenics
    conda install -c conda-forge fenics-dolfin mshr scipy ipywidgets matplotlib tikzplotlib

## Hardware Requirements

Modern NVIDIA GPU (we used a Titan XP with 24 GB VRAM). If you use a GPU with a smaller VRAM you might have to adjust the batch size settings.

## Generate training and validation data

You can generate all training and validation datasets by running:

    ./generate_waveeqn_data.sh

It is advisable to run the script in the background as the data generation might take a couple of hours to complete:

    nohup ./generate_waveeqn_data.sh > nohup_generate_waveeqn_data.txt&

The generated datasets can be found in the data/datasets_train_waveequation and data/datasets_test_waveequation directories (and respectively the _heatequation counterparts). The following naming convention is used:

  - for validation/test datasets - [entity_name]\_dataset_validation\_[identifier].npy
  - for training datasets - [entity_name]\_dataset\_[identifier].npy
  - data files of the following entity names are generated/saved:
    - 'bcs'  - boundary values in grid coordinates
    - 'config' - *.json file containing the configuration-dictionary of the respective dataset
    - 'domain' - binary domain mask in grid coordinates
    - 'field' - field evolution in grid coordinates
    - 'kappa' - domain parameterization related to mesh coordinates
    - 'material' - domain parameterization in grid coordinates
    - 'mesh' - coordinates of the underlying FEM mesh
    - 'steps' - field evolution related to mesh coordinates
    - 'surface' - field evolution in mesh coordinates
    - for 'bcs' and 'domain' also a '_surf' variant of the respective entity is generated. Contrary to the original entities, the grid coordinates right at the domain surface are not ascribed to the domain. 

You can visualize the generated datasets using the provided jupyter notebook 'notebook_visualize_train_and_test_data.ipynb'. Exchange 'waveeqn' by 'heateqn' in the commands above to generate the training and validation data for the heatequation case.

## Train the networks

You can run all training tasks by executing:

    ./train_waveeqn.sh

or (to run the script in the background):

    nohup ./train_waveeqn.sh > nohup_train_waveeqn.txt&
    nohup ./train_heateqn.sh > nohup_train_heateqn.txt&

Again, exchange 'waveeqn' by 'heateqn' in the commands above to train the networks for the heatequation case.

**The due to the size of the networks and the amount of data, the training will take several days or even weeks to complete (depending on the used hardware).**

This script will run the training for six predefined network architectures: 

- U^p-Net with constant filter dimensions
- U^p-Net with doubling filter dimensions*
- U-Net with constant filter dimensions
- U-Net with doubling filter dimensions*
- U-Net (without the skip connections) with constant filter dimensions
- U-Net (without the skip connections) with doubling filter dimensions*

*by default only included in the 'train_waveeqn.sh' skript.

The training process for each architecture is divided in the first 300 epochs with an exponential learningrate decay from 1e-4 to 1e-5 and another 50 epochs with an exponential learningrate decay from 1e-5 to 1e-7. The 

The trained models are saved in the 'data/model_waveequation' subdirectories (and respectively the '_heatequation' counterparts). You can visualize the training (loss) evolutions using the provided jupyter notebook.


## Make a prediction using a trained network

To run a simple prediction run:

    conda activate tf
    python predict_waveeqn.py  

or respectively the '_heateqn' counterpart.

Customize the prediction settings by using the config dictionaries in predict_waveeqn.py:
- validation_case = the validation-/test-case to predict. All validation case datasets contain data for 800 timesteps.
- model_config = Trained model used for predictions
- E.g. using config['prediciton']['timestep_start'] = 10 and config['prediciton']['timesteps'] = 5 the prediction will start using timesteps 10,11 and 12 to predict timestep 13 and move on until predicting timestep 17 using the field data at timesteps 14,15,16 as input.
- If config['prediciton']['iterative_feedback'] is True, the predicted field data is used as field input of the next prediction. Otherwise all predictions are performed using the ground truth as input data.

The predicted fields are saved in the data/prediction_waveequation directory (and respectively the _heatequation counterpart). You can visualize the predicted data using the provided jupyter notebook.



