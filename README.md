# DDE - Data Driven Deep Density Estimation

This package implements the DDE method of Puchert et al. `cite once available`.
Currently the package contains trained models for 1D, 2D, 3D, 5D, 10D and 30D, where *n*D is the dimensionality of the domain for the distribution whose PDF is to be estimated. 
The DDE Method uses trained neural networks to predict the PDF value for any query point given a sample distribution. 
These methods were trained on synthetic data.

The code contains the method for estimating PDFs given sample distriubtions, for training new models given pairs of sample distributions 
and pdf values and for generating synthetic probability distributions containing x and p(x).

For Examples of the respective methods, we refer to the projects [https://github.com/trikpachu/DDE](https://github.com/trikpachu/DDE).
There you find a script for every use case along a dockerfile with every requirement.

DDE is implemented as python pip pyckage in the PyPi library. To install it just use:

`pip install deep_density_estimation`
## Requirements
The package is tested for the following versions:

Python3.9 <br />
numpy>=1.18.5 <br />
pandas>=1.1.4 <br />
Pillow>=7.0.0 <br />
scikit-learn>=0.23.2 <br />
scipy>=1.5.4 <br />
tensorflow-gpu>=2.5.0 (or tensorflow>=2.5.0) <br />

Note that the gpu support of Tensorflow requires a Nvidia GPU with CUDA and cuDNN. For further details please see the installation requirements of [Tensorflow-GPU](https://www.tensorflow.org/install/gpu) 
Except for Tensorflow, all dependencies are listed in the setup file. As Tensorflow can be installed with or without GPU support and the latter having CUDA dependencies, you have to install it manually.

## Estimating a PDF Given Sample Distributions
This is an implementation of the estimation routines.
Imports NET from trainnet as the network operations and acts as a wrapper for data preprocessing steps.

For an example refer to one of the scripts: 

        examples/testing_synset.py (small dataset provided)

        examples/testing_stocks.py (data must be obtained from [https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/version/3](https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/version/3) and transformed into the PDF samples)

        examples/testing_imagenet.py (data must be obtained and transformed into the PDF samples, could be any set of PNG images)

        examples/testing_deeplesion.py (data must be obtained from [https://www.nih.gov/news-events/news-releases/nih-clinical-center-releases-dataset-32000-ct-images](https://www.nih.gov/news-events/news-releases/nih-clinical-center-releases-dataset-32000-ct-images) and transformed into the PDF samples)
        
```
dde.estimate.estimator(batch_size=512, num_point=128, model='model_4', model_path=None, training_size=None,
                 renormalize=True, dist=None, estimate_at=None, with_nn=False, smoothing=True, verbose=False)
```
Builds the DDE estimator object.

Args:
    batch_size:         int, batch size for the neural net, default=512

    num_point:          int, number of drawn neighbors, i.e. kernel size of DDE, default=128. 

                        This is fix per trained models, all provided models are trained with num_point=128.

    model:              string, neural net model to use, located in Models.py, default = model_4. The trained models provided are all model_4.

    model_path:         string, trained model. Must be set for custom trained models.

    renormalize:        bool, Renormalizing the data?, default=True. Data must be renormalized, only set to false if the data is already renormalized.

    dist:               array_like or string, test data structure [num_funcs, dim, num_points]. 
                        if a string is provided, the data will be loaded from that, expecting a pickle file with python3 encoding in binary (saved with 'wb').
                        dim (and nn if provided) is expected to be constant over all funstions. num_points may vary.
                        While the first dimesnion maybe omitted if only one function is passed, the second dimension must be apparent, even for dim==1.
                        only if bot num_funcs and dim are 1, dim can be omitted.

    with_nn:            bool, set to True if the data already contains neighbours.
                        This is recommended for testing of several similar models to speed up the process.

    training_size:      int, size of the training samples. Must be provided for custom trained models if it is not 5000. 
                        Otherwise it will be inferred from dimensionality (1000 for dim==1, 5000 else.). Default=None.

    smoothing:          bool, if True smooth the 1D estimates using univariate splines. Default=True.

    estimate_at:        array-like, floats. If provided, the estimation based on dist if conducted at these positions. expects shape (num_funcs, dim, size).
                        num_funcs and dim must be same as in dist. Default=None

##### Methods:

**run**
Runs the estimation for the initialized estimator object.

    Returns: a list containing the estimates per function. [num_funcs, size]


## Training a new model:
This is an implementation of the training routines compatible to the new datageneration with function_generation_nd.
The script imports NEt from trainnet as the training operations and acts as a wrapper for data preprocessing steps.
Note that this trains only one model. To mimic the process described in the DDE paper, several models willl need to be trained, 
out of which the best performing (on validation) is selected.

For an example refer to the script: examples/training.py (small dataset provided)
``` 
dde.train.trainer(batch_size=512, num_point=128, model='model_4', max_epoch=1000, model_path=None, 
                 continue_training=False, decay_step=200000, decay_rate=0.7, learning_rate=0.001, renormalize=True, 
                 training_data=None, validation_data=None, dims_last=False, with_nn=False, name_add=None, verbose=False)
```
Builds the trainer object.

    Args:
        batch_size:         int, batch size for the neural net, default=512.

        num_point:          int, number of drawn neighbors, i.e. kernel size of DDE, default=128.

        model:              string, neural net model to use, located in Models.py, default = model_4.

        max_epoch:          int, maximum number of epochs if early stopping did not trigger, default = 1000.

        decay_step:         int, default=200000, decaystep of learning rate, default = 200000.

        decay_rate:         float, decayrate of learning rate, default = 0.7.

        learning_rate:      float, initial learning rate, default = 0.001.

        renormalize:        bool, Renormalizing the data?, default=True. Data must be renormalized, only set to false if the data is already renormalized.

        training_data:      array_like or string, training data. structure either [2, num_funcs, dim+1, num_points] or [num_funcs, dim+1, num_points]. 
                            In the former case it must be a list of training and validation data in that order. if a string is provided, 
                            the data will be loaded from that, expecting a pickle file with python3 encoding in binary (saved with 'wb').
                            If no validatio_data is provided and shape=[num_funcs, dim+1, num_points], 
                            the last quarter of the data will be used as validation data.
                            The last two dimensions may be swapped, in that case, dims_last needs to be provided.

        dims_last:          bool, indicates whether the data order is [num_funcs, dim+1, num_points] or [num_funcs, num_points, dim+1].

        validation_data:    array_like or string, training data. structure [num_funcs, dim+1, num_points].
                            If a string is provided, the data will be loaded from that, expecting a pickle file with python3 encoding in binary (saved with 'wb').
                            If no validatio_data is provided, it will be taken from training data.
                            The last two dimensions may be swapped, in that case, dims_last needs to be provided.
                            Same shape for trainig and validation data is expected.

        with_nn:            bool, set to True if the data already contains neighbours. (same expected for both training and validation data)
                            This is recommended for training of several similar models to speed up the process.

        name_add:           String, addition to the save_path of the trained model. otherwise it is determined by the model, 
                            training dataset dimensions, num_point and the network parameters. default=None

    
##### Methods:

**run**
Runs the training for the initialized trainer object.
The model is directly saved whenever acurracy on the validation set gets better and early stopping is applied after 10 epochs of consecutive non-improvement.


## Generating Data
For the generation of synthetic data the package contains 2 types of classes. The first two are the two methods described in the paper 
for generation of purely synthetic data and the other 3 are the ones to transorm 1D, 2D (images) and 3D (volumes) data to PDFs and draw samples out of them.

### Purely synthetic:
Builds a set of Probability Density Functions (PDF's) out of a set of base_functions. 
The implemented set of base_functions is chosen to ensure a as random as possible shape of the PDF, 
while also ensuring a relatively uniform distribution regarding the monotony (if falling or rising). 
The base functions are selected in a tree manner and are connected via a randomly chosen operator at each step. 
The set of operators is defined in 'tokens' and contains only '+' and '*' to ensure positive definiteness and finiteness. 
After initialization we select a random number on the scale of the respective PDF 
for each proposed sample point and only select those samples higher than this rn. 
This ensures the probability distribution. 
Afterwards we integrate the function and divide by the integral to get a normalized PDF.

#### Version 1

Fast version of the function_generation_nd class, which contains only addition operator for dimension coupling.
This function firtst builds DIM 1 dimensional functions, whose resulting functions are then coupled additvely to get the DIM dimensional function.
The maximum of the DIM dimensional function is then calculated in "function_generator", which defines the functions.
The generated functions are saved as pickled lists of size [num_funcs, dim+1, size].

For an example refer to the script: examples/Function_generation.py

```
PDF_Generation.function_generation(size=1000, num_funcs=1000, complexity=3, scale=1.0, dim=3, max_initials=None, verbose=True, naming='', 
                                    pdf_save_dir=None, sampled_data_save_dir=None, select_functions=['all_sub'], deselect_functions=[])
```
Initializes the generation-object. This includes sampling of the function space for the definition of the synthetic functions along with
the calculations of integrals and maxima, necessary for normalization and sampling.

Args:   
        size:                       INT, number of samples drawn per function.

        num_funcs:                  INT, number of functions to generate.

        complexity:                 INT or tuple/list of two INTs, number of base_functions composed to the resulting 1D function compositions. 
                                    If a tuple/list is provided, the complexity will vary for every function in the given range.

        scale:                      FLOAT or Tuple/List of two FLOATs, defines the range of the function support. Constant over all functions. 
                                    Support will be [0, scale]. If tuple the scale will be randomly chosen in the provided range.

        dim:                        INT, dimensionality of the generated functions support.

        max_initials:               INT, number of initial guesses for the maximum optimization to overcome discontinuities and local maxima.

        verbose:                    BOOL, adds some verbosity.

        naming:                     STRING, String appended to output filename to not overwrite other files with same selected functions, size, dim and num_funcs.

        pdf_save_dir:               String, directory of the saved PDFs. Contains lists to recreate the generated PDFs. 

        sampled_data_save_dir:      String, directory of the saved samples. Contains lists with size (num_funcs, dim+1, size).

        select_functions:           List of strings, group of functions with same characteristica to include in generation. 
                                    Must be of the following: 'all', 'all_sub', 'gaussian', 'sinusoidal', 'linear', 'smooth', 'monotone', 'non_monotone'.

        deselect_functions:         List of strings, group of functions with same characteristica to exclude in generation. 
                                    Must be of the following: 'all', 'all_sub', 'gaussian', 'sinusoidal', 'linear', 'smooth', 'monotone', 'non_monotone'
    
##### Methods:
**function_generation**
Generates the functions for the initialized generationobject and saves them in a pickled list.

#### Version 2:
Slow version of the function_generation_nd class, which couples 1d functions with '+' or '*'. As this means the dimensions can not 
be easily decoupled during integration and maxima calculation, it takes much longer for higher DIM.
The function_generation class grows linearly in time with DIM, while this class grows exponentially with DIM. 
In contrast to before, this function firtst builds DIM 1 dimensional tree, whose resulting functions are then coupled additvely to get the DIM dimensional function.
The maximum of the DIM dimensional function is then calculated in "function_generator", which defines the functions.

For an example refer to the script: examples/Function_generation.py

```
PDF_Generation.function_generation_more_time(size=1000, num_funcs=1000, complexity=3, scale=1.0, dim=3, max_initials=None, verbose=True, naming='', 
                                             pdf_save_dir=None, sampled_data_save_dir=None, select_functions=['all_sub'], deselect_functions=[])
```
Initializes the generation-object. This includes sampling of the function space for the definition of the synthetic functions along with
the calculations of integrals and maxima, necessary for normalization and sampling.

Args:   
        size:                       INT, number of samples drawn per function.

        num_funcs:                  INT, number of functions to generate.

        complexity:                 INT or tuple/list of two INTs, number of base_functions composed to the resulting 1D function compositions. 
                                    If a tuple/list is provided, the complexity will vary for every function in the given range.

        scale:                      FLOAT or Tuple/List of two FLOATs, defines the range of the function support. Constant over all functions. 
                                    Support will be [0, scale]. If tuple the scale will be randomly chosen in the provided range.

        dim:                        INT, dimensionality of the generated functions support.

        max_initials:               INT, number of initial guesses for the maximum optimization to overcome discontinuities and local maxima.

        verbose:                    BOOL, adds some verbosity.

        naming:                     STRING, String appended to output filename to not overwrite other files with same selected functions, size, dim and num_funcs.

        pdf_save_dir:               String, directory of the saved PDFs. Contains lists to recreate the generated PDFs. 

        sampled_data_save_dir:      String, directory of the saved samples. Contains lists with size (num_funcs, dim+1, size).

        select_functions:           List of strings, group of functions with same characteristica to include in generation. Must be of the following: 
                                    'all', 'all_sub', 'gaussian', 'sinusoidal', 'linear', 'smooth', 'monotone', 'non_monotone'.

        deselect_functions:         List of strings, group of functions with same characteristica to exclude in generation. Must be of the following: 
                                    'all', 'all_sub', 'gaussian', 'sinusoidal', 'linear', 'smooth', 'monotone', 'non_monotone'.

##### Methods:
**function_generation**
Generates the functions for the initialized generationobject and saves them in a pickled list.

### PDF eneration from real data:

#### 1D
Generates a probability density out of arbitrary 1-dimensional data and draws a sample distribution with size SIZE out of it.
The probability values at arbitrary points are interpolated from the surrounding values by linear interpolation.
Normalizes the domain to range [0,1]
Dumps a list per sample in a directory, containing samples or [samples, grid_samples], where each is of size [n_dim, n_samples], n_dim = 2

For an example refer to the script: examples/transformPDF_Stock_data.py

```
PDF_Generation.Prob_dist_from_1D(size, data=None, with_grid=False, grid_number=10000, verbose=False, readtxt=False, exclude_short=False, data_dir=None, savedir='new_data/real_1d')
```
Initializes the generation object.

Args:
    size: INT, size of the drawn sample distribution.

    data: array-like, list or array of data points (float or int), structures as [n_functions, n_points, dim], where dim = 2 for x and f(x). Required if readtxt==False.

    with_grid: BOOL, if True a grid with the corresponding true probability values is also generated (for nicer plots).

    grid_number: INT, number of samples in the grid.

    readtxt: BOOL, if True the dat is read from txt files in data_dir. Then data is expected to be tabulated in txt/csv files with 1 sample point per row structured x,y (float or int).  
    
    data_dir: STRING, Directory with the data, only required if readtxt == True. data_dir is directly searched with glob.glob, thus '/*' is appended to the directory, if not already there. 
                        If subdirectories have to be searched, give path as 'path/to/data/*/*', where the last asterisk is the directory of readable datafiles.

    exclude_short: BOOL, excludes files shorter than 100 entries.

    savedir: STRING, directory for the generated pdfs. Will be appended by /"size" and _with_grid if with_grid==True

##### Methods:

**get_pdf**
Turns the provided data into PDFs and samples distributions with *size* points.

#### 2D
Generates a probability density out of arbitrary 2-dimensional data and draws a sample distribution with size SIZE out of it.
The probability values at arbitrary points are interpolated from the surrounding values by linear interpolation.
Normalizes the domain to range [0,1]^2
Dumps a list per sample in a directory, containing samples or [samples, grid_samples], where each is of size [n_dim, n_samples], n_dim = 3
Currently only supports structured data (images), which can be passed as list or as directory containing the images.

For an example refer to the script: examples/transformPDF_Imagenet.py

```
PDF_Generation.Prob_dist_from_2D(size, data=None, with_grid=False, grid_number=500, verbose=False, data_dir=None, readimg=False, savedir='new_data/real_2d', imagenorm=255)
```
Initializes the generation object.

Args:
    size: INT, size of the drawn sample distribution.

    data: array-like, list or array of data points (float or int), structures as [n_functions, height, width]. Required if readimg==False.

    with_grid: BOOL, if True a grid with the corresponding true probability values is also generated (for nicer plots).

    grid_number: INT, number of samples in the grid.

    imagenorm: INT, normalization constant for images. Has to be adapted if image files or data aer not 8bit (maxvalue of 255).

    readimg: BOOL, if True the dat is read from txt files in data_dir. Then data is expected to be imagefiles.  

    data_dir: STRING, Directory with the data, only required if readtxt == True. data_dir is directly searched with glob.glob, thus '/*' is appended to the directory, if not already there. 
                        If subdirectories have to be searched, give path as 'path/to/data/*/*', where the last asterisk is the directory of readable datafiles.
                        
    savedir: STRING, directory for the generated pdfs. Will be appended by /"size" and _with_grid if with_grid==True

##### Methods:

**get_pdf**
Turns the provided data into PDFs and samples distributions with *size* points.

#### 3D
Generates a probability density out of arbitrary 3-dimensional data and draws a sample distribution with size SIZE out of it.
The probability values at arbitrary points are interpolated from the surrounding values by linear interpolation.
Normalizes the domain to range [0,1]^3
Dumps a list per sample in a directory, containing samples or [samples, grid_samples], where each is of size [n_dim, n_samples], n_dim = 3
Currently only supports structured data (list of images), which can be passed as list or as directory containing one folder for each volume
which contains images for each layer of the volume.

For an example refer to the script: examples/transformPDF_Deeplesion.py
    
```
PDF_Generation.Prob_dist_from_3D(size, data=None, with_grid=False, grid_number=100, verbose=False, data_dir=None, readimg=False, savedir='new_data/real_3d')
```
Initializes the generation object.

Args:
    size: INT, size of the drawn sample distribution.

    data: array-like, list or array of data points (float or int), structures as [n_functions, n_layers, height, width]. Required if readimg==False.

    with_grid: BOOL, if True a grid with the corresponding true probability values is also generated (for nicer plots).

    grid_number: INT, number of samples in the grid.

    readimg: BOOL, if True the data is read from txt files in data_dir. Then data is expected to be imagefiles. where each volume is contained in a subdirectory with an image per layer.  

    data_dir: STRING, Directory with the data, only required if readtxt == True. data_dir is directly searched with glob.glob, thus '/*' is appended to the directory, if not already there. 
                        If subdirectories have to be searched, give path as 'path/to/data/*/*', where the last asterisk is the directory of readable datafiles.

    savedir: STRING, directory for the generated pdfs. will be appended by /"size" and _with_grid if with_grid==True


##### Methods:

**get_pdf**
Turns the provided data into PDFs and samples distributions with *size* points.
