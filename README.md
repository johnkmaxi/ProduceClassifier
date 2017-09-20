# ProduceClassifier
Detect various fruit and vegetables in images
This project provides the data and code necessary to create and train a
convolutional neural network for recognizing images of produce. The code is
compatible with python 3.5.3. The following python packages are needed to run
the code:
1. tensorflow 1.1.0
2. matplotlib 2.0.2
3. numpy 1.12.1
4. IPython 6.0.0
5. scipy 0.19.0
6. keras 2.0.6
7. sklearn 0.18.1
8. pandas 0.20.1
9. pillow 4.1.1
10. jupyter 1.0.0

A .yml file is provided to create the virtual environment this project was
created is in included. The .yml file is only guaranteed to work on a Windows
machine. Not all of the packages in the file work on Mac. I recommend using
the Anaconda Python distribution to create the virtual environment.

After setting up the environment, simply cd into the directory holding the data
and Jupyter notebooks. Run `jupyter notebook` from the Anaconda command line,
open a notebook and run the cells to reproduce the necessary data/file structures
and train the different CNNs tested in this product.

If you would like to test your own images, run
'python predict_produce.py path/to/image'
