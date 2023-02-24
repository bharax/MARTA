# MARTA
**M**achine le**AR**ning **T**utori**A**l
- **Instructor:** Margaux Boxho
- **Co-authors:** Noémie Valminck, Lionel Salesses, Caroline Sainvitu, and Thibaut Van Hoof
- **When:** Thursday 09/03/2023 from 9h to 12h and Friday 10/03/2023 from 9h to 12h

## Requirements/Prerequisites
Before attempting the ML tutorial, we encourage you to install `PyTorch` and `jupyter-notebook`.

You have two main ways to perform the installation:
- `Anaconda` base installation
- `Pip` base installation. 

### `Anaconda` base installation
For this installation process, you need to have the `Anaconda` python package manager installed on your laptop. It will be required to create a virtual environment for running the tutorial. You can download `Anaconda` from this [link](https://docs.anaconda.com/anaconda/install/). Note that `Anaconda` comes with over 150 data science packages. Hence, if you do not have the time and disk space (a few minutes and 3GB), you can instead install `Miniconda` from this [link](https://docs.conda.io/en/latest/miniconda.html). Once you have `Anaconda` or `Miniconda` on your computer, you first have to install the virtual environment using the command:
```
conda env create -f MARTA_ENV.yml
```
The file MARTA_ENV.yml is given in the repository source above. For more information about `Anaconda` environments, you can read [this](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment). Once all the packages defined in MARTA_ENV.yml (note that jupyter-notebook is already included) have been installed, you can activate the environment with, 
```
conda activate marta
```
### `Pip` base installation
In this approach, you can skip the installation of `Anaconda` and directly install the jupyter-notebook and PyTorch using pip (or pip3). For the installation of jupyter-notebook, you can have a look at [this](https://jupyter.org/install). For the installation of PyTorch, you can follow the instruction [here](https://pytorch.org/get-started/locally/). Select `Stable (1.13.1)`, your OS, `Pip` package, `Python` language, and `CPU` platform. We will not train your network on GPUs for the moment. 

## Launching the jupyter-notebook
Now that all required packages installed, you can launch the jupyter-notebook as, 
```
jupyter-notebook PyTorch_Tutorial1.ipynb
```
The file PyTorch_Tutorial1.ipynb is given in the repository source above. 

## Outlines
The tutorial is composed of six sections. 
- The **first section** is dedicated to the [`Pytorch`](torch.autograd) data structures. As in the case of `python` which uses `numpy.ndarray`, `PyTorch` has [`torch.tensor`](https://pytorch.org/docs/stable/tensors.html), which is a multi-dimensional matrix containing elements of a single data type. Through this section, we will see how to construct a tensor and how to perform operations on them. 
- The **second section** is focus on [`torch.autograd`](https://www.python-engineer.com/courses/pytorchbeginner/03-autograd/) which is the automatic differentiation package. It is an essential package for our model optimization (see [Autograd - PyTorch Beginner 03](https://www.python-engineer.com/courses/pytorchbeginner/03-autograd/), if needed). 
- The **third section** is devoted to the construction of a neural network through the definition of [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html). What is great with `PyTorch` is that you can construct your network as LEGO blocks and it makes `PyTorch` very flexible.
- A neural network is nothing without its database. Hence, the **fourth section** is concerned with the definition of database (i.e., training, validation, and testing ones). We will see how to construct the [database](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) and how to split it. 
- The **fifth section** will discuss the [optimization](https://pytorch.org/docs/stable/optim.html) of a neural network (i.e., how to fit the parameters of the network based on the training data and the  proper definition of a loss function)
- The **sixth section** ... it's your turn. 

## Objectives
The main objective of this tutorial is not just to be able to construct a neural network and train it. The idea behind this is to have an engineering approach. Assuming that you face a real-world problem in your job. You will first try to identify the parameters of your problem as usual. Then, you may ask yourself if the use of a neural network to model your problem is a good idea. _Do I have enough data? Is my data of good quality? What is the nature of my data (e.g., an image, a time signal, text, Excel table, arrays, graphs, ...) ?_ Once these questions are answered, you can pick the corresponding neural networks (e.g., images may work with CNN, time signals may work with RNN, text may work with transformers, arrays may work with a simple MLP, graphs may work with graph neural network, and so on and so forth). Afterward, you will construct your database and train your selected network. Finally, you will test your network on unseen data and put it in production. In the end, a neural network is nothing else than a model. Roughly speaking, a model is a mathematical tool that approximates a real-world behavior. So do not see the network as a complicated beast, just see it as a black-box model having its pros and cons. 

## References
For a _first_ introduction to deep learning, we recommend you the YouTube channel [3Blue1Brown](https://www.youtube.com/@3blue1brown). The authors have made four videos explaining the concept of deep neural networks and how to train them: 
- [What is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk&ab_channel=3Blue1Brown)
- [Gradient descent and How machines learn?](https://www.youtube.com/watch?v=IHZwWFHWa-w&ab_channel=3Blue1Brown)
- [What is backpropagation really doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U&ab_channel=3Blue1Brown)
- [The backpropagation computation](https://www.youtube.com/watch?v=tIeHLnjs5U8&ab_channel=3Blue1Brown)

For a _deeper_ introduction to a wide variety of neural networks (e.g., convolutional neural networks, transformers, encoder-decoders, graph neural networks, auto-encoders, ...), we recommend you the course [INFO8010 - Deep Learning](https://github.com/glouppe/info8010-deep-learning) given at ULiege. Professor Gilles Louppe has made a very nice GitHub where he links to its lectures and YouTube videos. 

If you want a good book about Deep Learning, we recommend `Dive into Deep Learning` by Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola, available in pdf format [here](https://d2l.ai/d2l-en.pdf).

## Complementary information
This tutorial is "_inspired_" by homework given in the course INFO8010 of ULiege in 2020. So we encourage you to have a look at this [repository](https://github.com/glouppe/info8010-deep-learning) if you want more information. 
