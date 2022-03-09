# COMP551---Miniproject-3-Classification-of-Image-Data-
## Background
In this miniproject, you will implement a multilayer perceptron from scratch, and use it to classify image data.
The goal is to implement a basic neural network and its training algorithm from scratch and get hands-on experience
with important decisions that you have to make while training these models. You will also have a chance to experiment
with convolutional neural networks.

## Task 1: Acquire the data
Your first task is to acquire the image dataset. You will be using only one dataset in your experiments: Fashion-
MNIST. Use the Fashion-MNIST dataset with the default train and test partitions. You can use existing machine learning libraries to load the dataset. Note that while working with multilayer perceptrons, after loading the data, you
will have to vectorize it so that it can have the appropriate dimensions. Also do not forget to normalize the training
and test set (see https://cs231n.github.io/neural-networks-2/#datapre).
Based on your previous miniprojects, you might be asking the question: where are the features? Well, this is the whole
point of using neural nets: instead of hand-designing the features, you train the model so that the feature extractor is
also learned together with the classifier on top.

## Task 2: Implement a Multilayer Perceptron
In this mini-project, you will implement a multilayer perceptron (MLP) to classify image data. An MLP is composed
of three types of layers: (1) an input layer, (2) hidden layers, (3) an output layer (see Figure 1). You should implement
it from scratch based on the code available in the slides. Your implementation should include the backpropagation and
the mini-batch gradient descent algorithm used (e.g., SGD).
You are free to implement the MLP as you see fit, but you should follow the equations that are presented in the
lecture slides, and you must implement it from scratch (i.e., you cannot use TensorFlow or PyTorch or any other
library). Using the Numpy package is encouraged. Regarding the implementation, we recommend the following
approach:
• Implement the MLP as a python class. The constructor for the class should take as input the activation function
(e.g., ReLU), the number of hidden layers (e.g., 2) and the number of units in the hidden layers (e.g., [64, 64])
and it should initialize the weights and biases (with an initializer of your choice) as well as other important
properties of the MLP.
• The class should have (at least) two functions:
– A fit function, which takes the training data (i.e., X and y)—as well as other hyperparameters (e.g., the
learning rate and number of gradient descent iterations)—as input. This function should train your model
by modifying the model parameters.
– A predict function, which takes a set of input points (i.e., X) as input and outputs predictions (i.e., ˆy)
for these points.
• In addition to the model classes, you should also define a functions evaluate acc to evaluate the model
accuracy. This function should take the true labels (i.e., y), and target labels (i.e., ˆy) as input, and it should
output the accuracy score.
You are also free to use any Python libraries you like to tune the hyper-parameters; see for example https://
scikit-learn.org/stable/modules/grid_search.html.

## Task 3: Run the experiments and report
The goal of the experiments in this part is to have you explore the consequences of important decisions made while
training neural networks. Split the dataset into training and test sets. Use test set to estimate performance in all of the
experiments after training the model with training set. Evaluate the performance using accuracy. You are welcome to
perform any experiments and analyses you see fit (e.g., the effect of data augmentation / L1-L2 regularization / number
of hidden layers / . . . on accuracy), but at a minimum you must complete the following experiments in the order
stated below:
1. First of all, create three different models: (1) an MLP with no hidden layers, i.e., it directly maps the inputs
to outputs, (2) an MLP with a single hidden layer having 128 units and ReLU activations, (3) an MLP with 2
hidden layers each having 128 units with ReLU activations. It should be noted that since we want to perform
classification, all of these models should have a softmax layer at the end. After training, compare the test
accuracy of these three models on the Fashion-MNIST dataset. Comment on how non-linearity and network
depth affects the accuracy. Are the results that you obtain expected?
2. Take the last model above, the one with 2 hidden layers, and create two different copies of it in which the
activations are now tanh and Leaky-ReLU. After training these two models compare their test accuracies with
model having ReLU activations. Comment on the performances of these models: which one is better and why?
Are certain activations better than others? If the results are not as you expected, what could be the reason?
3. Create an MLP with 2 hidden layers each having 128 units with ReLU activations as above. However, this time,
add dropout regularization to the network and train the MLP in this way. How does this affect the accuracy? This
proportion can be varied as a tunable hyperparameter that can be explored as part of other project requirements.
4. Create an MLP with 2 hidden layers each having 128 units with ReLU activations as above. However, this time,
train it with unnormalized images. How does this affect the accuracy?
5. Using existing libraries such as TensorFlow or PyTorch, create a convolutional neural network (CNN) with
2 convolutional and 2 fully connected layers. Although you are free in your choice of the hyperparameters
of the convolutional layers, set the number of units in the fully connected layers to be 128. Also, set the
activations in all of the layers to be ReLU. Train this CNN on the Fashion-MNIST dataset. Does using a CNN
increase/decrease the accuracy compared to using MLPs? Provide comments on your results.
6. Using your MLP implementation, try to come up with an MLP architecture that performs as good as possible.
How does this MLP perform compared to the CNN in part 5? Justify your choice of network architecture
parameters through careful experiments.
7. You can report your findings either in the form of a table or a plot in the write-up. However, include in your
colab notebooks the plots of the test and train performance of the MLPs/CNN as a function of training epochs.
This will allow you to see how much the network should be trained before it starts to overfit to the training data.
Note 1: The above experiments are the minimum requirements that you must complete; however, this project
is open-ended.
For example, you might investigate the effect of the width (number of units in the hidden layers) of the MLP on its test
accuracy or the effect of the CNN’s convolutinal layer hyperparameters (number of filters, kernel size, stride, padding,
. . . ) on its test accuracy. It is also possible to examine the effect of different dropout node proportions on the final
performance. Another interesting thing to report might be training the MLP and CNN with 10k, k ∈ {0, 1, 2, 3, 4}
images and plotting the test accuracy. You do not need to do all of these things or tune every parameter, but you should
demonstrate creativity, rigour, and an understanding of the course material in how you run your chosen experiments
and how you report on them in your write-up.
Note 2: We expect you to provide plots/tables in your report that justifies your choice of hyperparameters (the
learning rates of the MLPs/CNNs in parts 1-6, the architectural parameters of the MLPs/CNNs in parts 5 & 6).
You are not required to perform cross-validation in this project.
