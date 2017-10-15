# Building an Optimal Premium Model in an insurance company #

We are interested in solving a CRM problem for an insurance company. The tasks to be achieved are:

* Finding the ideal target, in this case, people who are more likely to contract the insurance.

* Obtaining the premium we should offer to each client, it means, the optimal price that should be offered to the clients.

* Calculating the difference between offering the premium randomly and optimally using the information obtained in the model.

## Solution ##

We have implemented the solution a machine learning model in Python, using keras with tensorflow as backend and scikit-learn.

The strategy followed is:

1. Fit a random forest model with scikit-learn to study variable importance, and select the most important variables for the final model. The ouput of the model is the probability of converting a call with a given premium into a sale.
2. Fit a neural net with keras to predict the sales probability, using only as inputs the variables that the previous model found the most important.
3. Using the keras model to predict the sales probability, select the optimal subset of customers and calculate the optimal premium.

## Files ##

* *Building an Optimal Premium Model in an insurance company.ipynb*: Jupyter notebook used for calculations and plots.
* *descriptive_analysis.py*: Module with functions used in the notebook for statistical analysis and plots.
* *random_forest.py*: Random forest model
* *neural_network.py*: Neural network model
* *nn_premium_optimization.py*: Calculation of optimal premium with the neural network model.
* *rf_premium_optimization.py*: Calculation of optimal premium with the neural network model (not used in the solution)
* *mixture_premium_optimization.py*: Calculation of optimal premium using a mixture of the two models (not used in the solution)
* Other files with data (both for input and outputs, and documentation)
