class MomentumSGD:

  import random

  def progress_tracker(self, step: int, cost_function: float) -> None:
    '''
    The function allows you to track progress

    :param step: current step
    :param cost_function: the value of the cost function at the moment

    '''
    from IPython.display import clear_output
    if step==1 or step%5000==0:
        clear_output(wait=True)
        print('Step: {}'.format(step))
        print('Loss: {:.2f}'.format(cost_function))

  def mse_function(self, y_true: list, y_pred: list) -> float:
    '''
    Function that calculates MSE

    :param y_true: the y values we know from the actual data
    :param y_pred: the y values we got at the moment

    :return mse: MSE value
    '''
    # Number of values to compare with actual y
    n = len(y_true)

    pre_mse = 0
    for index, value in enumerate(y_true):
      pre_mse += (value - y_pred[index])**2
    mse = pre_mse/n
    return mse

  def gradient_descent_multi(self, X_true: list, y_true: list, \
                              weights: list = None, max_steps: int = 10000, \
                              learning_rate: float = 0.003, \
                              save_steps: int = 0) -> dict:
    '''
    Gradient descent for multiple variables

    :param X_true: actual attributes
    :param y_true: actual results
    :param weights: starting weights, if we don't want to start training from random
    :param learning_rate: learning rate
    :param max_steps: maximum number of steps at which the algorithm will stop
    :param save_steps: if 0, only last step will be saved
                       If not 0, every #save_steps will be saved
    
    :return {
      :return weights: regression weights
      :return mse: MSE
      :return steps: # of Steps
      :return mse_list: list of MSEs if save_steps > 0
      :return weights_list: list of weigtht lists if save_steps > 0
    }
    '''

    # For data with only one atribute
    if (type(X_true[0])==int) or (type(X_true[0])==float):
      for i, x in enumerate(X_true):
        X_true[i]=[x,1]
    elif (type(X_true[0])==list) and (len(X_true[0])==1):
      for i, x in enumerate(X_true):
        X_true[i].append(1)

    # Initialize weights
    if weights == None:
      weights = [self.random.random() for f in X_true[0]]

    if save_steps > 0:
      mse_list = []
      weights_list = []
    
    # MSE of the previous state
    mse_prev = 0
    mse = 999

    # Nubmer of experiments
    n = len(X_true)

    step = 0
    while (step <= max_steps) and (abs(mse_prev-mse)>1e-5):
      # Calculate gradients
      gradients = []
      for wi, w_value in enumerate(weights):
        current_gradient=0
        for yi, y_t_val in enumerate(y_true):
          current_gradient += -2*(y_t_val - sum([w*x for w,x in \
                                                 zip(weights,X_true[yi])]))* X_true[yi][wi]
        current_gradient = current_gradient/n
        gradients.append(current_gradient)

      # Change weights
      for gi, gr_value in enumerate(gradients):
        weights[gi] = weights[gi] - learning_rate*gr_value

      # Calculate y_pred
      y_pred = []
      for X_current in X_true:
        y_pred.append(sum([w*x for w,x in zip(weights,X_current)]))
      
      step +=1
      mse_prev = mse
      mse = self.mse_function(y_true, y_pred)
      self.progress_tracker(step, mse)

      if save_steps > 0:
        if step % save_steps == 0:
          mse_list.append(mse)
          weights_list.append(weights)

    if save_steps > 0:
      return_dict = {'weights': weights, 'mse':mse, 'steps': step-1, \
                      'mse_list': mse_list, 'weights_list': weights_list}
    else:
      return_dict = {'weights': weights, 'mse':mse, 'steps': step-1}

    return return_dict


new_grad = MomentumSGD()

import random
import numpy as np
import pandas as pd

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]
X_true = []
for i in X:
  x_s_list = [f for f in i]
  X_true.append(x_s_list)
y_true = [f for f in y]
del X, y
learning_rate = 0.0000003
max_steps = 50000
step = 0

# Momentum
gamma = 0.9


weights = [1] * len(X_true[0])

# Number of elements in row of X
n = len(X_true[0])

# Current gradient
gradient = []
# Previous move
v_t_previous = [0] * len(X_true[0])

all_mses = []

while step < max_steps:
  random.seed(step)
  index = random.randint(0, n-1)
  # X & y for current step
  X_current = X_true[index]
  y_current = y_true[index]
  gradient = []
  # Calculate current gradient
  for x_i in X_current:
    current_gradient = -2*(y_current - sum([w*x for w,x in \
                                          zip(weights,X_current)]))*x_i
    gradient.append(current_gradient)
  
  # Apply momentum for previous step
  momentum_v_t_previous = [f*gamma for f in v_t_previous]
  # Applying step for gradient
  step_gradient = [f*learning_rate for f in gradient]
  # New delta to move weights
  v_t = [a+b for a,b in zip(momentum_v_t_previous,step_gradient)]
  v_t_previous = v_t

  # Move weights
  for vti, vti_value in enumerate(v_t):
    weights[vti] = weights[vti] - vti_value

  y_pred = sum([w*x for w,x in zip(weights,X_current)])
  mse = new_grad.mse_function([y_pred], [y_current])

  step += 1

  new_grad.progress_tracker(step, mse)

  # progress
  y_pred_algo_1 = []
  for X_current in X_true:
    y_pred_algo_1.append(sum([w*x for w,x in zip(weights,X_current)]))

  mse_algo_1 = new_grad.mse_function(y_pred_algo_1, y_true)


  all_mses.append(mse_algo_1)

import matplotlib.pyplot as plt
steps = [i+1 for i, f in enumerate(all_mses)]
plt.rcParams['figure.figsize'] = (15.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# Plot the loss function
plt.plot(steps, all_mses)
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

# Plot the head of loss function
plt.plot(steps[:500], all_mses[:500])
plt.title('Head of loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

# Plot the tail of loss function
plt.plot(steps[-2000:], all_mses[-2000:])
plt.title('Tail of loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')
# plt.legend()
plt.show()