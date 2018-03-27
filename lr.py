#!/usr/bin/python
#
# CIS 472/572 - Logistic Regression Template Code
#
# Author: Daniel Lowd <lowd@cs.uoregon.edu>
# Date:   2/9/2018
#
# Please use this code as the template for your solution.
#
import sys
import re
from math import log
from math import exp
from math import sqrt

MAX_ITERS = 100

# Load data from a file
def read_data(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  data = []
  header = f.readline().strip()
  varnames = p.split(header)
  namehash = {}
  for l in f:
    example = [int(x) for x in p.split(l.strip())]
    x = example[0:-1]
    y = example[-1]
    data.append( (x,y) )
  return (data, varnames)

def sigmoid(row, w, b):
  z = 0
  for i in range (len(w)):
    z += w[i]*row[i]
  z += b
  if z > 750:
    z = 750  
  return 1.0 / (1 + exp(-z))

def get_gradient(gradients, w, x, y, b):
  dot_product = 0.0
  for i in range (0, len(x)):
    dot_product += x[i] * w[i]
  dot_product += b
  dot_product *= y
  #if dot_product > 750: # seems like nothing larger than this works with math.exp
  #  dot_product = 750
  dot_product = 1.0 / (1 + exp(dot_product))
  dot_product
  vec = [0.0]*len(x)
  for i in range (0, len(x)):
    vec[i] = x[i] * dot_product * y
    gradients[i] += vec[i]
  return gradients

def get_gradient_bias(gradient, w, x, y, b):
  dot_product = 0.0
  for i in range (0, len(x)):
    dot_product += x[i] * w[i]
  dot_product += b
  dot_product *= y
  #if dot_product > 0:
  #  dot_product = 750  
  dot_product = 1.0 / (1 + exp(dot_product))
  gradient = dot_product
  return gradient

def update_weights(w, eta, gradients):
  for i in range (0, len(w)):
    w[i] += eta * gradients[i]

def update_gradients(gradients, lam, w):
  for i in range (0, len(gradients)):
    gradients[i] -= lam*w[i]

def update_bias(b, eta, gradient_bias):
  b -= eta * gradient_bias
  return b

def get_gradient_magnitude(gradients):
  sum_vec = 0
  for i in range(0, len(gradients)):
    sum_vec += gradients[i]**2
  return sqrt(sum_vec)

# Train a logistic regression model using batch gradient descent
def train_lr(data, eta, l2_reg_weight):
  numvars = len(data[0][0])
  w = [0.0] * numvars
  b = 0.0
  lam = l2_reg_weight
  min_gradient = 0.0001

  for i in range (0, MAX_ITERS):
    gradients = [0.0] * numvars
    gradient_bias = 0
    for j in range (0, len(data)):
      gradients = get_gradient(gradients, w, data[j][0], data[j][1], b)
      gradient_bias = get_gradient_bias(gradient_bias, w, data[j][0], data[j][1], b)
      magnitude = get_gradient_magnitude(gradients)
      if magnitude < min_gradient:
        return (w,b)

    update_gradients(gradients, lam, w)
    update_weights(w, eta, gradients)
    b = update_bias(b, eta, gradient_bias)

  return (w,b)

# Predict the probability of the positive label (y=+1) given the
# attributes, x.
def predict_lr(model, x):
  (w,b) = model
  for j in range (len(w)):
    z = sigmoid(model[0], x, model[1])
  return z

# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
  if (len(argv) != 5):
    print('Usage: lr.py <train> <test> <eta> <lambda> <model>')
    sys.exit(2)
  (train, varnames) = read_data(argv[0])
  (test, testvarnames) = read_data(argv[1])
  eta = float(argv[2])
  lam = float(argv[3])
  modelfile = argv[4]

  # Train model
  (w,b) = train_lr(train, eta, lam)

  # Write model file
  f = open(modelfile, "w+")
  f.write('%f\n' % b)
  for i in range(len(w)):
    f.write('%s %f\n' % (varnames[i], w[i]))

  # Make predictions, compute accuracy
  correct = 0
  for (x,y) in test:
    prob = predict_lr( (w,b), x )
    #print(prob)
    if (prob - 0.5) * y > 0:
      correct += 1
  acc = float(correct)/len(test)
  print("Accuracy: ",acc)

if __name__ == "__main__":
  main(sys.argv[1:])
