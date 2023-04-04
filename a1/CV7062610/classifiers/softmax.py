from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # L(W) = (1/N)*sum 1 to N(Li(f(xi, W),yi) + gamaR(W))
    # L2 reg : R(W) = sum k, sum l, of  Wkl^2
    # P(Y = k | X = xi) = e^sk/ sum j(e^sj) - softmax function

    
    N = X.shape[0] # num of test images
    C = W.shape[1] # num of Classes C

    #for test image i in X
    for i in range(N):
      #make sure >= 0 as probabilitys must be >= 0
      exp_s = np.exp(X[i].dot(W))
      #normalize to value in range [0,1] as probabilitys must sum to 1
      probs = exp_s / np.sum(exp_s)
      #calculate loss
      #Li = -log(P(Y = yi | X = xi))
      loss += -np.log(probs[y[i]])

      #gradiant of W aka dW
      #for class in classes
      for j in range(C):
            #if actual class = class
            if(y[i] == j):
              # add to j col in dW (our nets prediction in probabilitys - 1) * X feature values of i-th input 
              dW[:,j] += (probs[j] - 1)*X[i,:]
            else:
              dW[:,j] += (probs[j])*X[i,:]
    #avaraging across N test image samples
    loss*=(1/N)
    dW*=(1/N)

    # adding Reg
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    '''
      Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    '''
    #scores - make sure >= 0 as probabilitys must be >= 0
    exp_s = np.exp(np.dot(X, W))
    #normalize to value in range [0,1] as probabilitys must sum to 1
    probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)

    # number of sample tests
    N = X.shape[0]
    #using fancy indexing geek.com
    logprobs = -np.log(probs[range(N),y])
    #avg loss
    avg_loss = np.sum(logprobs)/N
    # L2 reg : R(W) = sum k, sum l, of  Wkl^2
    reg_loss = 0.5*reg*np.sum(W*W)
    loss = avg_loss + reg_loss

    # Compute the gradient
    dscores = probs
    #fancy indexing geek.com
    dscores[range(N),y] -= 1 # where class predicted is actual class
    #avraging
    dscores /= N
    #getting dW via matrix multiplication
    dW = np.dot(X.T, dscores)
    dW += reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
