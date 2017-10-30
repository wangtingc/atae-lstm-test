import numpy as np
import logging

class ADAGRAD(object):
    def __init__(self, params, lr, lr_word_vector=0.1, epsilon=1e-10):
        logging.info('Optimizer ADAGRAD lr %f' % (lr, ))
        self.lr = lr
        self.lr_word_vector = lr_word_vector
        self.epsilon = epsilon
        self.acc_grad = {}
        for param in params:
            self.acc_grad[param] = np.zeros_like(param.get_value())

    def iterate(self, grads):
        lr = self.lr
        epsilon = self.epsilon
        for param, grad in grads.iteritems():
            if param.name == 'Vw':
                param.set_value(param.get_value() - grad.get_value() * self.lr_word_vector)
            else:
                self.acc_grad[param] = self.acc_grad[param] + grad.get_value()**2
                param_update = lr * grad.get_value() / (np.sqrt(self.acc_grad[param]) + epsilon)
                param.set_value(param.get_value() - param_update)

OptimizerList = {'ADAGRAD': ADAGRAD}
