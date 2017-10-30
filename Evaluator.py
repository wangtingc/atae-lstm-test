from sklearn.metrics import confusion_matrix
import numpy as np
import logging

class Evaluator2(object):
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.clear()

    def keys(self):
        return ['binary']

    def clear(self):
        self.cm2 = np.zeros((2, 2), dtype=int)

    def accumulate(self, solution, pred):
        def label_2to2(probs):
            assert len(probs.shape) == 2
            assert probs.shape[1] == 2
            preds = np.argmax(probs, axis=1)
            return preds

        def get_cm2(solution, pred):
            solution = label_2to2(solution)
            pred = label_2to2(pred)
            self.cm2 += confusion_matrix(solution, pred, [0, 1])
            return solution == pred

        return {'binary': int(get_cm2(solution, pred))}

    def evaluate(self, solution, pred):
        clear()
        accumulate(solution, pred)

    def statistic(self):
        cm2 = self.cm2
        binary_total = float(np.sum(cm2))
        ret = {}
        ret['binary'] = (cm2[0, 0] + cm2[1, 1]) / binary_total
        if self.verbose:
            logging.info('Cm2:\n%s' % self.cm2)
        return ret

class Evaluator3(object):
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.clear()

    def clear(self):
        self.cm3 = np.zeros((3, 3), dtype=int)
        self.cm2 = np.zeros((3, 3), dtype=int)

    def keys(self):
        return ['binary', 'three-way']

    def accumulate(self, solution, pred):
        def label_3to3(probs):
            assert len(probs.shape) == 2
            assert probs.shape[1] == 3
            preds = np.argmax(probs, axis=1)
            return preds

        def label_3to2(probs):
            assert len(probs.shape) == 2
            assert probs.shape[1] == 3
            probs_without2 = probs - np.array([0.0, 0.99999, 0.0])
            preds = np.argmax(probs_without2, axis=1)
            return preds

        def get_cm3(solution, pred):
            solution = label_3to3(solution)
            pred = label_3to3(pred)
            self.cm3 += confusion_matrix(solution, pred, [0, 1, 2])
            return solution == pred

        def get_cm2(solution, pred):
            solution = label_3to2(solution)
            pred = label_3to2(pred)
            self.cm2 += confusion_matrix(solution, pred, [0, 1, 2])
            return solution == pred

        return {'binary': int(get_cm2(solution, pred)), \
                'three-way': int(get_cm3(solution, pred))}

    def evaluate(self, solution, pred):
        clear()
        accumulate(solution, pred)

    def statistic(self):
        cm3, cm2 = self.cm3, self.cm2
        three_grained_total = float(np.sum(cm3))
        binary_total = float(np.sum(cm2) - np.sum(cm2[1, 0:3]))
        ret = {}
        ret['three-way'] = np.sum([cm3[i][i] for i in xrange(3)]) / three_grained_total
        ret['binary'] = (np.sum(cm2[0, 0]) + np.sum(cm2[2, 2])) / binary_total
        if self.verbose:
            logging.info('Cm3:\n%s' % self.cm3)
            logging.info('Cm2:\n%s' % self.cm2)
        return ret

Evaluators = {2: Evaluator2, 3: Evaluator3}