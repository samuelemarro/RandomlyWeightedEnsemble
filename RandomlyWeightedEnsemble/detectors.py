import tensorflow as tf
import numpy as np

class Detector:
    def detect(self):
        pass

class InputPCA(Detector):
    """
    Hendrycks et al., Early Methods for Detecting Adversarial Images, 2017
    """
    def __init__(self, dataset='cifar10'):
        assert dataset in ['mnist', 'cifar10', 'tiny_imagenet']

        #Using the values provided by the authors
        if dataset == 'mnist':
            self.min_component = 700
        elif dataset == 'cifar10':
            self.min_component = 2500
        elif dataset == 'tiny_imagenet':
            self.min_component = 10000
    def detect(self, data):
        def tf_cov(x):
            mean_x = tf.reduce_mean(x, axis=0, keep_dims=True)
            mx = tf.matmul(tf.transpose(mean_x), mean_x)
            vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
            cov_xx = vx - mx
            return cov_xx

        def apply_pca(image):
            #Preprocessing
            mean, _ = tf.nn.moments(image, axes=(0, 1, 2))
            image = image - mean #If the image has been preprocessed this will not have any effects

            #Compute the covariance matrix
            covariance_matrix = tf_cov(tf.reshape(image, [-1]))
            svd = tf.svd(covariance_matrix)

            #Remove the first min_component components
            #svd = tf.slice(svd, [self.min_component, 0, 0], [-1, -1, -1])
            svd = svd[self.min_component:, :, :]

            #Compute the variance (which is our prediction)
            _, variance = tf.nn.moments(svd, [0, 1, 2])
            return variance

        predictions = tf.map_fn(apply_pca, data)
        return predictions

class MaxProbability(Detector):
    """
    Hendrycks et al., A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks, 2016
    """
    def __init__(self, classifier):
        self.classifier = classifier
    def detect(self, data):
        return 1 - tf.reduce_max( data, axis=1)

class RandomlyWeightedEnsemble(Detector):
    def __init__(self, detectors, detector_weights=None):
        self.detectors = detectors

        if weights is None:
            self.detector_weights = [1.] * len(detectors)
        else:
            self.detector_weights = detector_weights

    def detect(self, data):
        total_weighted_predictions = None
        total_weights = None

        for detector, detector_weight in zip(self.detectors, self.detector_weights):
            weights = np.random.uniform(size=[data.shape[0]]) * detector_weight#Quale distribuzione usare?

            predictions = detector.detect(data)

            weighted_predictions = tf.multiply(predictions, weights)

            total_weighted_predictions = weighted_predictions if total_weighted_predictions is None else total_weighted_predictions + weighted_predictions
            total_weights = weights if total_weights is None else total_weights + weight

        return tf.divide(weighted_prediction, total_weight)
