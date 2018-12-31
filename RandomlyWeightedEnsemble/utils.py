import numpy as np
import cleverhans
import tensorflow as tf
import keras
import keras.backend as K
import sklearn.metrics as metrics
import detectors
from tensorflow.python.framework import graph_util
from tensorflow.tools.graph_transforms import TransformGraph

def filter_misclassified(model: keras.Model, x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Removes samples that the model cannot classify correctly
    """
    y = tf.convert_to_tensor(y)
    y = tf.argmax(y, 1)

    y_ = model.predict(x)
    y_ = tf.convert_to_tensor(y_)
    y_ = tf.argmax(y_, 1)


    mask = tf.equal(y, y_)
        
    filtered_x = tf.boolean_mask(x, mask, axis=0)
    filtered_y = tf.boolean_mask(y, mask, axis=0)

    return filtered_x, filtered_y

def test_detector(sess, input_placeholder, label_placeholder, iterator, model: keras.Model, attack: cleverhans.attacks.Attack, detector: detectors.Detector, remove_misclassified=True):
    """
    Computes the ROC (Receiver Operating Characteristic) Curve of the detector
    """
    genuine_confidences = None
    adversarial_confidences = None

    for x_data, y_data in iterator:
        
        x = input_placeholder
        y = label_placeholder

        if remove_misclassified:
            x, _ = filter_misclassified(model, x, y)

        #Test the detector on genuine images
        batch_genuine_confidences = detector.detect(x)

        #Test the detector on adversarial images
        adversarials = attack.generate(x)
        batch_adversarial_confidences = detector.detect(adversarials)

        #Run the test
        (batch_genuine_confidences, batch_adversarial_confidences) = sess.run([batch_genuine_confidences, batch_adversarial_confidences], feed_dict={})

        #Join the results with the results of previous batches
        batch_genuine_confidences = np.expand_dims(batch_genuine_confidences, 0)
        batch_adversarial_confidences = np.expand_dims(batch_adversarial_confidences, 0)

        if genuine_confidences is not None:
            print(genuine_confidences.shape)
        print(batch_genuine_confidences.shape)
        print('========')

        genuine_confidences = batch_genuine_confidences if genuine_confidences is None else np.concatenate([genuine_confidences, batch_genuine_confidences], 0)
        adversarial_confidences = batch_adversarial_confidences if adversarial_confidences is None else np.concatenate([adversarial_confidences, batch_adversarial_confidences], 0)


    #Compute the ROC Curve
    ground_truths = np.concatenate(np.zeros_like(genuine_confidences), np.ones_like(adversarial_confidences))
    predictions = np.concatenate(genuine_confidences, adversarial_confidences)
    false_positive_rates, true_positive_rates, _ = metrics.roc_curve(ground_truths, predictions, pos_label=1)

    #Compute the AUC of the curve
    area_under_curve = metrics.auc(false_positive_rate, true_positive_rate)

    return false_positive_rates, true_positive_rates, area_under_curve

def keras_to_tensorflow(model, theano_backend=False, num_outputs=1, quantize=False):
    K.set_learning_phase(False)#TODO: Reimpostare
    if theano_backend:
        K.set_image_data_format('channels_first')
    else:
        K.set_image_data_format('channels_last')

        keras.Sequential().inputs
    pred = [None] * num_outputs
    pred_node_names = [None] * num_outputs
    for i in range(num_outputs):
        pred_node_names[i] = 'output' if num_outputs == 1 else 'output_' + str(i)
        pred[i] = tf.identity(model.outputs[i], name=pred_node_names[i])
        output_op = pred[i]
    print('output nodes names are: ', pred_node_names)

    sess = K.get_session()

    if quantize:
        transforms = ["quantize_weights", "quantize_nodes"]
        transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [], pred_node_names, transforms)
        constant_graph = graph_util.convert_variables_to_constants(sess, transformed_graph_def, pred_node_names)
    else:
        #print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=''))
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)    

    f = 'graph_db.ascii'
    tf.train.write_graph(sess.graph.as_graph_def(), '.', f, as_text=True)
    input_op = None
    for op in sess.graph.get_operations():
        #print('{} {}'.format(op.name, op.type))
        if 'input' in op.name and op.type == 'Placeholder':
            input_op = sess.graph.get_tensor_by_name(op.name + ':0')

    K.set_image_data_format('channels_last')
    #return graph
    print('Found')
    return input_op, output_op