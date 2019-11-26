import time

import cv2
import numpy as np
import tensorflow as tf


input_tensor_name = 'import/input_1:0'
output_tensor_name = 'import/fcn17/truediv:0'

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            name="import"
        )
        for op in graph.get_operations():
            print(op.name)
        print('''''''''''')
        [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
        return graph

def result_map_to_img(res_map):
    img = np.zeros((256, 320, 3), dtype=np.uint8)
    res_map = np.squeeze(res_map)

    argmax_idx = np.argmax(res_map, axis=2)

    # For np.where calculation.
    road = (argmax_idx == 1)
    car = (argmax_idx == 2)
    traffic = (argmax_idx == 3)

    img[:, :, 0] = np.where(road, 255, 0)
    img[:, :, 1] = np.where(car, 255, 0)
    img[:, :, 2] = np.where(traffic, 255, 0)

    return img

def visualize(img):
    img = cv2.resize(img,(320,256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0)
    img = img / 127.5 - 1

    with tf.Session(config= tf.ConfigProto() , graph=graph) as sess:
        output_image = sess.run(output_tensor, feed_dict={
            input_tensor: img})
    print(output_image.shape)
    res_img = result_map_to_img(output_image[0])
    

    return res_img

img = cv2.imread('/home/vicker/Downloads/all_rgb_simulator/sim1_33.jpg')


graph = load_graph('/home/vicker/Documents/Segmentation/model/tf_model.pb')

input_tensor = graph.get_tensor_by_name(input_tensor_name)
output_tensor = graph.get_tensor_by_name(output_tensor_name)

res = visualize(img)
cv2.imwrite('/home/vicker/Desktop/trash.png',res)
print('Done')