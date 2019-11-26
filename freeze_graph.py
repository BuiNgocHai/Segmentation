import argparse
import os

import tensorflow as tf
from keras import backend as K

from PTIT_model import build_ptit

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph

dir = os.path.dirname(os.path.realpath(__file__))


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

def freeze_graph_keras(net, model_dir):
    """Extract the sub graph defined by the output nodes and convert 
    all its variables into constant 
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names, 
                            comma separated
    """

    # The export path contains the name and the version of the model
    tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
    file_name = os.path.basename(model_dir).replace('.hdf5', '.pb')
    model_dir = os.path.dirname(model_dir)
    print(os.path.join(model_dir, file_name))
    with tf.keras.backend.get_session() as sess:
        tf.initialize_all_variables().run()
        frozen_graph = freeze_session(K.get_session(),
                                      output_names=[out.op.name for out in net.outputs])
        tf.train.write_graph(frozen_graph, model_dir,
                             file_name, as_text=False)
    print('All input nodes:', net.inputs)
    print('All output nodes:', net.outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str,
                        default="/home/vicker/Downloads/pitmodel-034-0.48831.hdf5", help="Model folder to export")

    args = parser.parse_args()                    

    model = build_ptit((256,320,3), num_classes=4,
                    lr_init=1e-3, lr_decay=5e-4)
                    
    model.load_weights("/home/vicker/Downloads/pitmodel-034-0.48831.hdf5")
    freeze_graph_keras(model, args.model_dir)
