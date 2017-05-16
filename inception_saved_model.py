import os.path
import tensorflow as tf

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat
from inception import inception_model

tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/inception_train',
                           """Directory where to read training checkpoints.""")
tf.app.flags.DEFINE_string('output_dir', '/tmp/inception_output',
                           """Directory where to export inference model.""")
tf.app.flags.DEFINE_integer('model_version', 1,
                            """Version number of the model.""")
tf.app.flags.DEFINE_integer('image_size', 299, """Needs to provide same values as in training""")

FLAGS = tf.app.flags.FLAGS


NUM_CLASSES = 1000
NUM_TOP_CLASSES = 5

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
#SYNSET_FILE = os.path.join(WORKING_DIR, 'imagenet_lsvrc_2015_synsets.txt')
#METADATA_FILE = os.path.join(WORKING_DIR, 'imagenet_metadata.txt')


def export():
  with tf.Graph().as_default():
    # Step: One
    # Input transformation.
    jpegs = tf.placeholder(tf.string, name="tf_example")
    images = tf.map_fn(preprocess_image, jpegs, dtype=tf.float32)
    logits, _ = inception_model.inference(images, NUM_CLASSES + 1)

    # Transform output to topK result.
    values, indices = tf.nn.top_k(logits, NUM_TOP_CLASSES)

    # Restore variables from training checkpoint.
    # Step: Two
    variable_averages = tf.train.ExponentialMovingAverage(
        inception_model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)
    with tf.Session() as sess:
     # setf.InteractiveSession() 
     # Restore variables from training checkpoints.
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print 'Successfully loaded model from %s at step=%s.' % (
            ckpt.model_checkpoint_path, global_step)
      output_path = os.path.join(
          compat.as_bytes(FLAGS.output_dir),
          compat.as_bytes(str(FLAGS.model_version)))
      print 'Exporting trained model to', output_path
      builder = saved_model_builder.SavedModelBuilder(output_path)

      # Step: Three
      # Build the signature_def_map.
      scores_output_tensor_info = utils.build_tensor_info(values)
      
      predict_inputs_tensor_info = utils.build_tensor_info(jpegs)
      
      prediction_signature = signature_def_utils.build_signature_def(
          inputs={'images': predict_inputs_tensor_info},
          outputs={
              'scores': scores_output_tensor_info
          },
          method_name=signature_constants.PREDICT_METHOD_NAME)

      legacy_init_op = tf.group(
          tf.tables_initializer(), name='legacy_init_op')
      builder.add_meta_graph_and_variables(
          sess, [tag_constants.SERVING],
          signature_def_map={
              'predict_images':
                  prediction_signature,
          },
          legacy_init_op=legacy_init_op)

      builder.save()
      print 'Successfully exported model to %s' % FLAGS.output_dir


def preprocess_image(image_buffer):
  image = tf.image.decode_jpeg(image_buffer, channels=3) # 3- channels
  image = tf.image.convert_image_dtype(image, dtype=tf.float32) # type convert
  image = tf.image.central_crop(image, central_fraction=0.875)   # return central
  image = tf.expand_dims(image, 0)  # expand_dims
  image = tf.image.resize_bilinear(
      image, [FLAGS.image_size, FLAGS.image_size], align_corners=False)
  image = tf.squeeze(image, [0]) #
  image = tf.subtract(image, 0.5)#  
  
  image = tf.multiply(image, 2.0)
  return image


def main(unused_argv=None):
  export()


if __name__ == '__main__':
  tf.app.run()
