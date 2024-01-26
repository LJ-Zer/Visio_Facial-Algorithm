import sys
 
import tensorflow.keras
import tensorflow as tf
import numpy as np
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
test2 = tf.test.is_built_with_cuda()
print (test2)
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")
print("GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
test = tf.config.list_physical_devices('GPU')
print (test)
