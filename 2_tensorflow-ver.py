# Modify setup.py file to install the tf-models-official repository targeted at TF v2.8.0
import re
with open('models/research/object_detection/packages/tf2/setup.py') as f:
    s = f.read()

with open('models/research/setup.py', 'w') as f:
    # Set fine_tune_checkpoint path
    s = re.sub('tf-models-official>=2.5.1',
               'tf-models-official==2.9.0', s)
    f.write(s)