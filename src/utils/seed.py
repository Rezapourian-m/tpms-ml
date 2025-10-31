import numpy as np, random, os, tensorflow as tf

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass
