import sys
import tensorflow as tf

path = sys.argv[1]
ckpt = tf.train.load_checkpoint(path)
names = ckpt.get_variable_to_shape_map()
print(f"total vars: {len(names)}")
for n in sorted(names):
    print(f"{n}\t{ckpt.get_tensor(n).shape}")
