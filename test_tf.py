import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')

print("Is GPU Available: ", gpus)

for gpu in gpus:
  print("Name:", gpu.name, "  Type:", gpu.device_type)

a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)