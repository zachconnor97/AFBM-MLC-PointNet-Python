import tensorflow as tf

x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as g:
  #g.watch(x)
  y = x * x
  z = y * y
dz_dx = g.gradient(z, x)  # (4*x^3 at x = 3)
print(dz_dx)

dy_dx = g.gradient(y, x)
print(dy_dx)