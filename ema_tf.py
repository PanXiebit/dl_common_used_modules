# -*- encoding = utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt

tf.reset_default_graph()

g = tf.Graph()

with g.as_default():
    with tf.name_scope("MovingAverage"):
        count = tf.Variable(0, dtype=tf.float32, name="count")
        one  = tf.constant(1.0, dtype=tf.float32, name="one")
        add1 = tf.add(count, one, name="count")

        #
        tf.summary.scalar("count", count)

        add1_op = tf.assign(count, add1)

        ema = tf.train.ExponentialMovingAverage(decay=0.99)

        ema_op = ema.apply([count])

        moving_averaged_count = ema.average(count)

        tf.summary.scalar("moving_averaged_count", moving_averaged_count)

    with tf.name_scope("global_ops"):
        merged = tf.summary.merge_all()
        initializer = tf.global_variables_initializer()

summary_writer = tf.summary.FileWriter(logdir="./log", graph=g)

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    variables = []
    shadow_variables = []
    for i in range(100):
        (merged_summary, variable, shadow_variable) = sess.run([merged, count, moving_averaged_count])

        variables.append(variable)
        shadow_variables.append(shadow_variable)

        sess.run(add1_op)

        sess.run(ema_op)

        summary_writer.add_summary(merged_summary, global_step=i)
    print(variables)
    print(shadow_variables)
