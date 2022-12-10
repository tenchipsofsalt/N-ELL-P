import tensorflow as tf
import numpy as np

class TransformerScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, scale_lr, hidden_size, warmup_steps):
        self.scale_lr = tf.cast(scale_lr, tf.float32)
        self.hidden_size = tf.cast(hidden_size, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        lr = self.scale_lr / tf.sqrt(self.hidden_size)
        lr *= tf.minimum(1 / tf.sqrt(step), step * self.warmup_steps ** (-1.5))
        return lr 
