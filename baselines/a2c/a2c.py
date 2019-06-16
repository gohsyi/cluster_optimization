import functools
import numpy as np
import tensorflow as tf

from tensorflow import losses


from common.argparser import args
from common.util import get_logger

from baselines.common import tf_util
from baselines.common import set_global_seeds

from baselines.a2c.policy import build_policy
from baselines.a2c.utils import find_trainable_variables
from baselines.a2c.runner import Runner


class Model(object):
    """
    We use this class to :
        __init__:
        - Creates the step_model
        - Creates the train_model

        train():
        - Make the training part (feedforward and retropropagation of gradients)

        save/load():
        - Save load the model
    """

    def __init__(
            self,
            name,  # name of this model
            ob_size,  # dimension of observation vector
            act_size,  # dimension of action vector
            latents,  # network hidden layer sizes
            lr=1e-5,  # learning rate
            activation='relu',  # activation function
            optimizer='adam',  # optimization function
            vf_coef=0.1,  # vf_loss weight
            ent_coef=0.01,  # ent_loss weight
            max_grad_norm=0.5):  # how frequently the logs are printed out

        sess = tf_util.get_session()

        activation = tf_util.get_activation(activation)
        optimizer = tf_util.get_optimizer(optimizer)

        # lr = tf.train.polynomial_decay(
        #     learning_rate=lr,
        #     global_step=tf.train.get_or_create_global_step(),
        #     decay_steps=total_epoches,
        #     end_learning_rate=lr/10,
        # )

        # placeholders for use
        X = tf.placeholder(tf.float32, [None, ob_size], 'observation')
        A = tf.placeholder(tf.int32, [None], 'action')
        ADV = tf.placeholder(tf.float32, [None], 'advantage')
        R = tf.placeholder(tf.float32, [None], 'reward')

        with tf.variable_scope(name):
            policy = build_policy(
                observations=X,
                act_size=act_size,
                latents=latents,
                vf_latents=latents,
                activation=activation
            )

        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Policy loss
        neglogpac = policy.neglogp(A)
        # L = A(s,a) * -logpi(a|s)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(policy.entropy())

        # Value loss
        vf_loss = losses.mean_squared_error(tf.squeeze(policy.vf), R)

        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # gradients and optimizer
        params = find_trainable_variables(name)
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))

        # 3. Make op for one policy and value update step of A2C
        trainer = optimizer(learning_rate=lr)

        _train = trainer.apply_gradients(grads)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name))

        def step(obs):
            action, value = sess.run([policy.action, policy.vf], feed_dict={
                X: np.reshape(obs, (-1, ob_size))
            })
            return action, value

        def value(obs):
            return sess.run(policy.vf, feed_dict={
                X: np.reshape(obs, (-1, ob_size))
            })

        def train(obs, actions, rewards, values):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            advs = rewards - values

            td_map = {X:obs, A:actions, ADV:advs, R:rewards}
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )

            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            saver.save(sess, save_path)
            print(f'Model saved to {save_path}')

        def load(load_path):
            saver.restore(sess, load_path)
            print(f'Model restored from {load_path}')

        self.train = train
        self.step = step
        self.value = value
        self.save = save
        self.load = load

        tf.global_variables_initializer().run(session=sess)
