import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from Class_F_Functions    import  pareto_frontier, F_Functions,  scale_columns, igd

mb_size = 150  # Minibatch size
X_dim = 10  # Number of variables to approximate
z_dim = 30  # Dimension of the latent variable  (z_dim=30)
h_dim = 128
Function = "F" + str(8)  # 8-MO function to optimize, all in F1...F9, except F6
n=10    # Number of Variables of the MO function (n=10)
k = 1000  # Number of samples of the Pareto set for computing approximation (k=1000)
MOP_f = F_Functions(n, Function)  # Creates a class co1 D   ntaining details on MOP
ps_all_x = MOP_f.Generate_PS_samples(k)  # Generates k points from the Pareto Set
pf1, pf2 = MOP_f.Evaluate_MOP_Function(ps_all_x)  # Evaluate the points from the Pareto Set
ps_all_x = scale_columns(ps_all_x)  # Scales columns so it could be used for learning model

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, X_dim])

D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]

def batch_function(num, data, start):

    '''
    Return a total of `num` samples and labels.
    '''
    idx = np.arange(start, np.min([start+num,len(data)]))
    return data[idx,:]


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    out = tf.matmul(D_h1, D_W2) + D_b2
    return out


G_sample = generator(z)
D_real = discriminator(X)
D_fake = discriminator(G_sample)

D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
G_loss = -tf.reduce_mean(D_fake)

D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(-D_loss, var_list=theta_D))
G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(G_loss, var_list=theta_G))

clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0
lowest_igd= 10000
for it in range(100000):
    for _ in range(5):

        z_mb = sample_z(mb_size,z_dim)
        X_mb = batch_function(mb_size, ps_all_x, i)

        _, D_loss_curr, _ = sess.run(
            [D_solver, D_loss, clip_D],
            feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)}
        )

    _, G_loss_curr = sess.run(
        [G_solver, G_loss],
        feed_dict={z: sample_z(mb_size, z_dim)}
    )

    if it % 100 == 0:
        print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss_curr, G_loss_curr))

        if it % 1000 == 0:
            final_samples = sess.run(G_sample, feed_dict={z: sample_z(1000, z_dim)})
            nf1, nf2 = MOP_f.Evaluate_MOP_Function(final_samples)
            Tf1, Tf2 = pareto_frontier(nf1, nf2)
            igd_val = igd(np.vstack((Tf1, Tf2)).transpose(), np.vstack((pf1, pf2)).transpose())
            if(igd_val < lowest_igd):
                lowest_igd = igd_val
            print("igd_val ",igd_val)
            i += 1
print("lowest igd :",lowest_igd)