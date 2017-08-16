#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 19:39:16 2017
Incorporates rnn into structure of skeleton_rnn.py
At each time step, the input variables into the rnn are set to be 
a visible component of the state variable, along with past components of the state
variable. 
The parameters include:
n_state: sets the dimension of the state variable
n_visible: sets the visible component of the state variable. The first n_visible components
                                of the state variable are visible.
n_addon_state: sets the visible component of the add on variables. The first n_addon components
                                of the state variable are visible.
n_addon_step: sets the number of past states that are included in the n_addon varables
n_addon = n_addon_step*n_addon_state: The total number of add_on variables

feed_output: 0 don't feed output, 1 feed output
Later, we will adapt this code to accept other inputs.


Other parameters of interest
iterates: number of training itereates performed 
n_batch: training batch size
n_step: number of times we call the ode solver in an iterate

n_ode_step: number of steps in each call to ode solver, same as number of cells in each tf optimization call.
ode_inc: time increment for ode solver
delta: number of ode_incs in a single step

@author: arthur
"""
from __future__ import print_function, division
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
from scipy.integrate import odeint
#import random
#from tensorflow.python import debug as tf_debug

iterates = 1

n_step = 14000 #1024 #number of times we maintain the initial conditions for the ode solver.
n_batch = 1
n_config = 1
n_state = 2
n_visible = 1
n_addon_state = n_visible 
n_addon_step = 1
n_addon = n_addon_step*n_addon_state
feed_output = 1  # when set to 1, uses a cell's output as an input into the next cell. Only set to 0 or 1.
n_total_input = (1+feed_output)*n_visible + n_addon
n_output = n_visible


n_categories = 4 # number of different categories for setting the dynamics

n_ode_step = 16  #number of steps in each call to ode solver, same as number of cells in each tf optimization call.
ode_inc = 0.05 #time step for ode solver
delta = 8 #number of ode_incs in a single step

np.random.seed(113)
def f(y, t):#vector field for pendulum
    theta, omega = y      # unpack current values of y
    derivs = [omega,  -np.sin(theta)]
    return derivs


def deriv(y, t):#vector field for double pendulum
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    g = 9.8
    theta1, theta2, z1, z2 = y
    L1 = 1
    L2 = 1
    m1 = 1
    m2 = 1
    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    theta1dot = z1
    z1dot = (m2*g*np.sin(theta2) - m2*s*(L1*z1**2*c + L2*z2**2) -
             (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
    theta2dot = z2
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
             m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
    
    derivs = [theta1dot, theta2dot, z1dot, z2dot]

    return derivs


def generate_dynamics(init_conds, t_in, steps):

    x = np.ndarray(shape=(n_batch,steps,n_state))
    y = np.ndarray(shape=(n_batch,steps,n_state))
    x_next = np.ndarray(shape=(n_batch, n_state))
    
    # Call the ODE solver
    i = 0
    for y0 in init_conds:
        psoln = odeint(f, y0, t_in, full_output = False)
        for k in range(n_config):
            for j in range(steps - 1):
                x[i,j,k] = psoln[delta*j,k]
                y[i,j,k] = psoln[delta*(j+1),k]
                x[i,j,k + n_config] = psoln[delta*j,k + n_config]
                y[i,j,k+ n_config] = psoln[delta*(j+1),k + n_config]
            x[i,steps-1,k] = psoln[delta*(steps-1),k]
            y[i,steps-1,k] = psoln[delta*steps,k]
            x[i,steps-1,k + n_config] = psoln[delta*(steps-1),k + n_config]
            y[i,steps-1,k + n_config] = psoln[delta*steps,k + n_config]
            x_next[i,k] = psoln[delta*steps,k]
            x_next[i,k + n_config] = psoln[delta*steps,k + n_config]
        i += 1
   
    return (x,y,x_next)


batchX = tf.placeholder(tf.float32, [n_batch, n_ode_step, n_output])
batchX_add = tf.placeholder(tf.float32, [n_batch, n_addon])
batchX_feed = tf.placeholder(tf.float32, [feed_output*n_batch, feed_output*n_output])
batchY = tf.placeholder(tf.float32, [n_batch, n_ode_step, n_output])



W0 = tf.Variable(tf.random_normal([n_total_input, n_categories], stddev=.2, seed=113), dtype=tf.float32)
b0 = tf.Variable(np.zeros((1,n_categories)), dtype=tf.float32)

T = tf.Variable(tf.random_normal([n_categories, n_total_input, n_output], stddev=.2, seed = 213), dtype=tf.float32)

input_series = tf.unstack(batchX, axis=1)
true_series = tf.unstack(batchY, axis=1)

output_series = []
x_add = batchX_add
x_feed = batchX_feed

for x0_in in input_series:
    x0 = tf.concat((x_add, x0_in, x_feed), axis=1)
    x1 = tf.sigmoid(tf.matmul(x0, W0) + b0) #+ tf.tanh(tf.matmul(x0_in, W0b) + b0b)
    y0 = tf.tensordot(x0, T, [[1],[1]]) # + Tb
    y_out = []
    for i in range(n_batch):
        x = tf.slice(x1,[i,0],[1,n_categories])
        y = tf.squeeze(tf.slice(y0,[i,0,0],[1,n_categories,n_output]), 0)
        y_out.append(tf.matmul(x, y))
    x_feed =tf.reshape(y_out, [n_batch, n_visible])
    x_add = tf.slice(x0,[0,n_addon_state],[n_batch, n_addon]) 
    output_series.append(x_feed)
     

        

losses = [tf.nn.l2_loss(o_series - t_series) for o_series, t_series in zip(output_series,true_series)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(0.005).minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tStop = n_ode_step*delta*ode_inc + ode_inc
    t = np.arange(0., tStop, ode_inc)
    init_conds = np.zeros((n_batch,n_state))
    for i in range(iterates):
            for j in range(n_batch):
                for k in range (n_config):
                    init_conds[j][k] = (j+1)*np.pi/(n_batch)
                    init_conds[j][k+n_config] = np.random.normal()*.01                 
            ts_S = n_addon_step*delta*ode_inc + ode_inc
            ts = np.arange(0., ts_S, ode_inc)
            if n_addon > 0:
                x_state,_, init_conds = generate_dynamics(init_conds, ts, n_addon_step)
            if feed_output == 1:
                x_feed = []
                for k in range(n_batch):
                    x_list = []
                    for j in range(n_visible):
                        x_list.append(init_conds[k,j])
                    x_feed.append(x_list)
            x_addon = [[x_state[xi][xj][xk] for xk in range(n_addon_state) for xj in range(n_addon_step)] for xi in range(n_batch)]
            
            for j in range(n_step):
                x_state,y_state, init_conds = generate_dynamics(init_conds, t, n_ode_step)
                x = [[[x_state[xi][xj][xk] for xk in range(n_visible)] for xj in range(n_ode_step)] for xi in range(n_batch)]
                y = [[[y_state[yi][yj][yk] for yk in range(n_visible)] for yj in range(n_ode_step)] for yi in range(n_batch)]   
                _, t_loss, o_s = sess.run([train_step, total_loss, output_series], feed_dict={batchX : x, batchY : y, batchX_add: x_addon, batchX_feed: x_feed})
                x_feed = [o_s[xi][xj] for xi in range(n_ode_step-1,n_ode_step) for xj in range(n_batch)]
                x_addon = [[x_state[xi][xj][xk] for xk in range(n_addon_state) for xj in range (n_ode_step - n_addon_step, n_ode_step)] for xi in range(n_batch)]

                if j%1000 == 0:
                    for k in range(n_ode_step):                        
                        print(str(y[int(n_batch/2)][k][0]) + '  ' + str(o_s[k][int(n_batch/2)][0]))
                    print("\n\n" + str(t_loss)+ '  ' + str(j) + "\n\n" )
            if i == iterates - 1:
                for k in range(n_ode_step):
                    print(str(y[0][k][0]) + '  ' + str(o_s[k][0][0]))
                print("\n\n" + str(t_loss)+ '  ' + str(n_step) + "\n\n" )
