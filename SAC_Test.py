#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 01:51:00 2019

@author: brianhaney
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf

from tf_agents.agents.sac import sac_agent
from tf_agents.environnments import time_step as ts
from tf_agents.policies.policy_step import PolicyStep
from tf_agents.specs import tensor_spec

class _MockDrstribution(object):
    def __init__(self, action):
        self._action = action
    
    def sample(self):
        return self._action
    
    def log_prob(self, unused_sample):
        return tf.constant(10., shape=[1])
    
class DummyActorPolicy(object):
    def __init__(self, time_step_spec, action_spec, actor_network):
        del time_step_spec
        del actor_network
        single_action_spec = action_spec
        
    def action(self, time_step):
        del time_step
        action = tf.constant(self._action, dtype=tf.float32, shape=[1])
        return PolicyStep(action=action)
    
    def distribution(self, time_step):
        action = self.action(time_step).action
        return PolicyStep(action=_MockDistribution(action))
    
class DummyCriticNet(object):
    
    def copy(self, name=''):
        del name
        return copy.copy(self)
    def __call__(self, inputs, step_type):
        observation, actions = inputs
        del step_type
        actions = tf.cast(tf.nest.flatten(actions)[0], tf.float32)
        
        states = tf.cast(tf.nest.flatten(observation)[0], tf.float32)
        #Biggest state is best state.
        value = tf.reduce_max(input_tensor=states, axis=-1)
        value = tf.reshape(value, [-1])
        
        #Biggest action is best action
        q_value = tf.reduce_max(input_tensor=actions, axis=-1)
        q_value = tf.reshape(q_value, [-1])
        #Biggest state is best state
        return value + q_value, ()
    
class SacAgentTest(tf.test.TestCase):
    def setUp(self):
        super(SacAgentTest. self).setUp()
        tf.compat.v1.enable_resource_variables()
        self._obs_spec = [tensor_spec.TensorSpec([2], tf.float32)]
        self._time_step_spec = ts.time_step_spec(self._obs_spec)
        self._action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, -1, 1)
        
        
    def testCreateAgent(self):
        sac_agent.SacAgent(
                self._time_step_spec,
                self._action_spec,
                critic_network=DummyCriticNet(),
                actor_network=None,
                actor_optimizer=None,
                critic_optimizer=None,
                alpha_optimizer=None,
                actor_policy_ctor=DummyActorPolicy)
        
    def testCriticLoss(self):
        agent = sac_agent.SacAgent(
                self._time_step_spec,
                self._action_spec,
                critic_network=DummyCriticNet(),
                actor_network=None,
                actor_optimizer=None,
                critic_optimizer=None,
                alpha_optimizer=None,
                actor_policy_ctor=DummyActorPolicy)
        
        
        observations = [tf.constant([[1, 2], [3, 4]], dtype=tf.float32)]
        time_steps = ts.restart(observations)
        actions = tf.constant([5], [6], dtype=tf.float32)
        
        rewards = tf.constant([10, 20], dtype=tf.flaot32)

        discounts = tf.constant([0.9, 0.9], dtype=tf.float32)
        next_observatoins = [tf.constant([[5, 6], [7, 8]], dtype=tf.float32)]
        next_time_steps = ts.transition(next_observations, rewards, discounts)
        
        td_targets = [7.3, 19.1]
        pred_td_targets = [7., 10.]
        
        self.evaluate(tf.compat.v1.global_variables_initializer())
        
        #Expected critic loss has factor of 2, for the two TD3 critics.
        expected_loss = self.evaluate(2 * tf.compat.v1.losses.mean_squared_error(
                tf.constant(td_targets), tf.constant(pred_td_targets)))
        
        loss = agent.critic_loss(
                time_steps,
                actions,
                next_time_steps,
                td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error)
        
        self.evaluate(tf.compat.v1.global_variables_initializer())
        loss_ = self.evaluate(loss)
        self.assertAllClose(loss_, expected_loss)
        
    def testActorLoss(self):
        agent = sac_agent.SacAgent(
                self._time_step_spec,
                self._action_spec,
                critic_network=DummyCriticNet(),
                actor_network=None,
                actor_optimizer=None,
                critic_optimizer=None,
                alpha_optimizer=None,
                actor_policy_ctor=DummyActorPolicy)
        observations = [tf.constant([[1, 2], [3, 4]], dtype=tf.float32)]
        time_steps = ts.restart(observations, batch_size=2)
        
        expected_loss = (2 * 10 - (2+1) - (4+1))/2
        loss = agent.actor_loss(time_steps)
        
        self.evaluate(tf.compat.v1.global_variables_initializer())
        loss_ = self.evaluate(loss)
        self.assertAllClose(loss_, expected_loss)
        
        def testAlphaLoss(self):
            agent = sac_agent.SacAgent(
                    self._time_step_spec,
                    self._action_spec,
                    critic_network=DummyCriticNet(),
                    actor_network=None,
                    actor_optimizer=None,
                    critic_optimizer=None,
                    alpha_optimizer=None,
                    target_entropy=3.0,
                    initial_log_alpha=4.0,
                    actor_policy_ctor=DummyActorPolicy)
            
            observations = [tf.constant([1, 2], [3, 4], dtype=tf.float32)]
            time_steps = ts.restart(observations, batch_size=2)
            
            action_step = agent.policy.action(time_steps)
            
            self.evaluate(tf.compat.v1.global_variables_initializer())
            action_ = self.evaluate(action_step.action)
            self.assertLessEqual(action_, self._action_spec.maximum)
            self.assertGreaterEqual(action_, self._action_spec.minimum)
            
if __name__ == '__main__':
    tf.test.main()