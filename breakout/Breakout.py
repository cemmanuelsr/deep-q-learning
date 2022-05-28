import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.trajectories.trajectory import to_transition
from tf_agents.utils.common import function, Checkpointer
from tf_agents.policies import policy_saver
import argparse

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger().setLevel(logging.INFO)

from utils import *

parser = argparse.ArgumentParser(prog='Breakout')
parser.add_argument('-t', '--train', action='store_true')
parser.add_argument('-m', '--model', type=str, default='breakout')
args = parser.parse_args()

env = suite_atari.load(
    'BreakoutNoFrameskip-v4',
    max_episode_steps=30000,
    gym_env_wrappers=[AtariPreprocessingWithAutoFire, FrameStack4])
env.seed(42)
tf_env = TFPyEnvironment(env)

checkpoint_dir = os.path.join('../checkpoints/', args.model)
policy_dir = os.path.join('../policies', f'{args.model}_policy')

preprocessing_layer = keras.layers.Lambda(
                          lambda obs: tf.cast(obs, np.float32) / 255.)
conv_layer_params=[(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
fc_layer_params=[512]

q_net = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    preprocessing_layers=preprocessing_layer,
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params)

train_step = tf.Variable(0)
update_period = 4 # run a training step every 4 collect steps
optimizer = keras.optimizers.RMSprop(learning_rate=2.5e-4, rho=0.95, momentum=0.0,
                                     epsilon=0.00001, centered=True)
epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.0, # initial ε
    decay_steps=250000 // update_period, # <=> 1,000,000 ALE frames
    end_learning_rate=0.01) # final ε
agent = DqnAgent(tf_env.time_step_spec(),
                 tf_env.action_spec(),
                 q_network=q_net,
                 optimizer=optimizer,
                 target_update_period=2000, # <=> 32,000 ALE frames
                 td_errors_loss_fn=keras.losses.Huber(reduction="none"),
                 gamma=0.99, # discount factor
                 train_step_counter=train_step,
                 epsilon_greedy=lambda: epsilon_fn(train_step))
agent.initialize()

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=100000)

replay_buffer_observer = replay_buffer.add_batch

train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]
log_metrics(train_metrics)

collect_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=update_period)
initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
                                        tf_env.action_spec())
init_driver = DynamicStepDriver(
    tf_env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(20000)],
    num_steps=10000) # <=> 80,000 ALE frames
final_time_step, final_policy_state = init_driver.run()

tf.random.set_seed(9)
trajectories, buffer_info = next(iter(replay_buffer.as_dataset(
    sample_batch_size=2,
    num_steps=3,
    single_deterministic_pass=False)))
time_steps, action_steps, next_time_steps = to_transition(trajectories)

dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=3).prefetch(3)

collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)

train_losses = list()
def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        train_losses.append(train_loss.loss.numpy())
        print("\r{} loss:{:.5f}".format(
            iteration, train_loss.loss.numpy()), end="")
        if iteration % 1000 == 0:
            log_metrics(train_metrics)

if args.train:

    train_agent(n_iterations=100000)

    import matplotlib.pyplot as plt
    plt.plot(train_losses)
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('Loss vs Episodes')
    plt.savefig(f"../results/{args.model}.jpg")     
    plt.close()

    train_checkpointer = Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=train_step
    )
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)
    tf_policy_saver.save(policy_dir)
else:
    saved_policy = tf.saved_model.load(policy_dir)

    frames = []
    def save_frames(trajectory):
        global frames
        frames.append(tf_env.pyenv.envs[0].render(mode="rgb_array"))

    watch_driver = DynamicStepDriver(
        tf_env,
        saved_policy,
        observers=[save_frames, ShowProgress(1000)],
        num_steps=1000)
    final_time_step, final_policy_state = watch_driver.run()

    plot_animation(frames)
