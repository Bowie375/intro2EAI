import os
import sys

import matplotlib.pyplot as plt

xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import time
import logging
_start = time.time()
class ETFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        return f"{record.created - _start:.1f}s"

fmt = "%(asctime)s [%(levelname)s] - %(message)s"
handler = logging.StreamHandler()
handler.setFormatter(ETFormatter(fmt))

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(handler)

import jax
from jax import numpy as jp
from ml_collections import config_dict
import mujoco
import numpy as np
import random

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1.getup import Getup

DESIRED_BODY_HEIGHT = 0.33


def a2b(a, b):
    return a + (b-a)*random.random()

# [x, [0.2, 0.3], [1.0, 3.0], x, x]
W_shop = {
    0: [200.0, a2b(0.2,0.22), a2b(2.0, 2.5), 0.0, a2b(0.1, 0.12)],
    1: [200.0, a2b(0.2,0.22), a2b(2.0, 2.5), 0.0, a2b(0.1, 0.12)],
    2: [200.0, a2b(0.2,0.22), a2b(2.0, 2.5), 0.0, a2b(0.1, 0.12)],
    3: [200.0, a2b(0.2,0.22), a2b(2.0, 2.5), 0.0, a2b(0.1, 0.12)],
    4: [200.0, a2b(0.2,0.22), a2b(2.0, 2.5), 0.0, a2b(0.1, 0.12)],
    5: [200.0, a2b(0.2,0.22), a2b(2.0, 2.5), 0.0, a2b(0.1, 0.12)],
    6: [200.0, a2b(0.2,0.22), a2b(2.0, 2.5), 0.0, a2b(0.1, 0.12)],
    7: [200.0, a2b(0.2,0.22), a2b(2.0, 2.5), 1.0, a2b(0.1, 0.12)],
}
W = W_shop[0]
A = 0.0

class MyGetupEnv(Getup):
    def step(self, state: mjx_env.State, action: jax.Array):
        motor_targets = state.data.qpos[7:] + action * self._config.action_scale
        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)

        obs = self._get_obs(data, state.info)
        done = self._get_termination(data)

        # --- Key State Extraction ---

        # Unit vector pointing up in world frame (z-axis negative in many physics engines)
        up_vec = jp.array([0.0, 0.0, -1.0])

        # Extract the 3D position of the IMU site (typically mounted on the body), in world coordinates [m]
        body_pos = data.site_xpos[self._imu_site_id]

        # Compute the linear velocity of the body in its local (body) frame [m/s]
        body_lin_vel = self.get_local_linvel(data)

        # Retrieve the angular velocity (gyroscope data) of the body in its local frame [rad/s]
        body_ang_vel = self.get_gyro(data)

        # Get the gravity vector expressed in the local body frame, usually used to infer orientation
        gravity_vector = self.get_gravity(data)

        # Make a copy of the default pose (e.g., reference or rest configuration) of the full model
        default_qpos = self._default_pose.copy()

        # Extract joint positions (excluding root pose: typically first 7 elements are base position + orientation)
        joint_qpos = data.qpos[7:]

        # Extract joint velocities (excluding root velocities)
        joint_qvel = data.qvel[7:]

        # TODO: your code here.
        #  Hint: consider three import objective related to getup task:
        #   1. body height
        #   2. body orientation
        #   3. joint position (error to default pose)
        #   4. body angular velocity (error to zero)

        #rew_height = jp.exp(-10.0 * jp.square(body_pos[2] - DESIRED_BODY_HEIGHT))
        #rew_height = jp.exp(-3 * (DESIRED_BODY_HEIGHT - body_pos[2]) ** 2)
        rew_height = -jp.square(body_pos[2] - DESIRED_BODY_HEIGHT)

        #rew_orient = jp.exp(-5.0 * jp.sum(jp.square(gravity_vector - up_vec)))
        #rew_orient = jp.exp(-jp.sum(jp.square(up_vec - gravity_vector)))
        rew_orient = -jp.sum(jp.square(up_vec - gravity_vector))

        #weight = jp.array([1.0, 1.0, 0.1] * 4)
        #rew_qpos = jp.exp(-2.0 * jp.sum(jp.square(joint_qpos - default_qpos)*weight))
        #rew_qpos = jp.exp(-jp.sum(jp.square(joint_qpos - default_qpos) * weight))
        rew_qpos = -jp.sum(jp.square(joint_qpos - default_qpos))

        jointvel_penalty = jp.sum(jp.square(joint_qvel))
        anglevel_penalty = jp.sum(jp.square(body_ang_vel))
        #rew_vel = jp.exp(-0.1 * (jointvel_penalty+anglevel_penalty))
        #rew_angvel = jp.exp(-jp.sum(jp.square(body_ang_vel)))

        #w = [1.0,1.0,1.0,1.0]
        #w = [200.0, 0.01, 0.05, 0.1, 0.1] good
        #w = [200.0, 1.0, 0.1, 0.1, 0.1] bad
        #w = [200.0, 1.0, 0.1, 0.1, 1.0] bad
        #w = [200.0, 0.01, 0.1, 0.1, 0.1] good
        #w = [200.0, 0.05, 0.1, 0.1, 0.1]
        #w = [200.0, 0.05, 0.5, 0.1, 0.1]
        #w = [250.0, 0.5, 0.1, 0.1, 0.1]
        w = W

        reward = w[0]*rew_height + w[1]*rew_orient + w[2]*rew_qpos - w[3]*jointvel_penalty - w[4]*anglevel_penalty
        # TODO: End of your code.

        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action

        state.metrics["reward"] = reward
        state.metrics["rew_height"] = rew_height
        state.metrics["rew_orient"] = rew_orient
        state.metrics["rew_qpos"] = rew_qpos
        state.metrics["anglevel_penalty"] = anglevel_penalty
        state.metrics["jointvel_penalty"] = A

        done = jp.float32(done)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # Sample a random initial configuration with some probability.
        # rng, key1, key2 = jax.random.split(rng, 3)
        # qpos = jp.where(
        #     jax.random.bernoulli(key1, self._config.drop_from_height_prob),
        #     self._get_random_qpos(key2),
        #     self._init_q,
        # )
        qpos = self._init_q.copy()
        # Sample a random root velocity.
        # rng, key = jax.random.split(rng)
        qvel = jp.zeros(self.mjx_model.nv)
        # qvel = qvel.at[0:6].set(jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5))

        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])

        # Let the robot settle for a few steps.
        data = mjx_env.step(self.mjx_model, data, qpos[7:], self._settle_steps)
        data = data.replace(time=0.0)

        info = {
            "rng": rng,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
        }

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        metrics = {
            "reward": jp.zeros(()),
            "rew_height": jp.zeros(()),
            "rew_orient": jp.zeros(()),
            "rew_qpos": jp.zeros(()),
            #"rew_angvel": jp.zeros(()),
            "anglevel_penalty": jp.zeros(()),
            "jointvel_penalty": jp.zeros(()),
        }
        return mjx_env.State(data, obs, reward, done, metrics, info)


def create_env():
    # env config
    env_cfg = config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        Kp=35.0,
        Kd=0.5,
        episode_length=300,
        drop_from_height_prob=0.6,
        settle_time=0.5,
        action_repeat=1,
        action_scale=0.5,
        soft_joint_pos_limit_factor=0.95,
        energy_termination_threshold=np.inf,
        noise_config=config_dict.create(
            level=0.0,
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gyro=0.2,
                gravity=0.05,
            ),
        ),
    )
    env = MyGetupEnv(env_cfg)
    return env


def train_ppo(cuda_idx:int, t:str):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_idx)
    sys.stdout = open(f'experiments/part2/{t}/{cuda_idx}/stdout.txt', 'w')
    sys.stderr = sys.stdout
    W = W_shop[cuda_idx]
    A = 1.0

    import mediapy as media

    from datetime import datetime
    import functools
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    from ml_collections import config_dict

    ppo_params = config_dict.create(
        num_timesteps=40_000_000,
        num_evals=0,
        reward_scaling=1.0,
        episode_length=500,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=5e-4, # 3e-4
        entropy_cost=1e-2,
        num_envs=4096,
        batch_size=128, # 256
        max_grad_norm=1.0,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="privileged_state",
            value_obs_key="privileged_state",
        ),
    )

    start_t = datetime.now()

    ppo_training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory
        )

    train_fn = functools.partial(
        ppo.train, **dict(ppo_training_params),
        network_factory=network_factory,
        # progress_fn=progress,
        num_eval_envs=0,
        log_training_metrics=True,
        training_metrics_steps=1_000_000
    )

    env = create_env()
    eval_env = create_env()
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    end_t = datetime.now()
    print(f"time to train: {end_t - start_t}")

    render_length = 500
    _pre_render_length = 100

    # Enable perturbation in the eval env.
    eval_env = create_env()

    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    rng = jax.random.PRNGKey(0)
    rollout = []
    body_height = []

    # is_upright = self._is_upright(gravity)
    # is_at_desired_height = self._is_at_desired_height(torso_height)
    # gate = is_upright * is_at_desired_height

    state = jit_reset(rng)
    for i in range(render_length):
        if i < _pre_render_length:
            ctrl = env._default_pose.copy()
        else:
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)

        state = jit_step(state, ctrl)
        rollout.append(state)
        env_height = state.data.site_xpos[env._imu_site_id][2]
        body_height.append(env_height)

    body_height = jp.array(body_height)
    height_error = np.mean(np.abs(body_height - DESIRED_BODY_HEIGHT))
    plt.plot(body_height)
    # plot desired body height
    plt.axhline(DESIRED_BODY_HEIGHT, color='r', linestyle='--')
    plt.title(f"Height error: {height_error:.3f}")
    plt.xlabel("steps")
    plt.ylabel("body height")
    #plt.savefig("part2_height_error.png")

    plt.savefig(f'experiments/part2/{t}/{cuda_idx}/part2_height_error.png')

    render_every = 2
    fps = 1.0 / eval_env.dt / render_every
    print(f"fps: {fps}")

    traj = rollout[::render_every]
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True

    frames = eval_env.render(
        traj,
        camera="track",
        height=480,
        width=640,
        scene_option=scene_option,
    )
    media.write_video(f'experiments/part2/{t}/{cuda_idx}/part2_video.mp4', frames)
    #media.write_video('../experiments/solutions/part2_video.mp4', frames)
    print("video saved to part2.mp4")

if __name__ == '__main__':
    import multiprocessing as mp
    import time
    import random
    import json

    processes = []
    t = time.strftime("%Y%m%d_%H%M%S")
    for i in range(7,8):
        os.makedirs(f'experiments/part2/{t}/{i}', exist_ok=True)
        with open(f"experiments/part2/{t}/{i}/config.json", 'w') as f:
            json.dump(W_shop[i], f)
        p = mp.Process(target=train_ppo, args=(i,t))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()
    
    #train_ppo()