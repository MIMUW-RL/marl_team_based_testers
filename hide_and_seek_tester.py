import argparse
import os.path
import random
import time
from os import listdir
import numpy as np
import onnxruntime
import pandas as pd
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel import SideChannel
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
import uuid


class HnsSideChannel(SideChannel):
    def __init__(self):
        super().__init__(uuid.UUID("a1d8f7b7-cec8-50f9-b78b-d3e165a78520"))
        self.time_hidden = 0.0
        self.winratio = 0.0

    def on_message_received(self, msg):
        s = msg.read_string()
        if s == "Environment/TimeHidden":
            f = msg.read_float32()
            print("Time hidden:", f)
            self.time_hidden = f
        if s == "Environment/HiderWinRatio":
            f = msg.read_float32()
            self.winratio = f
            print("Hider winratio:", f)


parser = argparse.ArgumentParser()
parser.add_argument("env", help="environment file")
parser.add_argument("-p", "--Port", type=int, help="Port")
parser.add_argument("-s", "--Seed", type=int, help="Seed")
parser.add_argument("-e", "--Episodes", type=int, help="Number of epsiodes to play")
parser.add_argument("-a", "--Agents", type=int, help="Number of agents in one team")
parser.add_argument(
    "-t0", type=str, help="text file with checkpoint paths to load (team hiders)"
)
parser.add_argument(
    "-t1", type=str, help="text file with checkpoint paths to load (team seekers)"
)
parser.add_argument("-o", default="", help="Output file prefix (directory)")
parser.add_argument("--time_scale", type=float, default=10)
parser.add_argument("--log_folder", default=None, help="folder for Unity logs")
parser.add_argument(
    "--shuffle",
    help="optional if the team chp's should be shuffled",
    action="store_true",
)

args = parser.parse_args()
Port = 6000
seed = 1
log_folder = None
if args.Port:
    Port = args.Port
    seed = Port
if args.Episodes:
    EPISODES = args.Episodes
if args.Seed:
    seed = args.Seed
if args.Agents:
    agents = args.Agents
if args.log_folder:
    log_folder = args.log_folder

if os.path.isdir(args.t0):
    t0name = args.t0.split("/")[-2]
else:
    t0name = args.t0.split("/")[-1]
if os.path.isdir(args.t1):
    t1name = args.t1.split("/")[-2]
else:
    t1name = args.t1.split("/")[-1]
output_file = os.path.join(
    args.o,
    t0name + "_vs_" + t1name + "_" + str(seed % 1000) + ".csv",
)
print(f"output_file={output_file}")

ckpts = []
if os.path.isdir(args.t0):
    teams_blue = listdir(args.t0)
    dirn_blue = args.t0
else:
    teams_blue = [args.t0]
    dirn_blue = ""

if os.path.isdir(args.t1):
    teams_purple = listdir(args.t1)
    dirn_purple = args.t1
else:
    teams_purple = [args.t1]
    dirn_purple = ""

blue_nr = len(teams_blue)
purple_nr = len(teams_purple)

if args.env:
    env_file = args.env

sc = HnsSideChannel()
engine_cfg = EngineConfigurationChannel()
env = UnityEnvironment(
    file_name=env_file,
    no_graphics=True,
    base_port=Port,
    seed=seed,
    log_folder=log_folder,
    side_channels=[sc, engine_cfg],
)
engine_cfg.set_configuration_parameters(time_scale=float(args.time_scale))

env.reset()
# get behavior_names (two agent teams) behaviour_specs is a 2d array
behavior_names = list(env.behavior_specs)

# print some environment specs
spec = env.behavior_specs[behavior_names[0]]
print("Number of observations : ", len(spec.observation_specs))
outputs = env._communicator.exchange(env._generate_reset_input(), env._poll_process)
if spec.action_spec.continuous_size > 0:
    print(f"There are {spec.action_spec.continuous_size} continuous actions")
if spec.action_spec.is_discrete():
    print(f"There are {spec.action_spec.discrete_size} discrete actions")

game_results = []
no_interrupted = 0
times = []
total_time = 0
startTotal = time.time()
ep_idx = 0
total_time_hidden = 0.0
total_reward_solo_seekers = 0.0
total_reward_solo_hiders = 0.0
total_reward_team_seekers = 0.0
total_reward_team_hiders = 0.0
for episode in range(EPISODES):
    ep_idx += 1
    print("Episode", ep_idx)
    brain_list = []
    tbi = np.random.randint(0, blue_nr)
    tpi = np.random.randint(0, purple_nr)
    tb = teams_blue[tbi]
    tp = teams_purple[tpi]
    with open(dirn_purple + tp, "r") as f:
        Lines = f.readlines()
        for l in Lines:
            brain_list.append(l.rstrip("\n"))
    with open(dirn_blue + tb, "r") as f:
        Lines = f.readlines()
        for l in Lines:
            brain_list.append(l.rstrip("\n"))
    print(f"brain_list:{brain_list}")
    # list of policies (agent brains) for both teams that will be loaded sequentially
    policies = [[], []]
    for i in range(agents):
        policies[1].append(brain_list[i])
    for i in range(agents):
        policies[0].append(brain_list[i + agents])
    print(policies)
    sc.time_hidden = 0.0
    sc.winratio = 0.0
    ort_sessions = {}
    decision_steps = {}
    terminal_steps = {}
    teampolicies = ""
    startTime = time.time()
    # decision_steps# is a vector of agents requesting decision (alive) in team
    # terminal_steps# is a vector of agents that terminated (dead) in team
    # (nested dictionary per team and per agent)
    agent_rewards = {}
    team_rewards = {}
    eliminated = {}
    for team_idx, team_id in enumerate(behavior_names):
        (
            decision_steps[team_id],
            terminal_steps[team_id],
        ) = env.get_steps(team_id)

        if args.shuffle:
            random.shuffle(policies[0])
            random.shuffle(policies[1])
        print(team_id)
        ort_sessions[team_id] = {}
        for idx, agent_id in enumerate(decision_steps[team_id]):
            curpolicy = policies[team_idx][idx]
            print(f"team {team_id}, agent id={agent_id} assigned policy {curpolicy}")
            # load onnx policies for getting an action for the current agent
            opts = onnxruntime.SessionOptions()
            opts.intra_op_num_threads = 1
            ort_sessions[team_id][agent_id] = onnxruntime.InferenceSession(
                curpolicy, sess_options=opts
            )
        for agent_id in terminal_steps[team_id]:
            print(
                f"sanity check, agent with id={agent_id} is initially in terminal step"
            )
    for team_id in behavior_names:
        agent_rewards[team_id] = {}
        team_rewards[team_id] = 0
        eliminated[team_id] = 0
    done = False
    steps = 0
    while not done:
        for team_id in behavior_names:
            # tuple for current team actions
            actions = ActionTuple()
            cont_acts = []
            disc_acts = []
            # iterate over all active agents and execute their policy on their obs vectors
            for n, agent_decision_id in enumerate(decision_steps[team_id]):
                if agent_decision_id not in ort_sessions[team_id].keys():
                    print(f"new agent {agent_decision_id} in team {team_id}")
                    done = True
                    break
                obs = decision_steps[team_id][agent_decision_id].obs
                action_mask = np.ones((1, 5))
                ort_inputs = {
                    ort_sessions[team_id][agent_decision_id]
                    .get_inputs()[0]
                    .name: np.expand_dims(obs[0], axis=0)
                    .astype(np.float32),
                    ort_sessions[team_id][agent_decision_id]
                    .get_inputs()[1]
                    .name: np.expand_dims(obs[1], axis=0)
                    .astype(np.float32),
                    ort_sessions[team_id][agent_decision_id]
                    .get_inputs()[2]
                    .name: np.expand_dims(obs[2], axis=0)
                    .astype(np.float32),
                    ort_sessions[team_id][agent_decision_id]
                    .get_inputs()[3]
                    .name: np.expand_dims(obs[3], axis=0)
                    .astype(np.float32),
                    ort_sessions[team_id][agent_decision_id]
                    .get_inputs()[4]
                    .name: action_mask.astype(np.float32),
                }
                names = [
                    ort_sessions[team_id][agent_decision_id].get_outputs()[i].name
                    for i in range(
                        len(ort_sessions[team_id][agent_decision_id].get_outputs())
                    )
                ]
                cont_ind = names.index("continuous_actions")
                det_cont_ind = names.index("deterministic_continuous_actions")
                disc_ind = names.index("discrete_actions")
                det_disc_ind = names.index("deterministic_discrete_actions")
                ort_outs = ort_sessions[team_id][agent_decision_id].run(
                    [], ort_inputs
                )
                cont_acts.append(ort_outs[det_cont_ind])
                disc_acts.append(ort_outs[det_disc_ind])
            if done:
                env.reset()
                break
            if len(decision_steps[team_id]) > 0:
                actions.add_continuous(np.vstack(cont_acts))
                actions.add_discrete(np.vstack(disc_acts))
                # Set the actions
                env.set_actions(team_id, actions)
        # Move the simulation forward
        experiment_data = []
        env.step()
        steps += 1
        for i, team_id in enumerate(behavior_names):
            # Get the new simulation results per each team
            (
                decision_steps[team_id],
                terminal_steps[team_id],
            ) = env.get_steps(team_id)

            for agent_id in decision_steps[team_id]:
                if agent_id not in agent_rewards[team_id]:
                    agent_rewards[team_id][agent_id] = 0
                else:
                    agent_rewards[team_id][agent_id] += decision_steps[team_id][
                        agent_id
                    ].reward
                team_rewards[team_id] += decision_steps[team_id].group_reward[0]

            for agent_id in terminal_steps[team_id]:
                if terminal_steps[team_id].group_reward[0] <= 0:
                    eliminated[team_id] += 1
                if agent_id not in agent_rewards[team_id]:
                    agent_rewards[team_id][agent_id] = 0
                else:
                    agent_rewards[team_id][agent_id] += terminal_steps[team_id][
                        agent_id
                    ].reward
                team_rewards[team_id] += terminal_steps[team_id].group_reward[0]
            if len(terminal_steps[team_id]) > 0:
                done = True
                if terminal_steps[team_id].interrupted[0] == True:
                    no_interrupted += 1
                    print(f"game got interrupted after {steps} steps")
            if done and i == 1:
                print(f"resetting env (file {env_file})")
                env.reset()
    endTime = time.time()
    reward_solo_hiders = (
        sum(agent_rewards["HideAndSeekAgent?team=0"].values()) / agents
    )
    reward_solo_seekers = (
        sum(agent_rewards["HideAndSeekAgent?team=1"].values()) / agents
    )
    reward_team_hiders = team_rewards["HideAndSeekAgent?team=0"] / agents
    reward_team_seekers = team_rewards["HideAndSeekAgent?team=1"] / agents
    total_reward_solo_hiders += reward_solo_hiders
    total_reward_solo_seekers += reward_solo_seekers
    total_reward_team_hiders += reward_team_hiders
    total_reward_team_seekers += reward_team_seekers
    print(f"episode {episode} game took {steps} steps in {endTime - startTime}")
    game_time = endTime - startTime
    times.append(game_time)
    game_results.append(
        (
            args.t0,
            args.t1,
            sc.time_hidden,
            reward_solo_hiders,
            reward_solo_seekers,
            game_time,
        )
    )
    total_time_hidden += sc.time_hidden
print("Avg time hidden:", total_time_hidden / EPISODES)
print("Avg hiders solo reward:", total_reward_solo_hiders / EPISODES)
print("Avg seekers solo reward:", total_reward_solo_seekers / EPISODES)
env.close()
endTotal = time.time()
print(f" there were {no_interrupted} games interrupted")
print(f" avg episode wall time: {np.mean(times)}")
print(f" total wall time={endTotal-startTotal}")

df = pd.DataFrame(
    game_results,
    columns=[
        behavior_names[0],
        behavior_names[1],
        "time hidden",
        "hiders solo reward",
        "seekers solo reward",
        "game time",
    ],
)
df.to_csv(output_file, index=False)
