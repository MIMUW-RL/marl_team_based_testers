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

parser = argparse.ArgumentParser()
parser.add_argument("env", help="env file")
parser.add_argument("-p", "--Port", type=int, help="Port")
parser.add_argument("-s", "--Seed", type=int, help="Seed")
parser.add_argument("-e", "--Episodes", type=int, help="Number of epsiodes to play")
parser.add_argument("-a", "--Agents", type=int, help="Number of agents in one team")
parser.add_argument(
    "-t0", help="optional dir with txt with policies for team0 (blue side)"
)
parser.add_argument(
    "-t1", help="optional dir with txt with policies for team1 (purple side)"
)
parser.add_argument(
    "--shuffle",
    help="optional if the team chp's should be shuffled",
    action="store_true",
)
parser.add_argument("--outdir", default="", required=False)

args = parser.parse_args()
Port = 6000
seed = 1
if args.Port:
    Port = args.Port
    seed = Port
if args.Episodes:
    EPISODES = args.Episodes
if args.Seed:
    seed = args.Seed
if args.Agents:
    agents = args.Agents

if os.path.isdir(args.t0):
    t0name = args.t0.split("/")[-2]
else:
    t0name = args.t0.split("/")[-1]
if os.path.isdir(args.t1):
    t1name = args.t1.split("/")[-2]
else:
    t1name = args.t1.split("/")[-1]
output_file = os.path.join(
    args.outdir,
    t0name + "_vs_" + t1name + "_" + str(seed % 1000) + ".csv",
)
print(f"output_file={output_file}")

if os.path.exists(output_file):
    print(f"the file with output {output_file} already exists exiting script")
    exit(1)

brain_list = []
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
env = UnityEnvironment(
    file_name=env_file,
    no_graphics=True,
    base_port=Port,
    seed=seed,
)

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
result = {behavior_names[0]: 0, behavior_names[1]: 0}
no_interrupted = 0
times = []
total_time = 0
startTotal = time.time()
for episode in range(EPISODES):
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
        policies[0].append(brain_list[i])
    for i in range(agents):
        policies[1].append(brain_list[i + agents])
    print(policies)
    ort_sessions = {}
    decision_steps = {}
    terminal_steps = {}
    teampolicies = ""
    startTime = time.time()
    # decision_steps# is a vector of agents requesting decision (alive) in team
    # (nested dictionary per team and per agent)
    # terminal_steps# is a vector of agents that terminated (dead) in team
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
                f"sanity check, agent with id={agent_id} is initially in terminal step, what that means?"
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
                action_mask = np.ones((1, 4))
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
                    .name: np.expand_dims(obs[4], axis=0)
                    .astype(np.float32),
                    ort_sessions[team_id][agent_decision_id]
                    .get_inputs()[5]
                    .name: np.expand_dims(obs[5], axis=0)
                    .astype(np.float32),
                    ort_sessions[team_id][agent_decision_id]
                    .get_inputs()[6]
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
                print(f"resetting env (file {env_file})")
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
        if steps % 1000 == 0:
            print(f"step no {steps}")
        for i, team_id in enumerate(behavior_names):
            # Get the new simulation results per each team
            decision_steps[team_id], terminal_steps[team_id] = env.get_steps(team_id)
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
                    print(
                        f"agent from team {team_id} with id={agent_id} was eliminated, total eliminations={eliminated[team_id]}"
                    )
                if agent_id not in agent_rewards[team_id]:
                    agent_rewards[team_id][agent_id] = 0
                else:
                    agent_rewards[team_id][agent_id] += terminal_steps[team_id][
                        agent_id
                    ].reward
                team_rewards[team_id] += terminal_steps[team_id].group_reward[0]
            if len(terminal_steps[team_id]) > 0:
                if terminal_steps[team_id].interrupted[0] == True:
                    no_interrupted += 1
                    print(f"game got interrupted after {steps} steps")
                    done = True
                    team_won = -1
            if team_rewards[team_id] > 0:
                done = True
                team_won = int(team_id[-1])
                print(f"EOG team {team_id} WON! (step {steps})")
                result[team_id] += 1
                print(
                    "sanity check , this should always be vector size #bots_left with positive team reward entry"
                )
                print(terminal_steps[team_id].group_reward)
                print(f"current total score={result}")
            if done and i == 1:
                print(f"resetting env (file {env_file})")
                env.reset()
    endTime = time.time()
    print(f"episode {episode} game took {steps} steps in {endTime - startTime}")
    game_time = endTime - startTime
    times.append(game_time)
    print(eliminated)
    game_results.append(
        (
            team_won,
            eliminated[behavior_names[1]],
            eliminated[behavior_names[0]],
            game_time,
        )
    )
env.close()
endTotal = time.time()
print(f"final scores {result}, avg episode wall time: {np.mean(times)}")
print(f" thwere were {no_interrupted} games interrupted")
print(f"total wall time={endTotal-startTotal}")

df = pd.DataFrame(
    game_results, columns=["team_won", "eliminated_0", "eliminated_1", "time"]
)
df.to_csv(output_file, index=False)
