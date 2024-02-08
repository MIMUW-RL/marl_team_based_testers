## Python testers for team-based MARL environments
## Unity ML-Agents compatible

We present python scripts for testing agents in team-based MARL environments. 
Currently, the scripts are [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) compatible, 
however will work with any compiled team-based MARL environment with similar Python interface. 
The scripts fills the gap for team-based Unity ML-Agents MARL, where a proper testing framework has been
unavailable, and allows for comprehensive and thorough tests of agents in team-based fashion, in environments
like the cool [Unity Dodgeball](https://blog.unity.com/engine-platform/ml-agents-plays-dodgeball), and our 
new custom [Hide \& Seek and Predator-Prey environments](https://github.com/MIMUW-RL/unity-ml-agents_hide-and-seek).

### Usage

Teams are defined through input .txt files that indicate the behavioral policy checkpoint for each agent. 
These checkpoints are in the Open Neural Network Exchange (ONNX) format, ensuring compactness and framework independence.
Our script adeptly manages mixed discrete-continuous actions and varying agent numbers, efficiently handling in-game events like agent eliminations.
We include example txt teams and agent checkpoints in [examples/](examples/). Notably, non-uniform teams compositions 
can be also tested, allowing for testing out-of-distribution agent generalization capabilities. 

#### Unity Dodgeball Elimination
`
python unity_dodgeball_tester.py path_to_environment -e 1 -a 4 -t0 examples/dodgeball_elimination_team0.txt -t1 examples/dodgeball_elimination_team1.txt
`

where `path_to_environment` is the path to `elimination.x86_64` -- the Dodgeball Elimination compiled test environment, [downloadable from here](https://drive.google.com/drive/folders/1K0f3o4wpg87EaijXJEYmcZraOpwvtYif?usp=drive_link),
`-t0` and `-t1` are the paths to txt files with the team composition (`t0`=the blue team, `t1`=the purple team in this context), using the provided example checkpoints.

#### Unity Dodgeball Capture the Flag

`
python unity_dodgeball_tester.py path_to_environment -e 1 -a 3 -t0 examples/dodgeball_ctf_team0.txt -t1 examples/dodgeball_ctf_team1.txt
`
where `path_to_environment` is the path to `ctf_1arena.x86_64` -- the compiled DB CTF test environment, [downloadable from here](https://drive.google.com/drive/folders/1K0f3o4wpg87EaijXJEYmcZraOpwvtYif?usp=drive_link),
`-t0` and `-t1` are the paths to txt files with the team composition (`t0`=the blue team, `t1`=the purple team in this context), using the provided example checkpoints.

#### Hide \& Seek environment (ours)

`
python3 hide_and_seek_tester.py path_to_environment -e 1 -a 3 -t0 examples/hide_seek_team0.txt -t1 examples/hide_seek_team1.txt
`
where `path_to_environment` is the path to `hidenseek.x86_64` -- the compiled Hide \& Seek test environment, [downloadable from here](https://drive.google.com/drive/folders/1K0f3o4wpg87EaijXJEYmcZraOpwvtYif?usp=drive_link),
`-t0` and `-t1` are the paths to txt files with the team composition (`t0`=the hiders team, `t1`=the seekers team in this context), using the provided example checkpoints.

#### Predator-Prey environment (ours)

`
python predator_prey_tester.py path_to_environment -e 1 -a 3 -t0 examples/predator_prey_team0.txt -t1 examples/predator_prey_team1.txt --game_params test_environments/predator_prey_configs/game_params.json --arena_params test_environments/predator_prey_configs/arena_params.json
`
where `path_to_environment` is the path to `predprey.x86_64` -- the compiled Predator-Prey test environment, [downloadable from here](https://drive.google.com/drive/folders/1K0f3o4wpg87EaijXJEYmcZraOpwvtYif?usp=drive_link),
`-t0` and `-t1` are the paths to txt files with the team composition (`t0`=the prey team, `t1`=the predators team in this context), using the provided example checkpoints.

This environment requires additionally, the path to the config json files through the arguments `--game_params`, `--arena_params`, see for more details the 
[Hide \& Seek and Predator-Prey environments](https://github.com/MIMUW-RL/unity-ml-agents_hide-and-seek).

Testers can be used to reproduce results presented in the submitted paper 'FCSP: Fictitious Co-Self-Play for Team-based, Multi-agent Reinforcement Learning', preprint available on request.
