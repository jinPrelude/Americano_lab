- 본 내용은 아메리카노 팀의 목표 "스타크레프트 인공지능 제작"을 성취하기 위한 연습의 일환입니다. 중간 과정이 도움이 될 것이라고 생각하여 올립니다! Continuous space 환경에서 DDPG 알고리즘을 사용하였고, 코드를 이해하기 쉽도록 주석을 달아놓았습니다. 의식의 흐름대로 쓴 주석이라 지속적으로 수정하도록 하겠습니다ㅎ
- 본 주소 : https://github.com/jinprelude/reinforcement_learning

# Pendulum ai using DDPG algorithm
<div align="center">
	<img src=./readme/pendulum.gif width="600px">
</div>
This is a simple implementation of DDPG algorithm.

- DDPG paper : https://arxiv.org/abs/1509.02971
- reference : https://github.com/pemami4911/deep-rl/tree/master/ddpg

## Requirements
- To run this project, you need gym, numpy, absl, and tensorflow.
```shell
pip install gym
pip install numpy
pip install absl-py
pip install tensorflow
```

## Getting Started
Clone this repo :
```shell
git clone https://github.com/jinprelude/reinforcement_learning
```

Go to the DDPG directory :
```shell
cd DDPG/Pendulum
```

### Training
- You should make directory 'results' before training :
```shell
mkdir results
python3 main.py
```
training will terminate if the mean value of the last episodes' rewards is higher than -300. Rendered video will be shown once before the termination.

### Testing
- After training the model, run model_test.py to see it works.
```shell
python3 model_test.py
```

- You can also check the tensorboard :
```shell
tensorboard --logdir=./results/tf_ddpg
```
<div align="center">
	<img src=./readme/DDPG_Pendulum_130_iteration_Qmax.png width="270px">
	<img src=./readme/DDPG_Pendulum_130_iteration_reward.png width="270px">
</div>




