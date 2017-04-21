# Evolution Strategies

This is a PyTorch implementation of [Evolution Strategies](https://arxiv.org/abs/1703.03864). All testing was done with PyTorch 0.1.10 and Python 2.7.

# Previous work

This repository is an adaptation of this one: https://github.com/atgambardella/pytorch-es

# News

I have developed an Tic-tac-toe environment for the ES training process. As this game can be studied in full depth by a classical min-max tree, I've used this classic AI to move against our neural network model.
The last result is a model that thanks to the evolutionary computation can reach a zero-perfect game against the classical AI brute force strategy.

# Requirements

Python 2.7, PyTorch >= 0.1.10, numpy

# What is this? (For non-ML people)

A large class of problems in AI can be described as "Markov Decision Processes," in which there is an agent taking actions in an environment, and receiving reward, with the goal being to maximize reward. This is a very general framework, which can be applied to many tasks, from learning how to play video games to robotic control. For the past few decades, most people used Reinforcement Learning -- that is, learning from trial and error -- to solve these problems. In particular, there was an extension of the backpropagation algorithm from Supervised Learning, called the Policy Gradient, which could train neural networks to solve these problems. Recently, OpenAI had shown that black-box optimization of neural network parameters (that is, not using the Policy Gradient or even Reinforcement Learning) can achieve similar results to state of the art Reinforcement Learning algorithms, and can be parallelized much more efficiently. This repo is an implementation of that black-box optimization algorithm.

# Usage

Run `python main.py --help` to see all of the options and hyperparameters available to you.

Typical usage would be:

```
python main.py
```
which will run the training process untill the perfect zero-game is reached by our neural network.

```
python main.py --test --restore ./checkpoints/latest.pth
```
which will render the environment and the performance of the agent saved in the checkpoint. Checkpoints are saved once per gradient update in training, always overwriting the old file.

# Tips

* If you increase the batch size, `n`, you should increase the learning rate as well.

* Sigma is a tricky hyperparameter to get right -- higher values of sigma will correspond to less variance in the gradient estimate, but will be more biased. At the same time, sigma is controlling the variance of our perturbations, so if we need a more varied population, it should be increased. It might be possible to adaptively change sigma based on the rank of the unperturbed model mentioned in the tip above. I tried a few simple heuristics based on this and found no significant performance increase, but it might be possible to do this more intelligently.

# Contributions

Please feel free to make Github issues or send pull requests.

# License

MIT
