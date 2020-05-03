# mentalRL


![mentalRL](./img/mentalRL.png "mentalRL")

(image credit to [HBR](https://hbr.org/2018/10/ais-potential-to-diagnose-and-treat-mental-illness))

 

Code for our *AAMAS 2020* paper: 

**"A Story of Two Streams: Reinforcement Learning Models from Human Behavior and Neuropsychiatry"** 

by [Baihan Lin](http://www.columbia.edu/~bl2681/) (Columbia), [Guillermo Cecchi](https://researcher.watson.ibm.com/researcher/view.php?person=us-gcecchi) (IBM Research), [Djallel Bouneffouf](https://scholar.google.com/citations?user=i2a1LUMAAAAJ&hl=en) (IBM Research), [Jenna Reinen](http://campuspress.yale.edu/jennareinen/) (IBM Research) and [Irina Rish](https://sites.google.com/site/irinarish/) (Mila, UdeM). 



For the latest full paper: https://arxiv.org/abs/1906.11286

For my oral talk at AAMAS 2020: https://youtu.be/CQBdQz1bmls



All the experimental results can be reproduced using the code in this repository. Feel free to contact me by doerlbh@gmail.com if you have any question about our work.


**Abstract**


Drawing an inspiration from behavioral studies of human decision making, we propose here a more general and flexible parametric framework for reinforcement learning that extends standard Q-learning to a two-stream model for processing positive and negative rewards, and allows to incorporate a wide range of reward-processing biases -- an important component of human decision making which can help us better understand a wide spectrum of multi-agent interactions in complex real-world socioeconomic systems, as well as various neuropsychiatric conditions associated with disruptions in normal reward processing. From the computational perspective, we observe that the proposed Split-QL model and its clinically inspired variants consistently outperform standard Q-Learning and SARSA methods, as well as recently proposed Double Q-Learning approaches, on simulated tasks with particular reward distributions, a real-world dataset capturing human decision-making in gambling tasks, and the Pac-Man game in a lifelong learning setting across different reward stationarities.


## Info

Language: Python3, Python2, bash


Platform: MacOS, Linux, Windows

by Baihan Lin, Sep 2018


## Citation

If you find this work helpful, please try the models out and cite our works. Thanks!

**Reinforcement Learning case** (main paper):

    @inproceedings{lin2020astory,
      title={A Story of Two Streams: Reinforcement Learning Models from Human Behavior and Neuropsychiatry},
      author={Lin, Baihan and Bouneffouf, Djallel and Reinen, Jenna and Rish, Irina and Cecchi, Guillermo},
      booktitle = {Proceedings of the Nineteenth International Conference on Autonomous Agents and Multi-Agent Systems, {AAMAS-20}},
      publisher = {International Foundation for Autonomous Agents and Multiagent Systems},             
      pages     = {},
      year      = {2020},
      month     = {5},
      doi       = {},
      url       = {},
    }

**Contextual Bandit case:**

    @inproceedings{lin2020unified,
      title={Unified Models of Human Behavioral Agents in Bandits, Contextual Bandits, and RL},
      author={Lin, Baihan and Bouneffouf, Djallel and Cecchi, Guillermo},
      booktitle={under review},
      pages={},
      year={},
      organization={}
    }

## Tasks

* Markov Decision Process (MDP) example with multi-modal reward distributions
* Multi-Armed Bandits (MAB)  example with multi-modal reward distributions 
* Iowa Gambling Task (IGT) example scheme 1 and 2
* PacMan RL game with different stationarities



## Requirements

* Python 3 for MDP and IGT tasks, and Python 2.7 for PacMan task.
* [PyTorch](http://pytorch.org/)
* numpy and scikit-learn



## Videos of mental agents playing PacMan



* AD ("Alzheimer's Disease")

![AZ](./img/AD.gif "AD")


* ADD ("addition")

![ADD](./img/ADD.gif "ADD")


* ADHD ("ADHD")

![ADHD](./img/ADHD.gif "ADHD")


* bvFTD (the behavioral variant of Frontotemporal dementia)

![bvFTD](./img/bvFTD.gif "bvFTD")


* CP ("Chronic Pain")

![CP](./img/CP.gif "CP")


* PD ("Parkinson's Disease")

![PD](./img/PD.gif "PD")


* M ("moderate")

![M](./img/M.gif "M")


* SQL ("Split Q-Learning")

![SQL](./img/SQL.gif "SQL")


* PQL ("Positive Q-Learning")

![PQL](./img/PQL.gif "PQL")


* NQL ("Negative Q-Learning")

![NQL](./img/NQL.gif "NQL")


* QL ("Q-Learning")

![QL](./img/QL.gif "QL")


* DQL ("Double Q-Learning")

![DQL](./img/DQL.gif "DQL")



## Acknowledgements 

The PacMan game was built upon Berkeley AI Pac-Man http://ai.berkeley.edu/project_overview.html. We modify many of the original files and included our comparison.

