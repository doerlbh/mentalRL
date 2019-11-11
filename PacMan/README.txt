
# example usage

python2.7 pacman.py -p QL -a extractor=SimpleExtractor,gamma=0.8,alpha=0.2 -x 10 -n 10 -l smallClassic
python2.7 pacman.py -p DQL -a extractor=SimpleExtractor,gamma=0.8,alpha=0.2 -x 10 -n 10 -l smallClassic
python2.7 pacman.py -p SQL -a extractor=SimpleExtractor,gamma=0.8,alpha=0.2 -x 10 -n 10 -l smallClassic
python2.7 pacman.py -p PQL -a extractor=SimpleExtractor,gamma=0.8,alpha=0.2 -x 10 -n 10 -l smallClassic
python2.7 pacman.py -p NQL -a extractor=SimpleExtractor,gamma=0.8,alpha=0.2 -x 10 -n 10 -l smallClassic

# for mental agents, e.g. AD

python2.7 pacman.py -p SQL -a extractor=SimpleExtractor,gamma=0.8,alpha=0.2,p1=1,p2=1,n1=0.5,n2=1 -x 1000 -n 1000 -l mediumClassic

# Notes:

-p flag defines which pacman agent to use
-a flag defines arguments for the agent to use. For approximate Q-learning, the main ones would be
    extractor: which feature extractor to use for the (s,a) pairs
    epsilon: epsilon parameter of the epsilon-greedy exploration
    alpha: learning rate in the q-learning
    gamma: discount factor to use
-x flag is the number of training games play
-n flag is the total number of games to play
-l defines the file name to pick out the layout from

