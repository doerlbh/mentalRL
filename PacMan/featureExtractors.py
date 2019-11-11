# featureExtractors.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

COLLISION_TOLERANCE = 0.7 # RN ADDED: How close ghosts must be to Pacman to kill

"Feature extractors for Pacman game states"

from game import Directions, Actions
import util
from util import manhattanDistance
from util import nearestPoint   # RN ADDED: To be able to compute state feature vector

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

# RN ADDED
def closestScaredGhost(pos, scaredGhosts, walls):
    """
    closestScaredGhost -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a scared ghost at this location then exit
        for ghostPosition in scaredGhosts:  # Check if collision
            if manhattanDistance( ghostPosition, (pos_x, pos_y) ) <= COLLISION_TOLERANCE:
                return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no scared ghost found
    return None

# RN ADDED: Added more features to this to make it more sophisticated
class SimpleExtractor(FeatureExtractor):
    """
    Returns features for Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent (RN: The one step away seems to capture this directly)
    - number of scared/unscared ghosts one step away
    - distance of the closest scared ghost
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # RN ADDED: look at the state of ghosts, since it's bonus if eat scared, but killed otherwise
        ghost_states = state.getGhostStates()
        num_ghosts = len(ghost_states)

        # RN ADDED: count number of scared/unscared ghosts 1-step away
        features["#-of-unscared-ghosts-1-step-away"] = 0
        features["#-of-scared-ghosts-1-step-away"] = 0
        for g in range(num_ghosts):
            if (next_x, next_y) in Actions.getLegalNeighbors(ghosts[g], walls): # This ghost is 1-step away!
                if ghost_states[g].scaredTimer > 0:
                    features["#-of-scared-ghosts-1-step-away"] += 1
                else:
                    features["#-of-unscared-ghosts-1-step-away"] += 1

        # RN ADDED: find closest distance of scared ghost
        scaredGhostPos = []
        for g in range(num_ghosts):
            if ghost_states[g].scaredTimer > 0: # Ghost g is scared
                scaredGhostPos.append(ghosts[g])
        if scaredGhostPos:  # There is at least one scared ghost
            dist = closestScaredGhost((next_x, next_y), scaredGhostPos, walls)
            if dist is None:
                print("Why is the minimum distance from scared ghost None?")    # May want to make more sophisticated raise exception
            features["closest-scared-ghost"] = float(dist) / (walls.width * walls.height)

        # # count the number of ghosts 1-step away
        # features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if food[next_x][next_y]:#not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

# RN ADDED: Similar to SimpleExtractor, except that returns a vector instead of dictionary of features
class SAVectorExtractor(FeatureExtractor):
    """
    Returns features for Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - number of scared/unscared ghosts one step away
    - distance of the closest scared ghost
    """

    def getFeatures(self, state, action):

        # extract the grid of food and wall locations
        food = state.getFood()
        walls = state.getWalls()

        # get the ghost & pacman locations
        ghosts = state.getGhostPositions()
        x, y = state.getPacmanPosition()

        # look at the state of ghosts, since it's bonus if eat scared, but killed otherwise
        ghost_states = state.getGhostStates()
        num_ghosts = len(ghost_states)

        # constuct list with positions of all scared ghosts
        scaredGhostPos = []
        for g in range(num_ghosts):
            if ghost_states[g].scaredTimer > 0:  # Ghost g is scared
                scaredGhostPos.append(ghosts[g])

        feat_vec = []
        feat_vec.append(1.0)    # Adding a bias term

        # compute the location of pacman after he takes the action
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count number of scared/unscared ghosts 1-step away, i.e. if there's a ghost action that leads to contact
        numUnscared_1step = 0
        numScared_1step = 0
        for g in range(num_ghosts):
            if (next_x, next_y) in Actions.getLegalNeighbors(ghosts[g], walls):  # This ghost is 1-step away!
                if ghost_states[g].scaredTimer > 0:
                    numScared_1step += 1
                else:
                    numUnscared_1step += 1
        feat_vec.append(numUnscared_1step)
        feat_vec.append(numScared_1step)

        # find closest distance of scared ghost
        closestGhost = 0.0
        if scaredGhostPos:  # There is at least one scared ghost
            dist = closestScaredGhost((next_x, next_y), scaredGhostPos, walls)
            closestGhost = float(dist) / (walls.width * walls.height)
        feat_vec.append(closestGhost)

        # whether food is consumed
        eats_food = 0.0
        if food[next_x][next_y]:
            eats_food = 1.0
        feat_vec.append(eats_food)

        # distance of the closest food
        food_dist = 1.0 # set extremely far by default
        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update will diverge wildly
            food_dist = float(dist) / (walls.width * walls.height)
        feat_vec.append(food_dist)

        num_feat = len(feat_vec)
        for i in range(num_feat):
            feat_vec[i] /= 10.0

        return feat_vec



# RN ADDED: gets features for (s,a,s') pair that are to be used as features for the reward function (whose weights are learnt by IRL)
class RewardFeatureExtractor:
    """
    Returns features of (s,a,s') that are to be used as features for the reward function
    - a bias term, constant 1
    - whether food has been eaten
    - whether pacman has won
    - number of scared ghosts eaten
    - whether collide wth unscared ghost, i.e. whether pacman has lost
    [Not including the spurious features below initially]
    - how far away the next food is
    - distance of the closest scared ghost
    - distance of the closest unscared ghost
    """

    def getFeatures(self, state, action, new_state):
        num_feat = 5
        bias_ind = 0
        food_ind = 1
        won_ind = 2
        ghostsEaten_ind = 3
        lost_ind = 4
        # Not including the spurious features right now

        features = [0.0]*num_feat

        features[bias_ind] = 1.0

        features[food_ind] = state.getNumFood() - new_state.getNumFood()    # whether food was eaten in this step

        if new_state.data._win:
            features[won_ind] = 1.0

        for ind in range(1, len(new_state.data._eaten)):
            if new_state.data._eaten[ind]:
                features[ghostsEaten_ind] += 1

        if new_state.data._lose:
            features[lost_ind] = 1

        return features

# RN ADDED: gets features for (s,a,s') pair that are to be used as features for the reward function (whose weights are learnt by IRL)
class RewardNoBiasFeatureExtractor:
    """
    Returns features of (s,a,s') that are to be used as features for the reward function
    - no bias term
    - whether food has been eaten
    - whether pacman has won
    - number of scared ghosts eaten
    - whether collide wth unscared ghost, i.e. whether pacman has lost
    [Not including the spurious features below initially]
    - how far away the next food is
    - distance of the closest scared ghost
    - distance of the closest unscared ghost
    """

    def getFeatures(self, state, action, new_state):
        num_feat = 4
        food_ind = 0
        won_ind = 1
        ghostsEaten_ind = 2
        lost_ind = 3
        # Not including the spurious features right now

        features = [0.0]*num_feat

        features[food_ind] = state.getNumFood() - new_state.getNumFood()    # whether food was eaten in this step

        if new_state.data._win:
            features[won_ind] = 1.0

        for ind in range(1, len(new_state.data._eaten)):
            if new_state.data._eaten[ind]:
                features[ghostsEaten_ind] += 1

        if new_state.data._lose:
            features[lost_ind] = 1

        return features


# RN ADDED: gets features for a state, to be used for a neural network
class StateFeatureExtractor:

    # BL ADDED: to make it compatible for IRL
    # def getFeatures(self, state):
    def getFeatures(self, state, action):
        """
          Returns a vector of features, used as input for the Deep Q Network. A complete one would contain
          - positions of all food particles
          - positions of all walls
          - coordinates of all ghosts
          - coordinates of pacman
        """
        util.raiseNotDefined()

# RN ADDED: gets features of a state, as contents of every grid square
class MapExtractor(StateFeatureExtractor):
    """
    Returns the map of the current state, with contents of each grid square
    """

    # BL ADDED: to make it compatible for IRL
    # def getFeatures(self, state):
    def getFeatures(self, state, action):

        # extract the grid of food and wall locations, as well as capsule coordinates
        food = state.getFood()
        walls = state.getWalls()
        capsules = state.getCapsules()

        # Get pacman position rounded to nearest grid point
        pacman_pos = nearestPoint(state.getPacmanPosition())

        # Get ghosts state, and their rounded grid positions
        ghosts = state.getGhostStates()
        ghosts_pos = [nearestPoint(g.getPosition()) for g in ghosts]
        ghosts_num = len(ghosts)

        width = state.data.layout.width
        height = state.data.layout.height

        # The origin (0,0) of walls, ghosts, etc is positioned at the bottom left corner of PacMan board

        # Using feature vector as all locations of board, and value (x,y) of this feature would be "what" is there in this location, i.e. either food, capsule, pacman, unscared ghost, scared ghost, wall or empty
        # Current mapping:
        # 0 - unscared ghost
        # 1 - scared ghost
        # 2 - wall
        # 3 - empty
        # 4 - pacman position
        # 5 - capsule
        # 6 - food

        feat_vec = []
        for x in range(width):
            for y in range(height):
                if (x,y) == pacman_pos:
                    feat_vec.append(4)
                elif (x,y) in ghosts_pos:
                    ghost_type = 1  # Use default ghost type as "scared" ghost
                    for ind in range(ghosts_num):
                        if (x,y) == ghosts_pos[ind]:
                            if ghosts[ind].scaredTimer == 0:
                                ghost_type = 0  # If there's even one unscared ghost in the grid location, switch to unscared ghost type
                                break
                    feat_vec.append(ghost_type)
                elif (x,y) in capsules:
                    feat_vec.append(5)
                elif walls[x][y]:
                    feat_vec.append(2)
                elif food[x][y]:
                    feat_vec.append(6)
                else:
                    feat_vec.append(3)


        # # Stuff just to print the feature vector on screen
        # for y in range(height-1, -1, -1):
        #     line_str = ""
        #     for x in range(width):
        #         if (x,y) == pacman_pos:
        #             line_str += '4'
        #         elif (x,y) in ghosts_pos:
        #             ghost_type = 1  # Use default ghost type as "scared" ghost
        #             for ind in range(ghosts_num):
        #                 if (x,y) == ghosts_pos[ind]:
        #                     if ghosts[ind].scaredTimer == 0:
        #                         ghost_type = 0  # If there's even one unscared ghost in the grid location, switch to unscared ghost type
        #                         break
        #             line_str += str(ghost_type)
        #         elif (x,y) in capsules:
        #             line_str += '5'
        #         elif walls[x][y]:
        #             line_str += '2'
        #         elif food[x][y]:
        #             line_str += '6'
        #         else:
        #             line_str += '3'
        #     print line_str

        # An alternative feature representation is to give the boolean vector of all food, of all walls, etc, coordinates of pacman, coordinates and direction of ghosts, etc

        return feat_vec

# RN ADDED: gets features of a state, using relevant information of every state-action pair
class ActionPairsExtractor(StateFeatureExtractor):
    """
    Returns features of a state using relevant information of every state-action pair
    """

    # BL ADDED: to make it compatible for IRL
    # def getFeatures(self, state):
    def getFeatures(self, state, action):

        # extract the grid of food and wall locations
        food = state.getFood()
        walls = state.getWalls()

        # get the ghost & pacman locations
        ghosts = state.getGhostPositions()
        x, y = state.getPacmanPosition()

        # look at the state of ghosts, since it's bonus if eat scared, but killed otherwise
        ghost_states = state.getGhostStates()
        num_ghosts = len(ghost_states)

        # constuct list with positions of all scared ghosts
        scaredGhostPos = []
        for g in range(num_ghosts):
            if ghost_states[g].scaredTimer > 0:  # Ghost g is scared
                scaredGhostPos.append(ghosts[g])

        all_actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
        feat_vec = []

        # feat_vec.append(1.0)    # Adding a bias term  # Not required, since neural net automatically adds biases

        for action in all_actions:

            # compute the location of pacman after he takes the action
            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)

            # count number of scared/unscared ghosts 1-step away, i.e. if there's a ghost action that leads to contact
            numUnscared_1step = 0
            numScared_1step = 0
            for g in range(num_ghosts):
                if (next_x, next_y) in Actions.getLegalNeighbors(ghosts[g], walls):  # This ghost is 1-step away!
                    if ghost_states[g].scaredTimer > 0:
                        numScared_1step += 1
                    else:
                        numUnscared_1step += 1
            feat_vec.append(numUnscared_1step)
            feat_vec.append(numScared_1step)

            # find closest distance of scared ghost
            closestGhost = 0.0
            if scaredGhostPos:  # There is at least one scared ghost
                dist = closestScaredGhost((next_x, next_y), scaredGhostPos, walls)
                closestGhost = float(dist) / (walls.width * walls.height)
            feat_vec.append(closestGhost)

            # whether food is consumed
            eats_food = 0.0
            if food[next_x][next_y]:
                eats_food = 1.0
            feat_vec.append(eats_food)

            # distance of the closest food
            food_dist = 1.0 # set extremely far by default
            dist = closestFood((next_x, next_y), food, walls)
            if dist is not None:
                # make the distance a number less than one otherwise the update will diverge wildly
                food_dist = float(dist) / (walls.width * walls.height)
            feat_vec.append(food_dist)

        num_feat = len(feat_vec)
        for i in range(num_feat):
            feat_vec[i] /= 10.0

        return feat_vec
