#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import math
import az_quiz
from typing import List
from functools import reduce
import time
import os


class AlphaZeroConfig(object):
    def __init__(self):
        ### Self-Play
        self.num_sampling_moves = 0  # 30  # tells how many game moves will be possibly random
        self.max_moves = 30  # depth of the search 28 for az-kviz 512 for chess and shogi, 722 for Go.
        self.num_simulations = 500  # 800 number of paths in mcts

        # Root prior exploration noise.
        self.root_dirichlet_alpha = 0.03  # for chess, 0.03 for Go and 0.15 for shogi.
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652  # dle videa 15000
        self.pb_c_init = 1.25  # dle videa 2

        self.learning_rate = 0.002
        self.network = Network(self)

        self.init_player = -1
        self.games_per_training = 20  # nr of games for each training

        self.already_reached_end = False


class Node(object):
    def __init__(self, prior: float = None):
        self.visit_count = 0
        self.to_play = -1  # None
        self.prior = prior if prior is not None else 0.03571  # 1/28
        self.value_sum = 0  # "sum" of value function of the children
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return self.value_sum
        return self.value_sum / self.visit_count


class Network:
    def __init__(self, args):
        # Define suitable model. Apart from the model defined in `reinforce`,
        # define also another model `baseline`, which produces one output
        # (using a dense layer without activation).
        #
        # Use Adam optimizer with given `args.learning_rate` for both models.

        actions = 28
        inputs = actions + 1

        hidden_layer_size = 64

        input_layer = tf.keras.layers.Input(inputs)
        hidden_layer = tf.keras.layers.Dense(hidden_layer_size, activation='softmax')(input_layer)
        output_layer = tf.keras.layers.Dense(actions, )(hidden_layer)
        output_evaluation = tf.keras.layers.Dense(1, )(hidden_layer)
        outputs = [output_layer, output_evaluation]
        self.model = tf.keras.Model(inputs=[input_layer], outputs=outputs)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            experimental_run_tf_function=False
        )

        ###0###print(self.model.summary())

    def train(self, states, actions, returns):
        actions = np.array(list(reduce(lambda a, b: np.concatenate((a, b)), actions)))
        returns = np.array(list(reduce(lambda a, b: np.concatenate((a, b)), returns)))
        states = np.array(list(reduce(lambda a, b: np.concatenate((a, b)), states)))

        weight = returns  # - baselines (if there is baseline available)
        self.model.train_on_batch(states, actions, sample_weight=weight)

    def predict(self, states_in):
        states = np.array([states_in])

        result = self.model.predict(states)
        # first actions, then quality

        return result[0:1], result[1][0][0]  # it is double array of one value [[v]]

    @staticmethod
    def file_location():
        return os.path.realpath(__file__) + "_trained_network.txt"

    def save_network(self, number=0):
        filepath = self.file_location()
        f = open(filepath, 'a')
        f.write(time.localtime())
        if number != 0:
            f.write('saved at step: ' + str(number))
        f.write('\n')
        f.write(self.model.get_weights())
        f.write('\n')
        f.close()

    def load_latest_network(self):
        filepath = self.file_location()
        line = ""
        with open(filepath, 'rU') as f:
            for tmpLine in f:
                print(tmpLine)
                line = tmpLine

        weights = self.parse_weights(line)
        self.model.set_weights(weights)

    @staticmethod
    def parse_weights(line):
        return [0] * 28  # todo parse the line


def get_readible_date():
    t = time.localtime()
    return f"{t.tm_year}_{t.tm_mon}_{t.tm_yday}_{t.tm_hour}:{t.tm_min}:{t.tm_sec}"


##TRAINING CODE FROM ALPHA ZERO
def train_network(config: AlphaZeroConfig, batch):
    network = config.network
    optimizer = tf.train.MomentumOptimizer(config.learning_rate_schedule,
                                           config.momentum)
    for i in range(config.training_steps):
        update_weights(optimizer, network, batch, config.weight_decay)


def update_weights(optimizer: tf.optimizers, network: Network, batch, weight_decay: float):
    loss = 0
    for (state, target_policy, target_value) in batch:
        policy_logits, value = network.predict(state)
        loss += (
                tf.losses.mean_squared_error(value, target_value) +
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=policy_logits, labels=target_policy))

    for weights in network.get_weights():
        loss += weight_decay * tf.nn.l2_loss(weights)

    optimizer.minimize(loss)


# ######### End Training ###########
# ##################################


# AlphaZero training is split into two independent parts: Network training and self-play data generation.
def alphazero(config: AlphaZeroConfig):
    network = config.network

    i = 0
    save_at_step = 1
    histories = []  # game simulation takes a long time - so remember everything and train over and over on past games

    while True and i < 20:  # 10_000_000:
        i += 1
        # TODO while enough time OR not sufficient quality do [selfplay (even multiple for batch) -> train cycle]
        for i in range(config.games_per_training):
            history = play_game(config)

            network.predict(history)
            histories.append(history)

        # history :- game, state, action     //   game.clone(), game2array(game), action

        # saved :- state, policy, value

        # state - DONE
        # policy - je to action? nn -- je to output values??? tzn 28 length?
        # myslim si ze ano, ma to byt to co vraci jednotlive nody - jako node .value pro sve deti
        # value - target_value - node value

        if i > save_at_step:
            network.save_network(save_at_step)
            save_at_step *= 2

        # saved je pole -> trojic (state, target_policy, target_value) # TODO create this 3-tuple
        train_network(config, saved)

    network.save_network()
    print('saving network to:', Network.file_location())

    return


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: AlphaZeroConfig):
    game = az_quiz.AZQuiz(randomized=False)
    moves = 0
    history = []
    while not game.winner and moves < config.max_moves:
        moves += 1
        action, root = run_mcts(config, game)
        # TODO chybi tady simulace
        history.append(game.clone(), (game2array(game), action, root))  # root probably always the same...
    return history


def softmax_sample_index(z):
    """Compute softmax values for each sets of scores in x."""

    x = [z[0] for z in z]
    actions = [z[0] for z in z]

    result = np.exp(x) / np.sum(np.exp(x))

    probabilities = [0 if p < 0 else p for p in result]
    sum_p = sum(probabilities)
    if sum_p == 0:
        index = np.random.choice(len(probabilities))
    else:
        indexes = list(range(len(probabilities)))
        probabilities = list(map(lambda p: p / sum_p, probabilities))
        index = np.random.choice(indexes, 1, p=probabilities)[0]
    return actions[index]


def max_arg(x):
    maximum = -1
    maxi = -1
    for val, index in x:
        # print(val, index, max, maxi)
        if val > maximum:
            maximum = val
            maxi = index

    return maxi


def print_children(node: Node):
    print(node.children.keys())
    reduce(lambda x: x, [1])


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: AlphaZeroConfig, game: az_quiz):
    player = game.to_play

    root = Node(0)
    evaluate(root, game, config, player)
    add_exploration_noise(config, root)

    # print("run_mcts")
    # print_children(root)
    runs = 0

    for _ in range(config.num_simulations):
        node = root
        scratch_game = game.clone()
        search_path = [node]
        # search_path_actions = []

        # print("\nnew path")
        while node.expanded():
            action, node = sample_random_child(config, node)
            scratch_game.move(action)

            runs += 1
            search_path.append(node)
            # search_path_actions.append(action)

        value = evaluate(node, scratch_game, config, player)
        # print("chosen actinos:", search_path_actions)
        # print_children(node)
        backpropagate(search_path, value, player, scratch_game.winner)

    # print_mstc_tree(root)
    return select_action(config, root, runs), root


def print_mstc_node_info(node: Node, action, spaces):
    # print(spaces)
    print(spaces * " " + str(action) + ":-> val:" + str(node.value()) + ":sum=" + str(node.value_sum) + "visits:" + str(
        node.visit_count))


def print_mstc_tree(root: Node, action=None, spaces=0):
    # as a path in dir system
    if action is None:
        # print("root print")
        action = ""
    print_mstc_node_info(root, action, spaces)

    sublevel = spaces + 2
    for action, child in root.children.items():
        # print("something")
        print_mstc_tree(child, action, sublevel)


def select_action(config: AlphaZeroConfig, root: Node, runs: int):
    visit_counts = [(child.visit_count, action)
                    for action, child in root.children.items()]
    if runs < config.num_sampling_moves:
        print("runs", runs)
        action = softmax_sample_index(visit_counts)
    else:
        _, action = max(visit_counts, key=lambda x: x[0])
    return action


# Sample the child with probability according to the UCB score.
def random_child(config: AlphaZeroConfig, node: Node):
    triples = [(ucb_score(config, node, child), action, child) for action, child in node.children.items()]

    index = np.random.choice(len(triples))
    _, action, child = triples[index]
    return action, child


# Sample the child with probability according to the UCB score.
def sample_random_child(config: AlphaZeroConfig, node: Node):
    triples = [(ucb_score(config, node, child), action, child) for action, child in node.children.items()]

    triples_scores = [score if score > 0 else 0 for score, action, child in triples]
    triples_scores_sum = sum(triples_scores)

    if triples_scores_sum == 0:
        index = np.random.choice(len(triples))
        _, action, child = triples[index]
        return action, child

    triples_probability = list(map(lambda x: 0 if x < 0 else x / triples_scores_sum, triples_scores))

    indexes = list(range(len(triples)))

    index = np.random.choice(indexes, 1, p=triples_probability)[0]
    _, action, child = triples[index]

    return action, child


# Select the child with the highest UCB score.
def select_child(config: AlphaZeroConfig, node: Node):
    _, action, child = max((ucb_score(config, node, child), action, child)
                           for action, child in node.children.items())
    return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: AlphaZeroConfig, parent: Node, child: Node):
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                    config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = 1 - child.value()

    return prior_score + value_score


def legal_actions(game):
    legal = [a for a in range(0, game.actions) if game.valid(a)]
    return legal


# We use the neural network to obtain a value and policy prediction.
def evaluate(node: Node, game: az_quiz, config: AlphaZeroConfig, player):
    # don't expand finished game
    if game.winner is not None:
        if not config.already_reached_end:
            print("final node reached won?: [+ winning node, - loosing node]")
            print(os.path.realpath(__file__))
            print(time.localtime())
            config.already_reached_end = True

        if node.to_play == -1:
            node.to_play = None
            node.value_sum = 1 if game.winner == player else 0
            node.prior = 1 if game.winner == player else 0.001  # somewhat hotfix
            # print("changed to play to None", node.value_sum)

        # print("node.value_sum", node.value_sum)
        print("+" if node.value_sum > 0 else "-", end="")

        return node.value_sum  # TODO possible bug - we may need to update the sum (due to the count value goes in limits to 0)
        # node.value_sum += 1 if node.value_sum > 0 else 0 # TODO uncomment if bug

    node.to_play = game.to_play
    network = config.network

    gamestate = game2array(game)

    policy_logits, value = network.predict(gamestate)
    if not (isinstance(value, (np.float32, float))):
        print("Ha - value from network evaluation", value)
        print(type(value))

    policy_logits = policy_logits[0][0]

    # Expand the node.
    policy = {a: math.exp(policy_logits[a]) for a in legal_actions(game)}
    policy_sum = sum(policy.values())
    if not (isinstance(policy_sum, float)):
        print("problem", value)
        raise
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)

    return value

    # MCTS version if at the end set correctly else 0.5, it gets fixed on backprop...
    # Expand the node.
    # value = 0.5
    # legal_actions = legal_actions(game)
    # for action in legal_actions:
    #     node.children[action] = Node(1 / len(legal_actions))  # TO DO GET POLICY
    #     g = game.clone()
    #     g.move(action)
    #
    #     if g.winner is not None:
    #         # print_game_situation(g)
    #         leaf = node.children[action]
    #         if g.winner == player:
    #             leaf.value_sum = 1
    #             value = 1
    #         else:
    #             leaf.value_sum = -1
    #             value = -1
    #
    # # if value == 0.5:
    # #     value = keyboard_value_input(game)
    # # else:
    # #     print_game_situation(game)
    # #     print("result is", value, flush=True)
    #
    # return value


_REPRESENTATION = np.array([[0, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 1], [0, 1, 0, 1]], dtype=np.bool)
_ACTION_Y = np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6],
                     dtype=np.int8)
_ACTION_X = np.array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6],
                     dtype=np.int8)


def game2array(az_quiz: az_quiz.AZQuiz):
    board = az_quiz.board
    board28 = []
    for i in range(28):
        x = _ACTION_X[i]
        y = _ACTION_Y[i]

        found = False
        for j in range(4):
            differ = False
            for k in range(4):
                if _REPRESENTATION[j][k] != board[y, x][k]:
                    differ = True
                    break

            if not differ:
                found = True
                board28.append(j)
                break

        if not found:
            raise

    # print(board28)
    # print(az_quiz.to_play)
    board28.append(az_quiz.to_play)
    return board28


def print_game_situation(self):
    SYMBOLS = [".", "*", "O", "X"]

    board, action = [], 0
    for j in range(self.N):
        board.append("")
        for mode in range(2):
            board[-1] += "  " * (self.N - 1 - j)
            for i in range(j + 1):
                board[-1] += " " + (SYMBOLS[self._board[j, i]] * 2 if mode == 0 else "{:2d}".format(action + i)) + " "
            board[-1] += "  " * (self.N - 1 - j)
        action += j + 1

    print("\n".join(board), flush=True)


def keyboard_value_input(self):
    SYMBOLS = [".", "*", "O", "X"]
    print_game_situation(self)

    while True:
        try:
            action = float(input("Float result of this {}: ".format(SYMBOLS[2 + self.to_play])))
            return action
        except ValueError:
            print("wrong value", flush=True)


# At the end of a simulation, we propagate the evaluation all the way up the tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play, winner):
    for node in search_path:
        if node.to_play == None:  # winner known
            # value is taken from last node.value_sum (at beginning 0 or 1)

            # node.value_sum += value if winner == to_play else (1 - value) # TODO First player win rate after 100 games: 29.00% (24.00% and 34.00% when starting and not starting)
            # node.value_sum += (value) #SPATNE VYSLEDKY

            # sum_for_greater = node.value_sum + (1 if node.value_sum > 0 else 0) # TODO First player win rate after 100 games: 79.00% (88.00% and 70.00% when starting and not starting) take two: 79.00 % (84.00 % and 74.0)

            # sum_for_win = node.value_sum + (value if winner != to_play else (1 - value)) # TODO First player win rate after 100 games: 73.00% (82.00% and 64.00% when starting and not starting) take two:: 83.00 % (90.00 % and 76.00 %)
            # sum_for_win2 = node.value_sum + (1 if winner != to_play else 0) # TODO First player win rate after 100 games: 69.00% (72.00% and 66.00% when starting and not starting)
            # sum_for_win3 = node.value_sum + (1 if winner == to_play else 0) # same as sum > : TODO First player win rate after 100 games: 76.00% (78.00% and 74.00% when starting and not starting)

            # sum_for_one_minus = node.value_sum + (1 - value) #TODO First player win rate after 100 games: 73.00% (84.00% and 62.00% when starting and not starting) take two: 72.00 % (78.00 % and 66.00)
            # sum_for_one_minus equivalent to: node.value_sum++

            node.value_sum += (1 if node.value_sum > 0 else 0)
            node.visit_count += 1
            # # if(sum_for_greater != sum_for_win3):
            # #     print ("differ",sum_for_greater, sum_for_win3)
            # # node.value_sum = sum_for_win3
            #
            # # if (sum_for_greater != sum_for_win or sum_for_win != sum_for_one_minus):
            # #     print("difference")
            # #     print(sum_for_greater, sum_for_win, sum_for_one_minus)
            # #
            # #     if sum_for_greater != sum_for_win:
            # #         print(" > != win")
            # #     if sum_for_greater != sum_for_one_minus:
            # #         print(" > != 1-")
            # #     if sum_for_win != sum_for_one_minus:
            # #         print(" win != 1-")
            # # else:
            # #     print("same, ", end="")
            # #
            # # node.value_sum = sum([sum_for_greater, sum_for_win, sum_for_one_minus]) / 3.0 # First player win rate after 100 games: 67.00% (78.00% and 56.00% when starting and not starting)

        else:
            node.value_sum += value if node.to_play == to_play else (1 - value)
            node.visit_count += 1


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: AlphaZeroConfig, node: Node):
    actions = node.children.keys()
    noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


class Player:
    # CENTER = [12]
    # ANCHORS = [4, 16, 19]
    config = None
    init_player = -1

    def play(self, az_quiz: az_quiz.AZQuiz):

        if self.init_player != az_quiz.to_play:
            self.init_player = -1  # this changes when table is turned! that is why it has to be reevaluated

        if self.init_player == -1:
            self.config = AlphaZeroConfig()
            self.init_player = az_quiz.to_play
            print("(new? start) Player nr:", self.init_player)

        debug = False
        if debug:
            CENTER = [12]
            ANCHORS = [4, 16, 19]
            for action in CENTER + ANCHORS + [27] + list(range(11)):
                if az_quiz.valid(action):
                    return action

        # time.sleep(0.3)
        action, root = run_mcts(self.config, az_quiz)
        print("action from mcts: ", action)

        while action is None or not az_quiz.valid(action):
            print("chosen another action")
            action = np.random.randint(az_quiz.actions)  # something failed - try to play
        return action


if __name__ == "__main__":
    import az_quiz_evaluator_recodex
    # if False:
    if True:
        az_quiz_evaluator_recodex.evaluate(Player())
        exit()

    import az_quiz_evaluator
    import importlib

    deterministic = importlib.import_module("az_quiz_player_deterministic").Player()
    random = importlib.import_module("az_quiz_player_random").Player()
    heuristic = importlib.import_module("az_quiz_player_simple_heuristic").Player()

    # players = [Player(), deterministic]
    # players = [Player(), random]
    players = [Player(), heuristic]

    randomized = False
    render = True

    az_quiz_evaluator.evaluate(players, 1, randomized, render)
    players = [Player(), players[1]][-1::-1]
    # az_quiz_evaluator.evaluate(players, 1, randomized, render)
