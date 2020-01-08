#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import math
import az_quiz
from typing import List
from functools import reduce
import time


class AlphaZeroConfig(object):
    def __init__(self):
        ### Self-Play
        self.num_sampling_moves = 0  # 30  # tells how many game moves will be possibly random
        self.max_moves = 30  # depth of the search 28 for az-kviz 512 for chess and shogi, 722 for Go.
        self.num_simulations = 5000  # 800 number of paths in mcts

        # Root prior exploration noise.
        self.root_dirichlet_alpha = 0.03  # for chess, 0.03 for Go and 0.15 for shogi.
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652  # dle videa 15000
        self.pb_c_init = 1.25  # dle videa 2

        self.input_size = (29,)  # 28 fields + 1 for player
        self.learning_rate = 0.002
        self.hidden_layer_size = 1
        # self.network = Network(self)

        self.init_player = -1


class Node(object):
    def __init__(self, prior: float = None):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior if prior is not None else 0.03571  # 1/28
        self.value_sum = 0  # "sum" of value function of the children
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class Network:
    def __init__(self, args):
        # Define suitable model. Apart from the model defined in `reinforce`,
        # define also another model `baseline`, which produces one output
        # (using a dense layer without activation).
        #
        # Use Adam optimizer with given `args.learning_rate` for both models.
        actions = 28
        input_layer = tf.keras.layers.Input(shape=args.input_size)
        hidden_layer = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(input_layer)
        output_layer = tf.keras.layers.Dense(actions, )(hidden_layer)
        self.model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            experimental_run_tf_function=False
        )

        ###0###print(self.model.summary())

    def train(self, states, actions, returns):
        # games = state. return = revard of the state. actions - possible actions
        # actions = np.array(list(reduce(lambda a, b: np.concatenate((a, b)), actions)))
        # returns = np.array(list(reduce(lambda a, b: np.concatenate((a, b)), returns)))
        # states = np.array(list(reduce(lambda a, b: np.concatenate((a, b)), states)))
        # np.array(list(reduce)) #TODO MUSIME TRENOVAT!

        # Train the model using the states, actions and observed returns.
        # You should:
        # - compute the predicted baseline using the `baseline` model
        # baselines = self.baseline.predict(states).flatten()

        # - train the policy model, using `returns - predicted_baseline` as weights
        #   in the sparse crossentropy loss
        weight = returns  # - baselines
        self.model.train_on_batch(states, actions, sample_weight=weight)

        # - train the `baseline` model to predict `returns`
        # self.baseline.train_on_batch(states, returns)

    def predict(self, states):
        states = np.array(states, np.float32)
        return self.model.predict(states)


##TRAINING CODE FROM ALPHA ZERO
# def train_network(config: AlphaZeroConfig, storage: SharedStorage,
#                   replay_buffer: ReplayBuffer):
#   network = Network()
#   optimizer = tf.train.MomentumOptimizer(config.learning_rate_schedule,
#                                          config.momentum)
#   for i in range(config.training_steps):
#     if i % config.checkpoint_interval == 0:
#       storage.save_network(i, network)
#     batch = replay_buffer.sample_batch()
#     update_weights(optimizer, network, batch, config.weight_decay)
#   storage.save_network(config.training_steps, network)
#
#
# def update_weights(optimizer: tf.train.Optimizer, network: Network, batch,
#                    weight_decay: float):
#   loss = 0
#   for image, (target_value, target_policy) in batch:
#     value, policy_logits = network.inference(image)
#     loss += (
#         tf.losses.mean_squared_error(value, target_value) +
#         tf.nn.softmax_cross_entropy_with_logits(
#             logits=policy_logits, labels=target_policy))
#
#   for weights in network.get_weights():
#     loss += weight_decay * tf.nn.l2_loss(weights)
#
#   optimizer.minimize(loss)
#
#
# ######### End Training ###########
# ##################################


# AlphaZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def alphazero(config: AlphaZeroConfig):
    # storage = SharedStorage()
    # replay_buffer = ReplayBuffer(config)

    run_selfplay(config)  # , storage, replay_buffer)
    # train_network(config)#, storage, replay_buffer)

    return  # storage.latest_network()


# takes the latest network snapshot, produces a game and makes it available
# to the training job by writing it to a shared replay buffer.
def run_selfplay(config: AlphaZeroConfig):  # ,
    # storage: SharedStorage,
    # replay_buffer: ReplayBuffer):
    while True:
        # network = storage.latest_network())
        game = play_game(config)
        # replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: AlphaZeroConfig):
    game = az_quiz.AZQuiz(randomized=False)
    moves = 0
    while not game.winner and moves < config.max_moves:
        moves += 1
        # while not game.winner and len(game.history) < config.max_moves:
        action, root = run_mcts(config, game)

        # game.apply(action)  # self.history.append(action)
        # game.store_search_statistics(root)
    return game


def softmax_sample_index(z):
    """Compute softmax values for each sets of scores in x."""

    x = [z[0] for z in z]

    # result = np.exp(x) / np.sum(np.exp(x), axis=0)
    result = np.exp(x) / np.sum(np.exp(x))
    result = [(val, idx) for idx, val in enumerate(result)]
    # print(result)
    chosen_index = max_arg(result)
    # print(y)
    result.sort(key=lambda x: -x[0])
    # print (result)

    # print(y)

    # TODO result contains list of pst with indexes -sample from it
    while len(result) > 1:
        if np.random.random() > 0.8:
            result.pop(0)
            chosen_index = result[0][1]
        else:
            break

    if len(result) == 1:
        return result[0][1]

    return chosen_index


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
        backpropagate(search_path, value, player)

    root2 = root
    # print_mstc_tree(root2)
    return select_action(config, root, runs), root


def print_mstc_node_info(node: Node, action, spaces):
    # print(spaces)
    print(spaces*" " + str(action) + ":-> val:" + str(node.value()) + ":sum=" + str(node.value_sum) + "visits:" + str(node.visit_count))


def print_mstc_tree(root: Node, action = None, spaces=0):
    #as a path in dir system
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
        print(visit_counts)
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

    # three = np.random.choice(triples, 1, p=triples_probability)

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
    # network = config.network
    # value, policy_logits = network.model.GET POLICY ## TODO GET POLICY
    #
    # legal_actions = game.legal_actions()
    #
    #
    # # Expand the node.
    # node.to_play = game.to_play()
    # policy = {a: math.exp(policy_logits[a]) for a in legal_actions}
    # policy_sum = sum(policy.itervalues())
    # for action, p in policy.iteritems():
    #     node.children[action] = Node(p / policy_sum)
    # return value

    #don't expand finished game
    if game.winner is not None:
        print("visit of a known one :", node.value_sum)
        # print_game_situation(game)
        return node.value_sum # TODO possible bug - may need to update the sum (due to the count value goes in limits to 0)
        # TODO bug fix for the possible bug
        addition = -1
        if node.value_sum > 0:
            addition = 1
        node.value_sum += add_exploration_noise()

        # return node.value() # value sum / visit count

    # Expand the node.
    node.to_play = game.to_play
    legal_actionss = legal_actions(game)
    value = 0.5
    for action in legal_actionss:
        node.children[action] = Node(1 / len(legal_actionss))  # TODO GET POLICY
        g = game.clone()
        g.move(action)

        if g.winner is not None:
            # print_game_situation(g)
            leaf = node.children[action]
            if g.winner == player:
                leaf.value_sum = 1
                value = 1
            else:
                leaf.value_sum = -1
                value = -1

    # if value == 0.5:
    #     value = keyboard_value_input(game)
    # else:
    #     print_game_situation(game)
    #     print("result is", value, flush=True)

    return value

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


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play):
    for node in search_path:
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
    # CENTER = 12
    # ANCHORS = [4, 16, 19]
    config = None
    init_player = -1

    def play(self, az_quiz: az_quiz.AZQuiz):
        if self.init_player == -1:
            self.config = AlphaZeroConfig()
            self.init_player = az_quiz.to_play
            print("Player nr:", self.init_player)

        for action in [27] + list(range(0)):
            if az_quiz.valid(action):
                return action

        time.sleep(0.3)
        action, root = run_mcts(self.config, az_quiz)
        print("action from mcts: ", action)

        while action is None or not az_quiz.valid(action):
            print("chosen another action")
            action = np.random.randint(az_quiz.actions)  # something failed - try to play
        return action


if __name__ == "__main__":
    import az_quiz_evaluator_recodex

    if False:
        az_quiz_evaluator_recodex.evaluate(Player())
        exit()

    import az_quiz_evaluator
    import importlib

    deterministic = importlib.import_module("az_quiz_player_deterministic").Player()
    random = importlib.import_module("az_quiz_player_random").Player()
    # az_quiz_evaluator.evaluate([Player(), deterministic], 2, False, True)
    players = [Player(), deterministic]
    players = [Player(), random]

    randomized = False
    render = True
    az_quiz_evaluator.evaluate(players, 1, randomized, render)
    players = [Player(), deterministic][-1::-1]
    # az_quiz_evaluator.evaluate(players, 1, randomized, render)
