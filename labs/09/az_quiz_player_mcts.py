#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import math
import az_quiz
from typing import List
from functools import reduce


class AlphaZeroConfig(object):
    def __init__(self):
        ### Self-Play
        self.num_sampling_moves = 8  # 30
        self.max_moves = 500  # 512 for chess and shogi, 722 for Go.
        self.num_simulations = 500  # 800

        # Root prior exploration noise.
        self.root_dirichlet_alpha = 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652  # dle videa 15000
        self.pb_c_init = 1.25  # dle videa 2

        self.input_size = (29,)  # 28 fields + 1 for player
        self.learning_rate = 0.002
        self.hidden_layer_size = 1
        self.network = Network(self)

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
    def __init__(self,  args):
        # Define suitable model. Apart from the model defined in `reinforce`,
        # define also another model `baseline`, which produces one output
        # (using a dense layer without activation).
        #
        # Use Adam optimizer with given `args.learning_rate` for both models.
        actions = 28
        input_layer = tf.keras.layers.Input(shape=args.input_size)
        hidden_layer = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(input_layer)
        output_layer = tf.keras.layers.Dense(actions,)(hidden_layer)
        self.model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])

        self.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        experimental_run_tf_function=False
        )

        ###0###print(self.model.summary())

    def train(self, states, actions, returns):
        #games = state. return = revard of the state. actions - possible actions
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

    run_selfplay(config)#, storage, replay_buffer)
    # train_network(config)#, storage, replay_buffer)

    return #storage.latest_network()


# takes the latest network snapshot, produces a game and makes it available
# to the training job by writing it to a shared replay buffer.
def run_selfplay(config: AlphaZeroConfig):#,
                 #storage: SharedStorage,
                 #replay_buffer: ReplayBuffer):
    while True:
        # network = storage.latest_network())
        game = play_game(config, config.network)
        #replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: AlphaZeroConfig):
    game = az_quiz.AZQuiz()
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
    result.sort(key= lambda x: -x[0])
    # print (result)

    # print(y)

    #TODO result contains list of pst with indexes -sample from it
    while len(result) > 1 :
        if (np.random.random() > 0.8):
            result.pop(0)
            chosen_index = result[0][1]
        else:
            break

    if len(result) == 1:
        return result[0][1]

    return chosen_index

def max_arg(x):
    max = -1
    maxi = -1
    for val,index in x:
        # print(val, index, max, maxi)
        if(val > max):
            max = val
            maxi = index

    return maxi



# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: AlphaZeroConfig, game: az_quiz):
    root = Node(0)
    evaluate(root, game, config)
    add_exploration_noise(config, root)


    runs = 0

    for _ in range(config.num_simulations):
        node = root
        scratch_game = game.clone()
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node)
            runs += 1
            search_path.append(node)

        value = evaluate(node, scratch_game, config)
        backpropagate(search_path, value, scratch_game.to_play)

    return select_action(config, game, root, runs ), root


def select_action(config: AlphaZeroConfig, game: az_quiz, root: Node, runs: int):
    visit_counts = [(child.visit_count, action)
                    for action, child in root.children.items()]
    if runs < config.num_sampling_moves:
        ###0###print(visit_counts)
        action = softmax_sample_index(visit_counts)
    else:
        _, action = max(visit_counts, key= lambda x: x[0])
    return action


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
def evaluate(node: Node, game: az_quiz, config: AlphaZeroConfig):
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



    # Expand the node.
    node.to_play = game.to_play
    legal_actionss = legal_actions(game)
    for action in legal_actionss:
        node.children[action] = Node(1/len(legal_actionss)) #TODO GET POLICY
        g = game.clone()
        g.move(action)

        if g.winner != None:
            leaf = node.children[action]
            if g.winner == node.to_play:
                leaf.value_sum = 1
                leaf.prior = 1
                # print("won")
            else:
                print("lost")
                leaf.value_sum = -1
                leaf.prior = -1
    return node.value_sum

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

    def play(self, az_quiz: az_quiz.AZQuiz):

        config = AlphaZeroConfig()

        if (config.init_player == -1):
            config.init_player = az_quiz.to_play

        action, root = run_mcts(config, az_quiz)
        # print("action from mcts: ", action)

        while action is None or not az_quiz.valid(action):
            print("chosen another action")
            action = np.random.randint(az_quiz.actions)  # something failed - try to play
        return action


if __name__ == "__main__":
    import az_quiz_evaluator_recodex
    az_quiz_evaluator_recodex.evaluate(Player())
