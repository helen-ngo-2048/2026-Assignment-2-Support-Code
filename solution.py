import sys
import time
import numpy

from game_env import GameEnv
from game_state import GameState
from collections import deque

"""
solution.py

This file is a template you should use to implement your solution.

You should implement each of the method stubs below. You may add additional methods and/or classes to this file if you 
wish. You may also create additional source files and import to this file if you wish.

COMP3702 Assignment 2 "CrystalRover" Support Code

Last updated by vp 09/09/2026
"""


class Solver:

    STUDENT_NAME = "Helen Ngo" # replace with your name
    STUDENT_ID = "48814553"  # replace with your student ID
    GITHUB_USERNAME = "helen-ngo-2048" # replace with your GitHub username
    DELTAS = {
            'LEFT': (0, -1),
            'RIGHT': (0, 1),
            'UP': (-1, 0),
            'DOWN': (1, 0),
        }

    def __init__(self, game_env: GameEnv):
        self.game_env = game_env

        # MPD model, adapted from PISolverLinAlg.__init__
        self.states = None # list of reachable GameState objects; index = row/column in the arrays below
        self.state_indices = None # dict: GameState -> index in self.states
        self.terminal_mask = None # bool array |S|: True for solved / game-over states
        self.valid_mask = None # bool array |S| x |A|: True if action a is valid in state s
        self.t_model = None # float array |S| x |A| x |S|: P(s' | s, a)
        self.r_model = None # float array |S| x |A|: expected immediate reward R(s, a)
        self.action_indices = {a: i for i, a in enumerate(GameEnv.ACTIONS)} # action string -> column index

        # Fast lookup crystal position to avoid O(N)
        self.crystal_index = {pos: i for i, pos in enumerate(game_env.crystal_positions)}
        # cache of get_transition_outcomes results, key is (state, action)
        self.outcome_cache = {}

        # Value Iteration
        self.vi_values = None # numpy array |S|: V(s)
        self.vi_policy = None # list of actions, one per state
        self.vi_max_delta = numpy.inf # largest change in V(s) during the last iteration

        # Policy Iteration
        self.pi_policy = None # numpy int array |S|: index of the action for each state
        self.pi_values = None # numpy array |S|: value of the current policy
        self.pi_converged = False
        pass

    @staticmethod
    def testcases_to_attempt():
        """
        Return a list of testcase numbers you want your solution to be evaluated for.
        """
        # TODO: modify below if desired (e.g. disable larger testcases if you're having problems with RAM usage, etc)
        return [1, 2, 3, 4, 5]

    # === Value Iteration ==============================================================================================

    def vi_initialise(self):
        """
        Initialise any variables required before the start of Value Iteration.
        """
        self.build_model()
        self.vi_values = numpy.zeros(len(self.states)) # Helper guide
        self.vi_policy = [GameEnv.ACTIONS[0]] * len(self.states)
        self.vi_max_delta = numpy.inf

    def vi_is_converged(self):
        """
        Check if Value Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        return self.vi_max_delta < self.game_env.epsilon

    def vi_iteration(self):
        """
        Perform a single iteration of Value Iteration (i.e. loop over the state space once).
        Gen AI (Claude Sonnet 5.5) was used to debug during this method's implementation.
        """
        # adapted from VISolver.vi_iteration. Sync update, new values only use prev iteration's values
        gamma = self.game_env.gamma
        new_values = numpy.zeros(len(self.states))
        new_policy = list(self.vi_policy)
        for i, state in enumerate(self.states):
            if self.terminal_mask[i]:
                continue
            best_q = -numpy.inf
            best_action = None
            for action in self.get_valid_actions(state):
                # Q(s, a) = sum_s' P(s' | s, a) * (R(s, a, s') + gamma * V(s'))
                total = 0.0
                for prob, next_state, reward in self.get_transition_outcomes(state, action):
                    total += prob * (reward + gamma * self.vi_values[self.state_indices[next_state]])
                if total > best_q:
                    best_q = total
                    best_action = action
            new_values[i] = best_q
            new_policy[i] = best_action

        self.vi_max_delta = numpy.abs(new_values - self.vi_values).max()
        self.vi_values = new_values
        self.vi_policy = new_policy

    def vi_plan_offline(self):
        """
        Plan using Value Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.vi_initialise()
        while True:
            self.vi_iteration()

            # NOTE: vi_iteration is always called before vi_is_converged
            if self.vi_is_converged():
                break

    def vi_get_state_value(self, state: GameState):
        """
        Retrieve V(s) for the given state.
        :param state: the current state
        :return: V(s)
        """
        if self.vi_values is None:
            return 0.0
        index = self.state_indices.get(state)
        if index is None:
            return 0.0
        return float(self.vi_values[index])

    def vi_select_action(self, state: GameState):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ACTIONS)
        """
        index = self.state_indices.get(state)
        if index is None:
            return self.get_valid_actions(state)[0]
        return self.vi_policy[index]

    # === Policy Iteration =============================================================================================

    def pi_initialise(self):
        """
        Initialise any variables required before the start of Policy Iteration.
        """
        self.build_model()
        # initial policy is always WALK_RIGHT
        self.pi_policy = numpy.full(len(self.states), self.action_indices[GameEnv.WALK_RIGHT], dtype = numpy.int64)
        self.pi_values = numpy.zeros(len(self.states))
        self.pi_converged = False

    def pi_is_converged(self):
        """
        Check if Policy Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        return self.pi_converged

    def pi_iteration(self):
        """
        Perform a single iteration of Policy Iteration (i.e. perform one step of policy evaluation and one step of
        policy improvement).
        """
        # Adapted from PISolverLinAlg.pi_iteration
        self.pi_values = self.policy_evaluation()
        self.policy_improvement(self.pi_values)

    def pi_plan_offline(self):
        """
        Plan using Policy Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.pi_initialise()
        while True:
            self.pi_iteration()

            # NOTE: pi_iteration is always called before pi_is_converged
            if self.pi_is_converged():
                break

    def pi_select_action(self, state: GameState):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ACTIONS)
        """
        index = self.state_indices.get(state)
        if index is None:
            return self.get_valid_actions(state)[0]
        return GameEnv.ACTIONS[self.pi_policy[index]]

    # === Helper Methods ===============================================================================================
    
    def move(self, state: GameState, action, distance):
        # Re-implementation of the movement part of GameEnv.apply_dynamics
        env = self.game_env
        reward = -1 * env.ACTION_COST[action]
        delta_row, delta_col = self.DELTAS[env._action_direction(action)]
        row, col = state.row, state.col

        for _ in range(distance):
            candidate_row = row + delta_row
            candidate_col = col + delta_col
            if not (0 <= candidate_row < env.n_rows and 0 <= candidate_col < env.n_cols) \
                    or env.grid_data[candidate_row][candidate_col] == env.ROCK_TILE:
                reward -= env.collision_penalty
                break

            row, col = candidate_row, candidate_col

            # fall into a crater
            if env.grid_data[row][col] == env.CRATER_TILE:
                break

            # fall into lava
            if env.grid_data[row][col] == env.LAVA_TILE:
                reward -= env.game_over_penalty
                break

        crystal_status = state.crystal_status
        crystal_idx = self.crystal_index.get((row, col))
        if crystal_idx is not None and crystal_status[crystal_idx] == 0:
            crystal_status = crystal_status[:crystal_idx] + (1,) + crystal_status[crystal_idx + 1:]
        game_over = env.grid_data[row][col] == env.LAVA_TILE
        return GameState(row, col, crystal_status), reward, game_over

    def get_action_outcomes(self, state: GameState, action):
        """
        Outcomes of 1 action as (probability, next_state, reward, game_over) tuples.
        Walk and jump are deterministic, a boost branches over the distance (0 to 4 tiles) with
        env.boost_probabilities.
        """
        env = self.game_env
        on_crater = env.grid_data[state.row][state.col] == env.CRATER_TILE

        if action in env.JUMP_ACTIONS:
            if not on_crater:
                return [(1.0, state, 0.0, False)]
            distances = [(1, 1.0)]
        elif action in env.WALK_ACTIONS:
            if on_crater:
                return [(1.0, state, 0.0, False)]
            distances = [(1, 1.0)]
        else:
            if on_crater:
                return [(1.0, state, 0.0, False)]
            distances = [(d, p) for d, p in enumerate(env.boost_probabilities) if p > 0.0]

        outcomes = []
        for distance, prob in distances:
            next_state, reward, game_over = self.move(state, action, distance)
            outcomes.append((prob, next_state, reward, game_over))
        return outcomes

    def get_noise_outcomes(self, action):
        """
        Enumerate the drift/ double-move outcomes of the original action as (probability, [movements]) pairs.
        """
        env = self.game_env
        p_drift = env.random_drift_prob
        p_double = env.random_double_prob
        # (movement, probability of that direction): original, or drift to one of the 2 perpendicular directions
        directions = [(action, 1.0 - p_drift)]
        for perpendicular in env.PERPENDICULAR_ACTIONS[action]:
            directions.append((perpendicular, p_drift / 2.0))

        outcomes = []
        for movement, p_direction in directions:
            outcomes.append((p_direction * (1.0 - p_double), [movement])) # single movement
            outcomes.append((p_direction * p_double, [movement, movement])) # double movement
        return [(p, m) for p, m in outcomes if p > 0.0]

    def apply_actions(self, state: GameState, action_sequence):
        """
        Apply a sequence of (1 or 2 same) actions. Returns [(prob, state, reward)].
        Reward adds up, stops early if game over.
        """
        current_outcomes = [(1.0, state, 0.0, False)]
        
        for action in action_sequence:
            next_outcomes = []
            
            for prob, current_state, total_reward, game_over in current_outcomes:
                if game_over:
                    # If already in lava, stop executing further actions in the sequence
                    next_outcomes.append((prob, current_state, total_reward, game_over))
                    continue
                
                for step_prob, next_state, step_reward, step_game_over in self.get_action_outcomes(current_state, action):
                    next_outcomes.append((
                        prob * step_prob,
                        next_state,
                        total_reward + step_reward,
                        step_game_over
                    ))

            current_outcomes = next_outcomes
            
        # Strip game_over flag
        return [(p, s, r) for p, s, r, _ in current_outcomes]

    def is_terminal_state(self, state: GameState):
        return self.game_env.is_solved(state) or self.game_env.is_game_over(state)

    def get_transition_outcomes(self, state: GameState, action):
        """
        Return a list of tuples for every possible outcome of performing the given
        action in the given state.
        Enumerate the drift and double-move combinations and their probabilities,
        for each combination apply the actions, branching on the boost distance.
        Gen AI (Claude Sonnet 5.5) was used to implement this method
        """
        key = (state, action)
        cached = self.outcome_cache.get(key)
        if cached is not None:
            return cached

        if self.is_terminal_state(state):
            outcomes = []
        else:
            merged = {} # (next_state, rounded reward) -> [probability, reward]
            for noise_prob, movements in self.get_noise_outcomes(action):
                for prob, next_state, reward in self.apply_actions(state, movements):
                    merge_key = (next_state, round(reward, 9))
                    if merge_key in merged:
                        merged[merge_key][0] += noise_prob * prob
                    else:
                        merged[merge_key] = [noise_prob * prob, reward]
            outcomes = [(p, next_state, reward) for (next_state, _), (p, reward) in merged.items()]

        self.outcome_cache[key] = outcomes
        return outcomes

    def get_valid_actions(self, state: GameState):
        """
        Valid actions in the given state
        """
        env = self.game_env
        on_crater = env.grid_data[state.row][state.col] == env.CRATER_TILE
        # Jump only valid in crater, walk and boost outside crater
        return [a for a in GameEnv.ACTIONS if (a in env.JUMP_ACTIONS) == on_crater]

    def build_model(self):
        """
        Build the MDP model: the reachable state space (bfs), the transition tensor
        t_model[s, a, s'], the expected reward array r_model[s, a] and the valid action mask.
        Gen AI (Claude Sonnet 5.5) was used to help generate line 356 - 370 of this method.
        """
        if self.states is not None:
            return

        # states/state_indices adapted from GridworldEnv.states/ PISolverLinAlg.state_indices
        init_state = self.game_env.get_init_state()
        self.states = [init_state]
        self.state_indices = {init_state: 0}
        queue = deque([init_state])

        while queue:
            state = queue.popleft()
            if self.is_terminal_state(state):
                continue
            for action in self.get_valid_actions(state):
                for _, next_state, _ in self.get_transition_outcomes(state, action):
                    if next_state not in self.state_indices:
                        self.state_indices[next_state] = len(self.states)
                        self.states.append(next_state)
                        queue.append(next_state)

        # Transition tensor + expected reward
        n_states = len(self.states)
        n_actions = len(GameEnv.ACTIONS)
        self.t_model = numpy.zeros([n_states, n_actions, n_states])
        self.r_model = numpy.zeros([n_states, n_actions])
        self.valid_mask = numpy.zeros([n_states, n_actions], dtype = bool)
        self.terminal_mask = numpy.zeros(n_states, dtype = bool)

        for i, state in enumerate(self.states):
            if self.is_terminal_state(state):
                self.terminal_mask[i] = True
                continue
            valid_actions = set(self.get_valid_actions(state))
            for j, action in enumerate(GameEnv.ACTIONS):
                self.valid_mask[i, j] = action in valid_actions
                for prob, next_state, reward in self.get_transition_outcomes(state, action):
                    self.t_model[i, j, self.state_indices[next_state]] += prob
                    self.r_model[i, j] += prob * reward

    def policy_evaluation(self):
        # Adapted from PISolverLinAlg.policy_evaluation
        state_numbers = numpy.arange(len(self.states))
        t_pi = self.t_model[state_numbers, self.pi_policy]
        r_pi = self.r_model[state_numbers, self.pi_policy]
        a_matrix = numpy.identity(len(self.states)) - self.game_env.gamma * t_pi
        return numpy.linalg.solve(a_matrix, r_pi)

    def policy_improvement(self, values):
        """
        pi'(s) = argmax_a Q(s, a), with Q = R + gamma * (T @ V^pi);
        Adapted from PISolverLinAlg.policy_improvement
        """
        q_values = self.r_model + self.game_env.gamma * (self.t_model @ values)
        q_values[~self.valid_mask] = -numpy.inf

        new_policy = numpy.argmax(q_values, axis = 1)
        new_policy[self.terminal_mask] = self.pi_policy[self.terminal_mask]
        policy_changed = not numpy.array_equal(new_policy, self.pi_policy)

        self.pi_policy = new_policy
        self.pi_converged = not policy_changed
