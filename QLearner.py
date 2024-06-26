""""""  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	   			  		 			     			  	 
All Rights Reserved  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	   			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	   			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	   			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   			  		 			     			  	 
or edited.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	   			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	   			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   			  		 			     			  	 
GT honor code violation.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Student Name: Tucker Balch (replace with your name)  		  	   		 	   			  		 			     			  	 
GT User ID: tb34 (replace with your User ID)  		  	   		 	   			  		 			     			  	 
GT ID: 900897987 (replace with your GT ID)  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import random as rand  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import numpy as np  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
class QLearner(object):
    """  		  	   		 	   			  		 			     			  	 
    This is a Q learner object.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    :param num_states: The number of states to consider.  		  	   		 	   			  		 			     			  	 
    :type num_states: int  		  	   		 	   			  		 			     			  	 
    :param num_actions: The number of actions available..  		  	   		 	   			  		 			     			  	 
    :type num_actions: int  		  	   		 	   			  		 			     			  	 
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		 	   			  		 			     			  	 
    :type alpha: float  		  	   		 	   			  		 			     			  	 
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		 	   			  		 			     			  	 
    :type gamma: float  		  	   		 	   			  		 			     			  	 
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		 	   			  		 			     			  	 
    :type rar: float  		  	   		 	   			  		 			     			  	 
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		 	   			  		 			     			  	 
    :type radr: float  		  	   		 	   			  		 			     			  	 
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		 	   			  		 			     			  	 
    :type dyna: int  		  	   		 	   			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	   			  		 			     			  	 
    :type verbose: bool  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    def __init__(  		  	   		 	   			  		 			     			  	 
        self,  		  	   		 	   			  		 			     			  	 
        num_states=1728, #8^3*3
        num_actions=3,
        alpha=0.3,
        gamma=0.9,  		  	   		 	   			  		 			     			  	 
        rar=0.65,
        radr=0.99,
        dyna=0,
        verbose=False,  		  	   		 	   			  		 			     			  	 
    ):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        Constructor method  		  	   		 	   			  		 			     			  	 
        """
        self.gamma = gamma
        self.alpha = alpha
        self.radr = radr
        self.rar = rar
        self.verbose = verbose
        self.num_actions = num_actions  		  	   		 	   			  		 			     			  	 
        self.s = 0
        self.a = 0
        self.q_table = np.zeros((num_states, num_actions))
        # Dyna
        self.dyna = dyna
        self.model = {}
  		  	   		 	   			  		 			     			  	 
    def querysetstate(self, s):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        Update the state without updating the Q-table  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        :param s: The new state  		  	   		 	   			  		 			     			  	 
        :type s: int  		  	   		 	   			  		 			     			  	 
        :return: The selected action  		  	   		 	   			  		 			     			  	 
        :rtype: int  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        self.s = s  		  	   		 	   			  		 			     			  	 
        action = rand.randint(0, self.num_actions - 1)  		  	   		 	   			  		 			     			  	 

        if self.verbose:
            print(f"s = {s}, a = {action}")

        return action  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    def query(self, s_prime, r):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        Update the Q table and return an action  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        :param s_prime: The new state  		  	   		 	   			  		 			     			  	 
        :type s_prime: int  		  	   		 	   			  		 			     			  	 
        :param r: The immediate reward  		  	   		 	   			  		 			     			  	 
        :type r: float  		  	   		 	   			  		 			     			  	 
        :return: The selected action  		  	   		 	   			  		 			     			  	 
        :rtype: int  		  	   		 	   			  		 			     			  	 
        """

        # 1) Update Q Table
        a_prev = self.a
        s_prev = self.s
        max_future_reward = np.max(self.q_table[s_prime])
        improved_estimate = (r + self.gamma * max_future_reward)

        self.q_table[s_prev][a_prev] = (1 - self.alpha) * self.q_table[s_prev][a_prev] + self.alpha * improved_estimate

        # Dyna ------ START
        self.model[(self.s, self.a)] = (s_prime, r)
        if len(self.model) > 0:
            num_samples = min(self.dyna, len(self.model))
            keys = list(self.model.keys())
            indices = np.random.choice(len(keys), size=num_samples, replace=True)
            sampled_keys = [keys[i] for i in indices]  # Keep sampled keys as list of tuples
            sampled_results = [self.model[key] for key in sampled_keys]  # Get results as list

            sampled_states = np.array([key[0] for key in sampled_keys])
            sampled_actions = np.array([key[1] for key in sampled_keys])
            sampled_next_states = np.array([result[0] for result in sampled_results])
            sampled_rewards = np.array([result[1] for result in sampled_results])

            if num_samples > 0:  # Only perform vectorized update if there are samples
                max_future_rewards = np.max(self.q_table[sampled_next_states], axis=1)
                self.q_table[sampled_states, sampled_actions] = (
                        (1 - self.alpha) * self.q_table[sampled_states, sampled_actions] +
                        self.alpha * (sampled_rewards + self.gamma * max_future_rewards)
                )
        # Dyna ------ END

        # 2) Determine action
        # Roll dice on random action
        if rand.uniform(0.0, 1.0) <= self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            # Get best action for new state based on highest Q value
            action = np.argmax(self.q_table[s_prime])

        # 3) Update previous action, new state, rar
        self.s = s_prime
        self.a = action
        self.rar = self.rar * self.radr

        if self.verbose:
            print(f"s = {s_prime}, a = {action}, r={r}")

        return action

    def author(self):
        return "fadam6"

def author():
  return 'fadam6'

if __name__ == "__main__":  		  	   		 	   			  		 			     			  	 
    print("Remember Q from Star Trek? Well, this isn't him")  		  	   		 	   			  		 			     			  	 

