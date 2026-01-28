# -*- coding: utf-8 -*-
"""
Created on Mon May 16 00:27:50 2016

@author: Hossam Faris
"""

import random
import numpy
import numpy as np
import math
import time
# from test import RingBuffer
from scipy.special import gamma
from solution import solution


class RingBuffer:
    """ Class that implements a not-yet-full buffer. """
    def __init__(self, bufsize):
        self.bufsize = bufsize
        self.data = []

    class __Full:
        """ Class that implements a full buffer. """
        def add(self, x):
            """ Add an element overwriting the oldest one. """
            self.data[self.currpos] = x
            self.currpos = (self.currpos+1) % self.bufsize
        def get(self):
            """ Return list of elements in correct order. """
            return self.data[self.currpos:]+self.data[:self.currpos]

        def get_e_findGwo(self):
            elements = [str(self.data[i]) for i in range(self.bufsize)]
            # print(elements)
            return "".join(elements)

    def add(self,x):
        """ Add an element at the end of the buffer"""
        self.data.append(x)
        if len(self.data) == self.bufsize:
            # Initializing current position attribute
            self.currpos = 0
            # Permanently change self's class from not-yet-full to full
            self.__class__ = self.__Full

    def get(self):
        """ Return a list of elements from the oldest to the newest. """
        return self.data

    def get_e_findGwo(self):
        elements = [str(self.data[i]) for i in range(self.bufsize)]
        print(elements)
        return "".join(elements)


def updateLeaderByAction(action, objf, a, Alpha_pos, Beta_pos, Delta_pos, Alpha_score, Beta_score, Delta_score, lb, ub):
    reward = -2
    state = 2
    dim = dimension
    # a = 0.01
    population = [Alpha_pos, Beta_pos, Delta_pos]
    Lambda = 1.5
    # levy distribution
    if action == 0:
        x = numpy.linspace(0, 30, dim)
        # r = stats.gamma.pdf(x, a=3, scale=5)
        shape, scale = 0.95, 0.4  # mean=4, std=2*sqrt(2)
        shape, scale = 1.5, 0.4  # mean=4, std=2*sqrt(2)
        r = numpy.random.gamma(shape, scale, dim)
        # r = r * a

    # cauchy distribution
    if action == 1:
        x = numpy.linspace(0, 30, dim)
        # r = stats.cauchy.pdf(x)
        r = numpy.random.standard_exponential(dim)
        # r = r * a

    if a < 0.7:
        q = 1
    else:
        q= -1

    if action == 2:
        return

    alphatemp_pos = Alpha_pos - q * Alpha_pos * r
    alphatemp_pos = numpy.clip(alphatemp_pos, lb, ub)

    if Alpha_score > objf(alphatemp_pos):
        # print("--")
        state = 0
        # reward += (Alpha_score - objf(alphatemp_pos))
        Alpha_pos = alphatemp_pos
        Alpha_score = objf(alphatemp_pos)
        reward += 30

    betatemp_pos = Beta_pos - Beta_pos * r
    betatemp_pos = numpy.clip(betatemp_pos, lb, ub)
    if Beta_score > objf(betatemp_pos):
        Beta_pos =  betatemp_pos
        Beta_score = objf(betatemp_pos)
        reward += 15
        if state == 2:  # none of them update
            state = 1   # except Alpha, others update
        # reward += (objf(Beta_pos) - objf(betatemp_pos))

    deltatemp_pos = Delta_pos - Delta_pos * r
    deltatemp_pos = numpy.clip(deltatemp_pos, lb, ub)
    if Delta_score > objf(deltatemp_pos):
        Delta_pos = deltatemp_pos
        Delta_score = objf(deltatemp_pos)
        reward += 5
        if state == 2:  # none of them update
            state = 1   # except Alpha, others update

    return Alpha_pos, Beta_pos, Delta_pos,Alpha_score, Beta_score, Delta_score, reward, state


def ProposedMethod(objf, lb, ub, dim, SearchAgents_no, Max_iter):

    # dim = 50
    global a
    global dimension
    dimension = dim
    a = 2.0

    # Initialize parameter
    tolerance = 1e-6  # مقدار حداقلی که تغییرات باید از آن کمتر باشد تا همگرا تلقی شود
    convergence_iteration = None
    previous_best = None
    current_best = None
    stability_count = 0
    stable_iter = 20

    # initialize alpha, beta, and delta_pos
    Alpha_pos = numpy.zeros(dim)
    Alpha_score = float("inf")

    Beta_pos = numpy.zeros(dim)
    Beta_score = float("inf")

    Delta_pos = numpy.zeros(dim)
    Delta_score = float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim


    # Q-Learning initial parameters
    state_space_update = 3  # -1, 1 (the fitness after the action_update gets better or not)
    action_space_update = 3  # 0: levy update function, 1: cauchy update function 3: normally update leader
    q_table_updteLeader = numpy.zeros((state_space_update, action_space_update))
    learning_rate = 0.01
    discount_factor = 0.85
    epsilon = 1.0
    decay = 0.995
    min_epsilon = 0.15
    state_update = 0     #[0: Alpha ok, 1: beta or delta ok, 2: none of them}


    # Create 30 different instances of RingBuffer
    buffers = [RingBuffer(3) for _ in range(SearchAgents_no)]

    state_mapping = {"1115": 0, "1112": 1, "11-15": 2, "11-12": 3, "1-115": 4, "1-112": 5, "1-1-15": 6, "1-1-12": 7, "-1115": 8, "-1112": 9, "-11-15": 10, "-11-12": 11, "-1-115": 12, "-1-112": 13, "-1-1-15": 14, "-1-1-12": 15}   # last state of grey wolf (if fitness gets better: +1 else -1) 5: 50 25 25  2: 25 50 25

    # State including 4 actions
    # state_mapping = {"1115": 0, "1112": 1, "11-15": 2, "11-12": 3, "1-115": 4, "1-112": 5, "1-1-15": 6, "1-1-12": 7, "-1115": 8, "-1112": 9, "-11-15": 10, "-11-12": 11, "-1-115": 12, "-1-112": 13, "-1-1-15": 14, "-1-1-12": 15,
    #                  "1116": 16, "1117": 17, "11-16": 18, "11-17": 19, "1-116": 20, "1-117": 21, "1-1-16": 22, "1-1-17": 23, "-1116": 24, "-1117": 25, "-11-16": 26, "-11-17": 27, "-1-116": 28, "-1-117": 29, "-1-1-16": 30, "-1-1-17": 31}   # last state of grey wolf (if fitness gets better: +1 else -1) 5: 50 25 25  2: 25 50 25

    # state_mapping = {"115": 0, "112": 1, "1-15": 2, "1-12": 3,"-115": 4, "-112": 5, "-1-15": 6, "-1-12": 7, "116": 8, "117": 9, "1-16": 10, "1-17": 11, "-116": 12, "-117": 13, "-1-16": 14, "-1-17": 15}  # last state of grey wolf (if fitness gets better: +1 else -1) 5: 50 25 25  2: 25 50 25

    # state_mapping = {"111": 0, "11-1": 1, "1-11": 2, "1-1-1": 3, "-111": 4, "-11-1": 5, "-1-11": 6, "-1-1-1": 7}   # last state of grey wolf (if fitness gets better: +1 else -1) 5: 50 25 25  2: 25 50 25
    # state_mapping = {"11115": 0, "11-115": 1, "1-1115": 2, "1-1-115": 3, "-11115": 4, "-11-115": 5, "-1-1115": 6, "-1-1-115": 7, "111-15": 8, "11-1-15": 9, "1-11-15": 10, "1-1-1-15": 11, "-111-15": 12, "-11-1-15": 13, "-1-11-15": 14, "-1-1-1-15": 15,
    #                  "11112": 16, "11-112": 17, "1-1112": 18, "1-1-112": 19, "-11112": 20, "-11-112": 21, "-1-1112": 22, "-1-1-112": 23, "111-12": 24, "11-1-12": 25, "1-11-12": 26, "1-1-1-12": 27, "-111-12": 28, "-11-1-12": 29, "-1-11-12": 30, "-1-1-1-12": 31}   # last state of grey wolf (if fitness gets better: +1 else -1) 5: 50 25 25  2: 25 50 25
    # state_mapping = {"115": 0, "112": 1, "1-15": 2, "1-12": 3, "-115": 4, "-112": 5, "-1-15": 6, "-1-12": 7}   # last state of grey wolf (if fitness gets better: +1 else -1) 5: 50 25 25  2: 25 50 25
    # state_mapping = {"11": 0, "1-1": 1, "-11": 2, "-1-1": 3}   # last state of grey wolf (if fitness gets better: +1 else -1) 5: 50 25 25  2: 25 50 25
    num_actions = 4  # a and change factor
    num_states = len(state_mapping)
    q_table = numpy.zeros((num_states, num_actions))
    last_state = numpy.zeros(SearchAgents_no, dtype=int)   # last value of each wolf state
    last_a = numpy.full(SearchAgents_no, 2.0)



    # Initialize the positions of search agents
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = (
                numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        )

    Convergence_curve = numpy.zeros(Max_iter)
    s = solution()
    timerStart = time.time()


    for i in range(0, SearchAgents_no):

        # Return back the search agents that go beyond the boundaries of the search space
        Positions[i] = numpy.clip(Positions[i], lb, ub)

        # Calculate objective function for each search agent
        fitness = objf(Positions[i, :])

        # Update Alpha, Beta, and Delta
        if fitness < Alpha_score:
            Delta_score = Beta_score  # Update delta
            Delta_pos = Beta_pos.copy()
            Beta_score = Alpha_score  # Update beta
            Beta_pos = Alpha_pos.copy()
            Alpha_score = fitness
            # Update alpha
            Alpha_pos = Positions[i, :].copy()

        if fitness > Alpha_score and fitness < Beta_score:
            Delta_score = Beta_score  # Update delta
            Delta_pos = Beta_pos.copy()
            Beta_score = fitness  # Update beta
            Beta_pos = Positions[i, :].copy()

        if fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score:
            Delta_score = fitness  # Update delta
            Delta_pos = Positions[i, :].copy()

    # Main loop
    for l in range(0, Max_iter):
        previous_best = current_best

        for i in range(0, SearchAgents_no):

            # Calculate objective function for each search agent
            fitness = objf(Positions[i, :])

            # Update Alpha, Beta, and Delta
            if fitness < Alpha_score:
                Delta_score = Beta_score  # Update delta
                Delta_pos = Beta_pos.copy()
                Beta_score = Alpha_score  # Update beta
                Beta_pos = Alpha_pos.copy()
                Alpha_score = fitness
                # Update alpha
                Alpha_pos = Positions[i, :].copy()

            if fitness > Alpha_score and fitness < Beta_score:
                Delta_score = Beta_score  # Update delta
                Delta_pos = Beta_pos.copy()
                Beta_score = fitness  # Update beta
                Beta_pos = Positions[i, :].copy()

            if fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score:
                Delta_score = fitness  # Update delta
                Delta_pos = Positions[i, :].copy()


        if numpy.random.rand() < epsilon:
            action_update = numpy.random.choice(action_space_update)
        else:
            action_update = numpy.argmax(q_table_updteLeader[state_update])

        next_state = 2
        leaderpop = [Alpha_pos, Beta_pos, Delta_pos]
        reward = -2
        if action_update == 2:
            for k in range(len(leaderpop)):
                A1, A2, A3 = a * (2 * random.random() - 1), a * (
                        2 * random.random() - 1), a * (2 * random.random() - 1)
                C1, C2, C3 = 2 * random.random(), 2 * random.random(), 2 * random.random()

                X1 = numpy.zeros(dim) # dim = 30
                X2 = numpy.zeros(dim)
                X3 = numpy.zeros(dim)
                Xnew = numpy.zeros(dim)
                for j in range(dim):
                    X1[j] = Alpha_pos[j] - A1 * abs(
                        C1 * Alpha_pos[j] - leaderpop[k][j])
                    X2[j] = Beta_pos[j] - A2 * abs(
                        C2 * Beta_pos[j] - leaderpop[k][j])
                    X3[j] = Delta_pos[j] - A3 * abs(
                        C3 * Delta_pos[j] - leaderpop[k][j])
                    Xnew[j] += (X1[j] + X2[j] + X3[j]) / 3
                if objf(leaderpop[k]) > objf(Xnew):
                    leaderpop[k] = Xnew
                    if k == 0:
                        # reward += 20
                        reward += (objf(leaderpop[k]) - objf(Xnew))
                        next_state = 0
                        # reward += (objf(leaderpop[k]) - objf(Xnew))
                    elif k == 1:
                        reward += 5
                        if next_state == 2:
                            next_state = 1
                    else:
                        reward += 3
                        if next_state == 2:
                            next_state = 1


        else:
            Alpha_pos, Beta_pos, Delta_pos,Alpha_score, Beta_score, Delta_score, reward, next_state = updateLeaderByAction(action_update, objf, a, Alpha_pos, Beta_pos,
                                                                          Delta_pos, Alpha_score, Beta_score, Delta_score, lb, ub)
            # print(a)
        # if reward < 0:  # state_update is bad (can't improve the fitness)
        #     next_state = 0  # 0: is bad state_update
        # else:
        #     next_state = 1

        if Beta_score < Alpha_score:
            Alpha_pos, Beta_pos = Beta_pos, Alpha_pos

        # Insert arr[2]
        if Delta_score < Beta_score:
            Beta_pos, Delta_pos = Delta_pos, Beta_pos
            if (Beta_score < Alpha_score):
                Beta_pos, Alpha_pos = Alpha_pos, Beta_pos


        max_q_next = numpy.max(q_table_updteLeader[next_state])
        new_q = q_table_updteLeader[state_update, action_update] + learning_rate * (
                    reward + discount_factor * max_q_next - q_table_updteLeader[state_update, action_update])
        q_table_updteLeader[state_update, action_update] = new_q
        epsilon = max(min_epsilon, epsilon * decay)

        state_update = next_state



        # Update the Position of search agents including omegas
        for i in range(0, SearchAgents_no):

            state = last_state[i]    # last state of i's wolf

            # Choosing action using e-greedy
            if random.uniform(0, 1) < epsilon:
                action = random.choice(range(num_actions))
            else:
                action = numpy.argmax(q_table[state, :])

            tt = 2 * math.cos((math.pi / 2) * (l / Max_iter))
            bb = abs(math.cos(l * math.pi / Max_iter)) * 2

            if action == 0:
                a = tt
                ad = 0.6
                bd = 0.3
                cd = 0.1
            elif action == 1:
                a = tt
                # ad = 0.4
                # bd = 0.3
                # cd = 0.3
                ad = 1
                bd = 1
                cd = 1

                # a = 2 - l * ((2) / Max_iter)
            elif action == 2:
                a = bb
                ad = 0.6
                bd = 0.3
                cd = 0.1

            elif action == 3:
                a = bb
                # a = 2 - math.cos((math.pi) * (l / Max_iter))
                # a = 0.5 * abs(math.sin(math.pi / (( l / Max_iter) + 1)))
                # a = abs(2 * math.cos((math.pi) * (l / Max_iter)))
                # ad = 0.4
                # bd = 0.3
                # cd = 0.3
                ad = 1
                bd = 1
                cd = 1

            last_a[i] = a

            old_fit = objf(Positions[i, :])

            for j in range(0, dim):
                r1 = random.random()  # r1 is a random number in [0,1]
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a
                # Equation (3.3)
                C1 = 2 * r2
                # Equation (3.4)

                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                # Equation (3.5)-part 1
                X1 = Alpha_pos[j] - A1 * D_alpha
                # Equation (3.6)-part 1

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a
                # Equation (3.3)
                C2 = 2 * r2
                # Equation (3.4)

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                # Equation (3.5)-part 2
                X2 = Beta_pos[j] - A2 * D_beta
                # Equation (3.6)-part 2

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a
                # Equation (3.3)
                C3 = 2 * r2
                # Equation (3.4)

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                # Equation (3.5)-part 3
                X3 = Delta_pos[j] - A3 * D_delta
                # Equation (3.5)-part 3

                # Positions[i, j] = (X1 + X2 + X3) / 3  # Equation (3.7)
                Positions[i, j] = ((ad * X1) + (bd * X2) + (cd * X3)) / (ad + bd + cd)

            # Return back the search agents that go beyond the boundaries of the search space
            Positions[i] = numpy.clip(Positions[i], lb, ub)
            new_fit = objf(Positions[i, :])

            # TODO: we should compare new fitness of wolf with old one
            if action == 0 or action == 2:
                t = 5
            else:
                t = 2


            if old_fit > new_fit:  # it gets better
                buffers[i].add(1)
                reward = 1
                reward = old_fit - new_fit
                # print("reward" + str(reward))
                # if i == 0:9

            elif old_fit <= new_fit:  # it gets worse
                buffers[i].add(-1)
                # reward = -3
                reward = 0
                # reward = old_fit - new_fit
                # print(reward)

            if l > 3:
                fit_action = str(buffers[i].get_e_findGwo()) + str(t)
                # fit_action = str(buffers[i].get_e_findGwo())
                next_state = state_mapping.get(fit_action)
            # print(buffers[i].get_e_findGwo())
            else:
                next_state = 0

            # TODO: update Q-value  using the Q-learning update rule
            q_table[state, action] = q_table[state, action] + learning_rate * (
                        reward + discount_factor * numpy.max(q_table[next_state, :])) - q_table[state, action]
            last_state[i] = next_state
            # learning_rate = epsilon

        Convergence_curve[l] = Alpha_score
        current_best = Alpha_score

        # Check is it converged or not
        if previous_best is not None and abs(current_best - previous_best) < tolerance:
            # print("current_best: " + str(current_best) + "   previous_best : " + str(previous_best) + "  minus: " + str(abs(current_best - previous_best)))
            stability_count += 1
        else:
            stability_count = 0

        if stability_count >= stable_iter and convergence_iteration is None:  # اگر 10 تکرار پشت سر هم تغییر قابل توجهی نداشته باشد
            convergence_iteration = l

        if l == Max_iter - 1:
            print(Alpha_score)
            # print(min(Convergence_curve))

    if convergence_iteration is None:
        convergence_iteration = l

    timerEnd = time.time()
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "ProposedMethod"
    s.objfname = objf.__name__
    s.best = Alpha_score
    # s.conviter = convergence_iteration
    # s.valueConv = Convergence_curve[convergence_iteration]
    # s.stopiter = l

    return s