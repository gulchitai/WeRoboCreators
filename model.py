import numpy as np
from heapq import heappop, heappush
import gym
from pogema.animation import AnimationMonitor
from pogema import GridConfig
from collections import Counter
from pogema.wrappers.metrics import MetricsWrapper


class Node:
    def __init__(self, coord: (int, int) = (0, 0), g: int = 0, h: int = 0):
        self.i, self.j = coord
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f or ((self.f == other.f) and (self.g < other.g))


class AStar:
    def __init__(self):
        self.start = (0, 0)
        self.goal = (0, 0)
        self.max_steps = 10000  # due to the absence of information about the map size we need some other stop criterion
        self.OPEN = list()
        self.CLOSED = dict()
        self.obstacles = set()
        self.other_agents = set()
        self.history = list()
        self.cntSkip = 0
        self.cntRandom = 0

    def compute_shortest_path(self, start, goal):
        self.start = start
        self.goal = goal
        self.CLOSED = dict()
        self.OPEN = list()
        heappush(self.OPEN, Node(self.start))
        u = Node()
        steps = 0
        while len(self.OPEN) > 0 and steps < self.max_steps and (u.i, u.j) != self.goal:
            u = heappop(self.OPEN)
            steps += 1
            for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                n = (u.i + d[0], u.j + d[1])
                if n not in self.obstacles and n not in self.CLOSED and n not in self.other_agents:
                    #h = abs(n[0] - self.goal[0]) + abs(
                    #    n[1] - self.goal[1])  # Manhattan distance as a heuristic function
                    h = np.sqrt(np.power(n[0] - self.goal[0], 2) + np.power(n[1] - self.goal[1], 2))
                    heappush(self.OPEN, Node(n, u.g + 1, h))
                    self.CLOSED[n] = (u.i, u.j)  # store information about the predecessor

    def get_next_node(self):
        next_node = self.start  # if path not found, current start position is returned
        if self.goal in self.CLOSED:  # if path found
            next_node = self.goal
            while self.CLOSED[next_node] != self.start:  # get node in the path with start node as a predecessor
                next_node = self.CLOSED[next_node]
            self.history.append(next_node)
        return next_node

    def update_obstacles(self, obs, other_agents, n):
        obstacles = np.transpose(np.nonzero(obs))  # get the coordinates of all obstacles in current observation
        for obstacle in obstacles:
            self.obstacles.add((n[0] + obstacle[0], n[1] + obstacle[1]))  # save them with correct coordinates
        self.other_agents.clear()  # forget previously seen agents as they move
        agents = np.transpose(np.nonzero(other_agents))  # get the coordinates of all agents that are seen
        for agent in agents:
            self.other_agents.add((n[0] + agent[0], n[1] + agent[1]))  # save them with correct coordinates


class Model:
    def __init__(self):
        self.agents = None
        self.actions = {tuple(GridConfig().MOVES[i]): i for i in
                        range(len(GridConfig().MOVES))}  # make a dictionary to translate coordinates of actions into id

    def act(self, obs, done, positions_xy, targets_xy) -> list:
        if self.agents is None:
            self.agents = [AStar() for _ in range(len(obs))]  # create a planner for each of the agents
        actions = []
        for k in range(len(obs)):
            if positions_xy[k] == targets_xy[k]:  # don't waste time on the agents that have already reached their goals
                actions.append(0)  # just add useless action to save the order and length of the actions
                continue

            done1 = sum([positions_xy[ag] == targets_xy[ag] for ag in range(len(obs))])
            cnt = Counter(self.agents[k].history)

            if (len(self.agents[k].history) > 0) and (cnt.most_common()[0][1] > 10): #если осталось меньше 10 роботов и они не могут разойтись, надо одному идти, а остальным остановиться на 5 ходов
                if (len(obs) - done1 <=10):
                    self.agents[k].history.clear()
                    for ag in range(len(obs)): #обходим всех роботов
                        if (positions_xy[ag] != targets_xy[ag]) and (ag != k): #робот еще не достиг цели и это не текущий робот
                            self.agents[ag].cntSkip = 3

            if self.agents[k].cntSkip > 0: # пропускаем нужное число ходов
                self.agents[k].cntSkip = self.agents[k].cntSkip - 1
                actions.append(0)
                continue

            #if self.agents[k].cntRandom > 0:
            #    self.agents[k].cntRandom = self.agents[k].cntRandom - 1
            #    actions.append(np.random.randint(5))
            #    continue

            if (len(self.agents[k].history) > 0) and (cnt.most_common()[0][1] > 10):
                self.agents[k].history.clear()
                # случайным образом выберем один из двух вариантов
                # 1. Сделать случайный ход
                # 2. Сделать 5 пропусков хода подряд
                r = np.random.randint(2)
                if r == 0:
                    #self.agents[k].cntRandom = 5
                    actions.append(np.random.randint(5))
                else:
                    self.agents[k].cntSkip = 5
                    actions.append(0)
                continue
            self.agents[k].update_obstacles(obs[k][0], obs[k][1], (positions_xy[k][0] - 5, positions_xy[k][1] - 5))
            self.agents[k].compute_shortest_path(start=positions_xy[k], goal=targets_xy[k])
            next_node = self.agents[k].get_next_node()
            actions.append(self.actions[(next_node[0] - positions_xy[k][0], next_node[1] - positions_xy[k][1])])
        return actions


def main():
    # Define random configuration
    grid_config = GridConfig(num_agents=64,  # количество агентов на карте
                             size=64,  # размеры карты
                             density=0.3,  # плотность препятствий
                             seed=7,  # сид генерации задания
                             max_episode_steps=256,  # максимальная длина эпизода
                             obs_radius=5,  # радиус обзора
                             )

    env = gym.make("Pogema-v0", grid_config=grid_config)
    env = AnimationMonitor(env)

    # обновляем окружение
    obs = env.reset()

    done = [False for k in range(len(obs))]
    solver = Model()
    steps = 0
    while not all(done):
        # Используем AStar
        obs, reward, done, info = env.step(solver.act(obs, done,
                                                      env.get_agents_xy_relative(),
                                                      env.get_targets_xy_relative()))
        steps += 1
        print(steps, np.sum(done))

    env = MetricsWrapper(env)
    CSR = info[0]['metrics']['CSR']
    ISR = np.mean([x['metrics']['ISR'] for x in info])
    print("CSR = ", CSR, " ISR = ", ISR)
    # сохраняем анимацию и рисуем ее
    env.save_animation("render.svg", egocentric_idx=None)


if __name__ == '__main__':
    main()
