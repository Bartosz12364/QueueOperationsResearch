from enum import Enum
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np


class Task:
    class State(Enum):
        BEFORE_QUEUE = "Before queue"
        IN_QUEUE = "In queue"
        EXECUTING = "Executing"
        DONE = "Done"

    def __init__(self, t_plus, b, state=State.BEFORE_QUEUE):
        self.state = state
        self.t_plus = t_plus
        self.b = b
        self.b_done = 0
        self.in_queue_id = None

    def do_task(self, b_count):
        assert (self.state == Task.State.EXECUTING)
        self.b_done += b_count
        self.b_done = round(self.b_done, 5)  # numerical problems
        if self.b <= self.b_done:
            self.b_done = self.b
            self.state = Task.State.DONE

    def move_before_queue(self, system, wtime):
        assert (self.state == Task.State.BEFORE_QUEUE)
        # self.t_plus -= t_delta
        if self.t_plus <= wtime:
            self.state = Task.State.IN_QUEUE
            self.in_queue_id = system.give_in_queue_id()


class Processor:
    def __init__(self, v):
        self.v = v
        self.current_task = None

    def process(self, t_delta):
        if self.current_task:

            b = self.v * t_delta
            self.current_task.do_task(b)
            if self.current_task.state == Task.State.DONE:
                self.current_task = None
        return self.current_task is not None

    def set_task(self, task):
        assert (self.current_task is None)
        self.current_task = task
        task.state = Task.State.EXECUTING

    def is_doing_task(self):
        return self.current_task is not None


class System:
    class GraphType(Enum):
        BOTH = "Both"
        N = "n"
        U = "u"

    def __init__(self, name, tasks, t_delta=0.01, time_limit=10, processors=None):
        self.name = name
        self.t_delta = t_delta
        self.time_limit = time_limit
        self.tasks = tasks
        self.in_queue_id = 0
        self.rev_t = 100
        self.processors = processors
        self.nt = []
        self.ut = []

    def give_in_queue_id(self):
        self.in_queue_id += 1
        return self.in_queue_id - 1

    def print_state(self):
        print("TASKS BEFORE QUEUE")
        print([self.tasks.index(task) for task in self.tasks if task.state == task.State.BEFORE_QUEUE])
        print("TASKS IN QUEUE")
        print([self.tasks.index(task) for task in self.tasks if task.state == task.State.IN_QUEUE])
        print("TASKS BEING EXECUTED")
        print([self.tasks.index(task) for task in self.tasks if task.state == task.State.EXECUTING])
        print("TASKS DONE")
        print([self.tasks.index(task) for task in self.tasks if task.state == task.State.DONE])

    def main_loop(self):
        for time in range(0, self.time_limit * self.rev_t, int(self.t_delta * self.rev_t)):
            print("\n\n\n")
            print(time / self.rev_t)

            for task in [task for task in self.tasks if task.state == task.State.BEFORE_QUEUE]:
                task.move_before_queue(self, time / self.rev_t)
            for processor in self.processors:
                if not processor.process(self.t_delta):
                    self.attach_task_to_processor(processor)
            self.calc_graphs()
            self.print_state()

    def attach_task_to_processor(self, processor):
        tasks_in_queue = [task for task in self.tasks if task.state == task.State.IN_QUEUE]
        if len(tasks_in_queue) > 0:
            task = \
                sorted(tasks_in_queue, key=lambda task: task.in_queue_id)[0]
            processor.set_task(task)

    def calc_graphs(self):
        current_u = 0
        tasks = [task for task in self.tasks if task.state in [Task.State.IN_QUEUE, Task.State.EXECUTING]]
        for task in tasks:
            current_u += task.b - task.b_done
        self.nt.append(len(tasks))
        self.ut.append(current_u)

    def plot_graphs(self, type):
        if type != System.GraphType.N:
            plt.plot([x / self.rev_t for x in range(0, self.time_limit * self.rev_t, int(self.t_delta * self.rev_t))],
                     self.ut, label='u(t) %s' % self.name)
        if type != System.GraphType.U:
            plt.plot([x / self.rev_t for x in range(0, self.time_limit * self.rev_t, int(self.t_delta * self.rev_t))],
                     self.nt, label='n(t) %s' % self.name)


def plot_graphs_from_multiple_systems(systems, type):
    plt.style.use('seaborn')
    for system in systems:
        system.plot_graphs(type)
    plt.xticks(np.arange(0, 20, step=1))
    plt.yticks(np.arange(0, 20, step=1))
    plt.xlabel("Time [s]")
    plt.ylabel("Values")
    plt.legend()
    plt.show()


tasks = [
    Task(t_plus=1, b=10),
    Task(t_plus=2, b=5),
    Task(t_plus=3, b=2),
    Task(t_plus=6, b=1),
]

systems = [
    System("Single Processor v=2", deepcopy(tasks), time_limit=20,
           processors=deepcopy([Processor(v=2)])),
    System("Single Processor v=1", deepcopy(tasks), time_limit=20,
           processors=deepcopy([Processor(v=1)])),
    System("Double Processor v=1", deepcopy(tasks), time_limit=20,
           processors=deepcopy([Processor(v=1), Processor(v=1)]))
]

for system in systems:
    system.main_loop()

plot_graphs_from_multiple_systems(systems, type=System.GraphType.U)
