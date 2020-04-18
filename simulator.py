import queue
from enum import Enum
from queue import SimpleQueue

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
        self.t_minus = None
        self.b = b
        self.b_done = 0
        self.in_queue_id = None

    def do_task(self, b_count, wtime):
        assert (self.state == Task.State.EXECUTING)
        self.b_done += b_count
        self.b_done = round(self.b_done, 5)  # numerical problems
        if self.b <= self.b_done:
            self.b_done = self.b
            self.state = Task.State.DONE

            self.t_minus = wtime
            print("TIMEL:", self.t_plus, self.t_minus)

    def move_before_queue(self, wtime, system):
        assert (self.state == Task.State.BEFORE_QUEUE)
        # self.t_plus -= t_delta
        if self.t_plus <= wtime:
            self.state = Task.State.IN_QUEUE
            system.scheduler.add_task(self)


class Scheduler:
    def __init__(self):
        pass

    def add_task(self, task: Task):
        pass

    def get_task_to_process(self, processor):

        pass


class RRScheduler(Scheduler):
    def __init__(self):
        self.queue = queue.Queue(500)
        super().__init__()

    def add_task(self, task: Task):
        self.queue.put(task)

    def set_processor_task(self, processor):
        if not processor.current_task and not self.queue.empty():
            task = self.queue.get()
            processor.set_task(task)


class Processor:
    def __init__(self, v, qtime=None):
        self.v = v
        self.current_task = None
        self.time_limit = qtime
        self.time_spent = 0

    def set_task(self, task):
        assert (self.current_task is None)
        assert (task.state == Task.State.IN_QUEUE)
        self.time_spent = 0
        task.state = Task.State.EXECUTING
        self.current_task = task

    def process(self, t_delta, scheduler, wtime):
        if self.current_task:
            b = self.v * t_delta
            self.current_task.do_task(b, wtime)
            if self.current_task.state == Task.State.DONE:
                self.current_task = None
                return False
            self.time_spent += t_delta
            self.time_spent = round(self.time_spent, 4)
            if self.time_limit is not None and self.time_limit <= self.time_spent:
                self.current_task.state = Task.State.IN_QUEUE
                scheduler.add_task(self.current_task)
                self.current_task = None
                return False
        return self.current_task is not None


class System:
    class GraphType(Enum):
        BOTH = "Both"
        N = "n"
        U = "u"

    def __init__(self, name, tasks, t_delta=0.01, time_limit=10, processors=None, scheduler=None):
        self.scheduler = scheduler
        self.name = name
        self.t_delta = t_delta
        self.time_limit = time_limit
        self.tasks = tasks
        self.in_queue_id = 0
        self.rev_t = 100
        self.qtime = qtime
        self.average_delay = None
        self.processors = processors
        self.nt = []
        self.ut = []

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
            # print("\n\n\n")
            # print(time / self.rev_t)

            for task in [task for task in self.tasks if task.state == task.State.BEFORE_QUEUE]:
                task.move_before_queue(time / self.rev_t, self)
            for processor in self.processors:
                if not processor.process(self.t_delta, self.scheduler, time / self.rev_t):
                    self.scheduler.set_processor_task(processor)
            self.calc_graphs()
            self.print_state()

    def calc_average_delay(self):
        sum = 0
        for task in self.tasks:
            sum += task.t_minus-task.t_plus
        self.average_delay = sum/len(self.tasks)


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
                     self.ut, label='%s%s' % ("u(t) " if type == System.GraphType.BOTH else "", self.name))
        if type != System.GraphType.U:
            plt.plot([x / self.rev_t for x in range(0, self.time_limit * self.rev_t, int(self.t_delta * self.rev_t))],
                     self.nt, label='%s%s' % ("n(t) " if type == System.GraphType.BOTH  else "", self.name))


def plot_graphs_from_multiple_systems(systems, type):
    plt.style.use('seaborn')
    for system in systems:
        system.plot_graphs(type)
    plt.xticks(np.arange(0, 13, step=1))
    plt.yticks(np.arange(0, 13, step=1))
    plt.xlabel("Time [s]")
    if type == System.GraphType.U:
        plt.ylabel("U(t)")
    elif type == System.GraphType.N:
        plt.ylabel("N(t)")
    else:
        plt.ylabel("values (U(t) or N(t))")
    plt.legend()
    plt.show()


tasks = [
    Task(t_plus=1, b=7),
    Task(t_plus=1, b=1),
    Task(t_plus=1, b=3),
]

RRScheduler = RRScheduler()

qtime = 2
systems = [
    System("1x Processor v=1, RR", deepcopy(tasks), time_limit=13,
           processors=deepcopy([Processor(v=1, qtime=qtime)]), scheduler=RRScheduler),
    # System("Single Processor v=1", deepcopy(tasks), time_limit=20,
    #        processors=deepcopy([Processor(v=1, qtime=qtime), Processor(v=1, qtime=qtime)]), scheduler=fifoScheduler),
    # System("Double Processor v=1", deepcopy(tasks), time_limit=20,
    #        processors=deepcopy([Processor(v=2, qtime=qtime)]), scheduler=fifoScheduler),
    # System("1x Processor v=1, FIFO", deepcopy(tasks), time_limit=13,
    #        processors=deepcopy([Processor(v=1, qtime=None)]), scheduler=RRScheduler),
    # System("2x Processor v=1, FIFO", deepcopy(tasks), time_limit=10,
    #        processors=deepcopy([Processor(v=1, qtime=None), Processor(v=1, qtime=None)]), scheduler=RRScheduler),
    # System("Double Processor v=1", deepcopy(tasks), time_limit=20,
    #        processors=deepcopy([Processor(v=2, qtime=None)]), scheduler=fifoScheduler),
]

for system in systems:
    system.main_loop()
    system.calc_average_delay()

for system in systems:
    print(system.name)
    print(system.average_delay)
plot_graphs_from_multiple_systems(systems, type=System.GraphType.U)
plot_graphs_from_multiple_systems(systems, type=System.GraphType.N)