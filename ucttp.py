from copy import deepcopy
import random

import pandas as pd
import numpy as np
import time


class Chromosome(list):
    def fitness_value(self):
        num_of_conflicts = 0
        n = len(self)
        for i in range(n):
            if i < n / 2 and self[i][1] != self[i + int(n / 2)][1]:  # different profs for same course
                num_of_conflicts += 1
            if free_times[self[i][1]][self[i][0] % num_of_timeslots] == 0:  # no free time for prof
                num_of_conflicts += 1
            for j in range(i):
                if self[i][1] == self[j][1] and self[i][0] % num_of_timeslots == self[j][0] % num_of_timeslots:
                    #  one profs in two rooms at same time
                    num_of_conflicts += 1
        # print(len(self.conflicts))
        return 1 / (1 + num_of_conflicts)

    def compute_conflicts(self):
        conflicts = []
        n = len(self)
        for i in range(n):
            if i < n / 2 and self[i][1] != self[i + int(n / 2)][1]:  # different profs for same course
                conflicts.append((1, i, i + int(n / 2)))
            if free_times[self[i][1]][self[i][0] % num_of_timeslots] == 0:  # no free time for prof
                conflicts.append((2, i))
            for j in range(i):
                if self[i][1] == self[j][1] and self[i][0] % num_of_timeslots == self[j][0] % num_of_timeslots:
                    #  one profs in two rooms at same time
                    conflicts.append((3, i, j))
        # print(len(self.conflicts))
        return conflicts

    def when_where(self):
        return [i[0] for i in self]


class Population(list):
    def __init__(self, num=0, length=0):
        super().__init__()
        for i in range(length):
            while True:
                rnd = random.sample(range(len(rooms_timetable)), 2 * num)
                if len(rnd) == len(set(rnd)):
                    break
            c = [-1] * (2 * num)
            for j in range(num):
                rnd2 = random.sample(course_prof[courses[j]], 1).__getitem__(0)
                c[j] = rnd2
                c[j + num] = rnd2
            self.append(Chromosome(list(zip(rnd, c))))

    def fit(self):
        f = []
        for p in self:
            f.append(Chromosome(p).fitness_value())
            return f


skill_file = "./information/Prof_Skill.xlsx"
free_time_file = "./information/Proffosor_FreeTime.xlsx"
am = "./information/Amoozesh.xlsx"


def load_files():
    prof_skill = pd.read_excel(skill_file, sheet_name=None)
    prof_free_time = pd.read_excel(free_time_file, sheet_name=None)
    amoozesh = pd.read_excel(am, sheet_name=None)
    return prof_skill, prof_free_time, amoozesh


def selection(pop):
    selected = pop[:round(0.2 * len(pop))]
    selected += random.sample(pop[round(0.2 * len(pop)):], round(0.4 * len(pop)))
    return selected


def proportional_fitness_selection(populationn):
    maxx = sum([Chromosome(c).fitness_value() for c in populationn])
    pick = random.uniform(0, maxx)
    current = 0
    for chromosome in populationn:
        current += Chromosome(chromosome).fitness_value()
        if current > pick:
            return chromosome


def crossover(pop):
    children = Population()
    while True:
        if len(children) >= 0.78 * len(population):
            break
        rnd = random.choices(pop, weights=Population(pop).fit(), k=2)
        chrm1 = Chromosome(deepcopy(rnd[0]))
        chrm2 = Chromosome(deepcopy(rnd[1]))
        room_timeslot_chrm1 = chrm1.when_where()
        room_timeslot_chrm2 = chrm2.when_where()
        c_points = random.sample(range(len(courses)), 2)
        for x in range(c_points[0], c_points[1], -1 if c_points[0] > c_points[1] else 1):
            chrm1[x], chrm2[x + len(courses)] = chrm2[x + len(courses)], chrm1[x]
            # chrm2[x], chrm1[x + len(courses)] = chrm1[x + len(courses)], chrm2[x]
            # chrm1[x], chrm2[x] = chrm2[x], chrm1[x]
            # chrm1[x + len(courses)], chrm2[x + len(courses)] = chrm2[x + len(courses)], chrm1[x + len(courses)]

            if x < len(courses) and chrm1[x][1] != chrm1[x + len(courses)][1]:
                if free_times[chrm1[x][1]][chrm1[x + len(courses)][0] % num_of_timeslots] == 1:
                    chrm1[x + len(courses)] = (chrm1[x + len(courses)][0], chrm1[x][1])
                else:
                    chrm1[x] = (chrm1[x][0], chrm1[x + len(courses)][1])
            if x > len(courses) and chrm1[x][1] != chrm1[x - len(courses)][1]:
                if free_times[chrm1[x][1]][chrm1[x - len(courses)][0] % num_of_timeslots] == 1:
                    chrm1[x - len(courses)] = (chrm1[x - len(courses)][0], chrm1[x][1])
                else:
                    chrm1[x] = (chrm1[x][0], chrm1[x - len(courses)][1])
            if x < len(courses) and chrm2[x][1] != chrm2[x + len(courses)][1]:
                if free_times[chrm2[x][1]][chrm2[x + len(courses)][0] % num_of_timeslots] == 1:
                    chrm2[x + len(courses)] = (chrm2[x + len(courses)][0], chrm2[x][1])
                else:
                    chrm2[x] = (chrm2[x][0], chrm2[x + len(courses)][1])
            if x > len(courses) and chrm2[x][1] != chrm2[x - len(courses)][1]:
                if free_times[chrm2[x][1]][chrm2[x - len(courses)][0] % num_of_timeslots] == 1:
                    chrm2[x - len(courses)] = (chrm2[x - len(courses)][0], chrm2[x][1])
                else:
                    chrm2[x] = (chrm2[x][0], chrm2[x - len(courses)][1])
            while chrm1[x][0] in room_timeslot_chrm1:
                chrm1[x] = (random.sample(range(len(rooms_timetable)), 1)[0], chrm1[x][1])
            room_timeslot_chrm1[x] = chrm1[x][0]
            while chrm2[x][0] in room_timeslot_chrm2:
                chrm2[x] = (random.sample(range(len(rooms_timetable)), 1)[0], chrm2[x][1])
            room_timeslot_chrm2[x] = chrm2[x][0]
        children.append(chrm1)
        children.append(chrm2)
    return children


def mutation(pop, rate=1.):
    children = Population()
    c1 = random.sample(pop, round(rate * len(population)))
    for k in deepcopy(c1[:]):
        c = deepcopy(k[:])
        while True:
            rnd = random.sample(range(len(k)), 2)
            if abs(rnd[0] - rnd[1]) != len(k) / 2:
                break
        c[rnd[0]], c[rnd[1]] = c[rnd[1]], c[rnd[0]]
        children.append(Chromosome(c))
    return children


def find_free_prof(time_room):
    ttime = time_room % num_of_timeslots
    for prof in free_times.keys():
        if free_times[prof][ttime] == 1:
            return prof
    return -1


def all_same(arr):
    val = Chromosome(arr[0]).fitness_value()
    for x in arr:
        if Chromosome(x).fitness_value() != val:
            return False
    return True


iters = []
ts = []
for a in range(10):
    startTime = time.time()
    professorSkill, professorFreeTime, classes = load_files()

    professorSkill = professorSkill[[i for i in professorSkill.keys()].__getitem__(0)]

    profs = [i for i in professorFreeTime.keys()]

    courses = list(professorSkill.columns)

    course_prof = {}
    for i in courses:
        course_prof[i] = [j for j in profs if professorSkill[i][j] == 1]
        if not course_prof[i]:
            del course_prof[i]
            courses.remove(i)

    days = list(professorFreeTime[profs[0]].index)
    times = list(professorFreeTime[profs[0]].columns)

    num_of_timeslots = len(np.ravel(professorFreeTime[profs[0]]))
    free_times = {}
    for i in profs:
        free_times[i] = np.ravel(professorFreeTime[i])

    classes = list(classes[[i for i in classes.keys()].__getitem__(0)].columns)

    rooms_timetable = [np.NAN] * (num_of_timeslots * len(classes))

    population = Population(len(courses), 100)
    population.sort(key=Chromosome.fitness_value, reverse=True)
    last_20_best = []

    iteration = 0
    while True:
        p_m = 1
        r = 0.01
        iteration += 1
        print(iteration, 1 / population[0].fitness_value() - 1, len(population))
        if population[0].fitness_value() == 1:
            print('no conflicts anymore')
            break

        # if len(last_20_best) < 20:
        #     last_20_best.append(population[0])
        #     last_20_best.sort(key=Chromosome.fitness_value, reverse=True)
        # else:
        #     last_20_best.sort(key=Chromosome.fitness_value, reverse=True)
        #     last_20_best[-1] = population[0]
        #     last_20_best.sort(key=Chromosome.fitness_value, reverse=True)
        #
        # if len(last_20_best) == 20 and all_same(last_20_best):
        #     print('hellooooooooooooooooo')
        #     # p_m = len(courses)
        #     r = 0.04
        #     if len(population) > 100:
        #         r = 0
        #         population += Population(len(courses), 10)
        #         population.sort(key=Chromosome.fitness_value, reverse=True)

        selected_population = selection(population)

        crossover_children = crossover(selected_population)

        mutated_children = mutation(crossover_children, rate=r)

        population = population[:round(0.2 * len(population))] + crossover_children + mutated_children
        population.sort(key=Chromosome.fitness_value, reverse=True)

    t = time.time() - startTime
    print("--- %s seconds ---" % t)
    ts.append(t)
    iters.append(iteration)

    for i in range(len(population[0])):
        rooms_timetable[population[0][i][0]] = (courses[i if i < len(courses) else i - len(courses)],
                                                population[0][i][1])

    print(rooms_timetable)

print('------------------------------------\n\n')
print(ts)
print(iters)
print('time average: ', np.mean(ts))
print('# of iterations average: ', np.mean(iters))
