import collections
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

        prof_num_of_courses = {p: 0 for p in profs}
        for i in self:
            prof_num_of_courses[i[1]] += 1
        # print(len(self.conflicts))
        f = 1 / (1 + (np.var(list(prof_num_of_courses.values())) + 1) * (num_of_conflicts))
        return f

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

    def mutate_chrm(self):
        flag = True
        conflict = self.compute_conflicts()
        for conf in conflict:
            if conf[0] == 2:
                self[conf[1]] = (self[conf[1]][0], find_free_prof(conf[1] if conf[1] < len(courses)
                                                    else conf[1] - len(courses), self[conf[1]][0]))
                flag = False
        if flag:
            while True:
                rnd = random.sample(range(len(self)), 2)
                if abs(rnd[0] - rnd[1]) != len(self) / 2:
                    break
            self[rnd[0]], self[rnd[1]] = self[rnd[1]], self[rnd[0]]
            if self[rnd[0]][1] not in course_prof[courses[rnd[0] if rnd[0] < len(courses) else rnd[0] - len(courses)]]:
                self[rnd[0]] = (self[rnd[0]][0], find_free_prof(rnd[0] if rnd[0] < len(courses) else rnd[0] - len(courses),
                                                                self[rnd[0]][0]))
            if self[rnd[1]][1] not in course_prof[courses[rnd[1] if rnd[1] < len(courses) else rnd[1] - len(courses)]]:
                self[rnd[1]] = (self[rnd[1]][0], find_free_prof(rnd[1] if rnd[1] < len(courses) else rnd[1] - len(courses),
                                                            self[rnd[1]][0]))


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
                rnd2 = find_free_prof(j, rnd[j], rnd[j + num])
                c[j] = rnd2
                c[j + num] = rnd2
            self.append(Chromosome(list(zip(rnd, c))))

    def fit(self):
        f = []
        for p in self:
            f.append(Chromosome(p).fitness_value())
            return f

    def resolve_repeated(self):
        for i in range(len(self) - 1):
            if collections.Counter(self[i]) == collections.Counter(self[i + 1]):
                self[i].mutate_chrm()


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


# def proportional_fitness_selection(populationn):
#     maxx = sum([Chromosome(c).fitness_value() for c in populationn])
#     pick = random.uniform(0, maxx)
#     current = 0
#     for chromosome in populationn:
#         current += Chromosome(chromosome).fitness_value()
#         if current > pick:
#             return chromosome


def crossover(pop, rate=0.8):
    children = Population()
    while True:
        if len(children) >= rate * len(population):
            if len(children) > rate * len(population):
                children.pop(np.argmax(children.fit()))
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

            if chrm1[x][1] != chrm1[x + len(courses)][1]:
                prof = find_free_prof(x, chrm1[x][0], chrm1[x + len(courses)][0])
                chrm1[x] = (chrm1[x][0], prof)
                chrm1[x + len(courses)] = (chrm1[x + len(courses)][0], prof)
                # if free_times[chrm1[x][1]][chrm1[x + len(courses)][0] % num_of_timeslots] == 1:
                #     chrm1[x + len(courses)] = (chrm1[x + len(courses)][0], chrm1[x][1])
                # else:
                #     chrm1[x] = (chrm1[x][0], chrm1[x + len(courses)][1])
            # if x > len(courses) and chrm1[x][1] != chrm1[x - len(courses)][1]:
            #     if free_times[chrm1[x][1]][chrm1[x - len(courses)][0] % num_of_timeslots] == 1:
            #         chrm1[x - len(courses)] = (chrm1[x - len(courses)][0], chrm1[x][1])
            #     else:
            #         chrm1[x] = (chrm1[x][0], chrm1[x - len(courses)][1])
            if chrm2[x][1] != chrm2[x + len(courses)][1]:
                prof = find_free_prof(x, chrm2[x][0], chrm2[x + len(courses)][0])
                chrm2[x] = (chrm2[x][0], prof)
                chrm2[x + len(courses)] = (chrm2[x + len(courses)][0], prof)
                # if free_times[chrm2[x][1]][chrm2[x + len(courses)][0] % num_of_timeslots] == 1:
                #     chrm2[x + len(courses)] = (chrm2[x + len(courses)][0], chrm2[x][1])
                # else:
                #     chrm2[x] = (chrm2[x][0], chrm2[x + len(courses)][1])
            # if x > len(courses) and chrm2[x][1] != chrm2[x - len(courses)][1]:
            #     if free_times[chrm2[x][1]][chrm2[x - len(courses)][0] % num_of_timeslots] == 1:
            #         chrm2[x - len(courses)] = (chrm2[x - len(courses)][0], chrm2[x][1])
            #     else:
            #         chrm2[x] = (chrm2[x][0], chrm2[x - len(courses)][1])
            while chrm1[x][0] in room_timeslot_chrm1:
                chrm1[x] = (random.sample(range(len(rooms_timetable)), 1)[0], chrm1[x][1])
            room_timeslot_chrm1[x] = chrm1[x][0]
            while chrm2[x][0] in room_timeslot_chrm2:
                chrm2[x] = (random.sample(range(len(rooms_timetable)), 1)[0], chrm2[x][1])
            room_timeslot_chrm2[x] = chrm2[x][0]
        # f1 = chrm1.fitness_value()
        # f2 = chrm2.fitness_value()
        # if f1 == f2:
        children.append(chrm1)
        children.append(chrm2)
        # elif f1 > f2:
        #     children.append(chrm1)
        # else:
        #     children.append(chrm2)
    return children


def mutation(pop, rate=1.):
    children = Population()
    c1 = random.sample(pop, round(rate * len(population)))
    for k in deepcopy(c1[:]):
        c = Chromosome(deepcopy(k[:]))
        # flag = True
        # conflict = Chromosome(c).compute_conflicts()
        # for conf in conflict:
        #     if conf[0] == 2:
        #         prof = ''
        #         for p, lst in free_times.items():
        #             if c[conf[1]][0] % num_of_timeslots in lst:
        #                 prof = p
        #                 break
        #         if prof != '':
        #             c[conf[1]] = (c[conf[1]][0], prof)
        #             flag = False
        c.mutate_chrm()
        children.append(c)
    return children


def find_free_prof(j, time_room1, time_room2=None):
    if time_room2 is not None:
        time1 = time_room1 % num_of_timeslots
        time2 = None
        time2 = time_room2 % num_of_timeslots
        lst = []
        for prof in course_prof[courses[j]]:
            if free_times[prof][time1] == 1 and free_times[prof][time2] == 1:
                lst.append(prof)
        if len(lst) > 0:
            return random.choice(lst)
    return random.sample(course_prof[courses[j]], 1).__getitem__(0)


def all_same(arr):
    val = Chromosome(arr[0]).fitness_value()
    for x in arr:
        if Chromosome(x).fitness_value() != val:
            return False
    return True


iters = []
ts = []
for a in range(1):
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

    population = Population(len(courses), 2 * len(courses))
    population.sort(key=Chromosome.fitness_value, reverse=True)
    last_20_best = []

    iteration = 0
    while True:
        Population(population).resolve_repeated()
        p_m = 1
        r_m = 0.02
        r_c = 0.78
        iteration += 1
        print(iteration, 1 / Chromosome(population[0]).fitness_value() - 1, len(population))
        if population[0].fitness_value() == 1:
            print('no conflicts anymore')
            break

        # if len(last_20_best) < 10:
        #     last_20_best.append(population[0])
        #     last_20_best.sort(key=Chromosome.fitness_value, reverse=True)
        # else:
        #     last_20_best.sort(key=Chromosome.fitness_value, reverse=True)
        #     last_20_best[-1] = population[0]
        #     last_20_best.sort(key=Chromosome.fitness_value, reverse=True)
        #
        # if len(last_20_best) == 10 and all_same(last_20_best):
        #     print('hellooooooooooooooooo')
        #     r_m = 0.05
        #     r_c = 0.75
        #     # p_m = len(courses)
        #     # r = 0.04
        #     # if len(population) > 100:
        #     #     r = 0
        #     #     population += Population(len(courses), 10)
        #     #     population.sort(key=Chromosome.fitness_value, reverse=True)

        selected_population = selection(population)

        crossover_children = crossover(selected_population, rate=r_c)

        mutated_children = mutation(crossover_children, rate=r_m)

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

    best_chromosome = population[0]

    timetable_per_room = []

    table_dict = {}

    for i in range(len(best_chromosome)):
        rooms_timetable[best_chromosome[i][0]] = (
        courses[i if i < len(courses) else i - len(courses)], best_chromosome[i][1])

    print(rooms_timetable)
    writer = pd.ExcelWriter('table.xlsx')
    df = pd.DataFrame(index=days, columns=times)
    counter = 0
    for i in range(len(rooms_timetable) + 1):
        if i // num_of_timeslots != counter:
            df.to_excel(writer, sheet_name=str(classes[counter]))
            counter += 1
            del df
            df = pd.DataFrame(index=days, columns=times)
        if i != len(rooms_timetable):
            df.iat[(i % num_of_timeslots) // len(times), i % len(times)] = rooms_timetable[i]

    writer.save()

    writer = pd.ExcelWriter('timeTable.xlsx')
    df = pd.DataFrame(index=days, columns=times)
    for i in range(len(rooms_timetable)):
        if rooms_timetable[i] is not np.NAN:
            data = str(rooms_timetable[i][0]) + ", " + str(rooms_timetable[i][1]) + ", " + str(
                classes[i // num_of_timeslots])
            if type(df.iat[(i % num_of_timeslots) // len(times), i % len(times)]) is str:
                df.iat[(i % num_of_timeslots) // len(times), i % len(times)] += ('\n' + data)
            else:
                df.iat[(i % num_of_timeslots) // len(times), i % len(times)] = data

    df.to_excel(writer, sheet_name='table')
    worksheet = writer.sheets['table']
    worksheet.set_column(1, len(times) + 1, 50)
    for i in range(len(days)):
        worksheet.set_row(i + 1, 20 * len(classes))
    writer.save()

print('------------------------------------\n\n')
print(ts)
print(iters)
print('time average: ', np.mean(ts))
print('# of iterations average: ', np.mean(iters))
