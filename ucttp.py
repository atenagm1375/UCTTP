from copy import deepcopy
import heapq
import random

import pandas as pd
import numpy as np
import time

skill_file = "./information/Prof_Skill.xlsx"
free_time_file = "./information/Proffosor_FreeTime.xlsx"
am = "./information/Amoozesh.xlsx"


def load_files():
    prof_skill = pd.read_excel(skill_file, sheet_name=None)
    prof_free_time = pd.read_excel(free_time_file, sheet_name=None)
    amoozesh = pd.read_excel(am, sheet_name=None)
    return prof_skill, prof_free_time, amoozesh


def generate_initial_population(n):
    arr = []
    for i in range(1200):
        while True:
            rnd = random.sample(range(len(rooms_timetable)), 2 * n)
            if len(rnd) == len(set(rnd)):
                break
        c = [-1] * (2 * n)
        for j in range(n):
            rnd2 = random.sample(course_prof[courses[j]], 1).__getitem__(0)
            c[j] = rnd2
            c[j + n] = rnd2
        arr.append(list(zip(rnd, c)))
    return arr


def compute_fitness_value(chrm):
    num_of_conflicts = 0
    n = len(chrm)
    for i in range(n):
        if i < n / 2 and chrm[i][1] != chrm[i + int(n / 2)][1]:
            num_of_conflicts += 1
        if free_times[chrm[i][1]][chrm[i][0] % num_of_timeslots] == 0:
            num_of_conflicts += 1
        for j in range(i):
            if chrm[i][1] == chrm[j][1] and chrm[i][0] % num_of_timeslots == chrm[j][0] % num_of_timeslots:
                num_of_conflicts += 1
    return 1 / (1 + num_of_conflicts)


def weighted_random_choice(choices):
    max = sum(choices.values())
    pick = random.uniform(0, max)
    current = 0
    for key, value in choices.items():
        current += value
        if current > pick:
            return key


def selection_function():
    global best_old_parents
    fs = heapq.nlargest(int((1 / 5) * len(population)), fitness_value)
    print(iteration, 1/max(fs) - 1)

    best_old_parents = [population[fitness_value.index(i)] for i in fs]

    choices = {chrm: fitness_value[chrm] for chrm in range(len(population))
               if fitness_value[chrm] not in fs}
    arr = []
    for i in range(int(len(choices) / 2)):
        arr.append(population[weighted_random_choice(choices)])
    return arr


def crossover_operation():
    children = []
    while True:
        if len(children) >= int((2 / 3) * len(selected_population)):
            break
        rnd = random.sample(selected_population, 2)
        chrm1 = deepcopy(rnd[0])
        chrm2 = deepcopy(rnd[1])
        # print(selected_population)
        when_where1 = []
        when_where2 = []
        for u, v in zip(chrm1, chrm2):
            when_where1.append(u[0])
            when_where2.append(v[0])
        c_points = random.sample(range(len(chrm1)), 2)
        if c_points[0] > c_points[1]:
            c_points.reverse()
        for x in range(c_points[0], c_points[1]):
            chrm1[x], chrm2[x] = chrm2[x], chrm1[x]
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
            while chrm1[x][0] in when_where1:
                chrm1[x] = (random.sample(range(len(rooms_timetable)), 1)[0], chrm1[x][1])
            when_where1[x] = chrm1[x][0]
            while chrm2[x][0] in when_where2:
                chrm2[x] = (random.sample(range(len(rooms_timetable)), 1)[0], chrm2[x][1])
            when_where2[x] = chrm2[x][0]
        children.append(chrm1)
        children.append(chrm2)
    return children


def mutation_operation(c1):
    children = []
    for k in deepcopy(c1[:]):
        rnd = random.sample(range(len(k)), 2)
        c = deepcopy(k[:])
        c[rnd[0]], c[rnd[1]] = c[rnd[1]], c[rnd[0]]
        children.append(c)
    return children


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

population = generate_initial_population(len(courses))

fitness_value = []
selected_population = []
crossover_children = []
best_old_parents = []
mutated_children = []

for iteration in range(4000):
    fitness_value = []
    for p in population:
        fitness_value.append(compute_fitness_value(p))

    if 1 in fitness_value:
        print('no conflict anymore')
        break

    selected_population = selection_function()
    selected_population += best_old_parents

    crossover_children = crossover_operation()

    mutated_children = mutation_operation(deepcopy(crossover_children))

    population = best_old_parents + crossover_children + mutated_children

print("--- %s seconds ---" % (time.time() - startTime))

fitness_values = []
for l in population:
    fitness_values.append(compute_fitness_value(l))

ind = np.argmax(fitness_values)

best_chromosome = population[ind]

timetable_per_room = []

table_dict = {}

for i in range(len(best_chromosome)):
    rooms_timetable[best_chromosome[i][0]] = (courses[i if i < len(courses) else i - len(courses)], best_chromosome[i][1])

print(rooms_timetable)
writer = pd.ExcelWriter('table.xlsx')
df = pd.DataFrame(index=days, columns=times)
counter = 0
for i in range(len(rooms_timetable)):
    if i // (len(days) * len(times)) != counter:
        df.to_excel(writer, sheet_name=str(classes[counter]))
        counter += 1
        del df
        df = pd.DataFrame(index=days, columns=times)
    df.iat[i % len(days), i % len(times)] = rooms_timetable[i]

writer.save()
