import copy
import heapq
import random

import pandas as pd
import numpy as np

skill_file = "./information/Prof_Skill.xlsx"
free_time_file = "./information/Proffosor_FreeTime.xlsx"
am = "./information/Amoozesh.xlsx"


def load_files():
    prof_skill = pd.read_excel(skill_file, sheet_name=None)
    prof_free_time = pd.read_excel(free_time_file, sheet_name=None)
    amoozesh = pd.read_excel(am, sheet_name=None)
    return prof_skill, prof_free_time, amoozesh


def generate_random_population():
    population = []
    for i in range(100):
        rnd = random.sample(range(len(rooms_timetable)), len(courses) * 2)
        c = [-1] * (2*len(courses))
        for j in range(len(courses)):
            c[j] = random.sample(course_prof[courses[j]], 1).__getitem__(0)
            c[j + len(courses)] = c[j]
        population.append(list(zip(rnd, c)))
    return population


def fitness_function(chrm):
    num_of_conflicts = 0
    for i in range(len(courses) * 2):
        # print(chrm[i][0])
        if i < len(courses) and chrm[i][1] != chrm[i + len(courses)][1]:
            num_of_conflicts += 1
        if free_times[chrm[i][1]][chrm[i][0] % len(classes)] == 0:
            num_of_conflicts += 1
        for j in range(i):
            if chrm[i][1] == chrm[j][1] and chrm[i][0] % len(classes) == chrm[j][0] % len(classes):
                num_of_conflicts += 1
    # print(iterations, num_of_conflicts)
    return 1/(1+num_of_conflicts)


def selection():
    global best_old_parents
    fs = heapq.nlargest(int((1 / 5) * len(population)), fitness_values)
    print(iterations, 1/fs[0] - 1)

    best_old_parents = [population[fitness_values.index(i)] for i in fs]

    nf = [j for j in range(len(fitness_values)) if fitness_values[j] not in fs or fs.remove(fitness_values[j])]
    choose = random.sample(nf, int(len(nf) / 2))
    for i in choose:
        selected_population.append(population[i])
    # print(iterations, len(best_old_parents), len(selected_population))
    return selected_population


def crossover():
    a = random.sample(range(len(selected_population)), int((1/10)*len(selected_population)))
    children = []
    for i in a:
        for j in a:
            if i < j:
                chrm1 = copy.deepcopy(population[i])
                chrm2 = copy.deepcopy(population[j])
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
                    while chrm1[x][0] in when_where1:
                        chrm1[x] = (random.sample(range(len(rooms_timetable)), 1)[0], chrm1[x][1])
                    when_where1[x] = chrm1[x][0]
                    while chrm2[x][0] in when_where2:
                        chrm2[x] = (random.sample(range(len(rooms_timetable)), 1)[0], chrm2[x][1])
                    when_where2[x] = chrm2[x][0]
                children.append(chrm1)
                children.append(chrm2)
    return children


def mutate(c1, c2):
    children = []
    for k in copy.deepcopy(c1[:]):
        rnd = random.sample(range(len(k)), 2)
        c = copy.deepcopy(k[:])
        c[rnd[0]], c[rnd[1]] = c[rnd[1]], c[rnd[0]]
        children.append(c)
    for k in copy.deepcopy(c2[:]):
        rnd = random.sample(range(len(k)), 2)
        c = copy.deepcopy(k[:])
        c[rnd[0]], c[rnd[1]] = c[rnd[1]], c[rnd[0]]
        children.append(c)
    return children


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


# print(courses)
# print(profs)
num_of_timeslots = len(np.ravel(professorFreeTime[profs[0]]))
free_times = {}
for i in profs:
    free_times[i] = np.ravel(professorFreeTime[i])

classes = list(classes[[i for i in classes.keys()].__getitem__(0)].columns)

rooms_timetable = [0] * (num_of_timeslots * len(classes))

population = generate_random_population()

fitness_values = []
selected_population = []
crossover_children = []
best_old_parents = []
mutation_children = []

for iterations in range(1000):
    arr = []
    for l in population[:20]:
        arr = fitness_function(l)
    print('checking', 1/np.max(arr) - 1)
    fitness_values = []
    selected_population = []
    crossover_children = []
    best_old_parents = []
    mutation_children = []
    for l in population:
        fitness_values.append(fitness_function(copy.deepcopy(l)))

    if 1 in fitness_values:
        print('hellooooooooooooooooooooooooooooooo')
        break
    selected_population = selection()

    arr = []
    for l in best_old_parents:
        arr = fitness_function(l)
    print('best old parent', 1/np.max(arr) - 1)

    selected_population += best_old_parents

    # print(len(selected_population))

    crossover_children = crossover()

    mutation_children = mutate(copy.deepcopy(crossover_children), copy.deepcopy(best_old_parents))

    population = best_old_parents + crossover_children + mutation_children
