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
        while True:
            rnd = random.sample(range(len(rooms_timetable)), len(courses) * 2)
            if len(set(rnd)) == len(rnd):
                break
        c = [-1] * (2*len(courses))
        for j in range(len(courses)):
            rnd2 = random.sample(course_prof[courses[j]], 1).__getitem__(0)
            c[j] = rnd2
            c[j + len(courses)] = rnd2
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
                    if x < len(courses) and chrm1[x][1] != chrm1[x + len(courses)][1]:
                        if free_times[chrm1[x][1]][chrm1[x + len(courses)][0] % len(classes)] == 1:
                            chrm1[x + len(courses)] = (chrm1[x + len(courses)][0], chrm1[x][1])
                        else:
                            chrm1[x] = (chrm1[x][0], chrm1[x + len(courses)][1])
                    if x > len(courses) and chrm1[x][1] != chrm1[x - len(courses)][1]:
                        if free_times[chrm1[x][1]][chrm1[x - len(courses)][0] % len(classes)] == 1:
                            chrm1[x - len(courses)] = (chrm1[x - len(courses)][0], chrm1[x][1])
                        else:
                            chrm1[x] = (chrm1[x][0], chrm1[x - len(courses)][1])
                    if x < len(courses) and chrm2[x][1] != chrm2[x + len(courses)][1]:
                        if free_times[chrm2[x][1]][chrm2[x + len(courses)][0] % len(classes)] == 1:
                            chrm2[x + len(courses)] = (chrm2[x + len(courses)][0], chrm2[x][1])
                        else:
                            chrm2[x] = (chrm2[x][0], chrm2[x + len(courses)][1])
                    if x > len(courses) and chrm2[x][1] != chrm2[x - len(courses)][1]:
                        if free_times[chrm2[x][1]][chrm2[x - len(courses)][0] % len(classes)] == 1:
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


def mutate(c1):
    children = []
    for k in copy.deepcopy(c1[:]):
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

days = list(professorFreeTime[profs[0]].index)
times = list(professorFreeTime[profs[0]].columns)

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

    selected_population += best_old_parents

    # print(len(selected_population))

    crossover_children = crossover()

    mutation_children = mutate(copy.deepcopy(crossover_children) + copy.deepcopy(best_old_parents))

    population = best_old_parents + crossover_children + mutation_children

fitness_values = []
for l in population:
    fitness_values.append(fitness_function(l))

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

# writer = pd.ExcelWriter('table.xlsx')
# for i in range(len(classes)):
#     df = pd.DataFrame(np.reshape(rooms_timetable[i:(i + 1) * num_of_timeslots], (5, 4)), index=days, columns=times)
#     df.to_excel(writer, classes[i])
#     writer.save()
#     gene = best_chromosome[i]
#     room = int(gene[0] / len(classes))
#     timeslot = gene[0] % len(classes)
#     day = int(timeslot / len(days))
#     hour = timeslot % len(days)
#     if (times[hour], days[day]) not in table_dict.keys():
#         table_dict[(times[hour], days[day])] = [(courses[i if i < len(courses) else i - len(courses)], gene[1], room)]
#     else:
#         table_dict[(times[hour], days[day])].append((courses[i if i < len(courses) else i - len(courses)], gene[1], room))
#
# for i in table_dict.keys():
#     timetable[i[0]][i[1]] = table_dict[i]
#
# writer = pd.ExcelWriter('table.xlsx')
# timetable.to_excel(writer, 'table')
# writer.save()
