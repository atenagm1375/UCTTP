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
        population.append(random.sample(range(len(rooms_timetable)), 100))
    return population


def fitness_function(chrm):
    num_of_conflicts = 0

    return 1/(1+num_of_conflicts)


professorSkill, professorFreeTime, classes = load_files()

professorSkill = professorSkill[[i for i in professorSkill.keys()].__getitem__(0)]

profs = [i for i in professorFreeTime.keys()]

courses = list(professorSkill.columns)

course_prof = [(i, j) for i in courses for j in profs if professorSkill[i][j] == 1]


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
for l in population:
    fitness_values.append(fitness_function(l))
