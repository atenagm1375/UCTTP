import pandas as pd
import numpy as np

skill_file = "./information/Prof_Skill.xlsx"
free_time_file = "./information/Proffosor_FreeTime.xlsx"
am = "./information/Amoozesh.xlsx"
sub = "./information/subjects.xlsx"


def load_files():
    prof_skill = pd.read_excel(skill_file, sheet_name=None)
    prof_free_time = pd.read_excel(free_time_file, sheet_name=None)
    amoozesh = pd.read_excel(am, sheet_name=None)
    subjects = pd.read_excel(sub, sheet_name=None)
    return prof_skill, prof_free_time, amoozesh, subjects


def generate_initial_state():
    # valid_profs = {}
    prof = professorSkill[[i for i in professorSkill.keys()].__getitem__(0)]
    s = set(subjects[[i for i in subjects.keys()].__getitem__(0)])
    for course in s.intersection(prof.columns):
        valid_profs = [i for i in prof[course].index if prof[course][i] == 1]
        print(course, valid_profs)


professorSkill, professorFreeTime, classes, subjects = load_files()
# a = np.zeros(professorSkill.shape)
# courseName = []
# print(professorSkill.head(2))
# a = a.reshape(36, 16)
# # print(a[0])
# for i , j in zip(professorSkill.keys(), range(a.shape[0])):
#     a[j] = professorSkill[i]
#     courseName.append(i)
# # a = a.reshape(16, 36)
# a = a.T
# print()
# a = professorSkill.keys()
# print(type(professorSkill))
# for i in a:
#     df = professorSkill[i]
#     for j in df.index:
#         print(professorSkill[i][j:j])


timeslots = [None] * 20
generate_initial_state()
empty_classes = dict(zip(classes[[i for i in classes.keys()].__getitem__(0)], [0] * len(classes)))
print(empty_classes)
