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
    sheet = professorSkill[[i for i in professorSkill.keys()].__getitem__(0)]
    s = set(subjects[[i for i in subjects.keys()].__getitem__(0)])
    f = [i for i in professorFreeTime.keys()]
    for course in s.intersection(sheet.columns):
        valid_profs = [i for i in sheet[course].index if sheet[course][i] == 1]
        # print(course)
        for i in set(f).intersection(valid_profs):
            l = [(j, k) for k in professorFreeTime[i].index
                 for j in professorFreeTime[i].columns if professorFreeTime[i][j][k] == 1]
            rnd = [None]*2
            while rnd[0] == rnd[1]:
                rnd = np.random.choice(range(len(l)), (2,))
            try:
                c1 = np.random.choice([j for j in empty_classes[(l[rnd[0]][0], l[rnd[0]][1])]])
                c2 = np.random.choice([j for j in empty_classes[(l[rnd[1]][0], l[rnd[1]][1])]])
            except ValueError:
                continue
            if timeTable[l[rnd[0]][0]][l[rnd[0]][1]] is not None:
                timeTable[l[rnd[0]][0]][l[rnd[0]][1]].append([course, i, c1])
            else:
                timeTable[l[rnd[0]][0]][l[rnd[0]][1]] = [[course, i, c1]]
            if timeTable[l[rnd[1]][0]][l[rnd[1]][1]] is not None:
                timeTable[l[rnd[1]][0]][l[rnd[1]][1]].append([course, i, c2])
            else:
                timeTable[l[rnd[1]][0]][l[rnd[1]][1]] = [[course, i, c2]]
            empty_classes[(l[rnd[0]][0], l[rnd[0]][1])].remove(c1)
            empty_classes[(l[rnd[1]][0], l[rnd[1]][1])].remove(c2)
            break
    print(timeTable)
    writer = pd.ExcelWriter("table.xlsx")
    timeTable.to_excel(writer, "table")
    writer.save()


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
days_of_week = professorFreeTime[[i for i in professorFreeTime.keys()].__getitem__(0)].index
times = professorFreeTime[[i for i in professorFreeTime.keys()].__getitem__(0)].columns
empty_classes = {}
t = [(j, k) for k in days_of_week for j in times]
for(i, j) in t:
    empty_classes[(i, j)] = list(classes[[i for i in classes.keys()].__getitem__(0)].columns)
# empty_classes.fromkeys(tuple(t), classes[[i for i in classes.keys()].__getitem__(0)].columns)
# empty_classes = dict(zip(classes[[i for i in classes.keys()].__getitem__(0)].columns,
#                          [1] * len(classes[[i for i in classes.keys()].__getitem__(0)].columns)))
timeTable = pd.DataFrame(np.reshape(timeslots, (-1, 4)), index=days_of_week, columns=times)
# print(timeTable)
print(t)
print(empty_classes)
generate_initial_state()
