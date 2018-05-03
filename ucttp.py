import pandas as pd

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


proffosorSkill, professorFreeTime, classes, subjects = load_files()
