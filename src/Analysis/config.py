import os


def check_if_cluster():
    path_here = os.path.abspath(__file__)
    if 'cluster' in path_here:
        return True
    else:
        return False


is_cluster = check_if_cluster()


if is_cluster:
    EXPERIMENT_RESULT_BASE_FOLDER = '/home/andrea/PhD/Experiment/Data/ExperimentResults'
    GRAPH_BASE_FOLDER = '/home/andrea/PhD/Experiment/Data/Graphs'
else:
    EXPERIMENT_RESULT_BASE_FOLDER = '/Users/andrea/Desktop/PhD/Projects/Current/Experiment/Data/ExperimentResults'
    GRAPH_BASE_FOLDER = '/Users/andrea/Desktop/PhD/Projects/Current/Experiment/Data/Graphs'