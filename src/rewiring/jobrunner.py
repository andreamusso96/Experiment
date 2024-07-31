import os


def launch_jobs(cluster):
    dims = [8, 10, 12]
    vision = [1, 2]

    if cluster:
        path_save = '/cluster/work/coss/anmusso/experiment/rewiring_dim_vision/experiment_{dims}_{vision}.csv'
    else:
        path_save = '/Users/andrea/Desktop/PhD/Projects/Current/Experiment/src/rewiring/experiment_attempts/experiment_{dims}_{vision}.csv'

    os.system('chmod +x run_script.sh')
    for d in dims:
        for v in vision:
            path_save_experiment = path_save.format(dims=d, vision=v)
            if cluster:
                os.system(f'sbatch run_script.sh {d} {v} {path_save_experiment}')
            else:
                os.system(f'./run_script.sh {d} {v} {path_save_experiment}')


if __name__ == '__main__':
    is_cluster = 'cluster' in os.getcwd()
    launch_jobs(cluster=is_cluster)
