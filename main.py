import utils
import time
import pandas as pd
import os

##### start of main function ####
start_time = time.time()

#### set and save hyperparams #####
pop_size = 1000
generations = 1000
mutation_rate = 0.1 # initial mutation rate
num_parents = 50
num_children = num_parents*2
max_mol_weight = 200 # g/mol
group_constraints = None #None, or 'acid stability'
SA_max = 3.5

# number of fittest molecules to be saved
num_fittest = 5000
# save dataframes of all postprocessed populations: 0 - false, 1- true
save_all_pops_flag = 0

# hyper params to dict
hyper_param_dict = {
    'population_size': pop_size,
    'number of generations': generations,
    'mutation rate': mutation_rate,
    'number of parent molecules': num_parents,
    'number of children': num_children,
    'maximum mol weight [g/mol]': max_mol_weight,
    'constraints on functional groups': group_constraints,
    'maximum synthetic accessibility score': SA_max
}

# save hyperparameters in dataframe
frame_path = './results/frames/'
if not os.path.exists(frame_path):
    os.makedirs(frame_path)
file_name = 'hyperparams.csv'

df_hyper = pd.DataFrame.from_dict(hyper_param_dict, orient='index')
df_hyper.to_csv(frame_path + file_name, sep=';')


##### start algorithm #####

# initialize start population
# df_pop = utils.generate_start_pop_hexane(pop_size)
df_pop = utils.generate_random_start_pop(pop_size, SA_max, max_mol_weight)

## initalize frame to store the fittest molecules
df_fittest = df_pop

# initialize statistics frame
df_stats = utils.ini_statistics_frame()

# initialize group frame
df_groups = utils.ini_group_frame()

for gen in range(generations):
    print('------ generation %s is running ------' %(str(gen + 1)))

    # select the fittest and perform cross over
    print('start cross-over')
    df_pop = utils.cross_over(df_pop, num_parents, num_children, max_mol_weight, group_constraints, SA_max)

    # perform mutation
    print('start mutation')
    df_pop = utils.mutation(df_pop, mutation_rate, max_mol_weight, group_constraints, SA_max)

    # get property prediction and fitness vals
    print('get fitness vals')
    df_pop = utils.get_fitness_vals(df_pop)

    print('get SA scores')
    df_pop = utils.get_SA_scores(df_pop)


    # sort the table according to descending fitness and only take the pop_size fittest
    print('sort pop and delete weakest')
    df_pop = df_pop.sort_values(by=['fitness value'], ascending=False)
    df_pop = df_pop.head(pop_size)



    ###### postprocessing #####

    # get fittest molecules of population and save
    print('save fittest')
    df_fittest = utils.get_all_time_fittest(df_fittest, df_pop, num_fittest)

    # get stats
    print('get stats')
    df_stats = utils.get_stats(df_stats, df_pop)

    # get group counts
    print('get group counts')
    df_groups = utils.get_group_counts(df_groups, df_pop)

    # post-process
    if gen % 5 == 0:
        print('post process')
        utils.post_process(df_pop, df_fittest, df_stats, df_groups, save_all_pops_flag, gen)

stop_time = time.time()

seconds_passed = stop_time - start_time
mins_passed = seconds_passed / 60

# plot fittest mols
path_fittest = './results/frames/fittest.csv'
utils.plot_mols(path_fittest)

import ipdb; ipdb.set_trace()







