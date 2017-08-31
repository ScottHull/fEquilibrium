import pandas as pd
from radioactivity.Radioactivity import decay


isotopes = pd.read_csv('chem.csv', index_col=0) # loads isotope data from a local source
time = 0 # time, in years
time_resolution = 1000000.0 # time resolution, in years (i.e. the time increment between model updates)
time_limit = (4.5 * 10**9) # limits the time evolution to the age of the Earth
iteration = 0 # the number of iterations the model has been exposed to
max_iterations = round(time_limit / time_resolution) # the maximum number of iterations (default: age of Earth 4.5Gya)


hf_182 = decay(element='182-Hf')
while time < 55000000:
    hf_182.rad_decay(isotope_df=isotopes, time_resolution=time_resolution)
    time += time_resolution
    iteration += 1
    # print("\nIterations: {}\nTime: {} years\n182-Hf: {} mol\n182-Ta: {} mol\n182-W: {} mol".format(iteration, time,
    #                                                                                               isotopes['Abundance'][
    #                                                                                                   '182-Hf'],
    #                                                                                               isotopes['Abundance'][
    #                                                                                                   '182-Ta'],
    #                                                                                               isotopes['Abundance'][
    #                                                                                                   '182-W']))
print("\nIsotope Dataframe:")
print(isotopes)
print("Time: {}, Iterations: {}".format(time, iteration))






# hf_182 = decay(element='182-Hf') # opens an instance of the decay class for the primordial isotope, keep formatting!
# for i in range(3):
#     if isotopes['Abundance']['182-Hf'] != 0:
#         run_decay = hf_182.rad_decay(isotope_df=isotopes, time_resolution=time_resolution)
#         isotopes['Abundance'] = run_decay['Abundance']
#         time += time_resolution
#         iteration += 1
#         print("\nIterations: {}\nTime: {} years\n182-Hf: {} mol\n182-Ta: {} mol\n182-W: {} mol".format(iteration, time,
#                                                                                                       isotopes['Abundance'][
#                                                                                                           '182-Hf'],
#                                                                                                       isotopes['Abundance'][
#                                                                                                           '182-Ta'],
#                                                                                                       isotopes['Abundance'][
#                                                                                                           '182-W']))







