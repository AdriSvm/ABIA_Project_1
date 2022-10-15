from main import *
import pandas as pd
'''
Fitxer per realitzar els diversos experiments
'''

#Experiment 1
def exp1():
    df = pd.DataFrame()
    seeds = [399,289,393,387,410,591,906,986,31,51]
    gains_inits = []
    gains_fins = []
    timmings = []
    for i in seeds:
        seed = i
        initial_state, n = experiment('SIMULATED ANNEALING', 'ONLY GRANTED', [5, 10, 25], 1000, [0.2, 0.3, 0.5], 0.75, seed, False)
        timming, nothing = experiment('SIMULATED ANEALING', 'ONLY GRANTED', [5, 10, 25], 1000, [0.2, 0.3, 0.5], 0.75, seed, True)
        gains_init = initial_state.heuristic()
        gains_fin = n.heuristic()
        timming /= 5
        seeds.append(seed)
        gains_inits.append(gains_init)
        gains_fins.append(gains_fin)
        timmings.append(timming)
    df["Seed"] = seeds
    df["initial_gains"] = gains_inits
    df["final_gains"] = gains_fins
    df["mean_time"] = timmings
    return df

print(exp1())