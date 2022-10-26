from main import *
import pandas as pd
'''
File only for executing all experiments, used for tests.
It doesn't output all the experiments results at once, changes made after each executing, like changing operators.
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
        print(f"Doing exp1 with seed {seed}, number {seeds.index(seed)+1} of {len(seeds)}")
        initial_state, n = experiment('HILL CLIMBING', 'ONLY GRANTED', [5, 10, 25], 1000, [0.2, 0.3, 0.5], 0.75, seed, False)
        timming, nothing = experiment('HILL CLIMBING', 'ONLY GRANTED', [5, 10, 25], 1000, [0.2, 0.3, 0.5], 0.75, seed, True,1)
        gains_init = initial_state.heuristic()
        gains_fin = n.heuristic()
        print(timming)
        gains_inits.append(gains_init)
        gains_fins.append(gains_fin)
        timmings.append(timming)
    df["Seed"] = seeds
    df["initial_gains"] = gains_inits
    df["final_gains"] = gains_fins
    df["mean_time"] = timmings
    return df

#print(exp1())

def exp2():
    df = pd.DataFrame()
    seeds = [245,39,678,1345,239,29,568,1422,991,132]
    gains_inits_on = []
    gains_inits_or = []
    gains_fins_on = []
    gains_fins_or = []
    timmings_on = []
    timmings_or = []
    for i in seeds:

        seed = i
        print(f"fent experimentació en la llavor {seed}, temps emprat = {sum(timmings_on)+sum(timmings_or)}")
        initial_state_on, n_on = experiment('HILL CLIMBING', 'ONLY GRANTED', [5, 10, 25], 1000, [0.2, 0.3, 0.5], 0.75, seed, False)
        timming_on, nothing_on = experiment('HILL CLIMBING', 'ONLY GRANTED', [5, 10, 25], 1000, [0.2, 0.3, 0.5], 0.75, seed, True,5)
        initial_state_or, n_or = experiment('HILL CLIMBING', 'ORDERED', [5, 10, 25], 1000, [0.2, 0.3, 0.5], 0.75, seed,False)
        timming_or, nothing_or = experiment('HILL CLIMBING', 'ORDERED', [5, 10, 25], 1000, [0.2, 0.3, 0.5], 0.75,seed, True, 5)

        gains_init_on = initial_state_on.heuristic()
        gains_fin_on = n_on.heuristic()

        gains_init_or = initial_state_or.heuristic()
        gains_fin_or = n_or.heuristic()

        timming_on /= 5
        timming_or /= 5

        gains_inits_on.append(gains_init_on)
        gains_fins_on.append(gains_fin_on)
        timmings_on.append(timming_on)

        gains_inits_or.append(gains_init_or)
        gains_fins_or.append(gains_fin_or)
        timmings_or.append(timming_or)

    df["Seed"] = seeds
    df["initial_gains_on"] = gains_inits_on
    df["final_gains_on"] = gains_fins_on
    df["mean_time_on"] = timmings_on

    df["initial_gains_or"] = gains_inits_or
    df["final_gains_or"] = gains_fins_or
    df["mean_time_or"] = timmings_or

    return df

#print(exp2())

def exp4():
    df = pd.DataFrame()
    seeds = [134,34,556,9012,4321,12,346,762,987,222]
    it = []
    timmings = []

    n = 40
    for _ in range(15):
        for i in seeds:
            seed = i
            c1 = 0.13 * n
            c2 = 0.25 * n
            c3 = 0.63 * n
            c1 = round(c1)
            c2 = round(c2)
            c3 = round(c3)
            print(f"Iteració de la llavor {seed} amb {c1 + c2 + c3} centrals amb una distribució {[c1, c2, c3]}")
            timming, nothing = experiment('HILL CLIMBING', 'ONLY GRANTED', [c1, c2, c3], 1000, [0.2, 0.3, 0.5], 0.75,
                                          seed, True, 10)
            timming /= 10
            timmings.append(timming)
            print(timming)
            it.append((seed, c1 + c2 + c3))
        n += 40

    df["iteracions"] = it
    df["mean_time"] = timmings
    return df

d = exp4()