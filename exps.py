from main import *
'''
Fitxer per realitzar els diversos experiments
'''

#Experiment 1
def exp1():
    res1 = {}
    for i in range(5):
        seed = random.randint(0, 999)
        initial_state, n = experiment('HILL CLIMBING', 'ONLY GRANTED', [5, 10, 25], 1000, [0.2, 0.3, 0.5], 0.5, seed, False)
        mean_time = 0
        for _ in range(5):
            timming, nothing = experiment('HILL CLIMBING', 'ONLY GRANTED', [5, 10, 25], 1000, [0.2, 0.3, 0.5], 0.5, seed, True)
            mean_time += int(timming)
        mean_time /= 5
        gains_init = initial_state.heuristic()
        gains_fin = n.heuristic()
        res1[i] = (seed, gains_init,gains_fin,mean_time)
    return res1

print(exp1())