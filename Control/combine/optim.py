# from scipy.optimize import minimize, rosen, rosen_der
# from .objectif import main_dim
# from numpy.random import rand
#
# Dim = main_dim(2,1)
# r_min, r_max = [1,1], [17,12]
# pt = [r_min[0] + rand(2) * (r_max[0] - r_min[0]), r_min[1] + rand(2) * (r_max[1] - r_min[1])]


import pandas as pd
import numpy as np
# from geneticalgorithm import geneticalgorithm as ga
from objectif import main_dim
from taux_mensuels import taux_mensuels



# Définition du paramètre "stepsize" pour le recuit simulé (simulated annealing)



# simulated annealing algorithm
def simulated_annealing(objective = main_dim(2,1), varbound=np.array([[3, 11], [2, 12]]), n_iterations=1, stepsize=np.array([3,3]), temp=10, batt=battery):
    #The temperature has been taken as the opposite of the DQN agent score for test environment in [10,10] PV/batt sizing (we will try to minimize the opposite of the penalty here)
    taux_autoprod, taux_autocons = [], []
    # generate an initial point
    best = [5,5]#(np.random.rand(len(varbound))*(varbound[:,1]-varbound[:,0])+varbound[:,0]).astype(int)
    # best = (np.random.rand(len(varbound))*(varbound[:,1]-varbound[:,0])+varbound[:,0]).astype(int)
    # evaluate the initial point
    best_eval, candidate_score, tau_autoprod, tau_autocons = objective.iterations_dim(best[0], best[1])
    taux_autoprod.append(tau_autoprod)
    taux_autocons.append(tau_autocons)
    curr, curr_eval = best, best_eval
    # model=ga(function=Dim.iterations_dim,dimension=2,variable_type='int',variable_boundaries=varbound)
    list_candidate = [curr]
    list_eval = [curr_eval]
    score_memory = pd.DataFrame()
    for i in range(n_iterations):
        ### Au choix : Considérer une disitribution normale centrée réduite pour le déplacement ou considérer stepsize
        # candidate = curr + np.random.randn(len(varbound)).astype(int)
        candidate = curr + np.random.randint(-stepsize, stepsize)
        if 0 in candidate:
            #À changer car pas de contrainte forte empêchant de prendre un dimensionnement nul.
            continue
        ## Ajout partie penalité non respect bornes
        while not (candidate[0] in (range(varbound[0][0],varbound[0][1]))) & (candidate[1] in (range(varbound[1][0],varbound[1][1]))):
            # Si le candidat n'est pas dans l'intervalle, le point en question devoent égal à celui du candidat précédent.
            candidate_eval_pen = 0
            if candidate[0]<min(range(varbound[0][0],varbound[0][1])) or candidate[0]>max(range(varbound[0][0],varbound[0][1])):
                candidate[0] = list_candidate[-1][0]
            #     #Si le dimensionnement PV du candidat est en dessous de des bornes, on pénalise
            #     candidate_eval_pen += candidate_eval*(min(range(varbound[0][0],varbound[0][1]))-candidate[0])
            # if candidate[0]>max(range(varbound[0][0],varbound[0][1])):
            #
            #     #Si le dimensionnement PV du candidat est au dessus de des bornes, on pénalise
            #     candidate_eval_pen += candidate_eval*(candidate[0]-max(range(varbound[0][0],varbound[0][1])))
            if candidate[1]<min(range(varbound[1][0],varbound[1][1])) or candidate[1]>max(range(varbound[1][0],varbound[1][1])):
                candidate[1] = list_candidate[-1][1]
            #     #Si le dimensionnement batt du candidat est en dessous de des bornes, on pénalise
            #     candidate_eval_pen += candidate_eval*(min(range(varbound[1][0],varbound[1][1]))-candidate[1])
            # if candidate[1]>max(range(varbound[1][0],varbound[1][1])):
            #     #Si le dimensionnement batt du candidat est au dessus de des bornes, on pénalise
            #     candidate_eval_pen += candidate_eval * (candidate[1]-max(range(varbound[1][0], varbound[1][1])))
            # candidate_eval = candidate_eval_pen
        list_candidate.append(candidate)
        candidate_eval, candidate_score, tau_autoprod, tau_autocons = objective.iterations_dim(candidate[0], candidate[1])
        taux_autoprod.append(tau_autoprod)
        taux_autocons.append(tau_autocons)
        list_eval.append(candidate_eval)
        score_memory = pd.concat([score_memory, pd.DataFrame([{'dim_PV':candidate[0],'dim_batt':candidate[1],'coût_tot_dim':candidate_eval, 'score_contrôle': candidate_score, 'tau_autoprod':tau_autoprod, 'tau_autocons':tau_autocons}])])
        # tau_autoprod, tau_autocons = taux_mensuels(candidate)
        print("tau_autoprod : ", tau_autoprod, "tau_autocons : ", tau_autocons)
        if candidate_eval < best_eval:
            best, best_eval = candidate, candidate_eval
            # report progress
            print('>%d f(%s) = %.5f' % (i, best, best_eval))
            # difference between candidate and current point evaluation
        ## jan 2023 : test de réduire artificiellement la différence entre les candidats (par *0.1) pour augmenter le seuil d'acceptabilité
        diff = (candidate_eval - curr_eval)*0.1
        print("diff : ", diff)
        # calculate temperature for current epoch
        t = temp / float(i + 1)
        # calculate metropolis acceptance criterion
        metropolis = np.exp(-diff / t)
        # check if we should keep the new point
        score_memory.to_csv('optimization_scores.csv')
        if diff < 0 or np.random.rand() < metropolis:
        # store the new current point
            curr, curr_eval = candidate, candidate_eval
    return (best, best_eval, list_candidate, list_eval, taux_autoprod, taux_autocons)


if __name__ == "__main__":
    [a, b, candidates, evals, taux_autoprod, taux_autocons] = simulated_annealing()
    print('LISTE DES CANDIDATS : ', candidates, 'LISTE DES EVALUATIONS : ', evals)
    print('a :',a,'b :', b, 'candidates :', candidates,'evals :', evals, 'taux_autoprod :', taux_autoprod, 'taux_autocons :', taux_autocons)
