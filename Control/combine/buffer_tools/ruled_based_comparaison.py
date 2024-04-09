import pandas as pd
import numpy as np
import time
def rule_based_nul(dim, data, epsilon):#, seed):
    ### VP mars 23 : Totalement à revoir pour fonctioner avec l'algo d'optimisation et autre..
    #P est le dimensionnement du microgrid
    Dim_PV=dim[0]
    Dim_batt=dim[1]*1000
    eta_PAC=0.5
    eta_H2=0.65
    eta_batt=0.9
    # Importation des données/définition des constantes
    C_panne=2;
    cons = data[0]
    prod = data[2]
    prod = pd.DataFrame(prod);
    prod.rename(columns = {list(prod)[0]: 'Valeurs'}, inplace = True)
    cons = pd.DataFrame(cons);
    #print(cons)
    cons.rename(columns = {list(cons)[0]: 'Valeurs'}, inplace = True)
    #print(cons)
    SOC_min=0.1;
    SOC_max=0.8;
    E_H2_min=0;
    E_H2_max=1000*1000; #en Wh
    Ebatt_min=SOC_min*Dim_batt;
    Ebatt_max=SOC_max*Dim_batt;
    ## Démarrage des itérations de "pilotage automatique" pour relever un indicateur de panne
    indicateur_panne=np.zeros(len(cons));
    action=np.zeros(len(cons));
    SOC=0.1; # Définition du SOC de départ
    E_H2_list = []
    E_H2=0; #Stockage H2 de départ
    P_nominal_H2 = 1*1000;  # Puissance H2
    P_nominal_batt = Ebatt_max - Ebatt_min;  # Puissance batterie (C1)
    nb_explo = 0
    # np.random.seed(seed=int(time.time()))
    for i in range(len(prod)):
        Pbatt = cons['Valeurs'][i] - (prod['Valeurs'][i]); #Dnet, si positif : PAC, sinon electrolyseur
        P_H2_max = (max(Pbatt, 0) / Pbatt) * min(P_nominal_H2 , Pbatt*(1/eta_PAC)) + (min(Pbatt, 0) / Pbatt) * max(
            -P_nominal_H2, Pbatt)
        P_dech = max(P_H2_max, 0)/P_H2_max* min(E_H2,P_H2_max) #* (min(SOC * E_H2_max - Ebatt_min, P_H2_max * (1 / eta_batt)));
        P_charg = min(P_H2_max, 0)/P_H2_max*P_H2_max  #* (max(SOC * E_H2_max - Ebatt_max, P_H2_max));
        #SOC = (SOC * E_H2_max - P_dech - P_charg * eta_batt) / E_H2_max;
        E_H2 = E_H2-P_dech-P_charg*eta_H2 ##D'un point de vue du stockage, on charge moins que Dnet et on décharge plus que Dnet si le but est d'équilibrer le réseau
        E_H2_list.append(E_H2)
        Delta = Pbatt-(P_dech*eta_PAC)-P_charg
        if not (Delta==0):
            Pbatt_max = (max(Delta, 0)/Delta)*min(P_nominal_batt, Delta*(1/eta_batt))+(min(Delta, 0)/Delta)*max(-P_nominal_batt, Delta);
            Pbatt_dech = (max(Pbatt_max, 0)/Pbatt_max)*(min(SOC-Dim_batt, Pbatt_max));
            Pbatt_charg = (min(Pbatt_max, 0) / Pbatt_max) * (max(SOC*Dim_batt-Ebatt_max, Pbatt_max));
            SOC = ((SOC*Dim_batt)-Pbatt_dech-(Pbatt_charg*eta_batt))/Dim_batt;
            indicateur_panne[i] = (Pbatt_dech<Delta)*(Delta-(eta_batt*Pbatt_dech));
    return E_H2_list, indicateur_panne