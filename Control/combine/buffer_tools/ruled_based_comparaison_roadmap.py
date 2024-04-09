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
    E_H2_max=455000; #en Wh
    Ebatt_min=SOC_min*Dim_batt;
    Ebatt_max=SOC_max*Dim_batt;
    ## Démarrage des itérations de "pilotage automatique" pour relever un indicateur de panne
    indicateur_panne=np.zeros(len(cons));
    action=np.zeros(len(cons));
    SOC=0.1; # Définition du SOC de départ
    E_H2_list = [0]
    E_H2=0; #Stockage H2 de départ
    P_nominal_H2 = 1*1000;  # Puissance H2
    P_nominal_batt = Ebatt_max - Ebatt_min;  # Puissance batterie (C1)
    nb_explo = 0
    energy_bought=[]
    surplus = []
    # np.random.seed(seed=int(time.time()))
    for i in range(len(prod)):
        Dnet = cons['Valeurs'][i] - (prod['Valeurs'][i]); #Dnet, si positif : PAC, sinon electrolyseur
        if Dnet>0: #Si on a plus de demande que de production, l'algo lance la PAC
            if E_H2>P_nominal_H2*1: #Si on a assez d'hydrogene, on utilise la PAC a Pmax
                P_PAC = P_nominal_H2*eta_PAC
                E_H2 = E_H2-P_nominal_H2
            else : #Sinon, on décharge avec ce qu'il reste et si rien, on ne fait rien
                P_PAC = E_H2*eta_PAC
                E_H2 = 0
            power_needed_from_battery = Dnet-P_PAC #La batterie s'occupera de règler le déséquilibre manquant, en se chargeant si ce delta est
            #négatif et en se déchargeant si il est positif
        else: #Si Dnet négatif, electrolyseur
            P_elec = P_nominal_H2
            diff_H2 = P_nominal_H2 * eta_H2
            if E_H2+diff_H2>E_H2_max: #Si avec ce qu'on charge, on dépasse la capacité max de H2, alors il faut modifier la puissance de charge
                P_elec = (E_H2_max-E_H2)*(1/eta_H2) #On utilise l'efficacité de l'électrolyseur dans cette ligne pour être sûr que la capacité du tank ne soit jamais dépassée
                diff_H2 = E_H2_max - E_H2
            E_H2 = E_H2+diff_H2
            power_needed_from_battery = Dnet+P_elec #Pareil, la batterie s'occupe de l'équilibre. On ajoute P_elec car c'est une demande.
        if power_needed_from_battery>0: #Si on a besoin de puissance de la part de la batterie, on la décharge
            surplus.append(0)
            if SOC*Dim_batt*eta_batt>power_needed_from_battery:#Si il y a assez d'énergie dans la batterie, alors on l'utilise
                energy_bought.append(0)
                SOC = SOC-(power_needed_from_battery/(eta_batt*Dim_batt))
            else: #Si pas assez d'énergie stockée, la batterie décharge le maximum possible, le reste est acheté au réseau central
                energy_bought.append(power_needed_from_battery - (SOC*Dim_batt*eta_batt))
                print(power_needed_from_battery - (SOC*Dim_batt*eta_batt))
                SOC = 0
        else: #Si puissance demandée par la batterie est négative, on la recharge
            energy_bought.append(0)
            if min(1, SOC - (power_needed_from_battery / Dim_batt) * eta_batt) == 1:#Si la batterie est trop pleine pour se charger à puissance max
                surplus.append((-1*power_needed_from_battery) - ((1*Dim_batt-(SOC*Dim_batt))/eta_batt))
            else:
                surplus.append(0)
            SOC = min(1., SOC - (power_needed_from_battery / Dim_batt) * eta_batt)#Le SOC vaut soit 1 si il y a du surplus, soit on lui ajoute le pourcentage de batterie charger pour équilibrer si il n'y en a pas
        E_H2_list.append(E_H2)
    return E_H2_list, energy_bought



