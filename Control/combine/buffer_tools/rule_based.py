import pandas as pd
import numpy as np
import time
def rule_based_actions(dim, data, epsilon):#, seed):
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
        Pbatt=cons['Valeurs'][i]-(prod['Valeurs'][i]); #Demande nette, cette valeur est négative si la production dépasse la conso, positive sinon
        Pbatt_max=(max(Pbatt,0)/Pbatt)*min(P_nominal_batt*eta_batt,Pbatt)+(min(Pbatt,0)/Pbatt)*max(-P_nominal_batt*eta_batt,Pbatt);# (Dé)Charge de la batterie, on la limite si la puissance demandée est plus faible que la puissance nominale
        P_dech=(max(Pbatt_max,0)/Pbatt_max)*(min(SOC*Dim_batt-Ebatt_min,Pbatt_max*(1/eta_batt))); #Puissance de décharge en tenant compte du SOC.On prend 1/eta pour intégrer le rendement de décharge tout en atteigannt le Delta demandé: On est obligé de déchargé eta fois plus de puissance que demandé
        P_charg=(min(Pbatt_max,0)/Pbatt_max)*(max(SOC*Dim_batt-Ebatt_max,Pbatt_max)); #Puissance de charge en tenant compte du SOC (négative)
        #print(Pbatt, P_charg,P_dech, P_dech*eta_batt)
        SOC=(SOC*Dim_batt-P_dech-P_charg*eta_batt)/Dim_batt; #Mise à jour du SOC.
        Delta=Pbatt-P_dech*eta_batt-P_charg; #Manque/Surplus à combler par H2. Le rendement n'intervient pas en dehors du système c'est pourquoi on l'enlève ne multipliant par son inverse
        if not(Delta==0):#On considère que la puissance d'électrolyse est la même que la puissance de PAC
            # seed+=1
            PH2_max=(max(Delta,0)/Delta)*min(P_nominal_H2*eta_PAC,Delta)+(min(Delta,0)/Delta)*max(-P_nominal_H2,Delta);
            # insérer le rendement ?
            PH2_dech=(max(PH2_max,0)/PH2_max)*(min(E_H2-E_H2_min,PH2_max*(1/eta_PAC))); #Puissance de décharge en tenant compte de l'énergie stockée
            PH2_charg=(min(PH2_max,0)/PH2_max)*(max(E_H2-E_H2_max,PH2_max)); #Puissance de charge (négative)
            #print(PH2_charg+PH2_dech, Delta)
            #print(PH2_charg,Delta, PH2_dech, PH2_dech*eta_PAC)
            if np.random.rand() < epsilon:
                nb_explo+=1
                action[i] = np.random.randint(3)
            else:
                action[i]=((PH2_dech+PH2_charg)<-0.005)*2+((abs(PH2_dech+PH2_charg)<0.005)); # Action 0 = décharger, action 1 = Rien, action 2 = charger
            E_H2=E_H2-PH2_dech-PH2_charg*eta_H2;
            E_H2_list.append(E_H2)
            indicateur_panne[i]=(PH2_dech<Delta)*(Delta-(eta_PAC*PH2_dech)); # l'indicateur utilisé est l'énergie en déficit dans le système
        else:
            E_H2_list.append(E_H2)
            action[i]=1
    print('nb_explo : ', nb_explo)
    ##Il faut retoucher le programme pour qu'il soit plus proche de microgridenv: Selon le signe du premier delta, on charge ou on décharge H2 à puissance max (peut importe combien on doit charger ou décharger). Puis la batterie équilibre le reste.
    ## Fonction coût panne
    return action, cons, prod, E_H2_list, indicateur_panne