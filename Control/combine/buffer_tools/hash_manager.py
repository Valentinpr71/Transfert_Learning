import sys
import os
import time
import numpy as np
import json
import hashlib

class Dim_manager():
    def __init__(self, distance, nb_voisins):
        self.distance=distance
        self.dicto={}
        self.dim=None
        self.nb_voisins = nb_voisins

    def _dim(self, dim):
        self.dim = dim

    def check_exist(self):
        return self.dim in self.dicto.values()

    def _create_hashkey(self):
        dim = json.dumps(self.dim, sort_keys=True).encode('utf-8') #json ne lit que les liste, pas les arrays de numpy
        crypted_name = hashlib.md5(dim).hexdigest()
        return crypted_name

    def save_model(self, model):
        model.save("Result/" + self._create_hashkey())

    def add_to_dicto(self): #Cette fonction ajoute un dimensionnement "non-vu" au dictionnaire. Elle servira de condition pour poursuivre: Si l'environnement a déjà été traité, on a déjà stocké le résultat quelque part, la suite des démarches est inutile.
        if self.check_exist():
            return 0
        self.dicto[self._create_hashkey()]=self.dim
        return 1

    def choose_parents(self):
        # #arr=np.zeros(len(self.dicto))
        # arr=self.dicto
        # for i in self.dicto:
        #     arr[i] = (np.linalg.norm(self.dim - self.dicto[i]) < self.distance)
        self.parents = {k: v for k, v in self.dicto.items() if ((np.linalg.norm(np.array(self.dim) - np.array(v)) < self.distance) and (np.linalg.norm(np.array(self.dim) - np.array(v) != 0)))}
        return self.parents #renvoie un dictionnaire rempli seulement avec les dimensionnement remplissant les conditions de distance euclidienne

    def add_data_cons(self, data_cons, data_cons_norm):
        self.data_cons = data_cons
        self.data_cons_norm = data_cons_norm

    def add_data_prod(self, data_prod, data_prod_norm):
        self.data_prod = data_prod
        self.data_prod_norm = data_prod_norm
        self.data = [self.data_cons, self.data_cons_norm, self.data_prod, self.data_prod_norm]

    def path(self):
        fname = str(self.dim)
        # Create unique log dir
        log_dir = "Batch_RL_results/" + self._create_hashkey()  # .format(int(time.time()))
        if os.path.isfile(log_dir + "/best_model.zip"):
            carry = 1
        else:
            os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        return log_dir, carry