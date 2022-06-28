import numpy as np
import sys
import os
import time
import numpy as np
import json
import hashlib

class Dim_manager()
    def __init__(self):
        self.distance=None
        self.dicto={}
        self.dim=None
    def _dim(self,dim):
        self.dim=dim
    def check_exist(self):
        return self.dim in self.dicto.values()
    def _create_hashkey(self):
        dim = json.dumps(self.dim, sort_keys=True).encode('utf-8')
        crypted_name = hashlib.md5(dim).hexdigest()
        return crypted_name
    def save_model(self, model):
        model.save("Result/" + self._create_hashkey())
    def add_to_dicto(self, dim): #Cette fonction ajoute un dimensionnement "non-vu" au dictionnaire. Elle servira de condition pour poursuivre: Si l'environnement a déjà été traité, on a déjà stocké le résultat quelque part, la suite des démarches est inutile.
        if self.check_exist(self.dim):
            return 0
        self.dicto[self._create_hashkey()]=self.dim
        return 1
    def choose_parents(self):
        # #arr=np.zeros(len(self.dicto))
        # arr=self.dicto
        # for i in self.dicto:
        #     arr[i] = (np.linalg.norm(self.dim - self.dicto[i]) < self.distance)
        return {k: v for k, v in self.dicto.items() if (np.linalg.norm(self.dim - v) < self.distance)} #renvoie un dictionnaire rempli seulement avec les dimensionnement remplissant les conditions de distance euclidienne

