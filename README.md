# Transfert_Learning

Bienvenue dans ce git construit pour essayer de transferer des politiques d'un agent de contrôle à l'autre. Le but est de passer d'un dimensionnement de microgrid à un autre en minimisant le temps d'entraînement de l'agent avant la convergence de sa politique.

Dans un premier temps, on cherche juste à contruire un agent sur le même environnement qui initialise sa politique avec les paramètres finaux d'un autre agent. Une fois fait, on verra comment il se comporte si l'environnement évolue en trouvant une solution pour qu'il apprenne le plus vite possible, même dans un environnement différent.

### Quels sont les prérequis ?

* gym
* stable-baselines
* pandas
* numpy
* matplotlib
* tensorflow version 1.15

Une fois les installations faites, se placer dans le dossier Control/microgrid, sous l'environnement python qui contient les packages puis taper : 
```
pip install -e microgrid
```
