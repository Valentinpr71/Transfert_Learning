# Transfert_Learning

Bienvenue dans ce git construit pour essayer de transferer des politiques d'un agent de contrôle à l'autre. Le but est de passer d'un dimensionnement de microgrid à un autre en minimisant le temps d'entraînement de l'agent avant la convergence de sa politique.

### Principe :

Pour chaque itération du dimensionnement du micro-réseau (dont on ne s'intéresse pas pour le moment), on observe les dimensionnements sur lesquels un agent a déjà été entraîné.
Si la distance euclidienne entre les variables de dimensionnement de l'environnement actuel et d'autres environnements est inférieure à un seuil L, pour au moins N environnements, alors on réutilise la politique apprise sur ces environnements pour générer un buffer
et apprendre en batch RL. Si ce n'est pas le cas, l'agent apprend en on-line off-policy RL, il intéragit avec l'environnement directement et apprend grâce au DQN.

### Constitution du buffer

Le buffer doit venir des politique de contrôle des agents entraînés dans un environnement proche, ansi que sur un algorithme purement déterministe (rule based). Comme en BCQ, l'idée est de construire un buffer qui a un large pannel de couples actions/états visités. Pour celà, on rajoute du bruit (équivalent à de l'exploration), qu'importe l'origine de la politique
utilisée pour construire le buffer.

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
