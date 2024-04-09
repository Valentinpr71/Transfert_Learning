# Transfert_Learning
L'objectif de ce repo est de combiner le dimensionnement et le contrôle d'un micro-réseau électrique. Le script à executer est `/Control/combine/main_dim.py`. 
Le script importe des objets pour construire la modélisation du micro-réseau électrique. Le micro-réseau modélisé est constitué de panneaux photovoltaïques, d'une batterie LiFePO4, d'un stockage hydrogène (électrolyseuer/PAC) et d'un point de demande. Une optimisation du dimensionnement est lancée en itérant sur les valeurs de la puissance crête installée pour les panneaux PV et de la capacité de la batterie. L'optimisation minimise le coût économique du système décomposé en coût d'investissement (+maintenance) et coût d'opération (+ usure). Les coûts d'opération dépendent du pilotage du système de stockage, effectué par apprentissage par renforcement. Il est donc très fortement non-linéaire par rapport à la fonction objectif.

Un algorithme de recuit simulé est utilisé pour explorer l'espace des solutions. Pour chaque itération d'opitmisation, un dimensionnement candidat est généré et un apprentissage a lieu. L'apprentissage demandant des ressources conséquentes, le but ici est de parvenir à ré-utiliser l'entraînement d'agents issus d'itérations précédentes pour ne pas initialiser l'entraînement de l'agent associé au contrôle de l'EMS sur le dimensionnement en cours de manière aléatoire. Un agent sert de démonstrateur, et déploie sa politique sur le dimensionnement de l'itération en cours, tandis que l'agent apprenant met à jour sa politique décisionnelle de manière hors-ligne, avec l'observation de la politique apprise. C'est le *Transfert de Politique*.

Le but est de passer d'un dimensionnement de microgrid à un autre en minimisant le temps d'entraînement de l'agent avant la convergence de sa politique. L'apprentissage sans transfert de politique se fait avec l'algorithme de Deep Q-Learning. Lorsqu'un dimensionnement est "voisin", soit proche d'un micro-réseau déjà itéré dans le processus de dimensionnement, l'entraînement est effectué de manière hors-ligne avec l'algorithme de *Batch Constrained Q-learning*. 


### BCQ :

Pour introduire le BCQ, *Fujimoto, 2019, Off-Policy Deep Reinforcement Learning without Exploration* a orquestré une expérience. Il compare la performance d'un DDPG online et de 3 DDPG entraînés off-line avec chacun une configuration.
- **Final buffer**: On entraîne un agent DDPG sur 1M d'échantillonnage avec un taux d'exploration très élevé. Une fois qu'il est entraîné, on entraîne un autre agent DDPG avec le buffer rempli et seulement avec cette expérience. C'est lourd et le buffer doit couvrir beaucoup d'état-actions.
- **Concurrent**: On les entraîne toujours sur 1M de tuple mais en même temps. L'agent off-policy échantillonne des tuples dans le replay buffer en temps réel.
- **Imitation**: On échantillonne 1M de tuples venant d'un agent déjà entraîné, donc issus de la même politique. On fait de l'imitation learning sur ce buffer en le considérant comme les actions d'un expert pour le DDPG off-policy.

Une conclusion étonnante est établie suite à ces expériences: Dans tous les cas (même le concurrent !), l'agent on-line surpasse largement les agents qui s'entraînent avec le buffer. De plus, l'estimation des valeurs diverge en off-policy alors qu'elles sont stables en on-policy. 

Enfaite, en off-policy, les actions qui n'ont jamais été prises sont extrapolées avec une valeur supérieure à leur valeur réelle. C'est à cause du biais, qui est combiné à l'objectif de maximisation dans le RL. Avec un espace d'état et d'action continus, on contribue à augmenter cette erreur d'extrapolation. Même avec de nombreuses données, l'effet appelé *Catastrophic forgetting* (l'agent oublie ce qu'il a appris en début d'entraînement) engendre ce biais. Le RL off-line ne peut pas fonctionner dans le monde réel sans plus de garantie de fonctionnement et d'efficacité avec peu de données.

Avec le **Batch Constrained RL**, l'estimations des valeurs faites par un algo RL off-policy peut être bien faite dans une région de l'environnement pour laquelle on a des données. 
L'idée derrière cet algorithme est qu'une politique doit induire la même fréquence de visite état-action que le batch. Les politiques sont "Batch-Constrained". Les politiques sont entraînées pour sélectionner des actions selon 3 objectifs:

1. Minimiser la distance entre les actions sélectionnées et celles du batch.
2. Mener à des états ou des données familières peuvent êtres observées.
3. Maximiser la fonction valeur.

Si *1.* n'est pas respecté, *2.* et *3.* ne peuvent pas l'être. Pour assurer le respect de *1.*, un modèle génératif est utilisé. Il produit des actions en sortie avec une haute probabilité de choisir des actions du batch. On le combine avec un réseau de neurones (Q Network) qui perturbe le choix d'action en le poussant vers le choix le plus optimal. Puis, on entraîne 2 réseaux de neurones(Q-Networks). Le minimum des deux estimations des NN est utilisé afin d'éviter une sur-estimation des valeurs. 

Pour résumer, il y a 4 réseaux de neurones paramètrés. Un modèle génératif $G_\omega(s)$, un modèle de perturbation $\zeta_\Phi(s,a)$ et 2 Q-Networks $Q_{\theta_{1}}(s,a)$ et $Q_{\theta_{2}}(s,a)$ fonctionnant en double DQN (donc chacun a un réseau target comme en DQN, de même pour $\zeta_\Phi(s,a)$) pour éviter des sur-estimations des valeurs. 

$G_\omega(s)$ est un variational auto-encoder. Il n'est conditionné que par l'éatat dans lequel l'agent se situe. Il est composé d'un encodeur qui prends en entrée un couple état-action et donne en sortie une moyenne et un écart-type d'une distribution normale et d'un décodeur qui prend en entrée l'état et la distribution normale suivant ces paramètres et renvoit une action. La mise à jour de $\omega$ se fait avec la divergence KL entre la distribution normale renvoyée et une centré réduite, ainsi que la distance entre l'action dans le batch et l'action renvoyée. On ajoute une perturbation selon $\zeta_\Phi(s,a)$ dont la valeur dépend de la sortie de $G_\omega(s)$ et des valeurs de paires état-action. Cette valeur de perturbation est une petite action que l'on somme à la première action. **Ça ne peut donc fonctionner ainsi qu'en espace d'état-actions continus !!!** La mise à jour de $\Phi$ dépend de $\theta_1$ et $\theta_2$. La mise à jour des target se fait avec un $\tau$ compris entre 0 et 1 qui évite de changer brutalement les target selon lesde paramètre du réseau déployé.

### Discrete batch Constraint RL

La version discrète de BCQ a été introduite un peu plus tard, la même année. Au lieu de perturber une action échantillonnée dans une distribution, on a des actions discrètes. 
$G_\omega(a|s) \approx \pi_b(a|s)$
On peut donc simplement calculer leur probabilité selon la distribution, et supprimer des actions dont la probabilité est trop faible par rapport à l'action à probabilité maximale de la distribution (moyennant de choisir un seuil $\tau$ de probabilité relative).
On peut donc écrire:

$\underset{a\| \frac{G_\omega(a|s)}{\underset{\hat{a}}{max}G_\omega(\hat{a}|s)}\>\tau}{argmax}Q_{\theta}(s,a)$

Ainsi, avec $\tau = 0$, on a affaire à un DQN classique alors qu'avec $\tau=1$, il s'agit d'imitation Learning.

### Principe :

Pour chaque itération du dimensionnement du micro-réseau (dont on ne s'intéresse pas pour le moment), on observe les dimensionnements sur lesquels un agent a déjà été entraîné.
Si la distance euclidienne entre les variables de dimensionnement de l'environnement actuel et d'autres environnements est inférieure à un seuil L, pour au moins N environnements, alors on réutilise la politique apprise sur ces environnements pour générer un buffer
et apprendre en batch RL. Si ce n'est pas le cas, l'agent apprend en on-line off-policy RL, il intéragit avec l'environnement directement et apprend grâce au DQN.

### Constitution du buffer

Le buffer doit venir des politique de contrôle des agents entraînés dans un environnement proche, ansi que sur un algorithme purement déterministe (rule based). Comme en BCQ, l'idée est de construire un buffer qui a un large pannel de couples actions/états visités. Pour celà, on rajoute du bruit (équivalent à de l'exploration), qu'importe l'origine de la politique
utilisée pour construire le buffer.

***
### Différences effectives par rapport à BCQ

* Première chose : Dans BCQ, il n'y a qu'un environnement. L'utilisateur spécifie au programme s'il veut entraîner un modèle from scratch (on-line en DQN), auquel cas il passera l'argument `train_behavioral` lors de l'execution du main. S'il veut utiliser une quelquonque politique, il peut utiliser l'argument `generate_buffer`.
Si aucun des deux n'est sur `True`, alors le script va utiliser le buffer qu'il a (ou pas) à disposition pour entraîner un agent de BCQ.

![descriptif BCQ](Images/BCQmultienv.png)

* Dans notre cas d'étude, on a plusieurs environnements. Un par dimensionnement du micro-réseau. Le but n'est pas de montrer que l'on est performant en Off-line mais de l'admettre pour le moment afin d'améliorer le temps de calcul. L'offline RL semble une bonne idée puisqu'une fois tout les tuples du  buffer collectés, il ne reste plus qu'à faire des mises-à-jours.
Ainsi, contrairement au papier du BCQ, pour chaque environnement, on cherchera à faire soit `train_behavioral` (s'il n'y a pas assez de dimensionnement proches de l'actuel, voir Principe), soit `generate_buffer` (suivi de `train_BCQ`) mais pas les deux l'un après l'autre.


![descriptif BCQ](Images/BCQmultienv2.png)



Ainsi, il est nécessaire de créer un moyen de garder en mémoire les différents dimensionnement (=environnements) connus. Alors, pour chaque nouvelle itération, une fonction vérifiera le nombre de voisin et selon le résultat appelera `train_behavioral` ou `generate_buffer`. 
Les politiques doivent être sauvegardées dans tous les cas pour constituer un buffer dans le cas d'un ultérieur `generate_buffer`.
***
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

