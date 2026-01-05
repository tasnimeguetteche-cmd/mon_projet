# Optimisation de cellules standards CMOS par apprentissage par renforcement

Ce projet a pour objectif l’optimisation automatique de cellules standards CMOS à partir de simulations électriques transistor-niveau, en utilisant un algorithme d’apprentissage par renforcement (Reinforcement Learning).

L’optimisation vise à trouver un compromis entre :
- le délai de propagation,
- la consommation statique,
- la consommation dynamique,

en s’appuyant sur des simulations SPICE basées sur un PDK industriel open-source.

---

## Principe général

Le projet repose sur une boucle complète d’optimisation :
1. Une cellule logique est décrite par une netlist SPICE paramétrée.
2. Les paramètres géométriques des transistors (largeur W) sont modifiés automatiquement.
3. Chaque configuration est simulée avec NGSpice.
4. Les performances électriques sont extraites.
5. Un agent de Reinforcement Learning évalue ces performances via une fonction de récompense.
6. Le processus est répété jusqu’à convergence vers une configuration optimale.


---
#### Fonctionnalités principales

- lancement des simulations SPICE ;
- modification automatique des paramètres de dimensionnement (largeur **W**) ;
- récupération des mesures SPICE ;
- calcul de la récompense à partir :
  - du délai de propagation ;
  - de la consommation statique ;
  - de la consommation dynamique.
 
---

## Description des fichiers

### `environment.py`

Fichier central du projet.  
Il assure l’interface entre le code Python et le simulateur physique **NGSpice**.


### `rl_agent.py`

Contient l’implémentation de l’agent d’apprentissage par renforcement.  
L’agent repose sur une **Q-Table** permettant d’associer une valeur à chaque couple état–action.



### `main.py`

Point d’entrée du projet.  
Il permet de :
- sélectionner la porte logique à optimiser ;
- lancer la boucle d’apprentissage ;
- afficher les résultats et l’évolution des performances dans la console.

---

### `netlists/`

Contient les fichiers `.cir` décrivant le comportement physique des cellules standards à tester.  
Chaque fichier correspond à une porte logique implémentée au niveau transistor.

---

## Prérequis

- Linux ou macOS (recommandé)
- Python **3.12** ou **3.13**
- **NGSpice** installé
- Accès Internet pour le téléchargement des dépendances et du PDK

---

##  Spécification importante — Chemins des bibliothèques

Les netlists SPICE utilisent des directives `.include` pointant vers les bibliothèques du PDK.

**L’utilisateur doit impérativement modifier les chemins des bibliothèques dans les fichiers `.cir` afin qu’ils correspondent à l’emplacement local du PDK sur sa machine.**

### Exemple

```spice
.include /chemin/vers/le/pdk/sky130/libs.tech/ngspice/sky130.lib.spice
```

Le chemin dépend :
- du répertoire d’installation de `ciel`,
- ou de la variable d’environnement `PDK_ROOT` définie par l’utilisateur.

Sans cette modification, les simulations NGSpice ne pourront pas être exécutées.


