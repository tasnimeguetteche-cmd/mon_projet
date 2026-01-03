from environment import InverterEnv
from rl_agent import QAgent
import numpy as np
import os
import sys

def choisir_porte():
    """
    Scanne le dossier 'netlists', liste les fichiers .cir disponibles 
    et demande à l'utilisateur d'en choisir un.
    """
    dossier_netlists = "netlists"
    
    # Vérification que le dossier existe
    if not os.path.exists(dossier_netlists):
        print(f"Erreur : Le dossier '{dossier_netlists}' est introuvable.")
        sys.exit(1)

    # Récupérer tous les fichiers .cir (exclut params.spice et autres fichiers)
    fichiers = [f for f in os.listdir(dossier_netlists) if f.endswith('.cir')]
    
    # On retire l'extension pour l'affichage et on trie
    portes_dispo = sorted([os.path.splitext(f)[0] for f in fichiers])

    # Si aucune porte n'est trouvée
    if not portes_dispo:
        print(f"Aucun fichier .cir trouvé dans '{dossier_netlists}'.")
        sys.exit(1)

    # --- AFFICHAGE MENU ---
    print("\n" + "="*40)
    print("Quelle fonction logique souhaitez vous choisir ?")
    print("="*40)
    
    for i, porte in enumerate(portes_dispo):
        print(f"  {i + 1}. {porte}")
    
    print("-" * 40)

    while True:
        try:
            choix = input(f"Votre choix (1-{len(portes_dispo)}) : ")
            idx = int(choix) - 1
            if 0 <= idx < len(portes_dispo):
                gate_name = portes_dispo[idx]
                print(f"\nConfiguration chargée : {gate_name.upper()}")
                return gate_name
            else:
                print("Numéro invalide, réessayez.")
        except ValueError:
            print("Entrée invalide, veuillez entrer un numéro.")

def main():
    selected_gate = choisir_porte()
    
    #Initialisation de l'environnement avec la porte choisie
    env = InverterEnv(gate_name=selected_gate) 
    agent = QAgent()
    
    n_episodes = 100
    steps_per_episode = 25 
    
    print(f"--- DEBUT DE L'OPTIMISATION : {env.gate_name.upper()} ---")
    print(f"\n Nombre d'épisodes total : {n_episodes}\n")
    
    best_reward = -float('inf')
    best_config = (0, 0)
    best_metrics = {} 

    for ep in range(n_episodes):
        # Reset aléatoire de la position de départ tous les 10 épisodes pour explorer
        if ep % 10 == 0:
             env.w_n = np.random.uniform(env.w_n_min, env.w_n_max)
             env.w_p = np.random.uniform(env.w_p_min, env.w_p_max)
             
        s = env.get_state()
        total_ep_reward = 0
        
        for step_idx in range(steps_per_episode):
            a = agent.act(s)
            
            # On passe les 3 arguments attendus par environment.py
            sn, r, perf = env.step(a, step_idx, ep)
            
            agent.learn(s, a, r, sn)
            
            # Sauvegarde de la meilleure configuration
            if r > best_reward:
                best_reward = r
                best_config = (env.w_n, env.w_p)
                best_metrics = perf.copy()
                
            s = sn
            total_ep_reward += r
        
        if ep % 20 == 0:
            print(f"Episode {ep:4d} | Epsilon: {agent.eps:.2f} | Best Reward: {best_reward:.2f}")

    #Affichage des résultats
    print("\n" + "="*60)
    print(f"OPTIMISATION TERMINEE POUR : {env.gate_name.upper()}")
    print("="*60)
    
    wn_f = best_config[0]
    wp_f = best_config[1]
    
    print(f"MEILLEURE CONFIGURATION TROUVÉE :")
    print(f"  Wn : {wn_f*1e6:.3f} um")
    print(f"  Wp : {wp_f*1e6:.3f} um")

    
    print("-" * 30)
    print(f"PERFORMANCES PHYSIQUES :")
    if best_metrics:
        # On affiche les valeurs stockées dans best_metrics
        print(f"  Délai Moyen (tp)    : {best_metrics.get('delay', 0)*1e12:.2f} ps")
        print(f"  Conso Dynamique     : {best_metrics.get('power_dynamic', 0)*1e6:.2f} uW")
        print(f"  Conso Statique      : {best_metrics.get('power_static', 0)*1e9:.2f} nW")
    else:
        print("  Aucune donnée de simulation disponible.")
        
    print("-" * 30)
    print(f"  Score Reward Final  : {best_reward:.2f}")
    print(f"  Nombre de simulations : {len(env.cache)}")
    print("="*60)

if __name__ == "__main__":
    main()
