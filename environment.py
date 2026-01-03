import subprocess
import os
import re
import numpy as np

class InverterEnv:
    def __init__(self, gate_name="ET"):
        #On définit le dossier des netlists relatif au script python
        self.netlist_dir = "netlists"
        self.gate_name = gate_name
        # Le fichier template doit être dans netlists/
        self.template_path = os.path.join(self.netlist_dir, f"{gate_name}.cir")
        self.params_path = os.path.join(self.netlist_dir, "params.spice")
        
        #Valeurs limites pour l'excursion
        self.w_n_min, self.w_n_max = 0.2e-6, 5.0e-6 
        self.w_p_min, self.w_p_max = 0.2e-6, 8.0e-6
        
        #Valeurs initiales
        self.w_n = 0.5e-6
        self.w_p = 1.0e-6
        self.l = 0.15e-6
        
        self.cache = {}
        self.grid_size = 20

    def get_state(self):
        # Normalisation
        sn = int(np.clip((self.w_n - self.w_n_min) / (self.w_n_max - self.w_n_min) * (self.grid_size - 1), 0, self.grid_size - 1))
        sp = int(np.clip((self.w_p - self.w_p_min) / (self.w_p_max - self.w_p_min) * (self.grid_size - 1), 0, self.grid_size - 1))
        return sn * self.grid_size + sp

    def step(self, action, step_num, ep_num):
        pas_n = (self.w_n_max - self.w_n_min) / self.grid_size
        pas_p = (self.w_p_max - self.w_p_min) / self.grid_size
        
        if action == 0: self.w_n += pas_n
        elif action == 1: self.w_n -= pas_n
        elif action == 2: self.w_p += pas_p
        elif action == 3: self.w_p -= pas_p
        
        self.w_n = float(np.clip(self.w_n, self.w_n_min, self.w_n_max))
        self.w_p = float(np.clip(self.w_p, self.w_p_min, self.w_p_max))

        config_key = (round(self.w_n, 8), round(self.w_p, 8))
        if config_key in self.cache:
            perf = self.cache[config_key]
        else:
            perf = self.simulate()
            self.cache[config_key] = perf
        
        #Calcul de la récompense
        
        d_lh = perf['tpLH'] * 1e12  # ps (Montée - PMOS)
        d_hl = perf['tpHL'] * 1e12  # ps (Descente - NMOS)
        d_avg = (d_lh + d_hl) / 2
        
        
        p_stat_val = perf['power_static'] * 1e9  # nW 
        p_dyn_val = perf['power_dynamic'] * 1e6  # uW
        

        reward = - (1 * d_avg  + 0.5 * p_dyn_val + 0.1 * p_stat_val)

        
        return self.get_state(), reward, perf

    def simulate(self):
        #Écriture des paramètres dans le fichier params.spice
        with open(self.params_path, "w") as f:
            f.write(f".param VDD=1.8\n")
            f.write(f".param L={self.l}\n")
            f.write(f".param W_N={self.w_n}\n")
            f.write(f".param W_P={self.w_p}\n")
        
        #Lancement NGSPICE
        cmd = ["ngspice", "-b", self.template_path]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout
        except FileNotFoundError:
            print("ERREUR: ngspice n'est pas installé ou le chemin est incorrect.")
            return {'tpLH': 1e-8, 'tpHL': 1e-8, 'power_static': 1e-3, 'power_dynamic': 1e-3}
        
        #Parsing (Récupération tpLH et tpHL séparés)
        try:
            # tpLH
            tplh_match = re.search(r"tplh\s*=\s*([0-9.eE+-]+)", output, re.IGNORECASE)
            tp_lh = float(tplh_match.group(1)) if tplh_match else 1e-9
            
            # tpHL
            tphl_match = re.search(r"tphl\s*=\s*([0-9.eE+-]+)", output, re.IGNORECASE)
            tp_hl = float(tphl_match.group(1)) if tphl_match else 1e-9
            
            # P_static
            ps_match = re.search(r"p_static\s*=\s*([0-9.eE+-]+)", output, re.IGNORECASE)
            p_static = float(ps_match.group(1)) if ps_match else 1e-3
            
            # P_dynamic
            pd_match = re.search(r"p_dynamic\s*=\s*([0-9.eE+-]+)", output, re.IGNORECASE)
            p_dynamic = float(pd_match.group(1)) if pd_match else 1e-3
            
            return {
                'tpLH': max(1e-15, abs(tp_lh)),
                'tpHL': max(1e-15, abs(tp_hl)),
                'power_static': max(1e-15, abs(p_static)), 
                'power_dynamic': max(1e-15, abs(p_dynamic)),
                'delay': (abs(tp_lh) + abs(tp_hl))/2 # Pour compatibilité
            }
            
        except Exception as e:
            # print(f"❌ Erreur parsing: {e}")
            return {'tpLH': 1e-8, 'tpHL': 1e-8, 'power_static': 1e-3, 'power_dynamic': 1e-3, 'delay': 1e-8}
