# ----------------------------------------------------------------------------------------------------------------------------------
# Archivo: rl_lunar_lander_constants.py
# ----------------------------------------------------------------------------------------------------------------------------------
# Descripción: Constantes para la resolución del ambiente LunarLander V2
# ----------------------------------------------------------------------------------------------------------------------------------
# Implementado por: Felipe Ramírez Herrera
# Curso Aprendizaje por Reforzamiento. 
# Master de Inteligencia Artificial Avanzada y Aplicada (IA3)
# Universidad de Valencia / ADEIT
# Ultima revisión: 15/06/2024 
# ----------------------------------------------------------------------------------------------------------------------------------

# Define una constante DEFAULT_DISCOUNT con valor 0.99, que representa el factor de descuento por defecto para 
# el aprendizaje por refuerzo.
DEFAULT_DISCOUNT = 0.99
# Establece la constante DEFAULT_BATCHSIZE con valor 128, que indica el tamaño predeterminado del lote utilizado 
# en el entrenamiento de redes neuronales.
DEFAULT_BATCHSIZE = 128
# Define DEFAULT_REPLAY_MEMSIZE como 1000000, que es el tamaño predeterminado de la memoria de repetición utilizada 
# en métodos de aprendizaje por refuerzo que usan experiencias pasadas para el entrenamiento.
DEFAULT_REPLAY_MEMSIZE = 1000000

DEFAULT_EVAL_EPISODES = 10
DEFAULT_EVAL_FREQUENCY = 100
DEFAULT_AVG_SLIDING_WINDOW_SIZE = 100

EPISODE_MAX_LEN = 4000


# Meta para entornos discretos:
D_REWARD_THRESHOLD = 225.0  # Punto medio, generalmente en el rango de 200 a 250

# Meta para entornos continuos (se espera que sea menor a la discreta)
C_REWARD_THRESHOLD = 200.0

LEARN_EVERY = 4
n_episodes = 2500

output_dqn_filename = "rl_dqn_{0}.pt" 
output_ddqn_filename = "rl_ddqn_{0}.pt" 
output_ddpg_filename = "rl_ddpg_{0}.pt" 
output_td3_filename = "rl_td3_{0}.pt" 
output_sac_filename = "rl_sac_{0}.pt" 

erm_dqn_filename = "rl_dqn_{0}.erm" 
erm_ddqn_filename = "rl_ddqn_{0}.erm" 
erm_ddpg_filename = "rl_ddpg_{0}.erm" 
erm_td3_filename = "rl_td3_{0}.erm" 
erm_sac_filename = "rl_sac_{0}.erm" 

N_DECIMALS = 2

# ----------------------------------------------------------------------------------------------------------------------------------
# END OF FILE
# ----------------------------------------------------------------------------------------------------------------------------------