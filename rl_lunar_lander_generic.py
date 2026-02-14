# ----------------------------------------------------------------------------------------------------------------------------------
# Archivo: rl_lunar_lander_generic.py
# ----------------------------------------------------------------------------------------------------------------------------------
# Descripción: Rutinas de entrenanmiento y elementos comunes a todos los agentes
# ----------------------------------------------------------------------------------------------------------------------------------
# Implementado por: Felipe Ramírez Herrera
# Curso Aprendizaje por Reforzamiento. 
# Master de Inteligencia Artificial Avanzada y Aplicada (IA3)
# Universidad de Valencia / ADEIT
# Ultima revisión: 15/06/2024 
# ----------------------------------------------------------------------------------------------------------------------------------

import os
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt # type: ignore
from IPython.display import clear_output
import gymnasium as gym # type: ignore
from gymnasium.wrappers import RecordVideo # type: ignore
from collections import deque
from datetime import datetime
import numpy as np
from enum import Enum
import random
import time
import imageio # type: ignore


from rl_lunar_lander_ERM import ExperienceReplayMemory
import rl_lunar_lander_constants as constants


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------------------------------------------------------------
# Contador con ventana (Moving Average)
# ----------------------------------------------------------------------------------------------------------------------------------
class RunningAverage:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.q = deque(maxlen=maxlen)  

    def Add(self, val : torch.Tensor):
        self.q.append(val.detach())

    def Mean(self):
        # Avoid divide by 0
        return torch.mean(torch.as_tensor(self.q)).item()

# ----------------------------------------------------------------------------------------------------------------------------------
# Clase base para un agente genérico.
# ----------------------------------------------------------------------------------------------------------------------------------
class Agent(object):
    def __init__(self):
        super().__init__()

    def Act(self, state, evaluate = False):
        """
        Método abstracto para tomar una acción basada en el estado actual del entorno.

        Args:
            state (tensor): Estado actual del entorno.
            evaluate (bool): Indica si el agente está en modo de evaluación.

        Raises:
            NotImplementedError: Este método debe ser implementado por subclases específicas del agente.
        """
        raise NotImplementedError
    
    def Fix(self, state):
        """
        Método opcional para preprocesar el estado antes de pasarlo al entorno.

        Args:
            state (tensor): Estado que puede ser preprocesado.

        Returns:
            tensor: Estado preprocesado.
        """
        return state

    def StepAndUpdate(self, buffer : ExperienceReplayMemory):
        """
        Método abstracto para realizar un paso de entrenamiento del agente y actualizar sus parámetros.

        Args:
            buffer (ExperienceReplayMemory): Buffer de memoria de repetición de experiencias.

        Raises:
            NotImplementedError: Este método debe ser implementado por subclases específicas del agente.
        """
        raise NotImplementedError

    def getStats(self, episode):
        raise NotImplementedError
    
    def getStatNames(self):
        raise NotImplementedError
    
    def getName(self):
        """
        Método para obtener el nombre del agente.

        Raises:
            NotImplementedError: Este método debe ser implementado por subclases específicas del agente.
        """
        raise NotImplementedError
    
    def soft_update(self, target, source, tau):
        """
        Actualiza suavemente los parámetros de la red objetivo utilizando el método de suavizado.

        Args:
            target (torch.nn.Module): Red neuronal objetivo a actualizar.
            source (torch.nn.Module): Red neuronal fuente desde la cual se copian los parámetros.
            tau (float): Parámetro de suavizado.

        """
        with torch.no_grad():
            for t, s in zip(target.parameters(), source.parameters()):
                t.data.copy_(tau * s.data + (1.0 - tau) * t.data)

    def polyak_averaging_update(self, target, source, polyak=0.995):
        """
        Actualiza los parámetros de la red objetivo utilizando el promedio de Polyak.

        Args:
            target (torch.nn.Module): Red neuronal objetivo a actualizar.
            source (torch.nn.Module): Red neuronal fuente desde la cual se copian los parámetros.
            polyak (float): Factor de promediado de Polyak (predeterminado: 0.995).
        """
        with torch.no_grad():
            for t, s in zip(target.parameters(), source.parameters()):
                t.data.mul_(polyak)
                t.data.add_((1 - polyak) * s.data)

    def hard_update(self, target, source): 
        """
        Actualiza los parámetros de la red objetivo de manera directa (copia dura).

        Args:
            target (torch.nn.Module): Red neuronal objetivo a actualizar.
            source (torch.nn.Module): Red neuronal fuente desde la cual se copian los parámetros.
        """
        with torch.no_grad():
            for t, s in zip(target.parameters(), source.parameters()):
                t.data.copy_(s.data)

    def enable_gradients(self, network, value = False):
        """
        Habilita o deshabilita los gradientes para la red neuronal.

        Args:
            network (torch.nn.Module): Red neuronal para la cual se habilitarán o deshabilitarán los gradientes.
            value (bool): Valor booleano que indica si se habilitan (True) o deshabilitan (False) los gradientes.
        """
        for param in network.parameters():
            param.requires_grad = value

    def update_params(self, optim, loss):
        """
        Realiza la actualización de los parámetros del modelo de acuerdo al optimizador y la pérdida calculada.

        Args:
            optim (torch.optim.Optimizer): Optimizador utilizado para actualizar los parámetros del modelo.
            loss (tensor): Pérdida calculada que se utilizará para retropropagación.
        """
        optim.zero_grad()
        loss.backward()
        optim.step()

    def compute_bellman_equation(self, rewards, target, dones):
        """
        Calcula la ecuación de Bellman utilizada en aprendizaje por refuerzo.

        Args:
            rewards (tensor): Recompensas obtenidas en la transición.
            target (tensor): Valores objetivo calculados para la siguiente transición.
            dones (tensor): Tensor booleano que indica si la transición es terminal.

        Returns:
            tensor: Valores de la ecuación de Bellman calculados.
        """
        return rewards + (1.0 - dones) * self.gamma * target 
    
    def save_model(self, episode, metrics_by_episode, filename):
        """
        Método para guardar el modelo entrenado en un archivo.

        Args:
            episode (int): Número de episodio en el cual se guarda el modelo.
            metrics_by_episode (dict): Métricas asociadas al episodio.
            filename (str): Nombre del archivo donde se guardará el modelo.
        
        Raises:
            NotImplementedError: Este método debe ser implementado por subclases específicas del agente.
        """
        raise NotImplementedError
    
    def load_saved_model(self, episode, filename):
        """
        Método para cargar un modelo previamente guardado desde un archivo.

        Args:
            episode (int): Número de episodio del modelo a cargar.
            filename (str): Nombre del archivo desde donde se cargará el modelo.
        
        Raises:
            NotImplementedError: Este método debe ser implementado por subclases específicas del agente.
        """
        raise NotImplementedError
    
    # El agente solo funciona en entornos continuos o no.
    def IsContinuosEnvExpected(self):
        raise NotImplementedError
    

    # En RL, una política es un conjunto de reglas que determina la acción que un agente debe tomar en un 
    # entorno dado para maximizar una recompensa acumulada a largo plazo. El "experience rollout" aprovecha 
    # las experiencias reales que ocurren durante la ejecución de la política actual para mejorar la política 
    # misma.

    def IsPriorCollectingForExperienceRolloutRequired(self):
        raise NotImplementedError

    def set_eval(self):
        raise NotImplementedError
    
    def set_train(self):
        raise NotImplementedError

# ----------------------------------------------------------------------------------------------------------------------------------
# Inicializa los pesos de los módulos (políticas y criticos)
# ----------------------------------------------------------------------------------------------------------------------------------
def weights_init_(m):
    """
    Inicializa los pesos de una capa lineal utilizando la inicialización Xavier uniforme
    y los sesgos con un valor constante cero.

    Args:
        m (torch.nn.Module): Módulo de PyTorch que representa una capa lineal.

    Detalles:
        - La inicialización Xavier uniforme ajusta los pesos de manera que se mantenga la varianza
          de las activaciones y gradientes aproximadamente constante en todas las capas.
        - Los sesgos se inicializan con un valor constante cero para evitar sesgos no deseados en el modelo.

    Ejemplo:
        >>> model.apply(weights_init_)
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# ----------------------------------------------------------------------------------------------------------------------------------
# Realiza una evaluación tras un número determinado iteraciones del ciclo de entrenamiento
# ----------------------------------------------------------------------------------------------------------------------------------

def evaluate_policy(agent : Agent, eval_episodes = 10, max_episode_length = constants.EPISODE_MAX_LEN):

    agent.set_eval()

    continuos = agent.IsContinuosEnvExpected()

    """
    Evalúa la política del agente en un entorno específico durante un número dado de episodios de evaluación.

    Args:
        policy (Agent): Agente cuya política se va a evaluar.
        continuos (bool): Indica si el entorno es continuo o discreto (predeterminado: False).
        eval_episodes (int): Número de episodios de evaluación (predeterminado: 10).
        seed (int): Semilla para la generación de números aleatorios del entorno (predeterminado: 1234).

    Returns:
        tuple: Tuple que contiene la recompensa promedio por episodio, la recompensa acumulada total y el número de episodios evaluados.

    Detalles:
        - Si `continuos` es True, crea un entorno LunarLander-v2 continuo.
        - Si `continuos` es False (predeterminado), crea un entorno LunarLander-v2 discreto.
        - `policy.Act(state, evaluate=True)` se utiliza para obtener acciones del agente en modo de evaluación.
        - `policy.Fix(action)` se utiliza para procesar acciones antes de aplicarlas al entorno en entornos discretos.

    Ejemplo:
        >>> agent = MiAgente()
        >>> evaluate_policy(agent, continuos=True, eval_episodes=5)
        Evaluation over 5 episodes: 123.45
        (123.45, 617.25, 5)
    """
    if (continuos): 
        env = gym.make("LunarLander-v2", continuous = True)
    else:
        env = gym.make('LunarLander-v2')

    cum_reward = 0.0

    for _ in range(eval_episodes):
        state, _ = env.reset()
        done, truncated = False, False
        episode_length = 0
        while not (done or truncated):
            action = agent.Act(state, evaluate=True)
            if (continuos):
                state, reward, done, truncated, _ = env.step(action)
            else:
                state, reward, done, truncated, _ = env.step(agent.Fix(action))
            cum_reward += reward
            episode_length += 1
            if (episode_length > max_episode_length):
                break


    avg_reward = cum_reward / eval_episodes
    print ("Evaluación durante %d episodios: %f" % (eval_episodes, avg_reward))
    return avg_reward, cum_reward, eval_episodes

# ----------------------------------------------------------------------------------------------------------------------------------
# Obtiene las caracteristicas del entorno
# ----------------------------------------------------------------------------------------------------------------------------------
def get_enviroment_details(continuos = False):
    """
    Imprime detalles importantes del entorno LunarLander-v2.

    Args:
        continuos (bool): Indica si el entorno es continuo o discreto (predeterminado: False).

    Detalles:
        - Imprime la dimensión del espacio de estados (`states_dim`).
        - Imprime la dimensión del espacio de acciones (`actions_dim`).
        - Imprime el valor máximo de las acciones (`max_action`).
        - Si `continuos` es True, crea un entorno LunarLander-v2 continuo.
        - Si `continuos` es False (predeterminado), crea un entorno LunarLander-v2 discreto.
    """
    if (continuos): 
        env = gym.make("LunarLander-v2", continuous = True)
    else:
        env = gym.make('LunarLander-v2')

    print("states_dim = {0}".format( env.observation_space.shape[0]))
    print("actions_dim = {0}".format(env.action_space.shape[0]))
    print("max_action = {0}".format( env.action_space.high[0]))

# ----------------------------------------------------------------------------------------------------------------------------------
# Establece la semilla para la generación de números aleatorios en varias librerías y dispositivos.
# ----------------------------------------------------------------------------------------------------------------------------------
def seed_everything(seed=73):
    """
    Establece la semilla para la generación de números aleatorios en varias librerías y dispositivos.

    Args:
        seed (int): Semilla a utilizar para la inicialización de la generación de números aleatorios (predeterminado: 73).

    Detalles:
        - Establece la semilla para random, numpy y Torch para asegurar la reproducibilidad de los experimentos.
        - Configura la semilla para CUDA (GPU) si está disponible.
        - Configura cuDNN para ser determinista, asegurando que las operaciones sean reproducibles en GPU.

    Ejemplo:
        >>> seed_everything(123)
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

# ----------------------------------------------------------------------------------------------------------------------------------
# Hasta que hayan transcurrido start_steps, muestrea acciones al azar de una distribución uniforme para una mejor exploración. 
# Después, utiliza la política aprendida (con algo de ruido).
# ----------------------------------------------------------------------------------------------------------------------------------
def collect_rollouts(env, buffer: ExperienceReplayMemory, continuos = False, rollout_episodes = 50, episode_max_len = constants.EPISODE_MAX_LEN):
    """
    Recolecta muestras de experiencia para calentar el buffer de memoria.

    Args:
        env (gym.Env): Entorno de OpenAI Gym del cual recolectar muestras.
        buffer (ExperienceReplayMemory): Buffer de memoria de repetición de experiencias donde almacenar las muestras.
        continuos (bool): Indica si el entorno es continuo (True) o discreto (False) (predeterminado: False).
        warmming_episodes (int): Número de episodios para recolectar muestras (predeterminado: 50).
        single_value (bool): Indica si se utiliza una única acción (True) o un vector de acciones (False) en entornos discretos (predeterminado: False).

    Detalles:
        - Utiliza el método de muestreo aleatorio de acciones (`env.action_space.sample()`) para recolectar muestras antes de iniciar el aprendizaje.
        - Almacena las transiciones de experiencia (state, action, reward, next_state, terminated) en el buffer proporcionado.
    """
    for i in range(0, rollout_episodes):
        terminated, truncated =  False, False
        state, _ = env.reset()

        episode_length = 0
        while not (terminated or truncated):
            if (continuos):
                action = env.action_space.sample()        
                next_state, reward, terminated, truncated, _ = env.step(action)
            else:
                action = np.asarray([0, 0, 0, 0])                
                action[env.action_space.sample()] = 1          
                next_state, reward, terminated, truncated, _ = env.step(np.argmax(action))    

            buffer.store(state, action, reward, next_state, terminated) 

            episode_length += 1

            if (episode_length > episode_max_len):
                break

            


# ----------------------------------------------------------------------------------------------------------------------------------
# Ciclo de aprendizaje completo para un agente
# ----------------------------------------------------------------------------------------------------------------------------------
def run_agent_learning_cycle(agent: Agent, out_filename:str, erm_filename:str, eval_episodes = constants.DEFAULT_EVAL_EPISODES, eval_frequency = constants.DEFAULT_EVAL_FREQUENCY, 
                             seed = 1234, episode_max_len = constants.EPISODE_MAX_LEN):
    
    """
    Ejecuta el ciclo de aprendizaje del agente en un entorno de OpenAI Gym.

    Args:
        agent (Agent): Agente que realizará el aprendizaje.
        filename (str): Nombre del archivo para guardar los checkpoints del modelo entrenado.
        eval_episodes (int): Número de episodios de evaluación (predeterminado: constants.DEFAULT_EVAL_EPISODES).
        eval_frequency (int): Frecuencia de evaluación en episodios (predeterminado: constants.DEFAULT_EVAL_FREQUENCY).
        continuos (bool): Indica si el entorno es continuo (True) o discreto (False) (predeterminado: False).
        req_warmming_steps (bool): Indica si se requieren pasos de calentamiento del buffer de memoria (predeterminado: False).
        single_value (bool): Indica si se utiliza una única acción (True) o un vector de acciones (False) en entornos discretos (predeterminado: False).
        seed (int): Semilla para la generación de números aleatorios del entorno (predeterminado: 1234).

    Returns:
        tuple: Tupla que contiene la lista de recompensas por episodio y la lista de pasos por episodio.

    Detalles:
        - Ejecuta el aprendizaje del agente durante un número predeterminado de episodios.
        - Gestiona la inicialización del entorno, la recolección de muestras de experiencia, y el entrenamiento del agente.
        - Realiza evaluaciones periódicas del desempeño del agente y guarda los checkpoints del modelo entrenado.
        - Finaliza el entorno al finalizar el ciclo de aprendizaje.
    """

    print("Agente: {0}".format(agent.getName()))
   
    continuos = agent.IsContinuosEnvExpected()

    collecting_for_rollout = agent.IsPriorCollectingForExperienceRolloutRequired()

    if (continuos): 
        env = gym.make("LunarLander-v2", continuous = True)
    else:
        env = gym.make('LunarLander-v2')

    metrics_by_episode = {}    

    seed_everything(seed=1234)

    print("env:=", env.action_space)


    GOAL = 0

    if (continuos):
        GOAL = constants.C_REWARD_THRESHOLD
    else: 
        GOAL = constants.D_REWARD_THRESHOLD

    if (continuos):
        print("Espacio de acciones: {0} (continuo)".format(env.action_space.shape))
        print("Espacio de observaciones: {0}".format(env.observation_space.shape))
    else:
        print("Espacio de acciones: {0} (discreto)".format(env.action_space.n))
        print("Espacio de observaciones: {0}".format(env.observation_space.sample().shape))

    print("Meta: Recompensa Promedio > {0}".format(GOAL))


    
    
   
    reward_window = deque(maxlen = constants.DEFAULT_AVG_SLIDING_WINDOW_SIZE)

    length_window = deque(maxlen = constants.DEFAULT_AVG_SLIDING_WINDOW_SIZE)

    start = time.time()

    timesteps_since_eval = 0

    replay_buffer = ExperienceReplayMemory(constants.DEFAULT_REPLAY_MEMSIZE, constants.DEFAULT_BATCHSIZE, device)
   
    converges = False

    total_steps = 0

    base_episode = 0 


    model_backup_filename = out_filename.format(constants.n_episodes)
    buffer_backup_filename = erm_filename.format(constants.n_episodes)
                                      
    # Busca el último de los respaldos
    if os.path.exists(model_backup_filename) and os.path.exists(buffer_backup_filename):
        base_episode = constants.n_episodes
    else:
        # Si no lo encuentra busca los posibles anteriores
        for episode in reversed(range(1, constants.n_episodes)):

            model_backup_filename = out_filename.format(episode)
            buffer_backup_filename = erm_filename.format(episode)
        
            if os.path.exists(model_backup_filename) and os.path.exists(buffer_backup_filename):
                base_episode = episode
                break

    # Recargar un checkpoint puede ser complicado pues no se tiene un experience replay buffer
    if (base_episode > 0):
        # Carga la última version del agente
        metrics_by_episode = agent.load_saved_model(base_episode, out_filename.format(base_episode))
        # Carga la última version del buffer
        replay_buffer.Restore(base_episode, erm_filename.format(base_episode))

        # Asigna el estado previo del generador de números aleatorios
        # esto evita que repita el patron de los episodios anteriores.
        restored_random_state = metrics_by_episode[base_episode]["random_state"]

        # Restaura los contadores 
        reward_window =  metrics_by_episode[base_episode]["reward_window"]
        length_window =  metrics_by_episode[base_episode]["length_window"]
        total_steps =  metrics_by_episode[base_episode]["total_steps"]

        random.setstate(restored_random_state)

        scale =  random.random()

        seed = math.ceil(base_episode * constants.n_episodes * scale)

        seed_everything(seed)

        print("El agente {0} se recargo desde el último checkpoint en el episodio {1}.".format(agent.getName(), base_episode))
        print("El agente {0} se reinicia con la semilla aleatoria {1}.".format(agent.getName(), seed))    

        print("Historia:")

        total_mins = 0

        for episode in range(1, base_episode + 1):
            if episode % constants.DEFAULT_AVG_SLIDING_WINDOW_SIZE == 0 and episode > 0:

                e_lengths = []

                for i in range(1, episode + 1):
                    e_lengths.append(metrics_by_episode[episode]['episode_length'])

                if len(e_lengths) > 0:
                    current_avg_length = np.mean(e_lengths)
                else:
                    current_avg_length = 0.0

                current_cum_reward = round(metrics_by_episode[episode]['cum_reward'], constants.N_DECIMALS)
                current_avg_reward = round(metrics_by_episode[episode]['avg_reward'], constants.N_DECIMALS)
                current_ttl_time = round(metrics_by_episode[episode]['episode_total_time'], constants.N_DECIMALS)
                current_epi_length = round(metrics_by_episode[episode]['episode_length'], constants.N_DECIMALS)

                hist_fmt = "- Episodio {0} con Duración = {1} minutos, Recompensa = {2}, Longitud = {3}. Recompensa Promedio = {4} y Longitud Promedio = {5}"
                print(hist_fmt.format(episode, current_ttl_time, current_cum_reward , current_epi_length, current_avg_reward, current_avg_length))

                if (metrics_by_episode[episode]['eval_executed']):

                    eval_avg_rwd = metrics_by_episode[episode]['eval_avg_reward']

                    if (eval_avg_rwd) < GOAL:
                        hist_fmt_eval = "- Episodio {0} evaluado con Recompensa Promedio = {1}"
                        print(hist_fmt_eval.format(episode,  eval_avg_rwd))
                    else:
                        hist_fmt_eval = "- Episodio {0} evaluado con Recompensa Promedio = {1} y supera/iguala la meta de {2}"
                        print(hist_fmt_eval.format(episode,  eval_avg_rwd, GOAL))
                        
                total_mins += current_ttl_time
        
        if total_mins > 0:
            print("Tiempo total de entrenamiento es {0} minuto(s).".format(total_mins))
        

        times_reach_goal = 0

        hist_eval_avg_reward = []
        hist_eval_cum_reward = []

        first_goal = False
        first_goal_achievement = 0

        for episode in range(1, base_episode + 1):
            if metrics_by_episode[episode]['finished']:
                times_reach_goal += 1
                if not first_goal:
                    first_goal = True
                    first_goal_achievement = episode

            if metrics_by_episode[episode]['eval_executed']:
                cum = metrics_by_episode[episode]['eval_cum_reward']
                avg = metrics_by_episode[episode]['eval_avg_reward']
                hist_eval_cum_reward.append(cum)
                hist_eval_avg_reward.append(avg)

              
        
        if (times_reach_goal > 0):
            
            if (first_goal):
                print("El modelo ha alcanzado la meta en el episodio {0} por primera vez.".format(first_goal_achievement))

            print("El modelo ha alcanzado la meta rewards > {0} un total de {1} veces.".format(GOAL, times_reach_goal))

        if len(hist_eval_avg_reward) > 0:
            v = round(np.mean(hist_eval_avg_reward), constants.N_DECIMALS)
            l = len(hist_eval_avg_reward)
            print("Evaluación recompensa promedio (promedio) = {0} en {1} evaluaciones efectuadas.".format(v, l))

        if len(hist_eval_cum_reward) > 0:
            v = round(np.mean(hist_eval_cum_reward), constants.N_DECIMALS)
            l = len(hist_eval_cum_reward)
            print("Evaluación recompensa acumulada (promedio) = {0} en {1} evaluaciones efectuadas.".format(v, l))


    if (base_episode + 1 < constants.n_episodes + 1):
        
        if (base_episode == 0):
            if (collecting_for_rollout):
                collect_rollouts(env, replay_buffer, continuos=continuos)

        first = False
        if (base_episode == 0):
            first = True

        for episode in range(base_episode + 1, constants.n_episodes + 1):
            
            episode_s_time = time.time()

            cum_reward = 0

            terminated, truncated =  False, False

            episode_length = 0

            if (first):
                state, _ = env.reset(seed=seed)
                first = False
            else:
                state, _ = env.reset()

            while not (terminated or truncated):
            
                action = agent.Act(state)

                if (continuos):
                    next_state, reward, terminated, truncated, _ = env.step(action)
                else:
                    next_state, reward, terminated, truncated, _ = env.step(agent.Fix(action))

                replay_buffer.store(state, action, reward, next_state, terminated)                       

                cum_reward += reward    
                        
                agent.StepAndUpdate(replay_buffer)  
                
                total_steps += 1
                
                state = next_state

                episode_length += 1

                if (episode_length > episode_max_len):
                    break

            episode_e_time = time.time()

            episode_total_time = (episode_e_time - episode_s_time)

            td_error = agent.getStats(episode)
      
            reward_window.append(cum_reward)

            length_window.append(episode_length)

            avg_reward = np.mean(reward_window)

            avg_length = np.mean(length_window)

            timesteps_since_eval += 1
            
            if episode % constants.DEFAULT_AVG_SLIDING_WINDOW_SIZE == 0 and episode > 0:
                txt_format = 'Episodio {} ejecutado en {:.2f} minutos. ETA: {:.2f} minutos. Recompensa: {:.2f} Recompensa promedio: {:.2f} Longitud promedio: {:.2f}'
                total_time = (time.time() - start)/60
                time_left = (((time.time() - start) / episode)*constants.n_episodes)/60
                print(txt_format.format(episode, total_time, time_left, cum_reward, avg_reward, avg_length))

            
            eval_avg_reward = 0
            eval_cum_reward = 0 
            
            eval_executed = False
            if (timesteps_since_eval >= eval_frequency):
                timesteps_since_eval %= eval_frequency
                eval_executed = True
                eval_avg_reward, eval_cum_reward, _ = evaluate_policy(agent, eval_episodes)

            

            metrics_by_episode[episode] = dict(
                model_name = agent.getName(),
                model_epoch = episode,
                cum_reward = cum_reward,
                avg_reward = avg_reward,
                episode_length = episode_length,
                episode_total_time = episode_total_time,
                finished = avg_reward >= GOAL,
                eval_executed = eval_executed,
                eval_avg_reward = eval_avg_reward,
                eval_cum_reward = eval_cum_reward,
                random_state = random.getstate(),
                reward_window = reward_window.copy(),
                length_window = length_window.copy(),
                total_steps = total_steps
            )

            metrics_by_episode[episode].update(td_error)

            
            if (episode % 500 == 0 or episode == constants.n_episodes):
                # Almacena el modelo
                temp_out_filename = "temp_" + out_filename.format(episode)
                temp_erm_filename = "temp_" + erm_filename.format(episode)

                agent.save_model(episode, metrics_by_episode, temp_out_filename)
                os.rename(temp_out_filename, out_filename.format(episode))

                replay_buffer.Backup(episode, temp_erm_filename)
                os.rename(temp_erm_filename, erm_filename.format(episode))

                # Almacena el experience replay buffer


            if (not converges):   
                if avg_reward >= GOAL:
                    print('Algoritmo converge en {:d} episodios con recompensa promedio: {:.2f}'.format(episode, avg_reward))
                    converges = True
                
        
    env.close()

    return metrics_by_episode



# ----------------------------------------------------------------------------------------------------------------------------------
# Demostración en vivo de un agente entrenado
# ----------------------------------------------------------------------------------------------------------------------------------
def live_model_demostration(agent: Agent, n_episodes = 5, episode_max_len = constants.EPISODE_MAX_LEN):

    agent.set_eval()

    continuos = agent.IsContinuosEnvExpected()

    """
    Realiza una demostración en vivo del agente en el entorno LunarLander-v2 de OpenAI Gym.

    Args:
        Agent (Agent): Agente cuyo modelo se va a demostrar en vivo.
        n_episodes (int): Número de episodios de demostración (predeterminado: 5).
        continuos (bool): Indica si el entorno es continuo (True) o discreto (False) (predeterminado: False).

    Detalles:
        - Inicializa el entorno LunarLander-v2 con renderización en modo de arreglo RGB para la demostración visual.
        - Utiliza el método Act del agente para elegir acciones en modo de evaluación.
        - Renderiza el entorno después de cada acción elegida por el agente.
        - Termina la demostración al completar el número especificado de episodios o cuando el entorno indique término o truncamiento.
    """

    os.environ["SDL_VIDEODRIVER"] = "dummy"

    if (continuos): 
        env = gym.make("LunarLander-v2", continuous = True, render_mode="rgb_array")
    else:
        env = gym.make('LunarLander-v2', render_mode="rgb_array")

    state, info = env.reset(seed=1234)

    for _ in range(n_episodes):
        terminated = False
        truncated = False
        episode_length = 0
        while not (terminated or truncated):

            action = agent.Act(state, True)

            if (continuos):
                next_state, reward, terminated, truncated, _ = env.step(action)
            else:
                next_state, reward, terminated, truncated, _ = env.step(agent.Fix(action))

            state = next_state
            clear_output(wait=True)
            plt.imshow( env.render() )
            plt.show()

            episode_length += 1

            if (episode_length > episode_max_len):
                break

        state, _ = env.reset()
    env.close()


# ----------------------------------------------------------------------------------------------------------------------------------
# Demostración fuera de linea, graba un video MP4 que se incorpora al notebook
# ----------------------------------------------------------------------------------------------------------------------------------
def generate_MP4(agent : Agent, video_filename, num_episodes = 10, seed = 1234, episode_max_len = constants.EPISODE_MAX_LEN):
    
    agent.set_eval()

    continuos = agent.IsContinuosEnvExpected()

    if (continuos): 
        env = gym.make("LunarLander-v2", continuous = True, render_mode="rgb_array")
    else:
        env = gym.make('LunarLander-v2', render_mode="rgb_array")

    first = True
    with imageio.get_writer(video_filename, fps=60) as video:
        
        for _ in range(1, num_episodes + 1):
                     
            if (first):
                state, _ = env.reset(seed=seed)
                first = False
            else:
                state, _ = env.reset()
            
            video.append_data(env.render())

            terminated =  False
            truncated = False

            episode_length = 0

            while not (terminated or truncated):
                
                action = agent.Act(state, evaluate=True)

                if (continuos):
                    next_state, reward, terminated, truncated, _ = env.step(action)
                else:
                    next_state, reward, terminated, truncated, _ = env.step(agent.Fix(action))
                
                state = next_state

                video.append_data(env.render())

                episode_length += 1
                if (episode_length > episode_max_len):
                    break

    env.close()
# ----------------------------------------------------------------------------------------------------------------------------------
# END OF FILE
# ----------------------------------------------------------------------------------------------------------------------------------