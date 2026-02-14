# ----------------------------------------------------------------------------------------------------------------------------------
# Archivo: rl_lunar_lander_Q.py
# ----------------------------------------------------------------------------------------------------------------------------------
# Descripción: Deep Q Learning (DQN) y Double Deep Q-Network (DDQN)
# ----------------------------------------------------------------------------------------------------------------------------------
# Implementado por: Felipe Ramírez Herrera
# Curso Aprendizaje por Reforzamiento. 
# Master de Inteligencia Artificial Avanzada y Aplicada (IA3)
# Universidad de Valencia / ADEIT
# Ultima revisión: 15/06/2024 
# ----------------------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from enum import Enum
import rl_lunar_lander_constants as constants
from rl_lunar_lander_generic import Agent
from rl_lunar_lander_generic import RunningAverage
from rl_lunar_lander_generic import weights_init_
from rl_lunar_lander_ERM import ExperienceReplayMemory
# ----------------------------------------------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------------------------------------------------------------------------------------------------------
Method = Enum('Method', ['DQN', 'DDQN'])
# ----------------------------------------------------------------------------------------------------------------------------------
# La idea de Double Q-learning es reducir las sobreestimaciones descomponiendo la operación máxima en el 
# objetivo en la selección de acción y la evaluación de # acción. Aunque no estén completamente desacopladas, 
# la red objetivo en la arquitectura DQN proporciona un candidato natural para la segunda función de valor, 
# sin necesidad de introducir redes adicionales. Se propone evaluar la política de acuerdo con la QNN Online, 
# pero utilizando la QNN Target para estimar su valor. 
# Basado en: https://arxiv.org/abs/1509.06461

# ----------------------------------------------------------------------------------------------------------------------------------
# CONSTANTES GENERALES
# ----------------------------------------------------------------------------------------------------------------------------------


# Establece DEFAULT_STATES en 8, que representa el número predeterminado de estados en el entorno de aprendizaje.
DEFAULT_DISCRETE_STATES = 8
# Define DEFAULT_ACTIONS como 4, que indica el número predeterminado de acciones posibles que el agente puede tomar 
# en el entorno.
DEFAULT_DISCRETE_ACTIONS = 4
DEFAULT_DISCRETE_MAX_ACTION = DEFAULT_DISCRETE_ACTIONS - 1


# ----------------------------------------------------------------------------------------------------------------------------------
# CONSTANTES PARA DQN y DDQN
# ----------------------------------------------------------------------------------------------------------------------------------

# Representa la tasa de aprendizaje para el optimizador Adam utilizado en el modelo de red neuronal.
XQN_LR = 0.001
# Es un parámetro importante en la actualización suave (soft update) del modelo de red neuronal.
# Propuesto en https://arxiv.org/pdf/1509.02971
XQN_TAU = 0.001 
# Indica cada cuántos pasos de tiempo se actualiza el modelo de red neuronal objetivo en el algoritmo de 
# aprendizaje por refuerzo.
XQN_UPDATE_EVERY = 4
# La tasa de exploración inicial para el agente en el método de aprendizaje.
XQN_EPSILON_START = 1.0
# Es el factor de decaimiento utilizado para reducir gradualmente la tasa de exploración a medida que el agente 
# adquiere experiencia.
XQN_EPSILON_DECAY_FACTOR = 0.995
# Es el valor mínimo que puede alcanzar la tasa de exploración durante el proceso de aprendizaje.
XQN_EPSILON_MINIMUM = 0.01
# Indica cada cuántas actualizaciones de modelo se reemplazan los pesos del modelo objetivo por los pesos del modelo en línea.
XQN_TARGET_REPLACE = 2

# ----------------------------------------------------------------------------------------------------------------------------------
# Representando una red neuronal para aprender la función Q en el aprendizaje por refuerzo.
# ----------------------------------------------------------------------------------------------------------------------------------

# QNN_INPUT_DIM (state size)
# QNN_NUM_ACTIONS (action size)
# QNN_HIDDEN_DIM (hidden dimension)

class QNN(nn.Module):
    def __init__(self, num_of_states, num_of_actions, hidden_dim = 128):
        super(QNN, self).__init__()
        self.fc1 = nn.Linear(num_of_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_of_actions)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.output(x)
       
# ----------------------------------------------------------------------------------------------------------------------------------
# Agente (DQN y DDQN)
# ----------------------------------------------------------------------------------------------------------------------------------

# Define una clase XQN que hereda de Agent, representando un agente en el entorno de aprendizaje por refuerzo.
class XQN(Agent):
    def __init__(self, method = Method.DDQN, num_of_states = DEFAULT_DISCRETE_STATES, num_of_actions = DEFAULT_DISCRETE_ACTIONS, max_action = DEFAULT_DISCRETE_MAX_ACTION, 
                 gamma=constants.DEFAULT_DISCOUNT, epsilon=XQN_EPSILON_START, batch_size=constants.DEFAULT_BATCHSIZE, learning_rate=XQN_LR,  epsilon_dec=XQN_EPSILON_DECAY_FACTOR,  

                 epsilon_min=XQN_EPSILON_MINIMUM, replace_q_target = XQN_TARGET_REPLACE, tau = XQN_TAU, noise_clip = 0.5, device = None):
        
        super(XQN, self).__init__()

        self.method = method
        self.gamma = gamma # alpha = learn rate, gamma = discount
        self.noise_clip = noise_clip
        self.num_of_states = num_of_states
        self.num_of_actions = num_of_actions
        self.max_action = max_action
        self.device = device
        # Exploración ávida 
        self.exploration_rate = epsilon
        self.exploration_rate_decay = epsilon_dec # decrement of epsilon for larger spaces
        self.exploration_rate_minimum = epsilon_min

        # Exploración basada en softmax policy
        self.tau = tau
        self.batch_size = batch_size
        self.number_of_steps_before_weight_update = replace_q_target
      
    	# (QNN en línea): Esta QNN actúa como un asesor inmediato, indicando qué acción tomar en una situación dada,
        # de manera similar a un consejero. Es el tomador de decisiones "aquí y ahora".

        self.online_q_func = QNN(DEFAULT_DISCRETE_STATES, DEFAULT_DISCRETE_ACTIONS).to(device)

        weights_init_(self.online_q_func)

        # (QNN objetivo): Esta QNN actúa como un mecanismo de doble verificación, verificando las decisiones tomadas 
        # por la QNN en línea. Asegura que la acción elegida sea realmente la más óptima. Es como tener un consejero 
        # que verifica el consejo del primer amigo.

        self.target_q_func = QNN(DEFAULT_DISCRETE_STATES, DEFAULT_DISCRETE_ACTIONS).to(device)
        self.target_q_func.load_state_dict(self.online_q_func.state_dict())

        # DDQN proporciona a al agente un dúo de consejeros: uno sugiere acciones y el otro las valida. 
        # Este enfoque minimiza errores y acelera el proceso de aprendizaje, lo que resulta en un tomador de decisiones 
        # más competente capaz de manejar situaciones complejas. Pero lo que distingue a DDQN de los DQNs está en la ingeniosa
        #  alteración en la ecuación de Bellman. 
        
        # En DDQN, la ecuación para actualizar los valores Q implica tanto a las redes en línea como a las redes objetivo, 
        # diferenciando entre la selección de acciones y la evaluación de acciones. Esta desvinculación reduce el sesgo de 
        # sobreestimación, convirtiendo a DDQN en una herramienta potente en la toma de decisiones dinámicas.

        self.optimizer = optim.Adam(self.online_q_func.parameters(), lr=learning_rate)

        self.td = RunningAverage(constants.DEFAULT_AVG_SLIDING_WINDOW_SIZE)

        self.steps = 0


    # Método Act que toma una acción basada en el estado actual del entorno, utilizando la política ε-greedy para la exploración.
    def Act(self, state, evaluate = False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        with torch.no_grad():
            action = self.online_q_func(state)  
            action = torch.argmax(action, dim = 1).squeeze()

        if (evaluate):
            # Exploiting: return an action from network
            return action.cpu().data.numpy()
        else:   
            r = np.random.uniform(0.0, 1.0)

            if r > self.exploration_rate: 
                # Exploring: return an action from network
                return action.cpu().data.numpy()
            else:
                # Exploring: return a random action
                return np.random.choice([i for i in range(DEFAULT_DISCRETE_ACTIONS)]) 



    # Método reduce_epsilon que reduce gradualmente la tasa de exploración epsilon del agente a medida que avanza en el entrenamiento.
    def reduce_epsilon(self):
        if self.exploration_rate > self.exploration_rate_minimum:
            self.exploration_rate = self.exploration_rate * self.exploration_rate_decay  
        else:
            self.exploration_rate = self.exploration_rate_minimum  

       
    # Método StepAndUpdate que realiza un paso de tiempo en el entorno y actualiza el modelo de red neuronal si es necesario, utilizando 
    # la memoria de repetición buffer.
    def StepAndUpdate(self, buffer : ExperienceReplayMemory):
        self.steps += 1     
        # Learn every UPDATE_EVERY time steps.
        if self.steps > 0 and self.steps % XQN_UPDATE_EVERY == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(buffer) > buffer.batch_size:
                experiences = buffer.get_random_sample()
                self.__learn(experiences)

                
    def getStats(self, episode):
        return {'Episode': episode,
                'TD': self.td.Mean()}
        
    def getStatNames(self):
        return ['TD']

    def getName(self):
        match self.method:
            case Method.DQN:
                return "Deep Q-Network (DQN)"
            case Method.DDQN:    
                return "Double Deep Q-Network (DDQN)"
            case _:
                raise Exception("Invalid method")
            
    # Método privado __learn que realiza la actualización del modelo de red neuronal utilizando la experiencia almacenada en experience, 
    # implementando el algoritmo específico (DQN o DDQN).
    
    # Este método __learn encapsula el proceso de aprendizaje del agente en el entorno de aprendizaje por refuerzo, utilizando una 
    # implementación específica según el método seleccionado (DQN o DDQN), actualizando los modelos y ajustando los hiperparámetros 
    # del agente en cada iteración de entrenamiento.

    def __learn(self, experience):
        
        self.set_train()

        # Este método privado __learn recibe una experiencia que consiste en tuplas de estados, acciones, recompensas, nuevos estados y 
        # banderas de terminación (dones). Estos datos son desempaquetados de la experiencia pasada al método.

        (states, actions, rewards, new_states, dones) = experience

        # En el algoritmo original de Double Q-learning, se aprenden dos funciones de valor asignando aleatoriamente 
        # cada experiencia para actualizar una de las dos funciones de valor, de modo que existan dos conjuntos de pesos, θ y θ₀. 
        # Para cada actualización, se utiliza un conjunto de pesos para determinar la política codiciosa y el otro para 
        # determinar su valor.


        # Expected calcula los valores Q esperados para las acciones elegidas en los estados actuales. Utiliza la red neuronal 
        # q_func_online para calcular los valores Q y gather para seleccionar los valores correspondientes a las acciones realizadas.

        current_Q = self.online_q_func(states).gather(1, actions.long()) 

        # maneja diferentes métodos de aprendizaje: DQN y DDQN. Dependiendo del método seleccionado, se calculan los valores Q 
        # maxQ usando la red neuronal objetivo q_func_target y se calcula el resultado esperado utilizando la ecuación de Bellman 
        # ajustada (compute_bellman_equation). 

        # En el caso de DDQN, se utilizan dos redes neuronales (online y target) para desacoplar la selección de acciones y la 
        # evaluación de las acciones, reduciendo así el sesgo de sobreestimación.

        match self.method:        
            case Method.DQN:
                with torch.no_grad():

                    # Importante: es el máximo valor
                    target_Q = self.target_q_func(new_states)
                    target_Q = torch.max(target_Q, dim = 1)[0].unsqueeze(1)             
                    target_Q = self.compute_bellman_equation(rewards, target_Q, dones)

            case Method.DDQN:    
                # Calcula los q_values para el próximo estado que usaremos para crear la pérdida y entrenar la red Q. 
                # Double DQN utiliza el índice local para seleccionar la acción con el q_value máximo y luego utiliza 
                # la red objetivo para calcular el q_value. La razón detrás de esto es ayudar a prevenir que la red 
                # sobreestime los valores q.

                with torch.no_grad():
                
                    # Importante: es índice del máximo valor
                    next_action_idx = self.online_q_func(new_states)
                    next_action_idx = torch.argmax(next_action_idx, dim = 1).unsqueeze(1)

                    # De la misma forma que el valor actual:
                    target_Q = self.target_q_func(new_states).gather(1, next_action_idx) 
                    target_Q = self.compute_bellman_equation(rewards, target_Q, dones) 
            case _:
                raise Exception("Invalid method")
            
        # calcula la pérdida utilizando el error cuadrático medio (mse_loss) entre los valores Q esperados (expected) y los valores 
        # Q obtenidos (obtained).

        loss = F.mse_loss(current_Q, target_Q)

        self.td.Add(loss)

        # Actualiza los parámetros del modelo utilizando el optimizador optimizer con la pérdida calculada.
        self.update_params(self.optimizer, loss)
                
        # Si el número de pasos (steps) es múltiplo de number_of_steps_before_weight_update, se realiza una actualización suave (soft_update)
        # de los pesos de la red neuronal objetivo (q_func_target) utilizando los pesos de la red neuronal en línea (q_func_online) con un factor tau.

        if (self.method == Method.DDQN):
            if self.steps % self.number_of_steps_before_weight_update == 0: 
                self.soft_update(self.target_q_func, self.online_q_func, self.tau)
        else:
            self.hard_update(self.target_q_func, self.online_q_func)
                
        # Reduce la tasa de exploración (epsilon) del agente, promoviendo una mayor explotación de conocimientos aprendidos y menos exploración aleatoria.
        self.reduce_epsilon()


        
    # Método save_model que guarda los pesos del modelo de red neuronal en un archivo especificado por path.
    def save_model(self, episode, metrics_by_episode, filename):
        torch.save({
            'episode': episode,
            'epsilon': self.exploration_rate,
            'steps': self.steps,
            'online_q_func_state_dict': self.online_q_func.state_dict(),
            'target_q_func_state_dict': self.target_q_func.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics_by_episode' : metrics_by_episode
            }, filename)
        print("CHKPNT: {0}".format(filename))

    # Método load_saved_model que carga los pesos de un modelo de red neuronal previamente guardado desde un archivo 
    # especificado por path.
    def load_saved_model(self, episode, filename):

        checkpoint = torch.load(filename)      
        if (checkpoint['episode'] == episode):
            print("Cargando archivo {0} con episodio {1}".format(filename, episode))
            self.steps = checkpoint['steps']
            self.exploration_rate = checkpoint['epsilon']
            self.online_q_func.load_state_dict(checkpoint['online_q_func_state_dict'], False)
            self.target_q_func.load_state_dict(checkpoint['target_q_func_state_dict'], False)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.set_eval()
            return checkpoint['metrics_by_episode']
        else:
            raise RuntimeError("Checkpoint {0} is invalid".format(episode))

    def set_eval(self):
        self.online_q_func.eval()
        self.target_q_func.eval()
    
    def set_train(self):
        self.online_q_func.train()
        self.target_q_func.train()

    def IsContinuosEnvExpected(self):
        return False
    
    def IsPriorCollectingForExperienceRolloutRequired(self):
        return False
    
# ----------------------------------------------------------------------------------------------------------------------------------
# END OF FILE
# ----------------------------------------------------------------------------------------------------------------------------------