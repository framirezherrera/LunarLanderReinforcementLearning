# ----------------------------------------------------------------------------------------------------------------------------------
# Archivo: rl_lunar_lander_SAC.py
# ----------------------------------------------------------------------------------------------------------------------------------
# Descripción: Soft Actor-Critic (SAC)
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
from torch.optim import Adam
from torch.distributions import Normal
from rl_lunar_lander_generic import Agent
from rl_lunar_lander_generic import RunningAverage
from rl_lunar_lander_generic import weights_init_
from rl_lunar_lander_ERM import ExperienceReplayMemory
from rl_lunar_lander_constants import DEFAULT_AVG_SLIDING_WINDOW_SIZE
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------------------
# Define la red neuronal QNetwork que evalúa la función Q para el algoritmo SAC. 
# Consiste en dos capas ocultas (q_fc1 y q_fc2) seguidas por una capa de salida (q_output). 
# Se inicializa utilizando una función de inicialización de pesos weights_init_.
# ----------------------------------------------------------------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim = 256):
        super(QNetwork, self).__init__()
        self.q_fc1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.q_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_output = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, x):
        a = F.relu(self.q_fc1(x))
        a = F.relu(self.q_fc2(a))
        a = self.q_output(a)
        return a

# ----------------------------------------------------------------------------------------------------------------------------------
# Define la red TwinnedQNetwork que utiliza dos funciones Q (Q1 y Q2) para mitigar el 
# sesgo positivo en el paso de mejora de la política, conocido por degradar el rendimiento 
# de los métodos basados en el valor.
# ----------------------------------------------------------------------------------------------------------------------------------

class TwinnedQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim = 256):
        super(TwinnedQNetwork, self).__init__()
        # Q1 architecture
        self.Q1 = QNetwork(num_inputs, num_actions, hidden_dim)
        # Q2 architecture
        self.Q2 = QNetwork(num_inputs, num_actions, hidden_dim)

    # Calcula y devuelve las estimaciones Q de ambas redes Q (Q1 y Q2).    
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        a = self.Q1.forward(x)
        b = self.Q2.forward(x)
        return a, b
# ----------------------------------------------------------------------------------------------------------------------------------
# Define constantes para los valores máximo y mínimo de log_std, utilizados en la política para 
# controlar la desviación estándar del modelo de distribución normal.
# ----------------------------------------------------------------------------------------------------------------------------------
LOG_STD_MAX = 2
LOG_STD_MIN = -20 

# ----------------------------------------------------------------------------------------------------------------------------------
# Define la red Policy, que modela la política del agente. Consiste en dos capas ocultas (fc1 y fc2), 
# seguidas por capas de salida (output_mean_of y output_log_std) para las medias y los logaritmos de 
# las desviaciones estándar de la distribución normal de las acciones.
# ----------------------------------------------------------------------------------------------------------------------------------
class Policy(nn.Module):

    def __init__(self, num_inputs=8, num_actions = 2, hidden_dim = 256, device = None):
        super(Policy, self).__init__()      
        self.device = device
        self.action_size = num_actions
        self.fc1 = nn.Linear(num_inputs, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_mean_of = nn.Linear(hidden_dim, num_actions)
        self.output_log_std = nn.Linear(hidden_dim, num_actions)
        self.apply(weights_init_)

    # Calcula las medias y los logaritmos de las desviaciones estándar de la distribución normal de 
    # las acciones 

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.output_mean_of(x)
        log_std = self.output_log_std(x)
        log_std = torch.clip(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        return mean, log_std

    # Genera muestras de acciones usando el truco de reparametrización y calcula el logaritmo de la 
    # probabilidad de esas acciones

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1.0 - action.square() + epsilon)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    # Define métodos para explorar (explore) y explotar (exploit) acciones basadas en la política. 
    
    # Explore retorna acciones y logaritmos de probabilidad de exploración (training).
    def explore(self, state):
        action, logprob, _ = self.sample(state)
        return action, logprob
    
    # Exploit retorna acciones para explotación (evaluación).
    def exploit(self, state):
        _, _, action = self.sample(state)
        return action
# ----------------------------------------------------------------------------------------------------------------------------------
# Implementación del agente SAC
# ----------------------------------------------------------------------------------------------------------------------------------
class SAC(Agent):

    def __init__(self, num_of_states = 8, num_of_actions = 2,  max_action = 1.0,  lr=3e-4, gamma=0.99, polyak = 0.995,  noise_clip = 0.5, target_update_interval = 1, device = None):
        super().__init__()

        self.num_of_states = num_of_states
        self.num_of_actions = num_of_actions
        self.max_action = max_action
        self.device = device        
        self.gamma = gamma
        self.polyak = polyak
        self.noise_clip = noise_clip
        self.target_update_interval = target_update_interval
        self.policy = Policy(num_of_states, num_of_actions, device=self.device).to(self.device)

        self.online_critic = TwinnedQNetwork(num_of_states, num_of_actions).to(device=self.device)
        self.target_critic = TwinnedQNetwork(num_of_states, num_of_actions).to(device=self.device)

        # Copy parameters of the learning network to the target network.
        self.target_critic.load_state_dict(self.online_critic.state_dict())

        # Disable gradient calculations of the target network.
        self.enable_gradients(self.target_critic, False)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.critic_optim = Adam(self.online_critic.parameters(), lr=lr)

        # Coeficiente de entropía
        # https://arxiv.org/abs/1812.05905

        # Originalmente es -log(1/|A|) * ratio (= maximum entropy * ratio).
        # self.target_entropy = torch.tensor(1.0 / num_of_actions, dtype=torch.float, device=device)
        # self.target_entropy = (target_entropy_ratio * -torch.log(self.target_entropy)).item()

        self.target_entropy = float(-np.prod(num_of_actions).astype(np.float32))    
        self.log_alpha = torch.log(torch.ones(1, device=self.device)).requires_grad_(True)
        self.alpha = torch.exp(self.log_alpha.detach())
        self.alpha_optim = Adam([self.log_alpha], lr=lr)
      
        # Estadísticas:

        self.td_Q1 = RunningAverage(DEFAULT_AVG_SLIDING_WINDOW_SIZE)
        self.td_Q2 = RunningAverage(DEFAULT_AVG_SLIDING_WINDOW_SIZE)
        self.td_critic = RunningAverage(DEFAULT_AVG_SLIDING_WINDOW_SIZE)
        self.td_policy = RunningAverage(DEFAULT_AVG_SLIDING_WINDOW_SIZE)
        self.td_entropy = RunningAverage(DEFAULT_AVG_SLIDING_WINDOW_SIZE)

        self.steps_before_update = 1
        self.steps = 0

        
    # Define el método Act para la clase SAC, que utiliza la política para seleccionar acciones basadas
    # en el estado dado, con opción para explorar o explotar dependiendo del modo de evaluación.

    def Act(self, state, evaluate = False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        if torch.no_grad():
            if evaluate:
                # Exploit: Act without randomness.        
                action = self.policy.exploit(state) # Exploit
            else:
                # Explore: Act with randomness.
                action, _ = self.policy.explore(state) # Explore
                action = self.__transform_with_clipped_noise(action)
        return action.cpu().data.numpy().flatten()

    def __transform_with_clipped_noise(self, x):
        x = x + self.noise_clip * torch.randn(self.num_of_actions, device= self.device)
        x = torch.clip(x, -self.max_action, self.max_action)
        return x

    # Define StepAndUpdate para la clase SAC, que toma muestras del búfer de repetición de experiencia y 
    # aprende de ellas utilizando el método __learn.

    def StepAndUpdate(self, buffer : ExperienceReplayMemory):
        
        # If enough samples are available in memory, get random subset and learn
        if len(buffer) > buffer.batch_size:
            (state, action, reward, next_state, done) = buffer.get_random_sample()
            self.__learn(state, action, reward, next_state, done)   

            
    def getStats(self, episode):
        return {'Episode': episode,
                'TD_Q1': self.td_Q1.Mean(), 
                'TD_Q2': self.td_Q2.Mean(), 
                'TD_Critic': self.td_critic.Mean(), 
                'TD_Policy': self.td_policy.Mean(), 
                'TD_Entropy': self.td_entropy.Mean()}


    def getStatNames(self):
        return ['TD_Q1','TD_Q2', 'TD_Critic', 'TD_Policy', 'TD_Entropy' ]

    #  Este método realiza la actualización de los modelos de red neuronal que representan 
    #  la política y las funciones Q del agente SAC, utilizando la información obtenida de 
    #  la memoria de repetición de experiencias 

    def __learn(self, states, actions, rewards, next_states, dones):

        self.set_train()

        # Calcula las estimaciones Q actuales (online_Q1 y online_Q2) utilizando las redes 
        # críticas en línea (online_critic) con los estados y acciones actuales.

        online_Q1, online_Q2 = self.online_critic(states, actions)      

        with torch.no_grad(): 

            # Este bloque asegura que las operaciones dentro de él no calcularán gradientes, ya 
            # que solo necesitamos las estimaciones de los valores Q sin afectar el grafo 
            # computacional para el retroceso automático.

            # Utiliza la política para explorar acciones (next_actions) y calcular los logaritmos 
            # de las probabilidades de las acciones exploradas (next_logprobs) para los estados 
            # siguientes (next_states).

            next_actions, next_logprobs = self.policy.explore(next_states)           

            next_actions = self.__transform_with_clipped_noise(next_actions)

            # Calcula las estimaciones Q objetivo (target_Q1 y target_Q2) utilizando las redes críticas 
            # objetivo (target_critic) con los estados siguientes (next_states) y las acciones obtenidas
            # de la política (next_actions).
            
            target_Q1, target_Q2 = self.target_critic(next_states, next_actions)

            # Calcula el mínimo entre las estimaciones Q objetivo (target_Q1 y target_Q2), lo cual 
            # es utilizado en el cálculo de la función objetivo.

            min_target_Q = torch.min(target_Q1, target_Q2)  

            # Aplica el término de entropía negativa (multiplicado por self.alpha) a min_target_Q, para 
            # fomentar la exploración del agente.
            min_target_Q = min_target_Q - self.alpha * next_logprobs
            
            #  Calcula la ecuación de Bellman para actualizar las estimaciones Q objetivo (target_Q) basadas 
            #  en las recompensas recibidas, min_target_Q y si el episodio terminó (dones).

            target_Q =  self.compute_bellman_equation(rewards, min_target_Q, dones) # Verificado contra algoritmo original
    
        #  Calcula las pérdidas de error cuadrático medio (MSE) entre las estimaciones Q actuales (online_Q1 y online_Q2) y 
        #  las estimaciones Q objetivo (target_Q) (error TD o error de Bellman).

        Q1_loss = F.mse_loss(online_Q1, target_Q)
        Q2_loss = F.mse_loss(online_Q2, target_Q)

        # Calcula la pérdida crítica total como el promedio de las pérdidas de los dos críticos Q (online_Q1 y online_Q2)
        # Respeta la versión original del autor, aunque otras implementaciones no consideran la multiplicación por 0.5.

        critic_loss = 0.5 * (Q1_loss + Q2_loss)

        #  Actualiza los parámetros de los optimizadores del crítico (critic_optim) utilizando la pérdida crítica calculada
        self.update_params(self.critic_optim, critic_loss)

        self.td_Q1.Add(Q1_loss)
        self.td_Q2.Add(Q2_loss)
        self.td_critic.Add(critic_loss)

        # Verifica si ya se han realizado algunos pasos de entrenamiento antes de continuar con la actualización 
        # de la política y el coeficiente de entropía.

        if self.steps > 0:
            
            # Desactiva el cálculo de gradientes para la red crítica en línea (online_critic), ya que no se actualizará 
            # en este momento.

            self.enable_gradients(self.online_critic, False)

            # Calcula las acciones (policy_actions) y los logaritmos de las probabilidades de esas acciones (policy_logprobs) 
            # para los estados actuales (states) utilizando la política para la exploración.

            policy_actions, policy_logprobs = self.policy.explore(states)

            # Calcula las estimaciones Q (policy_Q1 y policy_Q2) utilizando la red crítica en línea (online_critic) para las 
            # acciones generadas por la política

            policy_Q1, policy_Q2 = self.online_critic(states, policy_actions)
            
            # Calcula el mínimo entre las estimaciones Q (policy_Q1 y policy_Q2), que se utiliza en el cálculo de la pérdida 
            # de la política.

            min_policy_Q = torch.min(policy_Q1, policy_Q2) # Diferente del algoritmo original (usa mean)

            #  Calcula la pérdida de la política (policy_loss), que es el promedio ponderado de los logaritmos de las probabilidades 
            # de la política (policy_logprobs) y el mínimo de las estimaciones Q (min_policy_Q).

            policy_loss = torch.mean(self.alpha * policy_logprobs - min_policy_Q) # Verificado contra algoritmo original

            # Actualiza los parámetros del optimizador de la política (policy_optim) utilizando la pérdida de la política calculada
                        
            self.update_params(self.policy_optim, policy_loss)    

            self.td_policy.Add(policy_loss)

            # Actualiza el valor del coeficiente de entropía (alpha) basado en el logaritmo exponencial (log_alpha) después de 
            # desengancharlo del grafo computacional.

            self.alpha = torch.exp(self.log_alpha.detach())

            # Calcula la pérdida del coeficiente de entropía (alpha_loss), que es el negativo del producto del logaritmo del coeficiente de 
            # entropía (log_alpha) y la suma de los logaritmos de las probabilidades de la política (policy_logprobs) más la entropía objetivo 

            alpha_loss = -1.0 * (self.log_alpha * (policy_logprobs + self.target_entropy).detach()) # Verificado contra algoritmo original
            alpha_loss = torch.mean(alpha_loss)

            # Actualiza los parámetros del optimizador del coeficiente de entropía
            self.update_params(self.alpha_optim, alpha_loss)

            self.td_entropy.Add(alpha_loss)

            #  Habilita nuevamente el cálculo de gradientes para la red crítica en línea (online_critic)
            self.enable_gradients(self.online_critic, True)

            # Verifica si es el momento de actualizar la red crítica objetivo (target_critic) utilizando la técnica de promediado polinomial 
            # (polyak_averaging_update).
            if self.steps % self.target_update_interval == 0:
                self.polyak_averaging_update(self.target_critic, self.online_critic, self.polyak)
        
        self.steps += 1



    # Método save_model que guarda los pesos del modelo de red neuronal en un archivo especificado por path.
    def save_model(self, episode, metrics_by_episode, filename):
        torch.save({
            'episode': episode,
            'log_alpha': self.log_alpha,
            'alpha': self.alpha,
            'steps': self.steps,
            'policy_state_dict': self.policy.state_dict(),
            'online_critic_state_dict': self.online_critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'optimizer_policy_state_dict': self.policy_optim.state_dict(),
            'optimizer_critic_state_dict': self.critic_optim.state_dict(),
            'optimizer_alpha_state_dict': self.alpha_optim.state_dict(),
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
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = checkpoint['alpha']
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.online_critic.load_state_dict(checkpoint['online_critic_state_dict'])
            self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['optimizer_policy_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['optimizer_critic_state_dict'])
            self.alpha_optim.load_state_dict(checkpoint['optimizer_alpha_state_dict'])
            self.set_eval()
            self.enable_gradients(self.target_critic, False)
            return checkpoint['metrics_by_episode']
        else:
            raise RuntimeError("Checkpoint {0} is invalid".format(episode))
        
    def getName(self):
        return "Soft Actor-Critic (SAC)"
    
    def set_eval(self):
        self.policy.eval()
        self.online_critic.eval()
        self.target_critic.eval()
    
    def set_train(self):
        self.policy.train()
        self.online_critic.train()
        self.target_critic.train()

    def IsContinuosEnvExpected(self):
        return True
    
    def IsPriorCollectingForExperienceRolloutRequired(self):
        return True
    
# ----------------------------------------------------------------------------------------------------------------------------------
# END OF FILE
# ----------------------------------------------------------------------------------------------------------------------------------