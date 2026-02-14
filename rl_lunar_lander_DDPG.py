# ----------------------------------------------------------------------------------------------------------------------------------
# Archivo: rl_lunar_lander_DDPG.py
# ----------------------------------------------------------------------------------------------------------------------------------
# Descripción: Deep Deterministic Policy Gradients (DDPG)
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
from rl_lunar_lander_generic import Agent
from rl_lunar_lander_generic import RunningAverage
from rl_lunar_lander_generic import weights_init_
from rl_lunar_lander_ERM import ExperienceReplayMemory
from rl_lunar_lander_constants import DEFAULT_AVG_SLIDING_WINDOW_SIZE

# ----------------------------------------------------------------------------------------------------------------------------------
# Definición de la red neuronal para el actor en DDPG
# ----------------------------------------------------------------------------------------------------------------------------------
class DDPG_ActorNet(nn.Module):
    def __init__(self, num_of_actions = 2, num_of_states = 8, max_action = 1.0, hidden_dim_low = 256, hidden_dim_high = 256):
        super().__init__()
        self.fc1 = nn.Linear(num_of_states, hidden_dim_low)
        self.fc2 = nn.Linear(hidden_dim_low, hidden_dim_high)
        self.output = nn.Linear(hidden_dim_high, num_of_actions)
        self.max_action = max_action
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.output(x))
        return self.max_action * x
# ----------------------------------------------------------------------------------------------------------------------------------
# Definición de la red neuronal para el crítico en DDPG
# ----------------------------------------------------------------------------------------------------------------------------------
class DDPG_CriticNet(nn.Module):
    def __init__(self, num_of_actions = 2, num_of_states = 8, hidden_dim_low = 256, hidden_dim_high = 256):
        super().__init__()
        self.fc_1 = nn.Linear(num_of_states + num_of_actions, hidden_dim_low)
        self.fc_2 = nn.Linear(hidden_dim_low, hidden_dim_high)
        self.output = nn.Linear(hidden_dim_high, 1)
        self.apply(weights_init_)

    def forward(self, next_state, next_action): 
        x = torch.cat([next_state, next_action], 1)
        y = F.relu(self.fc_1(x))
        y = F.relu(self.fc_2(y))
        y = self.output(y)
        return y

# ----------------------------------------------------------------------------------------------------------------------------------
# Definición del agente DDPG que hereda de la clase base 'Agent'
# ----------------------------------------------------------------------------------------------------------------------------------
class DDPG(Agent):
    def __init__(self, num_of_actions = 2, num_of_states = 8, max_action = 1.0, gamma = 0.99, tau = 0.005, actor_lr = 1e-4, critic_lr = 1e-3,
                  lmbda = 0.8, policy_noise = 0.2, noise_clip = 0.5, policy_freq = 2, device = None):
        super(DDPG, self).__init__()
        # Parámetros y configuración del algoritmo DDPG
        self.gamma = gamma
        self.lmbda = lmbda
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.num_of_actions = num_of_actions
        self.max_action = max_action
        self.device = device

        # Red neuronal del actor (online y target)
        self.online_actor = DDPG_ActorNet(num_of_actions, num_of_states).to(device)
        self.target_actor = DDPG_ActorNet(num_of_actions, num_of_states).to(device)
        self.target_actor.load_state_dict(self.online_actor.state_dict())

        # Red neuronal del crítico (online y target)
              
        self.online_critic = DDPG_CriticNet(num_of_actions, num_of_states).to(device) 
        self.target_critic = DDPG_CriticNet(num_of_actions, num_of_states).to(device) 
        self.target_critic.load_state_dict(self.online_critic.state_dict())
        
        # Congelar las redes target respecto a los optimizadores
        self.enable_gradients(self.target_actor, False)
        self.enable_gradients(self.target_critic, False)

        # Inicialización de los optimizadores para el actor y el crítico
        self.actor_optimizer = torch.optim.AdamW(self.online_actor.parameters(), lr = actor_lr)
        self.critic_optimizer = torch.optim.AdamW(self.online_critic.parameters(), lr = critic_lr)

        # Estadísticas:

        self.td_critic = RunningAverage(DEFAULT_AVG_SLIDING_WINDOW_SIZE)
        self.td_actor = RunningAverage(DEFAULT_AVG_SLIDING_WINDOW_SIZE)

        # Inicialización de contador de iteraciones
        self.steps = 0
        
        
    def getName(self):
        return "Deep Deterministic Policy Gradients (DDPG)"

    # Método para que el agente actúe basado en el estado dado
    def Act(self, state, evaluate = False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if torch.no_grad():
            if evaluate:
                # Exploit:
                x = self.online_actor(state)    
            else:
                # Evaluate:
                x = self.online_actor(state)    
                x = self.__transform_with_clipped_noise(x)
        return x.cpu().data.numpy().flatten()
    
    # Método privado para aplicar transformaciones y clip de ruido al resultado del actor
    def __transform_with_clipped_noise(self, x):
        x = x + self.noise_clip * torch.randn(self.num_of_actions, device= self.device)
        x = torch.clip(x, -self.max_action, self.max_action)
        return x

    # Método para realizar un paso en el entorno y actualizar el agente
    def StepAndUpdate(self, buffer : ExperienceReplayMemory):  
        if len(buffer) > buffer.batch_size:
            (state, action, reward, next_state, done)  =  buffer.get_random_sample()
            self.__learn(state, action, reward, next_state, done)
     
            
    def getStats(self, episode):
        return {'Episode': episode,
                'TD_Critic': self.td_critic.Mean(), 
                'TD_Actor': self.td_actor.Mean()}

    def getStatNames(self):
        return ['TD_Critic', 'TD_Actor']

    # Método privado para realizar el aprendizaje en el agente DDPG
    def __learn(self, state, action, reward, next_state, done):

        self.set_train()

        with torch.no_grad():
            # Selección de acción según la política y adición de ruido (clipped noise)
            next_action = self.target_actor(next_state)
            next_action = self.__transform_with_clipped_noise(next_action)

            # Cálculo del valor Q objetivo
            target_Q = self.target_critic(next_state, next_action) 
            target_Q = self.compute_bellman_equation(reward, target_Q, done)

        # Obtención de las estimaciones Q actuales
        current_Q = self.online_critic(state, action)

        # Cálculo de la pérdida del crítico
        critic_loss = F.mse_loss(current_Q, target_Q) 
       
        # Optimización del crítico
        self.update_params(self.critic_optimizer, critic_loss)
      
        self.td_critic.Add(critic_loss)

        if self.steps > 0:

            # Congelar las redes Q para no desperdiciar esfuerzo computacional          
            self.enable_gradients(self.online_critic, False)

            # Cálculo de la pérdida del actor
            actor_loss = -self.online_critic(state, self.online_actor(state)).mean() 

            # Optimización del actor
            self.update_params(self.actor_optimizer, actor_loss)

            # Descongelar las redes Q para poder optimizarlas en el siguiente paso DDPG
            self.enable_gradients(self.online_critic, True)

            # Actualización de los modelos target con suavizado
            self.soft_update(self.target_critic, self.online_critic, self.tau)
            self.soft_update(self.target_actor, self.online_actor, self.tau)

            self.td_actor.Add(actor_loss)

        self.steps += 1     

   

    # Método save_model que guarda los pesos del modelo de red neuronal en un archivo especificado por path.
    def save_model(self, episode, metrics_by_episode, filename):
        torch.save({
            'episode': episode,
            'steps': self.steps,
            'online_actor_state_dict': self.online_actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'online_critic_state_dict': self.online_critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'optimizer_actor_state_dict': self.actor_optimizer.state_dict(),
            'optimizer_critic_state_dict': self.critic_optimizer.state_dict(),
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
            self.online_actor.load_state_dict(checkpoint['online_actor_state_dict'])
            self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
            self.online_critic.load_state_dict(checkpoint['online_critic_state_dict'])
            self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['optimizer_actor_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['optimizer_critic_state_dict'])
            self.set_eval()
             # Congelar las redes target respecto a los optimizadores
            self.enable_gradients(self.target_actor, False)
            self.enable_gradients(self.target_critic, False)
            return checkpoint['metrics_by_episode']
        else:
            raise RuntimeError("Checkpoint {0} is invalid".format(episode))
        
    def set_eval(self):
        self.online_actor.eval()
        self.target_actor.eval()
        self.online_critic.eval()
        self.target_critic.eval()
    
    def set_train(self):
        self.online_actor.train()
        self.target_actor.train()
        self.online_critic.train()
        self.target_critic.train()

    def IsContinuosEnvExpected(self):
        return True
    
    def IsPriorCollectingForExperienceRolloutRequired(self):
        return True

# ----------------------------------------------------------------------------------------------------------------------------------
# END OF FILE
# ----------------------------------------------------------------------------------------------------------------------------------
