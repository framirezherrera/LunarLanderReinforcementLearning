# ----------------------------------------------------------------------------------------------------------------------------------
# Archivo: rl_lunar_lander_TD3.py
# ----------------------------------------------------------------------------------------------------------------------------------
# Descripción: Twin Delayed Deep Deterministic Policy Gradients (TD3)
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
# Define la red neuronal QNetwork que evalúa la política para el algoritmo TD3. 
# ----------------------------------------------------------------------------------------------------------------------------------
class TD3_ActorNet(nn.Module):

    def __init__(self, num_of_actions = 2, num_of_states = 8, max_action = 1.0, hidden_dim_low = 256, hidden_dim_high = 256):
        super().__init__()
        # Capa completamente conectada 1: estado a dimensión oculta baja
        self.fc1 = nn.Linear(num_of_states, hidden_dim_low)
        # Capa completamente conectada 2: dimensión oculta baja a alta
        self.fc2 = nn.Linear(hidden_dim_low, hidden_dim_high)
         # Capa de salida: dimensión oculta alta a acciones
        self.output = nn.Linear(hidden_dim_high, num_of_actions)
        # Valor máximo de la acción permitida
        self.max_action = max_action

    def forward(self, state):
        # Activación ReLU en la primera capa oculta
        x = F.relu(self.fc1(state))
        # Activación ReLU en la segunda capa oculta
        x = F.relu(self.fc2(x))
        # Activación tanh en la capa de salida, escalada por max_action
        x = F.tanh(self.output(x))
        # Devuelve las acciones escaladas
        return self.max_action * x

# ----------------------------------------------------------------------------------------------------------------------------------
# Define la red neuronal QNetwork que evalúa la función Q para el algoritmo TD3. 
# ----------------------------------------------------------------------------------------------------------------------------------
class TD3_CriticNet(nn.Module):

    def __init__(self, num_of_actions = 2, num_of_states = 8, hidden_dim_low = 256, hidden_dim_high = 256):
        super().__init__()
        # Primer red neuronal para evaluar Q1
        self.fc_a_1 = nn.Linear(num_of_states + num_of_actions, hidden_dim_low)
        self.fc_a_2 = nn.Linear(hidden_dim_low, hidden_dim_high)
        self.output_a = nn.Linear(hidden_dim_high, 1)
        # Segunda red neuronal para evaluar Q2
        self.fc_b_1 = nn.Linear(num_of_states + num_of_actions, hidden_dim_low)
        self.fc_b_2 = nn.Linear(hidden_dim_low, hidden_dim_high)
        self.output_b = nn.Linear(hidden_dim_high, 1)

    def forward(self, next_state, next_action):
        # Concatenación del estado siguiente y la acción siguiente
        x = torch.cat([next_state, next_action], 1)
        # Computación de Q1
        q1 = F.relu(self.fc_a_1(x))
        q1 = F.relu(self.fc_a_2(q1))
        q1 = self.output_a(q1)
        # Computación de Q2
        q2 = F.relu(self.fc_b_1(x))
        q2 = F.relu(self.fc_b_2(q2))
        q2 = self.output_b(q2)
        # Devuelve Q1 y Q2
        return q1, q2

    def Q1(self, next_state, next_action):
        # Concatenación del estado siguiente y la acción siguiente
        q = torch.cat([next_state, next_action], 1)
        # Computación de Q1
        q = F.relu(self.fc_a_1(q))
        q = F.relu(self.fc_a_2(q))
        q = self.output_a(q)
        # Devuelve Q1
        return q
# ----------------------------------------------------------------------------------------------------------------------------------
# Implementación del agente
# ----------------------------------------------------------------------------------------------------------------------------------
class TD3(Agent):

    def __init__(self, num_of_actions = 2, num_of_states = 8, max_action = 1.0, gamma = 0.99, tau = 0.005, actor_lr = 1e-4, critic_lr = 1e-3, lmbda = 0.8, policy_noise = 0.2, noise_clip = 0.5, policy_freq = 2, device = None):
        super(TD3, self).__init__()

        # Parámetros de TD3
        self.gamma = gamma
        self.lmbda = lmbda
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.num_of_actions = num_of_actions
        self.max_action = max_action
        self.device = device

        # Redes actor y crítico en línea
        self.online_actor = TD3_ActorNet(num_of_actions, num_of_states).to(device)
        self.online_critic = TD3_CriticNet(num_of_actions, num_of_states).to(device) 

        # Inicialización de pesos
        weights_init_(self.online_actor)
        weights_init_(self.online_critic)

        # Redes actor y crítico objetivo
        self.target_actor = TD3_ActorNet(num_of_actions, num_of_states).to(device)
        self.target_actor.load_state_dict(self.online_actor.state_dict())
        
        self.target_critic = TD3_CriticNet(num_of_actions, num_of_states).to(device) 
        self.target_critic.load_state_dict(self.online_critic.state_dict())
        
        # Congelar las redes objetivo con respecto a los optimizadores
        self.enable_gradients(self.target_actor, False)
        self.enable_gradients(self.target_critic, False)

        # Optimizadores para el actor y el crítico
        self.actor_optimizer = torch.optim.AdamW(self.online_actor.parameters(), lr = actor_lr)
        self.critic_optimizer = torch.optim.AdamW(self.online_critic.parameters(), lr = critic_lr)
        
        # Estadísticas:

        self.td_critic = RunningAverage(DEFAULT_AVG_SLIDING_WINDOW_SIZE)
        self.td_actor = RunningAverage(DEFAULT_AVG_SLIDING_WINDOW_SIZE)

        # Inicialización del contador de iteraciones
        self.steps = 0
        
        
    def getName(self):
        return "Twin Delayed Deep Deterministic Policy Gradients (TD3)"


    def Act(self, state, evaluate = False):
        # Convertir estado a tensor y enviar a dispositivo
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        # Evaluar la red actor en línea
        if torch.no_grad():
            if evaluate:
                # Exploit:
                x = self.online_actor(state)    
            else:
                # Explore:
                x = self.online_actor(state)    
                x = self.__transform_with_clipped_noise(x)
                
        # Devolver las acciones como arreglo numpy
        return x.cpu().data.numpy().flatten()
    

    def __transform_with_clipped_noise(self, x):
        # Agregar ruido a las acciones y aplicar clipping
        x = x + self.noise_clip * torch.randn(self.num_of_actions, device= self.device)
        x = torch.clip(x, -self.max_action, self.max_action)
        return x


    def StepAndUpdate(self, buffer : ExperienceReplayMemory): 
        # Paso de aprendizaje y actualización del agente 
        if len(buffer) > buffer.batch_size:
            (state, action, reward, next_state, done)  =  buffer.get_random_sample()
            self.__learn(state, action, reward, next_state, done)


    def getStats(self, episode):
        return {'Episode': episode,
                'TD_Critic': self.td_critic.Mean(), 
                'TD_Actor': self.td_actor.Mean()}
        
    def getStatNames(self):
        return ['TD_Critic', 'TD_Actor']
    
    def __learn(self, state, action, reward, next_state, done):
       
        self.set_train()

        # Incrementar el contador de iteraciones
        self.steps += 1
        
        with torch.no_grad():
            # Seleccionar acción según la política y añadir ruido (clipped noise)
            next_action = self.target_actor(next_state)
            next_action = self.__transform_with_clipped_noise(next_action)
            
            # Calcular el valor Q objetivo
            target_Q1, target_Q2 = self.target_critic(next_state, next_action) 
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = self.compute_bellman_equation(reward, target_Q, done)

        # Obtener estimaciones Q actuales
        current_Q1, current_Q2 = self.online_critic(state, action)

        # Calcular la pérdida del crítico
        critic_loss_of_q1 = F.mse_loss(current_Q1, target_Q) 
        critic_loss_of_q2 = F.mse_loss(current_Q2, target_Q)
        critic_loss = critic_loss_of_q1 + critic_loss_of_q2

        # Optimizar el crítico
        self.update_params(self.critic_optimizer, critic_loss)

        self.td_critic.Add(critic_loss)

        # Actualizaciones de política demoradas
        if self.steps % self.policy_freq == 0:

            # Congelar las redes Q para no desperdiciar esfuerzo computacional
            # calculando gradientes para ellas durante el paso de aprendizaje de política.
            
            self.enable_gradients(self.online_critic, False)

            # Calcular la pérdida del actor
            actor_loss = -self.online_critic.Q1(state, self.online_actor(state)).mean() 
            
            # Optimizar el actor
            self.update_params(self.actor_optimizer, actor_loss)
            
            self.td_actor.Add(actor_loss)

            # Descongelar las redes Q para optimizarlas en el próximo paso.
            self.enable_gradients(self.online_critic, True)

            # Actualizar los modelos objetivo con un suavizado
            self.soft_update(self.target_critic, self.online_critic, self.tau)
            self.soft_update(self.target_actor, self.online_actor, self.tau)

    

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