# ----------------------------------------------------------------------------------------------------------------------------------
# Archivo: rl_lunar_lander_ERM.py
# ----------------------------------------------------------------------------------------------------------------------------------
# Descripción: Experience Replay Memory (ERM)
# ----------------------------------------------------------------------------------------------------------------------------------
# Implementado por: Felipe Ramírez Herrera
# Curso Aprendizaje por Reforzamiento. 
# Master de Inteligencia Artificial Avanzada y Aplicada (IA3)
# Universidad de Valencia / ADEIT
# Ultima revisión: 15/06/2024 
# ----------------------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from collections import namedtuple
import random

# Este código define una clase ExperienceReplayMemory que implementa un buffer de memoria de repetición de experiencias, 
# comúnmente utilizado en algoritmos de aprendizaje por refuerzo. La clase permite almacenar transiciones de experiencia, 
# recuperar muestras aleatorias de estas transiciones y gestionar la memoria de manera eficiente.


class ExperienceReplayMemory(object):
    """
    Implementación de un buffer de memoria de repetición de experiencias para aprendizaje por refuerzo.
    
    Args:
        max_length (int): Capacidad máxima del buffer de memoria.
        batch_size (int): Tamaño del lote a muestrear para el aprendizaje.
        device (torch.device): Dispositivo en el que se almacenarán los tensores (CPU o GPU).
    """
    
    def __init__(self, max_length, batch_size, device):        
        """
        Inicializa la memoria de repetición de experiencias con la capacidad máxima y otros parámetros.
        
        Args:
            max_length (int): Capacidad máxima del buffer de memoria.
            batch_size (int): Tamaño del lote a muestrear para el aprendizaje.
            device (torch.device): Dispositivo en el que se almacenarán los tensores (CPU o GPU).
        """
        self.max_length = max_length
        self.batch_size = batch_size
        self.content = deque(maxlen=self.max_length)
        self.device = device
        
    def __getitem__(self, idx):
        """
        Obtiene un elemento del buffer de memoria en la posición especificada por índice.

        Args:
            idx (int): Índice del elemento a obtener.

        Returns:
            Elemento de transición {state, action, reward, new_state, done}.
        """
        return self.content[idx]

    def __len__(self):
        """
        Retorna la cantidad actual de elementos almacenados en la memoria de repetición.

        Returns:
            int: Cantidad actual de elementos en la memoria.
        """
        return len(self.content)
    
    def clear(self):
        """
        Limpia todos los elementos del buffer de memoria.
        """
        self.content.clear();

    def store(self, state, action, reward, next_state, terminal):
        """
        Almacena una transición de experiencia en el buffer de memoria.

        Args:
            state (array): Estado actual del entorno.
            action (int): Acción realizada en el estado actual.
            reward (float): Recompensa obtenida por la acción realizada.
            next_state (array): Nuevo estado tras realizar la acción.
            done (bool): Indica si el estado es terminal (fin de episodio).
        """
        t = {'s': state, 'a': action, 'r': reward, 't': next_state, 'd': terminal}

        self.content.append(t)

    def get_random_sample(self):
        """
        Obtiene una muestra aleatoria de transiciones de experiencia del buffer de memoria.

        Returns:
            tuple: Tensores que contienen los estados, acciones, recompensas, nuevos estados y terminales.
        """
        assert len(self.content) >= self.batch_size 

        transitions = random.sample(self.content, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e['s'] for e in transitions if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e['a'] for e in transitions if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e['r'] for e in transitions if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e['t'] for e in transitions if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e['d'] for e in transitions if e is not None]).astype(np.uint8)).float().to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def Backup(self, episode, filename):
        torch.save({
            'episode': episode,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'content':  self.content
            }, filename)
        print("ERM CHKPNT: {0}".format(filename))

    def Restore(self, episode, filename):
        checkpoint = torch.load(filename)       
        if (checkpoint['episode'] == episode):
            print("Cargando ERM {0} con episodio {1}".format(filename, episode))
            self.max_length = checkpoint['max_length']
            self.batch_size = checkpoint['batch_size']
            self.content = checkpoint['content']
            print("ERM {0} con {1} ejemplos de {2}".format(filename, len(self.content), self.max_length))

        else:
            raise RuntimeError("ERM CHKPNT {0} is invalid".format(episode))

# ----------------------------------------------------------------------------------------------------------------------------------
# END OF FILE
# ----------------------------------------------------------------------------------------------------------------------------------