# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
#
# Adapted for BattleCity RL project.

import random
import math
import sys
import os

# Configurar paths para imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(src_dir)

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.utils import util
from src.agents.learningAgents import ReinforcementAgent

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qValues[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0
        return max(self.getQValue(state, action) for action in legalActions)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        bestAction = [a for a in legalActions if self.getQValue(state,a) == self.computeValueFromQValues(state)]
        return random.choice(bestAction)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if not legalActions:
          return None
        if util.flipCoin(self.epsilon):
          action = random.choice(legalActions)
        else:
          action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        self.qValues[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
    
    def saveQValues(self, filename):
        """Guarda los Q-values en un archivo para poder continuar el entrenamiento."""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.qValues), f)
        print(f"Q-values guardados en {filename}")
    
    def loadQValues(self, filename):
        """Carga los Q-values desde un archivo."""
        import pickle
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.qValues = util.Counter()
                self.qValues.update(data)
            print(f"Q-values cargados desde {filename}")
        except FileNotFoundError:
            print(f"Archivo {filename} no encontrado, iniciando con Q-values vacíos")


class BattleCityQAgent(QLearningAgent):
    """
    Agente Q-Learning específico para BattleCity.
    Hereda de QLearningAgent con parámetros por defecto adecuados para el juego.
    """

    def __init__(self, epsilon=0.1, gamma=0.9, alpha=0.2, numTraining=1000, **args):
        """
        Parámetros configurables:
            epsilon  - tasa de exploración (probabilidad de acción aleatoria)
            gamma    - factor de descuento
            alpha    - tasa de aprendizaje
            numTraining - número de episodios de entrenamiento
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # El jugador es siempre el agente 0
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Obtiene la acción a tomar y registra el estado/acción para el aprendizaje.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


# ==========================================
# EXTRACTORES DE CARACTERÍSTICAS PARA BATTLECITY
# ==========================================

class FeatureExtractor:
    """Clase base para extractores de características."""
    def getFeatures(self, state, action):
        """
        Retorna un Counter con las características para el par (estado, acción).
        """
        util.raiseNotDefined()


class IdentityExtractor(FeatureExtractor):
    """
    Extractor simple que usa el par (estado, acción) como única característica.
    """
    def getFeatures(self, state, action):
        features = util.Counter()
        features[(state, action)] = 1.0
        return features


class BattleCityFeatureExtractor(FeatureExtractor):
    """
    Extractor de características específico para BattleCity.
    Extrae características relevantes del estado del juego.
    """
    def getFeatures(self, state, action):
        from utils.util import manhattanDistance
        
        features = util.Counter()
        
        # Obtener información del estado
        try:
            # Si state es un BattleCityState
            player_pos = state.teamA_tank.getPos()
            base_pos = state.base.getPosition()
            enemies = [t for t in state.teamB_tanks if t.isAlive()]
            bullets = state.bullets
            board_size = state.board_size
        except AttributeError:
            # Si state tiene otro formato (ej: tupla hasheable)
            features[('state', action)] = 1.0
            return features
        
        # Característica de bias
        features['bias'] = 1.0
        
        # Distancia a la base (normalizada)
        dist_to_base = manhattanDistance(player_pos, base_pos) / board_size
        features['dist-to-base'] = dist_to_base
        
        # Distancia al enemigo más cercano
        if enemies:
            min_enemy_dist = min(manhattanDistance(player_pos, e.getPos()) for e in enemies)
            features['closest-enemy'] = min_enemy_dist / board_size
            
            # Número de enemigos vivos (normalizado)
            features['enemies-alive'] = len(enemies) / 4.0
            
            # Amenaza: enemigo cercano a la base
            min_threat_dist = min(manhattanDistance(e.getPos(), base_pos) for e in enemies)
            features['enemy-threat-to-base'] = 1.0 - (min_threat_dist / board_size)
        else:
            features['closest-enemy'] = 0.0
            features['enemies-alive'] = 0.0
            features['enemy-threat-to-base'] = 0.0
        
        # Balas enemigas cercanas (peligro)
        enemy_bullets = [b for b in bullets if hasattr(b, 'team') and b.team != 'A']
        if enemy_bullets:
            min_bullet_dist = min(manhattanDistance(player_pos, b.position) for b in enemy_bullets)
            features['bullet-danger'] = 1.0 if min_bullet_dist < 3 else 0.0
        else:
            features['bullet-danger'] = 0.0
        
        # Características basadas en la acción
        if action:
            # Si la acción es disparar
            if action.startswith('FIRE_'):
                direction = action.split('_')[1]
                dx, dy = {'UP': (0, 1), 'DOWN': (0, -1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}.get(direction, (0, 0))
                
                # Verificar si hay un enemigo en la línea de disparo
                x, y = player_pos
                has_target = False
                for i in range(1, board_size):
                    check_pos = (x + dx * i, y + dy * i)
                    for enemy in enemies:
                        if enemy.getPos() == check_pos:
                            has_target = True
                            features['fire-at-enemy'] = 1.0
                            break
                    if has_target:
                        break
                
                if not has_target:
                    features['fire-at-enemy'] = 0.0
            
            # Si la acción es moverse
            if action.startswith('MOVE_'):
                direction = action.split('_')[1]
                dx, dy = {'UP': (0, 1), 'DOWN': (0, -1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}.get(direction, (0, 0))
                new_pos = (player_pos[0] + dx, player_pos[1] + dy)
                
                # ¿Se acerca al enemigo más cercano?
                if enemies:
                    old_min = min(manhattanDistance(player_pos, e.getPos()) for e in enemies)
                    new_min = min(manhattanDistance(new_pos, e.getPos()) for e in enemies)
                    features['moves-toward-enemy'] = 1.0 if new_min < old_min else 0.0
                
                # ¿Se acerca a la base? (para defender)
                old_dist_base = manhattanDistance(player_pos, base_pos)
                new_dist_base = manhattanDistance(new_pos, base_pos)
                features['moves-toward-base'] = 1.0 if new_dist_base < old_dist_base else 0.0
        
        return features


class SimpleBattleCityExtractor(FeatureExtractor):
    """
    Extractor de características más simple usando el estado reducido (RLState).
    """
    def getFeatures(self, state, action):
        features = util.Counter()
        
        # Si el estado es un diccionario (de RLState.getGameState())
        if isinstance(state, dict):
            features['bias'] = 1.0
            features['health'] = state.get('healthTank', 0) / 3.0
            
            peligro = state.get('peligroDir', None)
            features['peligro-arriba'] = 1.0 if peligro == 'arriba' else 0.0
            features['peligro-abajo'] = 1.0 if peligro == 'abajo' else 0.0
            features['peligro-izquierda'] = 1.0 if peligro == 'izquierda' else 0.0
            features['peligro-derecha'] = 1.0 if peligro == 'derecha' else 0.0
            
            features['base-destroyed'] = 1.0 if state.get('destroyedBase', False) else 0.0
            
            # Peligro en la base (enemigos o balas cerca)
            peligro_base = state.get('peligroBase', 'lejos')
            features['peligro-base'] = 1.0 if peligro_base == 'cerca' else 0.0
            
            # Contar enemigos vivos
            enemy_health = state.get('healthEnemyTanks', [])
            enemies_alive = sum(1 for h in enemy_health if h > 0)
            features['enemies-alive'] = enemies_alive / 4.0
        
        # Característica de la acción
        if action:
            features[f'action-{action}'] = 1.0
        
        return features


class ApproximateQAgent(BattleCityQAgent):
    """
       ApproximateQLearningAgent para BattleCity
       
       Usa características lineales para aproximar la función Q.
       Esto permite generalizar a estados no vistos antes.
    """
    def __init__(self, extractor='BattleCityFeatureExtractor', **args):
        # Mapeo de nombres de extractores a clases
        extractors = {
            'IdentityExtractor': IdentityExtractor,
            'BattleCityFeatureExtractor': BattleCityFeatureExtractor,
            'SimpleBattleCityExtractor': SimpleBattleCityExtractor
        }
        
        if extractor in extractors:
            self.featExtractor = extractors[extractor]()
        else:
            self.featExtractor = BattleCityFeatureExtractor()
        
        BattleCityQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)
        return self.weights * features

    def update(self, state, action, nextState, reward):
        """
           Actualiza los pesos basándose en la transición observada.
        """
        dif = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
        features = self.featExtractor.getFeatures(state, action)
        for f in features:
            self.weights[f] += self.alpha * dif * features[f]

    def final(self, state):
        """Llamado al final de cada episodio."""
        # Llamar al método final de la clase padre
        BattleCityQAgent.final(self, state)

        # ¿Terminamos el entrenamiento?
        if self.episodesSoFar == self.numTraining:
            print("Pesos finales del agente:")
            print(self.weights)
    
    def saveWeights(self, filename):
        """Guarda los pesos en un archivo."""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.weights), f)
        print(f"Pesos guardados en {filename}")
    
    def loadWeights(self, filename):
        """Carga los pesos desde un archivo."""
        import pickle
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.weights = util.Counter()
                self.weights.update(data)
            print(f"Pesos cargados desde {filename}")
        except FileNotFoundError:
            print(f"Archivo {filename} no encontrado, iniciando con pesos vacíos")
