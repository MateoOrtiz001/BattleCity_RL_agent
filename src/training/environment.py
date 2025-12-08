# environment.py
# ---------------
# Entorno de entrenamiento para BattleCity RL
# Conecta el juego con los agentes de aprendizaje por refuerzo.

from src.gameClass.game import BattleCityState
from src.gameClass.gameState import RLState
from src.gameClass.scenarios.level1 import get_level1
from src.gameClass.scenarios.level2 import get_level2
from src.gameClass.scenarios.level3 import get_level3
from src.gameClass.scenarios.level4 import get_level4


class BattleCityEnvironment:
    """
    Entorno de entrenamiento para BattleCity.
    Proporciona una interfaz estándar para entrenar agentes de RL.
    """
    
    def __init__(self, level=1, use_reduced_state=True):
        """
        Inicializa el entorno.
        
        Args:
            level: Nivel del juego (1-4)
            use_reduced_state: Si True, usa RLState para estados reducidos
        """
        self.level = level
        self.use_reduced_state = use_reduced_state
        self.game_state = None
        self.previous_score = 0
        self.episode_rewards = 0
        self.steps = 0
        self.max_steps = 500  # Límite de pasos por episodio
        
        # Cargar el layout del nivel
        self.layout = self._get_layout(level)
        
    def _get_layout(self, level):
        """Obtiene el layout para el nivel especificado."""
        layouts = {
            1: get_level1,
            2: get_level2,
            3: get_level3,
            4: get_level4
        }
        return layouts.get(level, get_level1)()
    
    def reset(self):
        """
        Reinicia el entorno para un nuevo episodio.
        
        Returns:
            El estado inicial del juego
        """
        self.game_state = BattleCityState()
        self.game_state.initialize(self.layout)
        self.previous_score = 0
        self.episode_rewards = 0
        self.steps = 0
        
        return self._get_state()
    
    def _get_state(self):
        """
        Obtiene el estado actual del juego.
        
        Returns:
            Estado reducido (hasheable) o estado completo
        """
        if self.use_reduced_state:
            rl_state = RLState(self.game_state)
            # Convertir a tupla hasheable para usar como clave en Q-table
            return self._state_to_tuple(rl_state)
        else:
            return self.game_state
    
    def _state_to_tuple(self, rl_state):
        """
        Convierte un RLState a una tupla hasheable.
        Esto permite usarlo como clave en el diccionario de Q-values.
        """
        state_dict = rl_state.getGameState()
        
        # Crear una representación hasheable del estado
        health = state_dict['healthTank']
        enemy_positions = tuple(state_dict['posEnemyTanks'])
        enemy_health = tuple(state_dict['healthEnemyTanks'])
        base_destroyed = state_dict['destroyedBase']
        peligro_dir = state_dict['peligroDir']
        peligro_base = state_dict['peligroBase']
        
        return (health, enemy_positions, enemy_health, base_destroyed, peligro_dir, peligro_base)
    
    def get_legal_actions(self, agent_index=0):
        """
        Obtiene las acciones legales para un agente.
        
        Args:
            agent_index: Índice del agente (0 para el jugador)
            
        Returns:
            Lista de acciones legales
        """
        if self.game_state is None:
            return []
        return self.game_state.getLegalActions(agent_index)
    
    def step(self, action, agent_index=0):
        """
        Ejecuta una acción en el entorno.
        
        Args:
            action: La acción a ejecutar
            agent_index: Índice del agente que ejecuta la acción
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        if self.game_state is None:
            raise ValueError("El entorno debe ser reiniciado antes de dar pasos")
        
        # Guardar el estado anterior para calcular recompensa
        prev_state = self.game_state
        
        try:
            # Aplicar la acción del jugador
            self.game_state = self.game_state.getSuccessor(agent_index, action)
            
            # Si hay enemigos, aplicar sus acciones
            if not self.game_state.isTerminal():
                self._apply_enemy_actions()
            
        except Exception as e:
            print(f"Error al ejecutar acción {action}: {e}")
            # Si hay error, mantener el estado actual
            pass
        
        self.steps += 1
        
        # Calcular recompensa
        reward = self._calculate_reward(prev_state, self.game_state, action)
        self.episode_rewards += reward
        
        # Verificar si el episodio terminó
        done = self._is_done()
        
        # Obtener el nuevo estado
        next_state = self._get_state()
        
        # Información adicional
        info = {
            'steps': self.steps,
            'episode_rewards': self.episode_rewards,
            'win': self.game_state.isWin() if self.game_state else False,
            'lose': self.game_state.isLose() if self.game_state else False
        }
        
        return next_state, reward, done, info
    
    def _apply_enemy_actions(self):
        """
        Aplica las acciones de los enemigos.
        Por ahora usa una política simple/aleatoria.
        """
        import random
        
        num_agents = self.game_state.getNumAgents()
        
        for enemy_idx in range(1, num_agents):
            if self.game_state.isTerminal():
                break
                
            legal_actions = self.game_state.getLegalActions(enemy_idx)
            if legal_actions:
                # Política simple: elegir acción aleatoria con preferencia por disparar
                fire_actions = [a for a in legal_actions if a.startswith('FIRE_')]
                if fire_actions and random.random() < 0.3:
                    action = random.choice(fire_actions)
                else:
                    action = random.choice(legal_actions)
                
                try:
                    self.game_state = self.game_state.getSuccessor(enemy_idx, action)
                except Exception:
                    pass
    
    def _calculate_reward(self, prev_state, curr_state, action):
        """
        Calcula la recompensa basada en la transición de estados.
        
        Args:
            prev_state: Estado anterior
            curr_state: Estado actual
            action: Acción tomada
            
        Returns:
            float: Recompensa
        """
        if curr_state is None:
            return -100
        
        reward = 0
        
        # Recompensa terminal por ganar/perder
        if curr_state.isWin():
            return 1000  # Gran recompensa por ganar
        
        if curr_state.isLose():
            return -500  # Penalización por perder
        
        # Recompensa por tiempo (penalización pequeña por cada paso)
        reward -= 1
        
        # Recompensa por eliminar enemigos
        prev_enemies_alive = sum(1 for t in prev_state.teamB_tanks if t.isAlive())
        curr_enemies_alive = sum(1 for t in curr_state.teamB_tanks if t.isAlive())
        enemies_killed = prev_enemies_alive - curr_enemies_alive
        reward += enemies_killed * 100  # Recompensa por cada enemigo eliminado
        
        # Penalización por perder vida
        if prev_state.teamA_tank and curr_state.teamA_tank:
            health_lost = prev_state.teamA_tank.getHealth() - curr_state.teamA_tank.getHealth()
            reward -= health_lost * 20
        
        # Recompensa por disparar a un enemigo (aunque no lo mate)
        if action and action.startswith('FIRE_'):
            reward += 5  # Pequeña recompensa por intentar atacar
        
        # Recompensa por acercarse a enemigos (estrategia ofensiva)
        if prev_state.teamA_tank and curr_state.teamA_tank:
            from utils.util import manhattanDistance
            player_pos = curr_state.teamA_tank.getPos()
            
            enemies = [t for t in curr_state.teamB_tanks if t.isAlive()]
            if enemies:
                # Distancia al enemigo más cercano ahora
                min_dist_now = min(manhattanDistance(player_pos, e.getPos()) for e in enemies)
                
                # Distancia al enemigo más cercano antes
                prev_player_pos = prev_state.teamA_tank.getPos()
                prev_enemies = [t for t in prev_state.teamB_tanks if t.isAlive()]
                if prev_enemies:
                    min_dist_before = min(manhattanDistance(prev_player_pos, e.getPos()) for e in prev_enemies)
                    
                    # Recompensa si nos acercamos al enemigo
                    if min_dist_now < min_dist_before:
                        reward += 2
        
        return reward
    
    def _is_done(self):
        """Verifica si el episodio terminó."""
        if self.game_state is None:
            return True
        
        if self.steps >= self.max_steps:
            return True
        
        return self.game_state.isTerminal()
    
    def get_score(self):
        """Obtiene el puntaje actual."""
        if self.game_state:
            return self.game_state.evaluate_state()
        return 0
    
    def render(self):
        """
        Muestra el estado actual del juego en consola.
        Útil para depuración.
        """
        if self.game_state is None:
            print("Juego no inicializado")
            return
        
        board_size = self.game_state.board_size
        grid = [['.' for _ in range(board_size)] for _ in range(board_size)]
        
        # Colocar paredes
        for wall in self.game_state.walls:
            if not wall.isDestroyed():
                x, y = wall.getPosition()
                if 0 <= x < board_size and 0 <= y < board_size:
                    grid[board_size - 1 - y][x] = 'X' if wall.getType() == 'brick' else 'S'
        
        # Colocar base
        if self.game_state.base and not self.game_state.base.isDestroyed():
            bx, by = self.game_state.base.getPosition()
            if 0 <= bx < board_size and 0 <= by < board_size:
                grid[board_size - 1 - by][bx] = 'b'
        
        # Colocar tanques enemigos
        for tank in self.game_state.teamB_tanks:
            if tank.isAlive():
                tx, ty = tank.getPos()
                if 0 <= tx < board_size and 0 <= ty < board_size:
                    grid[board_size - 1 - ty][tx] = 'B'
        
        # Colocar tanque del jugador
        if self.game_state.teamA_tank and self.game_state.teamA_tank.isAlive():
            px, py = self.game_state.teamA_tank.getPos()
            if 0 <= px < board_size and 0 <= py < board_size:
                grid[board_size - 1 - py][px] = 'A'
        
        # Colocar balas
        for bullet in self.game_state.bullets:
            if bullet.isActive():
                bx, by = bullet.getPosition()
                if 0 <= bx < board_size and 0 <= by < board_size:
                    grid[board_size - 1 - by][bx] = '*'
        
        # Imprimir
        print("\n" + "=" * (board_size * 2 + 1))
        for row in grid:
            print(' '.join(row))
        print("=" * (board_size * 2 + 1))
        print(f"Paso: {self.steps}, Recompensa total: {self.episode_rewards:.1f}")
