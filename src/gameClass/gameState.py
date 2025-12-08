from .game import BattleCityState
from ..utils.util import manhattanDistance

class RLState:
    """
    Clase para los estados abstractos (reducidos) del entorno de juego para el
    agente de aprendizaje por refuerzo.
    """
    def __init__(self, game:BattleCityState):
        self.game = game
        self.posTank = game.teamA_tank.getPos()
        self.healthTank = game.teamA_tank.getHealth()
        self.liveTank = game.teamA_tank.isAlive()
        self.posEnemyTanks = [tank.getPos() for tank in game.teamB_tanks]
        self.healthEnemyTanks = [tank.getHealth() for tank in game.teamB_tanks]
        self.liveEnemyTanks = [tank.isAlive() for tank in game.teamB_tanks]
        self.posBullets = [bullet.getPos() for bullet in game.bullets if bullet.owner_id != 0]
        self.posBase = game.base.getPosition()
        self.destroyedBase = game.base.isDestroyed()
        
    def relativePosition(self, reference_pos=None, pos_obj=None, umbral=3):
        """Devuelve la posición relativa de un objeto con respecto al tanque del jugador."""
        # Calcular distancia Manhattan entre dos posiciones (tuplas)
        dist = manhattanDistance(pos_obj, reference_pos)
        if dist <= umbral:
            return 'cerca'
        else:
            return 'lejos'
    
    def peligroDir(self):
        """Determina si hay algún peligro (enemigo o bala) en línea de alguna de las direcciones del jugador."""
        # Si todos los enemigos están vivos, verificamos peligro
        if not any(self.liveEnemyTanks):
            return None
            
        for i, enemy_pos in enumerate(self.posEnemyTanks):
            if not self.liveEnemyTanks[i]:
                continue
            # Verificar si el enemigo está en la misma columna (eje X)
            if enemy_pos[0] == self.posTank[0]:
                dist = abs(enemy_pos[1] - self.posTank[1])
                if dist < 5:
                    if enemy_pos[1] > self.posTank[1]:
                        return 'arriba'
                    else:
                        return 'abajo'
            # Verificar si el enemigo está en la misma fila (eje Y)
            if enemy_pos[1] == self.posTank[1]:
                dist = abs(enemy_pos[0] - self.posTank[0])
                if dist < 5:
                    if enemy_pos[0] > self.posTank[0]:
                        return 'derecha'
                    else:
                        return 'izquierda'
        
        if not self.posBullets:
            return None
            
        for bullet_pos in self.posBullets:
            # Verificar si la bala está en la misma columna
            if bullet_pos[0] == self.posTank[0]:
                dist = abs(bullet_pos[1] - self.posTank[1])
                if dist < 5:
                    if bullet_pos[1] > self.posTank[1]:
                        return 'arriba'
                    else:
                        return 'abajo'
            # Verificar si la bala está en la misma fila
            if bullet_pos[1] == self.posTank[1]:
                dist = abs(bullet_pos[0] - self.posTank[0])
                if dist < 5:
                    if bullet_pos[0] > self.posTank[0]:
                        return 'derecha'
                    else:
                        return 'izquierda'
        return None
    
    def saludTank(self):
        """Devuelve la salud del tanque del jugador."""
        if self.liveTank == False:
            return 0
        else:
            return self.healthTank
        
    def saludEnemigo(self, index):
        """Devuelve la salud del tanque enemigo en la posición index."""
        if self.liveEnemyTanks[index] == False:
            return 0
        else:
            return self.healthEnemyTanks[index]
        
    def peligroBase(self):
        """Determina si hay enemigos o balas enemigas cerca de la base."""
        # Verificar tanques enemigos cerca de la base
        for t in self.game.teamB_tanks:
            if t.isAlive():
                dist = manhattanDistance(t.getPos(), self.posBase)
                if dist <= 5:
                    return 'cerca'
        
        # Verificar balas enemigas cerca de la base
        for bullet_pos in self.posBullets:
            dist = manhattanDistance(bullet_pos, self.posBase)
            if dist <= 3:
                return 'cerca'
        
        return 'lejos'
    
    def getGameState(self):
        """
        Devuelve el estado del juego en formato reducido para el agente de RL.
        Posiciones relativas simplificadas de los enemigos, balas y base con respecto al jugador.
        """
        state = {
            'healthTank': self.saludTank(),
            'posEnemyTanks': [self.relativePosition(self.posTank,pos) for pos in self.posEnemyTanks],
            'healthEnemyTanks': [self.saludEnemigo(i) for i in range(len(self.healthEnemyTanks))],
            'destroyedBase': self.destroyedBase,
            'peligroDir': self.peligroDir(),
            'peligroBase': self.peligroBase()
        }
        return state
        
    def isLose(self):
        """Devuelve True si el jugador ha perdido (tanque destruido o base destruida)."""
        return not self.liveTank or self.destroyedBase
    
    def isWin(self):
        """Devuelve True si el jugador ha ganado (todos los enemigos destruidos)."""
        return self.liveEnemyTanks.count(True) == 0
        
    def getScore(self):
        """Devuelve la puntuación de los estados finales."""
        if self.isWin():
            return 100
        elif self.isLose():
            return -500
        else:
            return 0