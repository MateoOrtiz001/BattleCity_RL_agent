
class Base:
    """Clase que representa una base en el juego Battle City."""
    __slots__ = ("position", "is_destroyed")
    
    def __init__(self, position):
        self.position = position    # Posición de la base
        self.is_destroyed = False   # Estado de la base
        
    def getPosition(self):
        return self.position
    
    def getPos(self):
        """Alias de getPosition para consistencia con otras clases."""
        return self.position
    
    def isDestroyed(self):
        return self.is_destroyed
    
    def takeDamage(self):
        self.is_destroyed = True