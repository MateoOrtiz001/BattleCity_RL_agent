

class RLState:
    """
    Clase para los estados abstractos (reducidos) del entorno de juego para el
    agente de aprendizaje por refuerzo.
    """
    def __init__(self, game):
        self.game = game