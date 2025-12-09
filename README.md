# BattleCity RL Agent

Implementación de agentes de Inteligencia Artificial para el juego Battle City utilizando técnicas de Aprendizaje por Refuerzo y algoritmos de búsqueda adversarial.
## Integrantes

- Mateo Ortiz
- Santiago Botero
- Juan Diego González

<div align="center">
  <h3> Video Presentación</h3>
  <a href="https://youtu.be/otL7HJ0bihU" target="_blank">
    <img src="https://img.youtube.com/vi/otL7HJ0bihU/0.jpg" alt="Video demostrativo" width="600">
  </a>
  <p><em>Vídeo subido a youtube</em></p>
</div>

## Descripción

Este proyecto implementa varios agentes inteligentes capaces de jugar una versión simplificada del clásico juego Battle City (NES, 1985). El objetivo del jugador es destruir todos los tanques enemigos mientras protege su base de ser destruida.

### Características del Juego

- Tablero de juego en cuadrícula (13x13)
- Tanque del jugador vs múltiples tanques enemigos
- Muros destructibles (ladrillo) e indestructibles (acero)
- Base que debe ser protegida
- Sistema de disparos y colisiones

## Agentes Implementados

| Agente | Tipo | Descripción |
|--------|------|-------------|
| `QLearningAgent` | Aprendizaje por Refuerzo | Q-Learning tabular para espacios de estados discretos |
| `ApproximateQAgent` | Aprendizaje por Refuerzo | Q-Learning con aproximación de funciones lineales |
| `MinimaxAgent` | Búsqueda Adversarial | Algoritmo Minimax para decisiones óptimas |
| `AlphaBetaAgent` | Búsqueda Adversarial | Minimax con poda alfa-beta |
| `ExpectimaxAgent` | Búsqueda Adversarial | Expectimax para enemigos estocásticos |
| `ReflexTankAgent` | Basado en Reglas | Agente reactivo con comportamiento ofensivo/defensivo |

## Estructura del Proyecto

```
BattleCity_RL_agent/
├── play_game.py              # Visualizar partidas con agente entrenado
├── train_agent.py            # Entrenar agentes de Q-Learning
├── models/                   # Agentes entrenados (.pkl)
├── src/
│   ├── agents/               # Implementación de agentes
│   │   ├── base_search.py    # Clase base para agentes de búsqueda
│   │   ├── qlearningAgents.py
│   │   ├── learningAgents.py
│   │   ├── minimax.py
│   │   ├── expectimax.py
│   │   ├── reflexAgent.py
│   │   └── enemyAgent.py
│   ├── gameClass/            # Lógica del juego
│   │   ├── game.py           # Estado principal del juego
│   │   ├── gameState.py      # Estados reducidos para RL
│   │   ├── tank.py
│   │   ├── bullet.py
│   │   ├── walls.py
│   │   ├── base.py
│   │   └── scenarios/        # Niveles del juego (1-4)
│   ├── training/             # Sistema de entrenamiento
│   │   ├── environment.py    # Entorno de RL
│   │   └── trainer.py        # Clase QLearningTrainer
│   ├── GUI/                  # Interfaz gráfica
│   │   └── menu.py
│   └── utils/                # Utilidades
│       └── util.py
├── PLAY_GAME_GUIDE.md        # Guía detallada para jugar
└── TRAINING_GUIDE.md         # Guía detallada para entrenar
```

## Requisitos

- Python 3.8+
- pygame
- numpy

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/BattleCity_RL_agent.git
cd BattleCity_RL_agent
```

2. Instalar dependencias:
```bash
pip install pygame numpy
```

## Uso Rápido

### Entrenar un Agente

```bash
# Entrenamiento básico (Q-Learning tabular, 1000 episodios)
python train_agent.py --mode basic

# Entrenamiento con aproximación de funciones (2000 episodios)
python train_agent.py --mode approximate

# Entrenamiento personalizado
python train_agent.py --episodes 5000 --epsilon 0.3 --alpha 0.2 --level 1
```

### Visualizar Partidas

```bash
# Jugar con el agente entrenado (interfaz gráfica)
python play_game.py

# Especificar un agente diferente
python play_game.py --agent models/agent_final.pkl

# Modo texto (sin interfaz gráfica)
python play_game.py --text

# Ajustar velocidad de visualización
python play_game.py --delay 200
```

### Controles (Modo Gráfico)

| Tecla | Acción |
|-------|--------|
| `ESC` | Salir |
| `SPACE` | Pausar/Reanudar |
| `+` / `-` | Ajustar velocidad |
| `ENTER` | Siguiente partida |

## Documentación Adicional

- [Guía de Entrenamiento](TRAINING_GUIDE.md) - Detalles sobre parámetros, función de recompensa y características
- [Guía de Visualización](PLAY_GAME_GUIDE.md) - Opciones de visualización y controles

## Ejemplo de Uso Programático

```python
from src.agents import MinimaxAgent, ExpectimaxAgent
from src.training import BattleCityEnvironment

# Crear entorno
env = BattleCityEnvironment(level=1)
state = env.reset()

# Crear agente
agent = MinimaxAgent(depth=2, time_limit=1.0)

# Obtener acción
game_state = env.game_state
action = agent.getAction(game_state)
print(f"Acción seleccionada: {action}")
```
