# BattleCity RL - Guía de Entrenamiento

Este documento explica cómo entrenar un agente de Q-Learning para jugar BattleCity Reducido.

## Estructura del Proyecto

```
BattleCity_RL_agent/
├── train_agent.py              # Script principal para entrenar
├── models/                     # Carpeta donde se guardan los agentes
├── src/
│   ├── agents/
│   │   ├── learningAgents.py   # Clases base para agentes de RL
│   │   ├── qlearningAgents.py  # Agentes Q-Learning y ApproximateQ
│   │   └── ...
│   ├── gameClass/
│   │   ├── game.py             # Estado del juego BattleCity
│   │   ├── gameState.py        # Estados reducidos para RL (RLState)
│   │   ├── tank.py             # Clase Tank
│   │   ├── bullet.py           # Clase Bullet
│   │   └── scenarios/          # Niveles del juego
│   ├── training/
│   │   ├── environment.py      # Entorno de entrenamiento
│   │   └── trainer.py          # Clase QLearningTrainer
│   └── utils/
│       └── util.py             # Utilidades (Counter, manhattanDistance, etc.)
```

## Agentes Implementados

### 1. BattleCityQAgent (Q-Learning Tabular)
- Usa una tabla Q para almacenar valores de (estado, acción)
- Bueno para espacios de estados pequeños/discretos
- Estados representados como tuplas hasheables

### 2. ApproximateQAgent (Approximate Q-Learning)
- Usa características lineales para aproximar Q(s,a)
- Puede generalizar a estados no vistos
- Usa extractores de características personalizados

## Cómo Entrenar

### Demo Rápido (100 episodios)
```bash
python train_agent.py --mode demo
```

### Entrenamiento Básico (1000 episodios)
```bash
python train_agent.py --mode basic
```

### Entrenamiento con Aproximación (2000 episodios)
```bash
python train_agent.py --mode approximate
```

### Entrenamiento Personalizado
```bash
python train_agent.py --episodes 5000 --epsilon 0.3 --alpha 0.2 --gamma 0.9 --level 1
```

## Parámetros de Entrenamiento

| Parámetro | Descripción | Valor por defecto |
|-----------|-------------|-------------------|
| `--episodes` | Número de episodios | 1000 |
| `--epsilon` | Tasa de exploración | 0.3 |
| `--alpha` | Tasa de aprendizaje | 0.2 |
| `--gamma` | Factor de descuento | 0.9 |
| `--level` | Nivel del juego (1-4) | 1 |
| `--approximate` | Usar ApproximateQAgent | False |
| `--save` | Ruta para guardar | `models/agent` |

## Cargar y Probar un Agente

```python
from src.training.trainer import QLearningTrainer

# Crear trainer
trainer = QLearningTrainer(level=1, num_episodes=0)

# Cargar agente guardado
trainer.load_agent('models/qlearning_basic_final.pkl')

# Probar el agente (sin aprendizaje)
stats = trainer.test(num_episodes=100, verbose=True)
```

## Función de Recompensa

El sistema de recompensas está diseñado para:
- **+1000**: Victoria (eliminar todos los enemigos)
- **-500**: Derrota (tanque destruido o base destruida)
- **+100**: Por cada enemigo eliminado
- **-20**: Por cada punto de vida perdido
- **+5**: Por disparar
- **+2**: Por acercarse a un enemigo
- **-1**: Penalización por tiempo (cada paso)

## Características para ApproximateQAgent

El `BattleCityFeatureExtractor` extrae:
- `bias`: Constante 1.0
- `dist-to-base`: Distancia normalizada a la base
- `closest-enemy`: Distancia al enemigo más cercano
- `enemies-alive`: Proporción de enemigos vivos
- `enemy-threat-to-base`: Amenaza de enemigos a la base
- `bullet-danger`: Hay balas enemigas cercanas
- `fire-at-enemy`: La acción dispara hacia un enemigo
- `moves-toward-enemy`: La acción acerca al enemigo

## Resultados Esperados

Con 1000 episodios de entrenamiento:
- Tasa de victorias: ~65-75%
- Estados-acciones aprendidos: ~4500-5000
- Tiempo de entrenamiento: ~2 minutos

## Próximos Pasos

1. **Más episodios**: Entrenar con 5000+ episodios para mejores resultados
2. **Ajustar recompensas**: Modificar `_calculate_reward` en `environment.py`
3. **Nuevos extractores**: Crear extractores de características más sofisticados
4. **Deep Q-Learning**: Implementar DQN para espacios de estados más grandes
5. **Múltiples niveles**: Entrenar en diferentes niveles para generalización
