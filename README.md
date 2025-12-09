# BattleCity RL Agent
<div align="center">
<img src="assets/banner.gif" alt="Banner">
</div>
Agente de Aprendizaje por Refuerzo (Q-Learning) para el videojuego clásico BattleCity. El proyecto implementa tanto Q-Learning tabular como aproximado con características lineales.


[![Vídeo de Muestra](https://img.youtube.com/vi/nLUwcIf3_e8/sddefault.jpg)](https://youtube.com/shorts/nLUwcIf3_e8)
## Descripción

Este proyecto entrena un agente inteligente para jugar una versión reducida de BattleCity, donde el objetivo es destruir todos los tanques enemigos mientras se protege la base aliada. El agente aprende a través de la interacción con el entorno, optimizando sus decisiones para maximizar la recompensa acumulada.

### Características principales

- **Q-Learning Tabular**: Aprende valores Q exactos para cada par estado-acción
- **Q-Learning Aproximado**: Usa características lineales para generalizar a estados no vistos
- **Visualización con Pygame**: Observa el comportamiento del agente en tiempo real
- **Análisis de Cadenas de Markov**: Extrae y analiza la política aprendida

## Requisitos

```bash
pip install pygame numpy
```

## Estructura del Proyecto

```
BattleCity_RL_agent/
├── train_agent.py          # Script de entrenamiento
├── play_game.py            # Visualización de partidas
├── extract_markov_chain.py # Análisis de la política
├── models/                 # Agentes entrenados (.pkl)
└── src/
    ├── agents/             # Implementaciones de agentes RL
    ├── gameClass/          # Lógica del juego y estados
    ├── training/           # Entorno y trainer
    └── utils/              # Utilidades
```

## Entrenamiento

### Modos predefinidos

```bash
# Demo rápido (100 episodios)
python train_agent.py --mode demo

# Entrenamiento básico (1000 episodios)
python train_agent.py --mode basic

# Entrenamiento con aproximación de funciones (2000 episodios)
python train_agent.py --mode approximate
```

### Entrenamiento personalizado

```bash
python train_agent.py --episodes 5000 --epsilon 0.3 --alpha 0.2 --gamma 0.9 --level 1
```

### Parámetros disponibles

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--episodes` | Número de episodios de entrenamiento | 1000 |
| `--epsilon` | Tasa de exploración (ε-greedy) | 0.3 |
| `--alpha` | Tasa de aprendizaje | 0.2 |
| `--gamma` | Factor de descuento | 0.9 |
| `--level` | Nivel del juego (1-4) | 1 |
| `--approximate` | Usar ApproximateQAgent | False |
| `--save` | Ruta para guardar el modelo | `models/agent` |

### Resultados esperados

Con ~1000 episodios:
- Tasa de victorias: 65-75%
- Tiempo de entrenamiento: ~2 minutos

## Visualización de Partidas

### Uso básico

```bash
# Ejecutar con configuración por defecto
python play_game.py

# Especificar un agente entrenado
python play_game.py --agent models/mi_agente.pkl

# Modo texto (sin ventana gráfica)
python play_game.py --text
```

### Opciones de visualización

| Opción | Descripción | Default |
|--------|-------------|---------|
| `--agent` | Ruta al agente entrenado | `models/qlearning_basic_final.pkl` |
| `--level` | Nivel del juego (1-4) | 1 |
| `--games` | Número de partidas | 1 |
| `--delay` | Delay entre frames (ms) | 150 |
| `--approximate` | Usar agente aproximado | False |
| `--text` | Modo texto | False |

### Ejemplos

```bash
# Partida lenta para analizar
python play_game.py --delay 300

# 10 partidas consecutivas
python play_game.py --games 10

# Usar agente aproximado
python play_game.py --agent models/approximate_final.pkl --approximate
```

### Controles (Pygame)

| Tecla | Acción |
|-------|--------|
| `ESC` | Salir |
| `SPACE` | Pausar/Reanudar |
| `+` / `-` | Ajustar velocidad |
| `ENTER` | Siguiente partida |

### Elementos visuales

- 🟩 **Verde**: Tu tanque (con barra de vida)
- 🟥 **Rojo**: Tanques enemigos
- 🟨 **Dorado**: Base a proteger
- 🟫 **Marrón**: Muros destructibles
- ⬜ **Gris**: Muros indestructibles

## Sistema de Recompensas

| Evento | Recompensa |
|--------|------------|
| Victoria | +1000 |
| Derrota | -500 |
| Eliminar enemigo | +100 |
| Perder vida | -20 |
| Disparar | +5 |
| Acercarse al enemigo | +2 |
| Cada paso (tiempo) | -1 |

## Análisis de la Política

Extrae la cadena de Markov de la política aprendida:

```bash
python extract_markov_chain.py --agent models/mi_agente.pkl --episodes 500
```

Esto permite analizar:
- Matriz de transición de estados
- Distribución estacionaria
- Probabilidades de victoria/derrota
- Tiempos esperados de absorción

## Licencia

Ver archivo [LICENSE](LICENSE) para más detalles.
