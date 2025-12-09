# BattleCity RL Agent

Implementación de agentes de Inteligencia Artificial para el juego Battle City utilizando técnicas de Aprendizaje por Refuerzo y algoritmos de búsqueda adversarial. El cuaderno con los experimentos reproducibles se encuentra en el cuaderno de Colab <a href="https://colab.research.google.com/drive/1FEEV7XqpNBkCPceiMQBcTx4-8pKv2aAw?usp=sharing" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>.
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

## Introducción y Objetivo del Proyecto
Este proyecto se enfoca en el **Aprendizaje por Refuerzo (RL)**, un área de la inteligencia artificial donde un agente interactúa con un entorno estocástico para encontrar una política óptima $\pi: S \rightarrow A$ que maximice su recompensa .

El objetivo fue analizar políticas resultantes del entrenamiento de un agente que juega una **versión reducida del clásico juego BattleCity**. Específicamente, buscamos evaluar la eficiencia del agente en función de su tiempo de entrenamiento a través del **análisis de las Cadenas de Markov (CM) inducidas por estas políticas**. Este enfoque permite cuantificar la calidad de las políticas aprendidas e identificar tendencias estructurales en el comportamiento del agente.

## Modelamiento y Metodología

El agente se entrenó utilizando el algoritmo de **Q-learning**, que se basa en la iteración de Q-valores para estimar la utilidad esperada de tomar una acción en un estado dado, aunque el agente no posea información previa sobre las funciones de transición ($T$) o recompensa ($R$) del entorno. Los algoritmos de aprendizaje por refuerzo están fundados en los **Procesos de Decisión de Markov (MDP)**.

### Cadena de Markov Inducida

Una política aprendida genera una sucesión de estados y acciones en una simulación, lo que a su vez induce una **Cadena de Markov sobre los posibles estados del mundo**. Al contar las frecuencias de transiciones observadas durante las simulaciones, se estiman las probabilidades de transición de esta cadena.

### Reducción del Espacio de Estados

El juego original de BattleCity presentaba un número de configuraciones posibles que lo hacía intratable ($\approx 2.6 \times 10^9$ estados). Para acelerar el aprendizaje y facilitar el análisis con CM, se optó por una **representación de estado abstracta y reducida**.

El estado se modeló como una tupla que contiene información parcial del juego, como la vida del jugador, la vida de los enemigos, la posición relativa de peligro a la base, y si hay peligro en la dirección del jugador. Esta abstracción redujo el espacio total a solo **1024 estados posibles**.

## Experimentos y Resultados Clave

Se analizaron tres modelos de agentes entrenados con diferentes cantidades de iteraciones de Q-learning: **Modelo 1 (10000 pasos), Modelo 2 (50000 pasos) y Modelo 3 (100000 pasos)**. Para cada modelo, se extrajo una Cadena de Markov a partir de **1500 trayectorias simuladas**.

Los análisis principales se centraron en el cálculo de la **Matriz Fundamental ($M$)** y la matriz de probabilidades terminales ($G = MA$).

### Hallazgos Principales

1.  **Eficiencia y Exploración:** El comportamiento del agente evoluciona de una fase de exploración a una de explotación eficiente.
    *   El **Modelo 2** (entrenamiento intermedio) exhibió el mayor número de pasos esperados para alcanzar un estado final (victoria o derrota), aproximadamente **89.36 pasos**, lo que sugiere que oscila y explora más tiempo alrededor del estado inicial.
    *   El **Modelo 3** (más entrenado) se estabiliza y converge al resultado en menos pasos esperados, aproximadamente **25.02 pasos**, demostrando una fase de explotación más decisiva .

2.  **Probabilidades Terminales:** Al aumentar el entrenamiento, la política se vuelve más exitosa.
    *   El **Modelo 3** logró la **mayor proporción de victorias (77.5 %)** y la menor proporción de derrotas (16.3 %).
    *   Además, el Modelo 3 redujo significativamente el número de estados únicos explorados (883), concentrándose en regiones del espacio de estados asociadas con trayectorias más estables y exitosas.

3.  **Análisis Riesgo-Recompensa:**
    *   El análisis demostró que el **Modelo 3** es el más robusto, ya que la mayoría de sus estados se agruparon en torno a recompensas esperadas positivas con **niveles de riesgo bajos o moderados**.
    *   Esto indica que, con más entrenamiento, el agente reduce la exposición a situaciones muy penalizadas (recompensas negativas) y mejora el balance riesgo-recompensa de su política.
## Documentación Adicional
En la [rama principal](https://github.com/MateoOrtiz001/BattleCity_RL_agent) se puede encontrar toda la documentación sobre el proyecto del agente en sí. Además de las siguientes guías de uso:
- [Guía de Entrenamiento](TRAINING_GUIDE.md) - Detalles sobre parámetros, función de recompensa y características
- [Guía de Visualización](PLAY_GAME_GUIDE.md) - Opciones de visualización y controles
