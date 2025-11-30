# Guía de Visualización de Partidas - play_game.py

Este script permite visualizar partidas del juego BattleCity utilizando un agente de Q-Learning entrenado, con interfaz gráfica pygame o modo texto.

## Requisitos

- Python 3.x
- pygame
- Agente entrenado (archivo `.pkl` en la carpeta `models/`)

## Uso Básico

```bash
# Ejecutar con configuración por defecto (pygame)
python play_game.py

# Especificar un agente diferente
python play_game.py --agent models/mi_agente.pkl

# Modo texto (sin ventana gráfica)
python play_game.py --text
```

## Opciones Disponibles

| Opción | Descripción | Valor por defecto |
|--------|-------------|-------------------|
| `--agent` | Ruta al archivo del agente entrenado | `models/qlearning_basic_final.pkl` |
| `--level` | Nivel del juego (1-4) | `1` |
| `--games` | Número de partidas a jugar | `1` |
| `--delay` | Delay entre frames en milisegundos | `150` |
| `--approximate` | Usar agente con aproximación de funciones | `False` |
| `--text` | Usar modo texto en lugar de pygame | `False` |

## Ejemplos de Uso

### Ver una partida rápida
```bash
python play_game.py --delay 50
```

### Ver una partida lenta (para analizar)
```bash
python play_game.py --delay 300
```

### Jugar 10 partidas consecutivas
```bash
python play_game.py --games 10
```

### Probar en nivel 2
```bash
python play_game.py --level 2
```

### Usar un agente aproximado
```bash
python play_game.py --agent models/approximate_final.pkl --approximate
```

### Modo texto (sin interfaz gráfica)
```bash
python play_game.py --text --delay 200
```

## Controles (Modo Pygame)

| Tecla | Acción |
|-------|--------|
| `ESC` | Salir del juego |
| `SPACE` | Pausar/Reanudar |
| `+` / `=` | Aumentar velocidad |
| `-` | Disminuir velocidad |
| `ENTER` | Continuar a siguiente partida (en pantalla de resultado) |

## Interfaz Gráfica

La ventana de pygame muestra:

### Elementos del Juego
- **Cuadrado verde**: Tu tanque (jugador)
  - Flecha blanca: Dirección actual
  - Barra verde arriba: Vida restante
- **Cuadrados rojos**: Tanques enemigos
  - Barra roja arriba: Vida restante
- **Cuadrado dorado**: Base a proteger
- **Cuadrados marrones**: Muros de ladrillo (destructibles)
- **Cuadrados grises**: Muros de acero (indestructibles)
- **Círculos**: Balas
  - Verde claro: Tus balas
  - Rojo claro: Balas enemigas

### Panel Inferior
Muestra información en tiempo real:
- **Paso**: Número de paso actual
- **Acción**: Última acción ejecutada
- **Enemigos**: Enemigos vivos restantes
- **Recompensa**: Recompensa acumulada

### Pantalla de Resultado
Al finalizar cada partida muestra:
- **¡VICTORIA!** (verde) o **DERROTA** (rojo)
- Pasos totales
- Recompensa final

## Resumen de Múltiples Partidas

Al ejecutar varias partidas (`--games N`), al final se muestra:
```
========================================
RESUMEN FINAL
========================================
Victorias: X/N (XX.X%)
Pasos promedio: XXX.X
Recompensa promedio: XXX.X
========================================
```

## Solución de Problemas

### "Archivo no encontrado"
Verifica que el archivo del agente existe:
```bash
dir models\*.pkl
```

### La ventana no aparece
- Asegúrate de tener pygame instalado: `pip install pygame`
- Prueba el modo texto: `python play_game.py --text`

### El juego va muy rápido/lento
Ajusta el delay:
```bash
python play_game.py --delay 200  # Más lento
python play_game.py --delay 50   # Más rápido
```

## Agentes Disponibles

Los agentes entrenados se guardan en la carpeta `models/`. Ejemplo:
- `models/qlearning_basic_final.pkl` - Agente Q-Learning tabular
- `models/approximate_final.pkl` - Agente con aproximación de funciones

Para entrenar nuevos agentes, usa `train_agent.py`.
