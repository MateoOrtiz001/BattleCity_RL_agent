#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
play_game.py
------------
Script para visualizar partidas usando un agente entrenado con interfaz gráfica pygame.

Uso:
    python play_game.py                              # Usar agente por defecto
    python play_game.py --agent models/mi_agente.pkl # Especificar agente
    python play_game.py --delay 200                  # Más lento (ms entre frames)
    python play_game.py --games 5                    # Jugar 5 partidas
    python play_game.py --text                       # Modo texto (sin pygame)
"""

import sys
import os
import time
import argparse
import pickle

# Configurar paths
project_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_dir, 'src')
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import pygame

from src.training.environment import BattleCityEnvironment
from src.agents.qlearningAgents import BattleCityQAgent, ApproximateQAgent
from src.utils.util import Counter
from src.GUI.menu import TILE_SIZE, COLORS


def load_agent(filename, use_approximate=False):
    """Carga un agente desde un archivo."""
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        if use_approximate or 'weights' in data:
            agent = ApproximateQAgent(epsilon=0, alpha=0)
            if 'weights' in data:
                agent.weights = Counter()
                agent.weights.update(data['weights'])
        else:
            agent = BattleCityQAgent(epsilon=0, alpha=0)
            if 'qValues' in data:
                agent.qValues = Counter()
                agent.qValues.update(data['qValues'])
        
        print(f"✓ Agente cargado desde {filename}")
        return agent
        
    except FileNotFoundError:
        print(f"✗ Archivo no encontrado: {filename}")
        return None
    except Exception as e:
        print(f"✗ Error al cargar agente: {e}")
        return None


def draw_game_enhanced(screen, game_state, action=None, step=0, reward=0, font=None):
    """
    Dibuja el estado del juego con información adicional.
    """
    screen.fill(COLORS['background'])
    size = game_state.getBoardSize()

    # Dibujar muros
    for wall in game_state.getWalls():
        if wall.isDestroyed():
            continue
        color = COLORS['wall_brick'] if wall.getType() == 'brick' else COLORS['wall_steel']
        x, y = wall.getPosition()
        pygame.draw.rect(screen, color, (x*TILE_SIZE, (size-1-y)*TILE_SIZE, TILE_SIZE, TILE_SIZE))

    # Dibujar base
    if not game_state.getBase().isDestroyed():
        x, y = game_state.getBase().getPosition()
        pygame.draw.rect(screen, COLORS['base'], (x*TILE_SIZE, (size-1-y)*TILE_SIZE, TILE_SIZE, TILE_SIZE))
        # Dibujar símbolo de base
        center_x = x*TILE_SIZE + TILE_SIZE//2
        center_y = (size-1-y)*TILE_SIZE + TILE_SIZE//2
        pygame.draw.polygon(screen, (200, 150, 0), [
            (center_x, center_y - 12),
            (center_x - 10, center_y + 8),
            (center_x + 10, center_y + 8)
        ])

    # Dibujar tanque A (jugador) con dirección
    tankA = game_state.getTeamATank()
    if tankA and tankA.isAlive():
        x, y = tankA.getPos()
        rect_x = x*TILE_SIZE
        rect_y = (size-1-y)*TILE_SIZE
        pygame.draw.rect(screen, COLORS['tank_A'], (rect_x, rect_y, TILE_SIZE, TILE_SIZE))
        
        # Dibujar indicador de dirección
        direction = tankA.getDirection() if hasattr(tankA, 'getDirection') else 'UP'
        center_x = rect_x + TILE_SIZE//2
        center_y = rect_y + TILE_SIZE//2
        arrow_points = {
            'UP': [(center_x, center_y-12), (center_x-8, center_y+6), (center_x+8, center_y+6)],
            'DOWN': [(center_x, center_y+12), (center_x-8, center_y-6), (center_x+8, center_y-6)],
            'LEFT': [(center_x-12, center_y), (center_x+6, center_y-8), (center_x+6, center_y+8)],
            'RIGHT': [(center_x+12, center_y), (center_x-6, center_y-8), (center_x-6, center_y+8)],
        }
        if direction in arrow_points:
            pygame.draw.polygon(screen, (255, 255, 255), arrow_points[direction])
        
        # Dibujar barra de vida
        health = tankA.getHealth() if hasattr(tankA, 'getHealth') else 3
        for i in range(health):
            pygame.draw.rect(screen, (0, 255, 0), (rect_x + i*12 + 2, rect_y - 8, 10, 5))
        for i in range(health, 3):
            pygame.draw.rect(screen, (80, 80, 80), (rect_x + i*12 + 2, rect_y - 8, 10, 5))

    # Dibujar tanques B (enemigos)
    for t in game_state.getTeamBTanks():
        if t.isAlive():
            x, y = t.getPos()
            rect_x = x*TILE_SIZE
            rect_y = (size-1-y)*TILE_SIZE
            pygame.draw.rect(screen, COLORS['tank_B'], (rect_x, rect_y, TILE_SIZE, TILE_SIZE))
            
            # Barra de vida enemigo
            health = t.getHealth() if hasattr(t, 'getHealth') else 3
            for i in range(health):
                pygame.draw.rect(screen, (255, 100, 100), (rect_x + i*12 + 2, rect_y - 8, 10, 5))

    # Dibujar balas
    for b in game_state.getBullets():
        if b.isActive():
            x, y = b.getPosition()
            center_x = x*TILE_SIZE + TILE_SIZE//2
            center_y = (size-1-y)*TILE_SIZE + TILE_SIZE//2
            # Color según equipo
            bullet_color = (150, 255, 150) if b.getTeam() == 'A' else (255, 150, 150)
            pygame.draw.circle(screen, bullet_color, (center_x, center_y), 6)
            pygame.draw.circle(screen, (255, 255, 255), (center_x, center_y), 3)

    # Panel de información en la parte inferior
    panel_y = size * TILE_SIZE
    panel_height = 60
    pygame.draw.rect(screen, (40, 40, 50), (0, panel_y, size * TILE_SIZE, panel_height))
    
    if font is None:
        font = pygame.font.SysFont(None, 24)
    
    # Información de la partida
    action_text = action.replace('_', ' ').title() if action else "---"
    enemies_alive = sum(1 for t in game_state.getTeamBTanks() if t.isAlive())
    
    texts = [
        f"Paso: {step}",
        f"Acción: {action_text}",
        f"Enemigos: {enemies_alive}",
        f"Recompensa: {reward:+.1f}"
    ]
    
    for i, text in enumerate(texts):
        surf = font.render(text, True, (255, 255, 255))
        screen.blit(surf, (10 + i * 150, panel_y + 10))
    
    # Instrucciones
    help_text = "ESC: Salir | SPACE: Pausar | +/-: Velocidad"
    help_surf = font.render(help_text, True, (150, 150, 150))
    screen.blit(help_surf, (10, panel_y + 35))
    
    pygame.display.flip()


def draw_result_screen(screen, win, steps, reward, font):
    """Dibuja pantalla de resultado final."""
    screen.fill((20, 20, 30))
    
    # Título
    if win:
        title = "¡VICTORIA!"
        title_color = (0, 255, 100)
    else:
        title = "DERROTA"
        title_color = (255, 100, 100)
    
    title_font = pygame.font.SysFont(None, 72)
    title_surf = title_font.render(title, True, title_color)
    title_rect = title_surf.get_rect(center=(screen.get_width()//2, 100))
    screen.blit(title_surf, title_rect)
    
    # Estadísticas
    stats = [
        f"Pasos: {steps}",
        f"Recompensa total: {reward:+.1f}"
    ]
    
    for i, stat in enumerate(stats):
        surf = font.render(stat, True, (255, 255, 255))
        rect = surf.get_rect(center=(screen.get_width()//2, 180 + i*30))
        screen.blit(surf, rect)
    
    # Instrucciones
    help_text = "Presiona ENTER para continuar o ESC para salir"
    help_surf = font.render(help_text, True, (150, 150, 150))
    help_rect = help_surf.get_rect(center=(screen.get_width()//2, 280))
    screen.blit(help_surf, help_rect)
    
    pygame.display.flip()


def play_game_pygame(agent, env, delay_ms=150, level=1):
    """
    Juega una partida con visualización pygame.
    
    Args:
        agent: Agente entrenado
        env: Entorno de juego
        delay_ms: Delay entre frames en milisegundos
        level: Nivel del juego
        
    Returns:
        tuple: (result dict, running bool)
    """
    pygame.init()
    pygame.font.init()
    
    state = env.reset()
    game_state = env.game_state
    
    # Configurar ventana
    board_size = game_state.getBoardSize()
    window_width = board_size * TILE_SIZE
    window_height = board_size * TILE_SIZE + 60  # Panel info adicional
    
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption(f"BattleCity RL - Nivel {level}")
    
    font = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()
    
    agent.actionFn = lambda s: env.get_legal_actions(0)
    
    done = False
    paused = False
    running = True
    last_action = None
    current_delay = delay_ms
    info = {}
    
    # Dibujar estado inicial
    draw_game_enhanced(screen, game_state, None, env.steps, env.episode_rewards, font)
    pygame.time.delay(500)
    
    while running and not done:
        # Procesar eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    current_delay = max(50, current_delay - 50)
                elif event.key == pygame.K_MINUS:
                    current_delay = min(1000, current_delay + 50)
        
        if paused:
            # Mostrar indicador de pausa
            pause_surf = font.render("PAUSADO", True, (255, 255, 0))
            pause_rect = pause_surf.get_rect(center=(window_width//2, 30))
            screen.blit(pause_surf, pause_rect)
            pygame.display.flip()
            clock.tick(10)
            continue
        
        # Obtener acción del agente
        legal_actions = env.get_legal_actions(0)
        if not legal_actions:
            break
        
        agent.actionFn = lambda s: legal_actions
        action = agent.getPolicy(state)
        
        if action is None:
            action = legal_actions[0] if legal_actions else None
        
        if action is None:
            break
        
        # Ejecutar acción
        next_state, reward, done, info = env.step(action)
        state = next_state
        last_action = action
        
        # Dibujar
        draw_game_enhanced(screen, env.game_state, action, env.steps, env.episode_rewards, font)
        
        pygame.time.delay(current_delay)
        clock.tick(60)
    
    # Pantalla de resultado
    result = {
        'win': info.get('win', False),
        'lose': info.get('lose', False),
        'steps': env.steps,
        'reward': env.episode_rewards
    }
    
    if running:
        draw_result_screen(screen, result['win'], result['steps'], result['reward'], font)
        
        # Esperar input del usuario
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        waiting = False
                    elif event.key == pygame.K_ESCAPE:
                        waiting = False
                        running = False
            clock.tick(30)
    
    return result, running


def render_game_text(env, clear_screen=True):
    """Renderiza el estado actual del juego en la consola (modo texto)."""
    if env.game_state is None:
        print("Juego no inicializado")
        return
    
    if clear_screen:
        os.system('cls' if os.name == 'nt' else 'clear')
    
    game = env.game_state
    board_size = game.board_size
    
    # Crear grid
    grid = [['·' for _ in range(board_size)] for _ in range(board_size)]
    
    # Colocar paredes
    for wall in game.walls:
        if not wall.isDestroyed():
            x, y = wall.getPosition()
            if 0 <= x < board_size and 0 <= y < board_size:
                grid[board_size - 1 - y][x] = '▓' if wall.getType() == 'brick' else '█'
    
    # Colocar base
    if game.base and not game.base.isDestroyed():
        bx, by = game.base.getPosition()
        if 0 <= bx < board_size and 0 <= by < board_size:
            grid[board_size - 1 - by][bx] = '⌂'
    
    # Colocar tanques enemigos
    for tank in game.teamB_tanks:
        if tank.isAlive():
            tx, ty = tank.getPos()
            if 0 <= tx < board_size and 0 <= ty < board_size:
                grid[board_size - 1 - ty][tx] = 'E'
    
    # Colocar tanque del jugador
    if game.teamA_tank and game.teamA_tank.isAlive():
        px, py = game.teamA_tank.getPos()
        direction = game.teamA_tank.getDirection()
        player_char = {'UP': '▲', 'DOWN': '▼', 'LEFT': '◄', 'RIGHT': '►'}.get(direction, 'P')
        if 0 <= px < board_size and 0 <= py < board_size:
            grid[board_size - 1 - py][px] = player_char
    
    # Colocar balas
    for bullet in game.bullets:
        if bullet.isActive():
            bx, by = bullet.getPosition()
            if 0 <= bx < board_size and 0 <= by < board_size:
                bullet_char = '○' if bullet.getTeam() == 'A' else '●'
                grid[board_size - 1 - by][bx] = bullet_char
    
    # Imprimir
    print("\n" + "═" * (board_size * 2 + 3))
    print("  BATTLECITY RL")
    print("═" * (board_size * 2 + 3))
    
    for row in grid:
        print("│ " + ' '.join(row) + " │")
    
    print("═" * (board_size * 2 + 3))
    
    player_health = game.teamA_tank.getHealth() if game.teamA_tank else 0
    enemies_alive = sum(1 for t in game.teamB_tanks if t.isAlive())
    
    print(f"│ Paso: {env.steps:3d} │ Vida: {'♥' * player_health}{'♡' * (3 - player_health)} │ Enemigos: {enemies_alive} │")
    print(f"│ Recompensa: {env.episode_rewards:+.1f}")
    print("═" * (board_size * 2 + 3))


def play_game_text(agent, env, delay=0.2):
    """Juega una partida con visualización en modo texto."""
    state = env.reset()
    agent.actionFn = lambda s: env.get_legal_actions(0)
    
    done = False
    info = {}
    
    render_game_text(env)
    time.sleep(delay * 2)
    
    while not done:
        legal_actions = env.get_legal_actions(0)
        if not legal_actions:
            break
        
        agent.actionFn = lambda s: legal_actions
        action = agent.getPolicy(state)
        
        if action is None:
            action = legal_actions[0] if legal_actions else None
        
        if action is None:
            break
        
        next_state, reward, done, info = env.step(action)
        state = next_state
        
        render_game_text(env)
        print(f"\n→ Acción: {action.replace('_', ' ').title()}")
        time.sleep(delay)
    
    result = {
        'win': info.get('win', False),
        'lose': info.get('lose', False),
        'steps': env.steps,
        'reward': env.episode_rewards
    }
    
    render_game_text(env, clear_screen=False)
    print("\n" + "=" * 40)
    if result['win']:
        print("🎉 ¡VICTORIA!")
    elif result['lose']:
        print("💀 DERROTA")
    else:
        print("⏱️ TIEMPO AGOTADO")
    print(f"Pasos: {result['steps']}, Recompensa: {result['reward']:.1f}")
    print("=" * 40)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Ver partidas con agente entrenado (pygame)')
    parser.add_argument('--agent', type=str, default='models/qlearning_basic_final.pkl',
                        help='Ruta al archivo del agente')
    parser.add_argument('--level', type=int, default=1, help='Nivel del juego (1-4)')
    parser.add_argument('--games', type=int, default=1, help='Número de partidas')
    parser.add_argument('--delay', type=int, default=150, help='Delay entre frames (ms)')
    parser.add_argument('--approximate', action='store_true', help='Agente usa aproximación')
    parser.add_argument('--text', action='store_true', help='Usar modo texto en lugar de pygame')
    
    args = parser.parse_args()
    
    # Cargar agente
    agent = load_agent(args.agent, args.approximate)
    if agent is None:
        print("\nNo se pudo cargar el agente. Verifica la ruta.")
        print("Agentes disponibles en models/:")
        if os.path.exists('models'):
            for f in os.listdir('models'):
                if f.endswith('.pkl'):
                    print(f"  - models/{f}")
        return
    
    # Crear entorno
    env = BattleCityEnvironment(level=args.level)
    
    wins = 0
    total_steps = 0
    total_reward = 0
    
    if args.text:
        # Modo texto
        for game_num in range(args.games):
            if args.games > 1:
                print(f"\n{'='*40}")
                print(f"PARTIDA {game_num + 1} de {args.games}")
                print('='*40)
            
            try:
                result = play_game_text(agent, env, delay=args.delay/1000)
                if result['win']:
                    wins += 1
                total_steps += result['steps']
                total_reward += result['reward']
                
                if args.games > 1 and game_num < args.games - 1:
                    input("\nPresiona Enter para la siguiente partida...")
                    
            except KeyboardInterrupt:
                print("\n\n⏹️ Partida interrumpida.")
                break
    else:
        # Modo pygame
        running = True
        for game_num in range(args.games):
            if not running:
                break
            
            try:
                result, running = play_game_pygame(agent, env, delay_ms=args.delay, level=args.level)
                if result['win']:
                    wins += 1
                total_steps += result['steps']
                total_reward += result['reward']
                
            except KeyboardInterrupt:
                print("\n\n⏹️ Partida interrumpida.")
                break
        
        # Cerrar pygame
        try:
            pygame.quit()
        except:
            pass
    
    # Resumen final
    if args.games > 1:
        print(f"\n{'='*40}")
        print(f"RESUMEN FINAL")
        print(f"{'='*40}")
        print(f"Victorias: {wins}/{args.games} ({100*wins/args.games:.1f}%)")
        print(f"Pasos promedio: {total_steps/args.games:.1f}")
        print(f"Recompensa promedio: {total_reward/args.games:.1f}")
        print('='*40)


if __name__ == '__main__':
    main()
