#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_markov_chain.py
-----------------------
Script para extraer la cadena de Markov de la política aprendida.

La cadena de Markov representa:
- Estados: Representaciones abstractas del juego
- Transiciones: Probabilidades de pasar de un estado a otro bajo la política
- Recompensas: Recompensas esperadas por transición

Uso:
    python extract_markov_chain.py                           # Valores por defecto
    python extract_markov_chain.py --agent mi_agente.pkl     # Agente específico
    python extract_markov_chain.py --episodes 500            # Más muestras
    python extract_markov_chain.py --output markov_data.pkl  # Guardar datos
"""

import sys
import os
import pickle
import numpy as np
from collections import defaultdict, Counter as StdCounter
import argparse

# Configurar paths
project_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_dir, 'src')
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src.training.environment import BattleCityEnvironment
from src.agents.qlearningAgents import BattleCityQAgent, ApproximateQAgent
from src.utils.util import Counter


class MarkovChainExtractor:
    """
    Extrae y analiza la cadena de Markov inducida por una política.
    """
    
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        
        # Contadores para estimar probabilidades de transición
        self.state_visits = Counter()           # N(s)
        self.transition_counts = defaultdict(Counter)  # N(s, s')
        self.action_counts = defaultdict(Counter)      # N(s, a)
        self.reward_sums = defaultdict(float)          # Suma de recompensas por (s, a, s')
        self.reward_counts = defaultdict(int)          # Conteo de (s, a, s')
        
        # Estados terminales
        self.terminal_states = set()
        self.winning_states = set()
        self.losing_states = set()
        
        # Trayectorias recolectadas
        self.trajectories = []
        
    def collect_trajectories(self, num_episodes=100, verbose=True):
        """
        Recolecta trayectorias siguiendo la política del agente.
        """
        if verbose:
            print(f"Recolectando {num_episodes} trayectorias...")
        
        for episode in range(num_episodes):
            trajectory = self._run_episode()
            self.trajectories.append(trajectory)
            
            if verbose and (episode + 1) % 50 == 0:
                print(f"  Episodios completados: {episode + 1}/{num_episodes}")
        
        if verbose:
            print(f"  Recolectadas {len(self.trajectories)} trayectorias")
            print(f"  Estados únicos visitados: {len(self.state_visits)}")
            print(f"  Estados terminales (victoria): {len(self.winning_states)}")
            print(f"  Estados terminales (derrota): {len(self.losing_states)}")
    
    def _run_episode(self):
        """Ejecuta un episodio y registra las transiciones."""
        state = self.env.reset()
        self.agent.actionFn = lambda s: self.env.get_legal_actions(0)
        
        trajectory = []
        done = False
        
        while not done:
            self.state_visits[state] += 1
            
            legal_actions = self.env.get_legal_actions(0)
            if not legal_actions:
                break
            
            self.agent.actionFn = lambda s: legal_actions
            action = self.agent.getPolicy(state)
            
            if action is None:
                break
            
            self.action_counts[state][action] += 1
            
            next_state, reward, done, info = self.env.step(action)
            
            # Registrar transición
            self.transition_counts[state][next_state] += 1
            
            # Registrar recompensa
            key = (state, action, next_state)
            self.reward_sums[key] += reward
            self.reward_counts[key] += 1
            
            trajectory.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state
            })
            
            # Marcar estados terminales
            if done:
                self.terminal_states.add(next_state)
                if info.get('win', False):
                    self.winning_states.add(next_state)
                elif info.get('lose', False):
                    self.losing_states.add(next_state)
            
            state = next_state
        
        return trajectory
    
    def get_transition_matrix(self):
        """
        Construye la matriz de transición P(s'|s) bajo la política.
        
        Returns:
            states: Lista de estados
            P: Matriz de transición numpy (n_states x n_states)
        """
        states = list(self.state_visits.keys())
        n_states = len(states)
        state_to_idx = {s: i for i, s in enumerate(states)}
        
        P = np.zeros((n_states, n_states))
        
        for state, next_state_counts in self.transition_counts.items():
            if state not in state_to_idx:
                continue
            i = state_to_idx[state]
            total = sum(next_state_counts.values())
            
            for next_state, count in next_state_counts.items():
                if next_state in state_to_idx:
                    j = state_to_idx[next_state]
                    P[i, j] = count / total
        
        return states, P
    
    def get_reward_matrix(self):
        """
        Construye la matriz de recompensas esperadas R(s, s').
        
        Returns:
            states: Lista de estados
            R: Matriz de recompensas numpy (n_states x n_states)
        """
        states = list(self.state_visits.keys())
        n_states = len(states)
        state_to_idx = {s: i for i, s in enumerate(states)}
        
        R = np.zeros((n_states, n_states))
        
        for (state, action, next_state), reward_sum in self.reward_sums.items():
            if state in state_to_idx and next_state in state_to_idx:
                i = state_to_idx[state]
                j = state_to_idx[next_state]
                count = self.reward_counts[(state, action, next_state)]
                R[i, j] = reward_sum / count if count > 0 else 0
        
        return states, R
    
    def get_policy_distribution(self):
        """
        Obtiene la distribución de acciones π(a|s) para cada estado.
        
        Returns:
            dict: {estado: {acción: probabilidad}}
        """
        policy = {}
        
        for state, action_counts in self.action_counts.items():
            total = sum(action_counts.values())
            policy[state] = {
                action: count / total 
                for action, count in action_counts.items()
            }
        
        return policy
    
    def compute_stationary_distribution(self, max_iter=1000, tol=1e-8):
        """
        Calcula la distribución estacionaria de la cadena de Markov.
        
        Returns:
            states: Lista de estados
            pi: Vector de distribución estacionaria
        """
        states, P = self.get_transition_matrix()
        n = len(states)
        
        if n == 0:
            return [], np.array([])
        
        # Inicializar con distribución uniforme
        pi = np.ones(n) / n
        
        for _ in range(max_iter):
            pi_new = pi @ P
            if np.linalg.norm(pi_new - pi) < tol:
                break
            pi = pi_new
        
        # Normalizar
        pi = pi / pi.sum() if pi.sum() > 0 else pi
        
        return states, pi
    
    def compute_hitting_times(self):
        """
        Calcula tiempos esperados de llegada a estados terminales.
        
        Returns:
            states: Lista de estados no terminales
            hitting_times_win: Tiempos esperados a victoria
            hitting_times_lose: Tiempos esperados a derrota
        """
        states, P = self.get_transition_matrix()
        state_to_idx = {s: i for i, s in enumerate(states)}
        
        # Estados no terminales
        non_terminal_idx = [
            i for i, s in enumerate(states) 
            if s not in self.terminal_states
        ]
        
        if not non_terminal_idx:
            return [], np.array([]), np.array([])
        
        # Submatriz para estados no terminales
        Q = P[np.ix_(non_terminal_idx, non_terminal_idx)]
        
        # Resolver (I - Q) * h = 1 para tiempos de llegada
        n = len(non_terminal_idx)
        I = np.eye(n)
        
        try:
            # Tiempo esperado a cualquier estado terminal
            hitting_times = np.linalg.solve(I - Q, np.ones(n))
        except np.linalg.LinAlgError:
            hitting_times = np.full(n, np.inf)
        
        non_terminal_states = [states[i] for i in non_terminal_idx]
        
        return non_terminal_states, hitting_times
    
    def get_state_statistics(self):
        """
        Calcula estadísticas por estado.
        
        Returns:
            dict: Estadísticas para cada estado
        """
        stats = {}
        
        for state in self.state_visits.keys():
            visits = self.state_visits[state]
            
            # Recompensa promedio desde este estado
            outgoing_rewards = []
            for (s, a, ns), reward_sum in self.reward_sums.items():
                if s == state:
                    count = self.reward_counts[(s, a, ns)]
                    if count > 0:
                        outgoing_rewards.append(reward_sum / count)
            
            avg_reward = np.mean(outgoing_rewards) if outgoing_rewards else 0
            
            # Acciones más comunes
            actions = dict(self.action_counts.get(state, {}))
            
            stats[state] = {
                'visits': visits,
                'avg_reward': avg_reward,
                'action_distribution': actions,
                'is_terminal': state in self.terminal_states,
                'is_winning': state in self.winning_states,
                'is_losing': state in self.losing_states
            }
        
        return stats
    
    def export_to_dict(self):
        """
        Exporta todos los datos de la cadena de Markov a un diccionario.
        
        Returns:
            dict: Datos completos de la cadena de Markov
        """
        states, P = self.get_transition_matrix()
        _, R = self.get_reward_matrix()
        _, pi = self.compute_stationary_distribution()
        policy = self.get_policy_distribution()
        state_stats = self.get_state_statistics()
        
        # Convertir estados a strings para serialización
        states_str = [str(s) for s in states]
        terminal_str = [str(s) for s in self.terminal_states]
        winning_str = [str(s) for s in self.winning_states]
        losing_str = [str(s) for s in self.losing_states]
        
        # Simplificar trayectorias para serialización
        simple_trajectories = []
        for traj in self.trajectories:
            simple_traj = []
            for step in traj:
                simple_traj.append({
                    'state': str(step['state']),
                    'action': step['action'],
                    'reward': step['reward'],
                    'next_state': str(step['next_state'])
                })
            simple_trajectories.append(simple_traj)
        
        # Convertir policy a strings
        policy_str = {str(k): v for k, v in policy.items()}
        state_stats_str = {str(k): v for k, v in state_stats.items()}
        state_visits_str = {str(k): v for k, v in self.state_visits.items()}
        
        return {
            'states': states_str,
            'states_original': states,  # Mantener original para análisis
            'transition_matrix': P,
            'reward_matrix': R,
            'stationary_distribution': pi,
            'policy': policy_str,
            'state_statistics': state_stats_str,
            'terminal_states': terminal_str,
            'winning_states': winning_str,
            'losing_states': losing_str,
            'trajectories': simple_trajectories,
            'state_visits': state_visits_str,
            'num_episodes': len(self.trajectories)
        }
    
    def print_summary(self):
        """Imprime un resumen del análisis."""
        import sys
        try:
            states, P = self.get_transition_matrix()
            _, pi = self.compute_stationary_distribution()
            
            print("\n" + "=" * 60)
            print("ANÁLISIS DE CADENA DE MARKOV")
            print("=" * 60)
            
            print(f"\n ESTADÍSTICAS GENERALES:")
            print(f"   Estados únicos: {len(states)}")
            print(f"   Estados terminales: {len(self.terminal_states)}")
            print(f"   - Victoria: {len(self.winning_states)}")
            print(f"   - Derrota: {len(self.losing_states)}")
            print(f"   Trayectorias recolectadas: {len(self.trajectories)}")
            
            # Estados más visitados
            print(f"\n ESTADOS MÁS VISITADOS:")
            sorted_states = sorted(self.state_visits.items(), key=lambda x: -x[1])[:5]
            for state, visits in sorted_states:
                print(f"   {state}: {visits} visitas")
            
            # Distribución estacionaria (top estados)
            if len(pi) > 0:
                print(f"\n DISTRIBUCIÓN ESTACIONARIA (top 5):")
                state_probs = list(zip(states, pi))
                sorted_probs = sorted(state_probs, key=lambda x: -x[1])[:5]
                for state, prob in sorted_probs:
                    if prob > 0.001:
                        print(f"   {state}: {prob:.4f}")
            
            # Acciones más comunes globalmente
            print(f"\n ACCIONES MÁS FRECUENTES:")
            global_actions = StdCounter()
            for state_actions in self.action_counts.values():
                for action, count in state_actions.items():
                    global_actions[action] += count
            
            total_actions = sum(global_actions.values())
            if total_actions > 0:
                for action, count in global_actions.most_common(5):
                    print(f"   {action}: {count} ({100*count/total_actions:.1f}%)")
            
            print("\n" + "=" * 60)
            sys.stdout.flush()
            
        except Exception as e:
            print(f"\n Error en print_summary: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()


def load_agent(filename):
    """Carga un agente desde archivo."""
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        if 'weights' in data:
            agent = ApproximateQAgent(epsilon=0, alpha=0)
            agent.weights = Counter()
            agent.weights.update(data['weights'])
        else:
            agent = BattleCityQAgent(epsilon=0, alpha=0)
            if 'qValues' in data:
                agent.qValues = Counter()
                agent.qValues.update(data['qValues'])
        
        print(f" Agente cargado desde {filename}")
        return agent
        
    except Exception as e:
        print(f" Error al cargar agente: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Extraer cadena de Markov de política')
    parser.add_argument('--agent', type=str, default='models/qlearning_basic_final.pkl',
                        help='Ruta al archivo del agente')
    parser.add_argument('--level', type=int, default=1, help='Nivel del juego')
    parser.add_argument('--episodes', type=int, default=200, 
                        help='Episodios para estimar transiciones')
    parser.add_argument('--output', type=str, default='markov_chain_data.pkl',
                        help='Archivo de salida')
    parser.add_argument('--csv', type=str, default=None,
                        help='Exportar matriz de transición a CSV')
    
    args = parser.parse_args()
    
    # Cargar agente
    agent = load_agent(args.agent)
    if agent is None:
        return
    
    # Crear entorno
    env = BattleCityEnvironment(level=args.level)
    
    # Extraer cadena de Markov
    extractor = MarkovChainExtractor(agent, env)
    extractor.collect_trajectories(num_episodes=args.episodes)
    
    # Mostrar resumen
    try:
        extractor.print_summary()
    except Exception as e:
        print(f"\n Error en print_summary: {e}")
        import traceback
        traceback.print_exc()
    
    # Exportar datos
    try:
        data = extractor.export_to_dict()
        
        with open(args.output, 'wb') as f:
            pickle.dump(data, f)
        print(f"\n Datos guardados en {args.output}")
    except Exception as e:
        print(f"\n Error al exportar datos: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Exportar a CSV si se solicita
    if args.csv:
        states, P = extractor.get_transition_matrix()
        
        # Guardar matriz de transición
        import csv
        with open(args.csv, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header con índices de estados
            writer.writerow([''] + list(range(len(states))))
            for i, row in enumerate(P):
                writer.writerow([i] + list(row))
        
        # Guardar mapeo de estados
        with open(args.csv.replace('.csv', '_states.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'state'])
            for i, state in enumerate(states):
                writer.writerow([i, str(state)])
        
        print(f" Matriz de transición exportada a {args.csv}")
    
    print("\n Para usar los datos en Python:")
    print(f"   import pickle")
    print(f"   with open('{args.output}', 'rb') as f:")
    print(f"       data = pickle.load(f)")
    print(f"   P = data['transition_matrix']  # Matriz de transición")
    print(f"   R = data['reward_matrix']      # Matriz de recompensas")
    print(f"   pi = data['stationary_distribution']  # Dist. estacionaria")
    
    # Limpiar pygame
    try:
        import pygame
        pygame.quit()
    except:
        pass
    
    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code if exit_code else 0)
    except SystemExit:
        pass
    except Exception as e:
        print(f"Error fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
