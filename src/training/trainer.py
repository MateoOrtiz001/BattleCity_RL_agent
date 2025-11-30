# trainer.py
# -----------
# Entrenador para agentes de Q-Learning en BattleCity

import sys
import os
import time
import pickle

# Configurar paths correctamente
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(src_dir)

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.training.environment import BattleCityEnvironment
from src.agents.qlearningAgents import QLearningAgent, BattleCityQAgent, ApproximateQAgent


class QLearningTrainer:
    """
    Clase para entrenar agentes de Q-Learning en BattleCity.
    """
    
    def __init__(self, agent=None, env=None, level=1, 
                 epsilon=0.3, gamma=0.9, alpha=0.2, 
                 num_episodes=1000, use_approximate=False):
        """
        Inicializa el entrenador.
        
        Args:
            agent: Agente pre-configurado (opcional)
            env: Entorno pre-configurado (opcional)
            level: Nivel del juego (1-4)
            epsilon: Tasa de exploración inicial
            gamma: Factor de descuento
            alpha: Tasa de aprendizaje
            num_episodes: Número de episodios de entrenamiento
            use_approximate: Si True, usa ApproximateQAgent con características
        """
        # Crear entorno
        self.env = env if env else BattleCityEnvironment(level=level)
        
        # Crear agente
        if agent:
            self.agent = agent
        elif use_approximate:
            self.agent = ApproximateQAgent(
                epsilon=epsilon,
                gamma=gamma,
                alpha=alpha,
                numTraining=num_episodes,
                extractor='BattleCityFeatureExtractor'
            )
        else:
            self.agent = BattleCityQAgent(
                epsilon=epsilon,
                gamma=gamma,
                alpha=alpha,
                numTraining=num_episodes
            )
        
        # Configurar la función de acciones legales del agente
        self.agent.actionFn = lambda state: self.env.get_legal_actions(0)
        
        self.num_episodes = num_episodes
        self.initial_epsilon = epsilon
        
        # Estadísticas
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = 0
        self.losses = 0
        
    def train(self, verbose=True, render_every=0, save_every=100, save_path=None):
        """
        Entrena el agente.
        
        Args:
            verbose: Si True, muestra información del entrenamiento
            render_every: Renderiza el juego cada N episodios (0 para desactivar)
            save_every: Guarda el agente cada N episodios
            save_path: Ruta para guardar el agente
            
        Returns:
            dict: Estadísticas del entrenamiento
        """
        start_time = time.time()
        
        for episode in range(self.num_episodes):
            # Decay de epsilon (exploración -> explotación)
            self.agent.epsilon = max(0.01, self.initial_epsilon * (1 - episode / self.num_episodes))
            
            # Iniciar episodio
            self.agent.startEpisode()
            state = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                # Obtener acciones legales
                legal_actions = self.env.get_legal_actions(0)
                
                if not legal_actions:
                    break
                
                # Actualizar la función de acciones del agente
                self.agent.actionFn = lambda s: legal_actions
                
                # El agente elige una acción
                action = self.agent.getAction(state)
                
                if action is None:
                    break
                
                # Ejecutar acción en el entorno
                next_state, reward, done, info = self.env.step(action)
                
                # Actualizar Q-values
                self.agent.observeTransition(state, action, next_state, reward)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                # Renderizar si es necesario
                if render_every > 0 and episode % render_every == 0:
                    self.env.render()
                    time.sleep(0.1)
            
            # Finalizar episodio
            self.agent.stopEpisode()
            
            # Guardar estadísticas
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            if info.get('win', False):
                self.wins += 1
            elif info.get('lose', False):
                self.losses += 1
            
            # Mostrar progreso
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = sum(self.episode_rewards[-100:]) / min(100, len(self.episode_rewards[-100:]))
                avg_length = sum(self.episode_lengths[-100:]) / min(100, len(self.episode_lengths[-100:]))
                elapsed = time.time() - start_time
                
                print(f"Episodio {episode + 1}/{self.num_episodes}")
                print(f"  Recompensa promedio (últimos 100): {avg_reward:.2f}")
                print(f"  Duración promedio (últimos 100): {avg_length:.1f}")
                print(f"  Victorias: {self.wins}, Derrotas: {self.losses}")
                print(f"  Epsilon actual: {self.agent.epsilon:.3f}")
                print(f"  Tiempo transcurrido: {elapsed:.1f}s")
                
                if hasattr(self.agent, 'qValues'):
                    print(f"  Estados-acciones en Q-table: {len(self.agent.qValues)}")
                elif hasattr(self.agent, 'weights'):
                    print(f"  Características aprendidas: {len(self.agent.weights)}")
                print()
            
            # Guardar checkpoint
            if save_path and save_every > 0 and (episode + 1) % save_every == 0:
                self.save_agent(f"{save_path}_episode_{episode + 1}.pkl")
        
        total_time = time.time() - start_time
        
        # Estadísticas finales
        stats = {
            'total_episodes': self.num_episodes,
            'total_time': total_time,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': self.wins / self.num_episodes if self.num_episodes > 0 else 0,
            'avg_reward': sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0,
            'avg_length': sum(self.episode_lengths) / len(self.episode_lengths) if self.episode_lengths else 0,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }
        
        if verbose:
            print("\n" + "=" * 50)
            print("ENTRENAMIENTO COMPLETADO")
            print("=" * 50)
            print(f"Episodios totales: {stats['total_episodes']}")
            print(f"Tiempo total: {stats['total_time']:.1f} segundos")
            print(f"Victorias: {stats['wins']} ({stats['win_rate']*100:.1f}%)")
            print(f"Derrotas: {stats['losses']}")
            print(f"Recompensa promedio: {stats['avg_reward']:.2f}")
            print(f"Duración promedio: {stats['avg_length']:.1f} pasos")
        
        return stats
    
    def test(self, num_episodes=100, render=False, verbose=True):
        """
        Prueba el agente entrenado sin aprendizaje.
        
        Args:
            num_episodes: Número de episodios de prueba
            render: Si True, renderiza el juego
            verbose: Si True, muestra información
            
        Returns:
            dict: Estadísticas de la prueba
        """
        # Desactivar aprendizaje
        original_epsilon = self.agent.epsilon
        original_alpha = self.agent.alpha
        self.agent.epsilon = 0.0
        self.agent.alpha = 0.0
        
        wins = 0
        losses = 0
        total_rewards = []
        total_lengths = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                legal_actions = self.env.get_legal_actions(0)
                
                if not legal_actions:
                    break
                
                self.agent.actionFn = lambda s: legal_actions
                action = self.agent.getAction(state)
                
                if action is None:
                    break
                
                next_state, reward, done, info = self.env.step(action)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if render:
                    self.env.render()
                    time.sleep(0.2)
            
            total_rewards.append(total_reward)
            total_lengths.append(steps)
            
            if info.get('win', False):
                wins += 1
            elif info.get('lose', False):
                losses += 1
        
        # Restaurar parámetros
        self.agent.epsilon = original_epsilon
        self.agent.alpha = original_alpha
        
        stats = {
            'episodes': num_episodes,
            'wins': wins,
            'losses': losses,
            'win_rate': wins / num_episodes if num_episodes > 0 else 0,
            'avg_reward': sum(total_rewards) / len(total_rewards) if total_rewards else 0,
            'avg_length': sum(total_lengths) / len(total_lengths) if total_lengths else 0
        }
        
        if verbose:
            print("\n" + "=" * 50)
            print("RESULTADOS DE PRUEBA")
            print("=" * 50)
            print(f"Episodios: {stats['episodes']}")
            print(f"Victorias: {stats['wins']} ({stats['win_rate']*100:.1f}%)")
            print(f"Derrotas: {stats['losses']}")
            print(f"Recompensa promedio: {stats['avg_reward']:.2f}")
            print(f"Duración promedio: {stats['avg_length']:.1f} pasos")
        
        return stats
    
    def save_agent(self, filename):
        """Guarda el agente en un archivo."""
        data = {
            'epsilon': self.agent.epsilon,
            'alpha': self.agent.alpha,
            'discount': self.agent.discount,
            'episodesSoFar': self.agent.episodesSoFar,
        }
        
        if hasattr(self.agent, 'qValues'):
            data['qValues'] = dict(self.agent.qValues)
        
        if hasattr(self.agent, 'weights'):
            data['weights'] = dict(self.agent.weights)
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Agente guardado en {filename}")
    
    def load_agent(self, filename):
        """Carga el agente desde un archivo."""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            self.agent.epsilon = data.get('epsilon', self.agent.epsilon)
            self.agent.alpha = data.get('alpha', self.agent.alpha)
            self.agent.discount = data.get('discount', self.agent.discount)
            self.agent.episodesSoFar = data.get('episodesSoFar', 0)
            
            if 'qValues' in data and hasattr(self.agent, 'qValues'):
                from utils.util import Counter
                self.agent.qValues = Counter()
                self.agent.qValues.update(data['qValues'])
            
            if 'weights' in data and hasattr(self.agent, 'weights'):
                from utils.util import Counter
                self.agent.weights = Counter()
                self.agent.weights.update(data['weights'])
            
            print(f"Agente cargado desde {filename}")
            
        except FileNotFoundError:
            print(f"Archivo {filename} no encontrado")
        except Exception as e:
            print(f"Error al cargar agente: {e}")


def main():
    """Función principal para entrenar desde línea de comandos."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenar agente Q-Learning para BattleCity')
    parser.add_argument('--level', type=int, default=1, help='Nivel del juego (1-4)')
    parser.add_argument('--episodes', type=int, default=1000, help='Número de episodios')
    parser.add_argument('--epsilon', type=float, default=0.3, help='Tasa de exploración')
    parser.add_argument('--alpha', type=float, default=0.2, help='Tasa de aprendizaje')
    parser.add_argument('--gamma', type=float, default=0.9, help='Factor de descuento')
    parser.add_argument('--approximate', action='store_true', help='Usar ApproximateQAgent')
    parser.add_argument('--save', type=str, default='battlecity_agent', help='Nombre para guardar')
    parser.add_argument('--load', type=str, default=None, help='Cargar agente existente')
    parser.add_argument('--test', type=int, default=0, help='Episodios de prueba después de entrenar')
    parser.add_argument('--render', action='store_true', help='Mostrar juego durante pruebas')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("BattleCity Q-Learning Trainer")
    print("=" * 50)
    print(f"Nivel: {args.level}")
    print(f"Episodios: {args.episodes}")
    print(f"Epsilon: {args.epsilon}, Alpha: {args.alpha}, Gamma: {args.gamma}")
    print(f"Tipo de agente: {'ApproximateQAgent' if args.approximate else 'BattleCityQAgent'}")
    print()
    
    # Crear entrenador
    trainer = QLearningTrainer(
        level=args.level,
        epsilon=args.epsilon,
        alpha=args.alpha,
        gamma=args.gamma,
        num_episodes=args.episodes,
        use_approximate=args.approximate
    )
    
    # Cargar agente existente si se especifica
    if args.load:
        trainer.load_agent(args.load)
    
    # Entrenar
    stats = trainer.train(
        verbose=True,
        save_every=100,
        save_path=args.save
    )
    
    # Guardar agente final
    trainer.save_agent(f"{args.save}_final.pkl")
    
    # Probar si se especifica
    if args.test > 0:
        print(f"\nProbando agente con {args.test} episodios...")
        trainer.test(num_episodes=args.test, render=args.render)


if __name__ == '__main__':
    main()
