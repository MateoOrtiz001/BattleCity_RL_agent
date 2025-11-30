#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_agent.py
--------------
Script principal para entrenar un agente de Q-Learning en BattleCity.

Uso:
    python train_agent.py                           # Entrenar con parámetros por defecto
    python train_agent.py --episodes 5000           # Entrenar 5000 episodios
    python train_agent.py --approximate             # Usar ApproximateQAgent
    python train_agent.py --level 2                 # Entrenar en nivel 2
    python train_agent.py --test 100 --render       # Probar con visualización
"""

import sys
import os

# Configurar paths correctamente
project_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_dir, 'src')

# Agregar directorios al path
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src.training.trainer import QLearningTrainer


def train_basic_agent():
    """
    Entrena un agente básico de Q-Learning tabular.
    Bueno para estados discretos simples.
    """
    print("\n" + "="*60)
    print("ENTRENAMIENTO - Q-Learning Tabular")
    print("="*60 + "\n")
    
    trainer = QLearningTrainer(
        level=1,
        epsilon=0.3,        # 30% exploración inicial
        gamma=0.9,          # Factor de descuento
        alpha=0.2,          # Tasa de aprendizaje
        num_episodes=1000,  # Episodios de entrenamiento
        use_approximate=False
    )
    
    stats = trainer.train(
        verbose=True,
        save_every=200,
        save_path='models/qlearning_basic'
    )
    
    trainer.save_agent('models/qlearning_basic_final.pkl')
    
    # Probar el agente entrenado
    print("\n" + "="*60)
    print("PROBANDO AGENTE ENTRENADO")
    print("="*60 + "\n")
    
    trainer.test(num_episodes=50, verbose=True)
    
    return trainer, stats


def train_approximate_agent():
    """
    Entrena un agente con aproximación de función.
    Mejor para generalizar a estados no vistos.
    """
    print("\n" + "="*60)
    print("ENTRENAMIENTO - Approximate Q-Learning")
    print("="*60 + "\n")
    
    trainer = QLearningTrainer(
        level=1,
        epsilon=0.4,        # Más exploración para aprender características
        gamma=0.95,         # Factor de descuento alto
        alpha=0.1,          # Tasa de aprendizaje más baja
        num_episodes=2000,  # Más episodios
        use_approximate=True
    )
    
    stats = trainer.train(
        verbose=True,
        save_every=500,
        save_path='models/approximate_agent'
    )
    
    trainer.save_agent('models/approximate_agent_final.pkl')
    
    # Probar el agente entrenado
    print("\n" + "="*60)
    print("PROBANDO AGENTE ENTRENADO")
    print("="*60 + "\n")
    
    trainer.test(num_episodes=50, verbose=True)
    
    return trainer, stats


def quick_demo():
    """
    Demo rápido de entrenamiento (pocos episodios).
    """
    print("\n" + "="*60)
    print("DEMO RÁPIDO - Entrenamiento de 100 episodios")
    print("="*60 + "\n")
    
    trainer = QLearningTrainer(
        level=1,
        epsilon=0.5,
        gamma=0.9,
        alpha=0.3,
        num_episodes=100,
        use_approximate=False
    )
    
    stats = trainer.train(verbose=True)
    
    print("\nProbando agente...")
    trainer.test(num_episodes=10, verbose=True)
    
    return trainer, stats


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenar agente Q-Learning para BattleCity')
    parser.add_argument('--mode', type=str, default='demo', 
                        choices=['demo', 'basic', 'approximate'],
                        help='Modo de entrenamiento')
    parser.add_argument('--level', type=int, default=1, 
                        help='Nivel del juego (1-4)')
    parser.add_argument('--episodes', type=int, default=None, 
                        help='Número de episodios de entrenamiento')
    parser.add_argument('--epsilon', type=float, default=0.3, 
                        help='Tasa de exploración')
    parser.add_argument('--alpha', type=float, default=0.2, 
                        help='Tasa de aprendizaje')
    parser.add_argument('--gamma', type=float, default=0.9, 
                        help='Factor de descuento')
    parser.add_argument('--test', type=int, default=50, 
                        help='Episodios de prueba después de entrenar')
    parser.add_argument('--save', type=str, default='models/agent', 
                        help='Ruta para guardar el agente')
    parser.add_argument('--save-every', type=int, default=500, 
                        help='Guardar modelo cada N episodios')
    
    args = parser.parse_args()
    
    # Crear directorio de modelos si no existe
    os.makedirs('models', exist_ok=True)
    
    if args.mode == 'demo':
        quick_demo()
    elif args.mode == 'basic':
        if args.episodes:
            # Entrenamiento personalizado básico
            trainer = QLearningTrainer(
                level=args.level,
                epsilon=args.epsilon,
                gamma=args.gamma,
                alpha=args.alpha,
                num_episodes=args.episodes,
                use_approximate=False
            )
            
            trainer.train(
                verbose=True,
                save_every=args.save_every,
                save_path=args.save
            )
            
            trainer.save_agent(f"{args.save}_final.pkl")
            
            if args.test > 0:
                trainer.test(num_episodes=args.test)
        else:
            train_basic_agent()
    elif args.mode == 'approximate':
        if args.episodes:
            # Entrenamiento personalizado aproximado
            print("\n" + "="*60)
            print(f"ENTRENAMIENTO - Approximate Q-Learning ({args.episodes} episodios)")
            print("="*60 + "\n")
            
            trainer = QLearningTrainer(
                level=args.level,
                epsilon=args.epsilon,
                gamma=args.gamma,
                alpha=args.alpha,
                num_episodes=args.episodes,
                use_approximate=True
            )
            
            trainer.train(
                verbose=True,
                save_every=args.save_every,
                save_path=args.save
            )
            
            trainer.save_agent(f"{args.save}_final.pkl")
            
            if args.test > 0:
                print("\n" + "="*60)
                print("PROBANDO AGENTE ENTRENADO")
                print("="*60 + "\n")
                trainer.test(num_episodes=args.test)
        else:
            train_approximate_agent()
    else:
        # Entrenamiento personalizado
        trainer = QLearningTrainer(
            level=args.level,
            epsilon=args.epsilon,
            gamma=args.gamma,
            alpha=args.alpha,
            num_episodes=args.episodes or 1000,
            use_approximate=(args.mode == 'approximate')
        )
        
        trainer.train(
            verbose=True,
            save_every=args.save_every,
            save_path=args.save
        )
        
        trainer.save_agent(f"{args.save}_final.pkl")
        
        if args.test > 0:
            trainer.test(num_episodes=args.test)
