"""
BattleCity RL Agent - Módulo principal

Este módulo configura los paths del proyecto para permitir imports relativos
correctos desde cualquier punto del proyecto.
"""

import os
import sys

# Configurar paths una sola vez al importar el paquete src
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_current_dir)

if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# Limpiar variables temporales del namespace
del _current_dir, _project_dir
