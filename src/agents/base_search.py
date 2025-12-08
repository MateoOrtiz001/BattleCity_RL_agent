# Base class for search agents (Minimax, AlphaBeta, Expectimax)
# Provides common functionality: time management, fallback behavior, node counting

import time
import random


class BaseSearchAgent:
    """Base class for game tree search agents.
    
    Provides:
    - Time limit management with is_time_exceeded()
    - Node expansion counter
    - Fallback to ReflexTankAgent when time exceeded
    """
    
    def __init__(self, depth=1, tankIndex=0, time_limit=1.0):
        self.index = tankIndex  # Tank index this agent controls
        self.depth = int(depth) if isinstance(depth, str) else depth
        self.expanded_nodes = 0
        self.start_time = 0
        self.time_limit = time_limit
    
    def reset_search(self):
        """Reset counters for a new search. Call at start of getAction()."""
        self.start_time = time.time()
        self.expanded_nodes = 0
    
    def increment_nodes(self):
        """Increment the node expansion counter."""
        self.expanded_nodes += 1
    
    def is_time_exceeded(self):
        """Check if the time limit has been exceeded."""
        return (
            self.time_limit is not None
            and (time.time() - self.start_time) > self.time_limit
        )
    
    def get_elapsed_time(self):
        """Return elapsed time since search started."""
        return time.time() - self.start_time if self.start_time else 0.0
    
    def fallback_action(self, gameState, log=True):
        """Return an action from ReflexTankAgent as fallback.
        
        Uses 50/50 probability between offensive and defensive modes.
        """
        try:
            from .reflexAgent import ReflexTankAgent
            rtype = 'offensive' if random.random() < 0.5 else 'defensive'
            reflex = ReflexTankAgent(script_type=rtype)
            
            if log:
                elapsed = self.get_elapsed_time()
                print(f"[FALLBACK] {self.__class__.__name__} exceeded time after "
                      f"{elapsed:.2f}s, nodes={self.expanded_nodes} -> ReflexTankAgent({rtype})")
            
            return reflex.getAction(gameState)
        except Exception:
            # If fallback fails, return STOP
            return 'STOP'
    
    def maybe_fallback(self, gameState):
        """Check if time exceeded and return fallback action if so.
        
        Returns:
            Action string if fallback triggered, None otherwise.
        """
        if self.is_time_exceeded():
            return self.fallback_action(gameState)
        return None
    
    def getAction(self, gameState):
        """Override in subclass to implement the search algorithm."""
        raise NotImplementedError("Subclasses must implement getAction()")
