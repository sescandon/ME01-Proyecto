"""
Simulación del Modelo de Segregación de Schelling
=================================================

Este módulo implementa una simulación básica del famoso modelo de Schelling
que demuestra cómo preferencias individuales moderadas pueden llevar a
segregación extrema a nivel colectivo.
"""

import random
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class AgentType(Enum):
    """Tipos de agentes en la simulación"""
    EMPTY = 0
    TYPE_A = 1
    TYPE_B = 2

@dataclass
class Agent:
    """
    Representa un agente individual en la simulación
    
    Attributes:
        agent_type: Tipo del agente (A o B)
        tolerance: Umbral de tolerancia (0.0 - 1.0)
        satisfied: Si el agente está satisfecho con su ubicación actual
    """
    agent_type: AgentType
    tolerance: float
    satisfied: bool = False
    
    def calculate_satisfaction(self, neighbors: List['Agent']) -> bool:
        """
        Calcula si el agente está satisfecho basándose en sus vecinos
        
        Args:
            neighbors: Lista de agentes vecinos
            
        Returns:
            True si está satisfecho, False en caso contrario
        """
        if self.agent_type == AgentType.EMPTY:
            return True
            
        # Filtrar vecinos no vacíos
        non_empty_neighbors = [n for n in neighbors if n.agent_type != AgentType.EMPTY]
        
        if not non_empty_neighbors:
            return True  # Si no hay vecinos, está satisfecho
            
        # Contar vecinos del mismo tipo
        same_type_count = sum(1 for n in non_empty_neighbors if n.agent_type == self.agent_type)
        
        # Calcular proporción de vecinos similares
        similarity_ratio = same_type_count / len(non_empty_neighbors)
        
        # Actualizar estado de satisfacción
        self.satisfied = similarity_ratio >= self.tolerance
        return self.satisfied

class SchellingModel:
    """
    Implementación del modelo de segregación de Schelling
    """
    
    def __init__(self, width: int = 100, height: int = 100, 
                 density: float = 0.8, minority_ratio: float = 0.3,
                 tolerance_a: float = 0.3, tolerance_b: float = 0.3):
        """
        Inicializa el modelo de Schelling
        
        Args:
            width: Ancho de la cuadrícula
            height: Alto de la cuadrícula
            density: Proporción de celdas ocupadas (0.0 - 1.0)
            minority_ratio: Proporción del grupo minoritario (0.0 - 1.0)
            tolerance_a: Tolerancia del grupo A (0.0 - 1.0)
            tolerance_b: Tolerancia del grupo B (0.0 - 1.0)
        """
        self.width = width
        self.height = height
        self.density = density
        self.minority_ratio = minority_ratio
        self.tolerance_a = tolerance_a
        self.tolerance_b = tolerance_b
        
        # Inicializar cuadrícula
        self.grid = [[Agent(AgentType.EMPTY, 0.0) for _ in range(width)] 
                     for _ in range(height)]
        
        # Estadísticas
        self.iteration = 0
        self.satisfaction_history = []
        self.segregation_history = []
        
        self._populate_grid()
    
    def _populate_grid(self):
        """Puebla la cuadrícula inicialmente con agentes distribuidos aleatoriamente"""
        total_cells = self.width * self.height
        occupied_cells = int(total_cells * self.density)
        minority_cells = int(occupied_cells * self.minority_ratio)
        majority_cells = occupied_cells - minority_cells
        
        # Crear lista de posiciones disponibles
        positions = [(i, j) for i in range(self.height) for j in range(self.width)]
        random.shuffle(positions)
        
        # Colocar agentes tipo A (mayoría)
        for i in range(majority_cells):
            row, col = positions[i]
            self.grid[row][col] = Agent(AgentType.TYPE_A, self.tolerance_a)
        
        # Colocar agentes tipo B (minoría)
        for i in range(majority_cells, majority_cells + minority_cells):
            row, col = positions[i]
            self.grid[row][col] = Agent(AgentType.TYPE_B, self.tolerance_b)
    
    def get_neighbors(self, row: int, col: int) -> List[Agent]:
        """
        Obtiene los vecinos de una celda (8-conectividad)
        
        Args:
            row: Fila de la celda
            col: Columna de la celda
            
        Returns:
            Lista de agentes vecinos
        """
        neighbors = []
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:  # Skip la celda central
                    continue
                    
                new_row, new_col = row + dr, col + dc
                
                # Verificar límites
                if (0 <= new_row < self.height and 0 <= new_col < self.width):
                    neighbors.append(self.grid[new_row][new_col])
        
        return neighbors
    
    def update_satisfaction(self):
        """Actualiza el estado de satisfacción de todos los agentes"""
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i][j].agent_type != AgentType.EMPTY:
                    neighbors = self.get_neighbors(i, j)
                    self.grid[i][j].calculate_satisfaction(neighbors)
    
    def get_unsatisfied_agents(self) -> List[Tuple[int, int]]:
        """
        Obtiene lista de posiciones de agentes insatisfechos
        
        Returns:
            Lista de tuplas (fila, columna) de agentes insatisfechos
        """
        unsatisfied = []
        for i in range(self.height):
            for j in range(self.width):
                agent = self.grid[i][j]
                if (agent.agent_type != AgentType.EMPTY and not agent.satisfied):
                    unsatisfied.append((i, j))
        return unsatisfied
    
    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """
        Obtiene lista de celdas vacías
        
        Returns:
            Lista de tuplas (fila, columna) de celdas vacías
        """
        empty = []
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i][j].agent_type == AgentType.EMPTY:
                    empty.append((i, j))
        return empty
    
    def move_agent(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]):
        """
        Mueve un agente de una posición a otra
        
        Args:
            from_pos: Posición origen (fila, columna)
            to_pos: Posición destino (fila, columna)
        """
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        # Mover agente
        self.grid[to_row][to_col] = self.grid[from_row][from_col]
        self.grid[from_row][from_col] = Agent(AgentType.EMPTY, 0.0)
    
    def step(self) -> bool:
        """
        Ejecuta un paso de la simulación
        
        Returns:
            True si se realizaron movimientos, False si converge
        """
        # Actualizar satisfacción
        self.update_satisfaction()
        
        # Obtener agentes insatisfechos y celdas vacías
        unsatisfied = self.get_unsatisfied_agents()
        empty_cells = self.get_empty_cells()
        
        if not unsatisfied or not empty_cells:
            return False  # No hay movimientos posibles
        
        # Realizar movimientos aleatorios
        random.shuffle(unsatisfied)
        movements_made = 0
        
        for agent_pos in unsatisfied:
            if not empty_cells:
                break
                
            # Elegir celda vacía aleatoria
            new_pos = random.choice(empty_cells)
            empty_cells.remove(new_pos)
            empty_cells.append(agent_pos)  # La antigua posición queda vacía
            
            # Mover agente
            self.move_agent(agent_pos, new_pos)
            movements_made += 1
        
        self.iteration += 1
        return movements_made > 0
    
    def calculate_satisfaction_rate(self) -> float:
        """
        Calcula el porcentaje de agentes satisfechos
        
        Returns:
            Tasa de satisfacción (0.0 - 1.0)
        """
        total_agents = 0
        satisfied_agents = 0
        
        for i in range(self.height):
            for j in range(self.width):
                agent = self.grid[i][j]
                if agent.agent_type != AgentType.EMPTY:
                    total_agents += 1
                    if agent.satisfied:
                        satisfied_agents += 1
        
        return satisfied_agents / total_agents if total_agents > 0 else 0.0
    
    def calculate_segregation_index(self) -> float:
        """
        Calcula un índice simple de segregación basado en homogeneidad local
        
        Returns:
            Índice de segregación (0.0 - 1.0)
        """
        total_similarity = 0.0
        agent_count = 0
        
        for i in range(self.height):
            for j in range(self.width):
                agent = self.grid[i][j]
                if agent.agent_type != AgentType.EMPTY:
                    neighbors = self.get_neighbors(i, j)
                    non_empty_neighbors = [n for n in neighbors if n.agent_type != AgentType.EMPTY]
                    
                    if non_empty_neighbors:
                        same_type = sum(1 for n in non_empty_neighbors 
                                      if n.agent_type == agent.agent_type)
                        similarity = same_type / len(non_empty_neighbors)
                        total_similarity += similarity
                        agent_count += 1
        
        return total_similarity / agent_count if agent_count > 0 else 0.0
    
    def run_simulation(self, max_iterations: int = 1000) -> dict:
        """
        Ejecuta la simulación completa
        
        Args:
            max_iterations: Número máximo de iteraciones
            
        Returns:
            Diccionario con estadísticas de la simulación
        """
        print("Iniciando simulación de Schelling...")
        print(f"Configuración: {self.width}x{self.height}, densidad={self.density:.2f}")
        print(f"Tolerancia A={self.tolerance_a:.2f}, B={self.tolerance_b:.2f}")
        print("-" * 50)
        
        # Estado inicial
        self.update_satisfaction()
        initial_satisfaction = self.calculate_satisfaction_rate()
        initial_segregation = self.calculate_segregation_index()
        
        self.satisfaction_history.append(initial_satisfaction)
        self.segregation_history.append(initial_segregation)
        
        print(f"Estado inicial - Satisfacción: {initial_satisfaction:.3f}, "
              f"Segregación: {initial_segregation:.3f}")
        
        # Ejecutar simulación
        for iteration in range(max_iterations):
            if not self.step():
                print(f"Convergencia alcanzada en iteración {iteration}")
                break
                
            # Registrar estadísticas cada 10 iteraciones
            if iteration % 10 == 0:
                satisfaction = self.calculate_satisfaction_rate()
                segregation = self.calculate_segregation_index()
                self.satisfaction_history.append(satisfaction)
                self.segregation_history.append(segregation)
                
                print(f"Iteración {iteration:3d} - Satisfacción: {satisfaction:.3f}, "
                      f"Segregación: {segregation:.3f}")
        
        # Estado final
        final_satisfaction = self.calculate_satisfaction_rate()
        final_segregation = self.calculate_segregation_index()
        
        print("-" * 50)
        print(f"Estado final - Satisfacción: {final_satisfaction:.3f}, "
              f"Segregación: {final_segregation:.3f}")
        
        return {
            'iterations': self.iteration,
            'initial_satisfaction': initial_satisfaction,
            'final_satisfaction': final_satisfaction,
            'initial_segregation': initial_segregation,
            'final_segregation': final_segregation,
            'satisfaction_history': self.satisfaction_history,
            'segregation_history': self.segregation_history
        }
    
    def visualize(self, title: str = "Modelo de Schelling"):
        """
        Visualiza el estado actual de la simulación
        
        Args:
            title: Título del gráfico
        """
        # Crear matriz para visualización
        vis_grid = np.zeros((self.height, self.width))
        
        for i in range(self.height):
            for j in range(self.width):
                agent = self.grid[i][j]
                if agent.agent_type == AgentType.TYPE_A:
                    vis_grid[i][j] = 1
                elif agent.agent_type == AgentType.TYPE_B:
                    vis_grid[i][j] = 2
                # Empty cells remain 0
        
        # Crear visualización
        plt.figure(figsize=(10, 8))
        
        # Mapa principal
        plt.subplot(2, 2, (1, 2))
        colors = ['white', 'blue', 'red']
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        plt.imshow(vis_grid, cmap=cmap, vmin=0, vmax=2)
        plt.title(f'{title} - Iteración {self.iteration}')
        plt.colorbar(ticks=[0, 1, 2], label='Tipo de Agente')
        
        # Gráfico de satisfacción
        if len(self.satisfaction_history) > 1:
            plt.subplot(2, 2, 3)
            plt.plot(self.satisfaction_history, 'g-', linewidth=2)
            plt.title('Tasa de Satisfacción')
            plt.xlabel('Iteración (x10)')
            plt.ylabel('Satisfacción')
            plt.grid(True, alpha=0.3)
        
        # Gráfico de segregación
        if len(self.segregation_history) > 1:
            plt.subplot(2, 2, 4)
            plt.plot(self.segregation_history, 'r-', linewidth=2)
            plt.title('Índice de Segregación')
            plt.xlabel('Iteración (x10)')
            plt.ylabel('Segregación')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def main():
    """Función principal para ejecutar una simulación de ejemplo"""
    
    # Crear modelo con parámetros por defecto
    model = SchellingModel(
        width=100,
        height=100,
        density=0.85,
        minority_ratio=0.4,
        tolerance_a=0.3,
        tolerance_b=0.3
    )
    
    # Visualizar estado inicial
    print("Estado inicial:")
    model.visualize("Estado Inicial")
    
    # Ejecutar simulación
    results = model.run_simulation(max_iterations=200)
    
    # Visualizar estado final
    print("\nEstado final:")
    model.visualize("Estado Final")
    
    # Mostrar resumen
    print("\n" + "="*60)
    print("RESUMEN DE LA SIMULACIÓN")
    print("="*60)
    print(f"Iteraciones ejecutadas: {results['iterations']}")
    print(f"Cambio en satisfacción: {results['initial_satisfaction']:.3f} → {results['final_satisfaction']:.3f}")
    print(f"Cambio en segregación: {results['initial_segregation']:.3f} → {results['final_segregation']:.3f}")
    print("\nEsto demuestra la paradoja de Schelling: preferencias individuales")
    print("moderadas (30% tolerancia) resultan en alta segregación!")


if __name__ == "__main__":
    main()