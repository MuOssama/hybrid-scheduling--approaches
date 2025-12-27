"""
models.py
Data structures for Hybrid HEFT-WOA Scheduler
Author: Mustapha Ossama Abdelhalim
Cairo University - December 2025
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Task:
    """Represents a computational task"""
    id: int
    computation_cost: List[float]  # Cost on each processor
    successors: List[int] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)
    rank: float = 0.0


@dataclass
class Processor:
    """Represents a heterogeneous computing resource"""
    id: int
    speed_factor: float
    energy_coefficient: float
    availability: float = 0.0


@dataclass
class ScheduleResult:
    """Results from scheduling"""
    schedule: Dict  # {task_id: (processor_id, start_time, finish_time)}
    makespan: float
    total_cost: float
    total_energy: float
    algorithm: str
    
    def improvement_over(self, baseline: 'ScheduleResult') -> float:
        """Calculate percentage improvement"""
        if baseline.makespan == 0:
            return 0.0
        return ((baseline.makespan - self.makespan) / baseline.makespan) * 100