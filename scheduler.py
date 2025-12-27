"""
scheduler.py
HEFT and WOA Algorithms for Hybrid Scheduling
Author: Mustapha Ossama Abdelhalim
Cairo University - December 2025
"""

import numpy as np
import random
from typing import List, Dict, Tuple
from models import Task, Processor, ScheduleResult


class HEFTScheduler:
    """Heterogeneous Earliest Finish Time Algorithm"""
    
    def __init__(self, tasks: List[Task], processors: List[Processor], 
                 comm_matrix: np.ndarray):
        self.tasks = tasks
        self.processors = processors
        self.comm_matrix = comm_matrix
        self.schedule = {}
        
    def compute_rank(self) -> None:
        """Compute upward rank for task prioritization"""
        rank_cache = {}
        
        def rank_recursive(task_id: int) -> float:
            if task_id in rank_cache:
                return rank_cache[task_id]
            
            task = self.tasks[task_id]
            avg_comp = np.mean(task.computation_cost)
            
            if not task.successors:
                rank_value = avg_comp
            else:
                max_succ_rank = 0
                for succ_id in task.successors:
                    avg_comm = np.mean(self.comm_matrix[task_id][succ_id])
                    succ_rank = rank_recursive(succ_id)
                    max_succ_rank = max(max_succ_rank, avg_comm + succ_rank)
                rank_value = avg_comp + max_succ_rank
            
            rank_cache[task_id] = rank_value
            return rank_value
        
        for task in self.tasks:
            task.rank = rank_recursive(task.id)
    
    def schedule_tasks(self) -> ScheduleResult:
        """Execute HEFT scheduling"""
        self.compute_rank()
        sorted_tasks = sorted(self.tasks, key=lambda t: t.rank, reverse=True)
        
        # Reset processors
        for proc in self.processors:
            proc.availability = 0.0
        
        for task in sorted_tasks:
            best_proc = None
            min_eft = float('inf')
            best_start = 0
            
            for proc in self.processors:
                est = proc.availability
                
                for pred_id in task.predecessors:
                    if pred_id in self.schedule:
                        pred_proc_id, _, pred_ft = self.schedule[pred_id]
                        comm_cost = (self.comm_matrix[pred_id][task.id][pred_proc_id]
                                    if pred_proc_id != proc.id else 0)
                        est = max(est, pred_ft + comm_cost)
                
                eft = est + task.computation_cost[proc.id]
                
                if eft < min_eft:
                    min_eft = eft
                    best_proc = proc
                    best_start = est
            
            self.schedule[task.id] = (best_proc.id, best_start, min_eft)
            best_proc.availability = min_eft
        
        return self._create_result()
    
    def _create_result(self) -> ScheduleResult:
        """Package results"""
        makespan = max(ft for _, _, ft in self.schedule.values())
        
        total_cost = 0.0
        total_energy = 0.0
        
        for task_id, (proc_id, start, finish) in self.schedule.items():
            duration = finish - start
            proc = self.processors[proc_id]
            total_cost += duration * proc.speed_factor
            total_energy += duration * proc.energy_coefficient
        
        return ScheduleResult(
            schedule=self.schedule.copy(),
            makespan=makespan,
            total_cost=total_cost,
            total_energy=total_energy,
            algorithm="HEFT"
        )


class WhaleOptimizationAlgorithm:
    """Whale Optimization Algorithm for schedule refinement"""
    
    def __init__(self, tasks: List[Task], processors: List[Processor],
                 comm_matrix: np.ndarray, n_whales: int = 30, max_iter: int = 80):
        self.tasks = tasks
        self.processors = processors
        self.comm_matrix = comm_matrix
        self.n_whales = n_whales
        self.max_iter = max_iter
        self.dim = len(tasks)
        self.n_proc = len(processors)
        
    def initialize_population(self, heft_solution: Dict) -> np.ndarray:
        """Initialize whale population"""
        population = np.zeros((self.n_whales, self.dim), dtype=int)
        
        # First whale is HEFT solution
        for task_id, (proc_id, _, _) in heft_solution.items():
            population[0][task_id] = proc_id
        
        # Generate diverse variations
        for i in range(1, self.n_whales):
            if random.random() < 0.5:
                # Perturb HEFT solution
                population[i] = population[0].copy()
                n_changes = random.randint(self.dim // 3, self.dim // 2)
                for _ in range(n_changes):
                    idx = random.randint(0, self.dim - 1)
                    population[i][idx] = random.randint(0, self.n_proc - 1)
            else:
                # Random solution
                population[i] = np.random.randint(0, self.n_proc, self.dim)
        
        return population
    
    def calculate_fitness(self, solution: np.ndarray) -> float:
        """Calculate makespan for a solution"""
        task_schedule = {}
        proc_availability = [0.0] * self.n_proc
        
        sorted_tasks = sorted(self.tasks, key=lambda t: t.rank, reverse=True)
        
        for task in sorted_tasks:
            proc_id = solution[task.id]
            est = proc_availability[proc_id]
            
            for pred_id in task.predecessors:
                if pred_id in task_schedule:
                    pred_proc_id = solution[pred_id]
                    _, _, pred_ft = task_schedule[pred_id]
                    comm_cost = (self.comm_matrix[pred_id][task.id][pred_proc_id]
                                if pred_proc_id != proc_id else 0)
                    est = max(est, pred_ft + comm_cost)
            
            exec_time = task.computation_cost[proc_id]
            finish_time = est + exec_time
            
            task_schedule[task.id] = (proc_id, est, finish_time)
            proc_availability[proc_id] = finish_time
        
        return max(proc_availability)
    
    def optimize(self, initial_solution: Dict) -> Tuple[np.ndarray, float, List[float]]:
        """Main WOA optimization loop"""
        population = self.initialize_population(initial_solution)
        
        # Evaluate initial population
        fitness = np.array([self.calculate_fitness(sol) for sol in population])
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        convergence_history = [best_fitness]
        
        print(f"  Initial best makespan: {best_fitness:.2f}")
        
        for iteration in range(self.max_iter):
            a = 2.0 - iteration * (2.0 / self.max_iter)
            
            for i in range(self.n_whales):
                r1, r2 = random.random(), random.random()
                A = 2 * a * r1 - a
                C = 2 * r2
                p = random.random()
                l = random.uniform(-1, 1)
                
                if p < 0.5:
                    if abs(A) < 1:
                        # Encircling prey
                        D = np.abs(C * best_solution - population[i])
                        population[i] = best_solution - A * D
                    else:
                        # Search for prey
                        rand_idx = random.randint(0, self.n_whales - 1)
                        D = np.abs(C * population[rand_idx] - population[i])
                        population[i] = population[rand_idx] - A * D
                else:
                    # Spiral updating
                    D = np.abs(best_solution - population[i])
                    population[i] = (D * np.exp(l) * np.cos(2 * np.pi * l) + 
                                    best_solution)
                
                population[i] = np.clip(population[i], 0, self.n_proc - 1).astype(int)
                
                new_fitness = self.calculate_fitness(population[i])
                
                if new_fitness < best_fitness:
                    best_solution = population[i].copy()
                    best_fitness = new_fitness
                    print(f"  Iteration {iteration+1}: New best = {best_fitness:.2f} "
                          f"({((convergence_history[0] - best_fitness) / convergence_history[0] * 100):.2f}% improvement)")
            
            convergence_history.append(best_fitness)
        
        return best_solution, best_fitness, convergence_history
    
    def solution_to_schedule(self, solution: np.ndarray) -> Dict:
        """Convert solution to schedule dictionary"""
        schedule = {}
        proc_availability = [0.0] * self.n_proc
        sorted_tasks = sorted(self.tasks, key=lambda t: t.rank, reverse=True)
        
        for task in sorted_tasks:
            proc_id = solution[task.id]
            est = proc_availability[proc_id]
            
            for pred_id in task.predecessors:
                if pred_id in schedule:
                    pred_proc_id = solution[pred_id]
                    _, _, pred_ft = schedule[pred_id]
                    comm_cost = (self.comm_matrix[pred_id][task.id][pred_proc_id]
                                if pred_proc_id != proc_id else 0)
                    est = max(est, pred_ft + comm_cost)
            
            finish_time = est + task.computation_cost[proc_id]
            schedule[task.id] = (proc_id, est, finish_time)
            proc_availability[proc_id] = finish_time
        
        return schedule
    
    def create_result(self, solution: np.ndarray, makespan: float) -> ScheduleResult:
        """Create result object from solution"""
        schedule = self.solution_to_schedule(solution)
        
        total_cost = 0.0
        total_energy = 0.0
        
        for task_id, (proc_id, start, finish) in schedule.items():
            duration = finish - start
            proc = self.processors[proc_id]
            total_cost += duration * proc.speed_factor
            total_energy += duration * proc.energy_coefficient
        
        return ScheduleResult(
            schedule=schedule,
            makespan=makespan,
            total_cost=total_cost,
            total_energy=total_energy,
            algorithm="Hybrid HEFT-WOA"
        )