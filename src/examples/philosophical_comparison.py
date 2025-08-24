"""
Compare the original and fundamental approaches to philosophical reasoning
"""
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import both approaches
from src.reasoning.philosophical_energy import PhilosophicalEnergy
from src.reasoning.fundamental_philosophy import FundamentalPhilosophy
from src.core.unified_energy import UnifiedEnergy

class ComparisonRunner:
    def __init__(self):
        # Initialize energy system
        self.energy_system = UnifiedEnergy()
        
        # Initialize both philosophical systems
        self.original = PhilosophicalEnergy()
        self.fundamental = FundamentalPhilosophy(self.energy_system)
        
        # Track memory and connection growth
        self.original_stats = {
            'iterations': [],
            'memory_usage': [],
            'connection_count': [],
            'processing_time': []
        }
        
        self.fundamental_stats = {
            'iterations': [],
            'memory_usage': [],
            'connection_count': [],
            'processing_time': []
        }
    
    def run_comparison(self, iterations=100):
        """Run comparison between original and fundamental approaches"""
        for i in range(iterations):
            # Generate test concepts
            concepts = self._generate_test_concepts(3)
            
            # Measure original approach
            self._measure_original_approach(concepts, i)
            
            # Measure fundamental approach
            self._measure_fundamental_approach(concepts, i)
            
            # Print progress
            if (i+1) % 10 == 0:
                print(f"Completed {i+1}/{iterations} iterations")
        
        # Plot results
        self._plot_results()
    
    def _generate_test_concepts(self, count):
        """Generate test concepts with energy signatures"""
        concepts = []
        
        for i in range(count):
            # Create concept with random energy signature
            concept = {
                'name': f"concept_{i}_{np.random.randint(1000)}",
                'energy_signature': {
                    'magnitude': {'value': np.random.random()},
                    'frequency': {'value': np.random.random()},
                    'entropy': {'value': np.random.random()},
                    'vector': {'value': [np.random.random() for _ in range(3)]}
                }
            }
            concepts.append(concept)
        
        return concepts
    
    def _measure_original_approach(self, concepts, iteration):
        """Measure performance of original approach"""
        # Measure time
        start_time = time.time()
        
        # Apply reasoning
        reasoning_type = np.random.choice(['dialectical', 'deductive', 'inductive', 
                                          'abductive', 'analogical', 'conceptual_blending'])
        result = self.original._dialectical_reasoning(concepts)
        
        # Measure elapsed time
        elapsed = time.time() - start_time
        
        # Estimate memory usage (simplified)
        memory_usage = sys.getsizeof(str(result))
        
        # Estimate connection count (simplified)
        connection_count = len(concepts) * (len(concepts) - 1) // 2 + len(concepts)
        
        # Record stats
        self.original_stats['iterations'].append(iteration)
        self.original_stats['memory_usage'].append(memory_usage)
        self.original_stats['connection_count'].append(connection_count)
        self.original_stats['processing_time'].append(elapsed)
    
    def _measure_fundamental_approach(self, concepts, iteration):
        """Measure performance of fundamental approach"""
        # Measure time
        start_time = time.time()
        
        # Apply reasoning
        reasoning_type = np.random.choice(['dialectical', 'deductive', 'inductive', 
                                          'abductive', 'analogical', 'hermeneutic'])
        lens_type = np.random.choice(['ontological', 'epistemological', 'ethical', 
                                     'logical', 'metaphysical'])
        result = self.fundamental.apply_reasoning(reasoning_type, concepts, lens_type)
        
        # Measure elapsed time
        elapsed = time.time() - start_time
        
        # Estimate memory usage (simplified)
        memory_usage = sys.getsizeof(str(result))
        
        # Estimate connection count
        connection_count = len(concepts)  # One connection per concept to result
        
        # Record stats
        self.fundamental_stats['iterations'].append(iteration)
        self.fundamental_stats['memory_usage'].append(memory_usage)
        self.fundamental_stats['connection_count'].append(connection_count)
        self.fundamental_stats['processing_time'].append(elapsed)
    
    def _plot_results(self):
        """Plot comparison results"""
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
        # Plot memory usage
        axs[0].plot(self.original_stats['iterations'], self.original_stats['memory_usage'], 
                   'r-', label='Original Approach')
        axs[0].plot(self.fundamental_stats['iterations'], self.fundamental_stats['memory_usage'], 
                   'g-', label='Fundamental Approach')
        axs[0].set_title('Memory Usage Comparison')
        axs[0].set_xlabel('Iterations')
        axs[0].set_ylabel('Memory Usage (bytes)')
        axs[0].legend()
        
        # Plot connection count (cumulative)
        original_cumulative = np.cumsum(self.original_stats['connection_count'])
        fundamental_cumulative = np.cumsum(self.fundamental_stats['connection_count'])
        
        axs[1].plot(self.original_stats['iterations'], original_cumulative, 
                   'r-', label='Original Approach')
        axs[1].plot(self.fundamental_stats['iterations'], fundamental_cumulative, 
                   'g-', label='Fundamental Approach')
        axs[1].set_title('Cumulative Connection Count')
        axs[1].set_xlabel('Iterations')
        axs[1].set_ylabel('Total Connections')
        axs[1].legend()
        
        # Plot processing time
        axs[2].plot(self.original_stats['iterations'], self.original_stats['processing_time'], 
                   'r-', label='Original Approach')
        axs[2].plot(self.fundamental_stats['iterations'], self.fundamental_stats['processing_time'], 
                   'g-', label='Fundamental Approach')
        axs[2].set_title('Processing Time Comparison')
        axs[2].set_xlabel('Iterations')
        axs[2].set_ylabel('Processing Time (seconds)')
        axs[2].legend()
        
        plt.tight_layout()
        plt.savefig('philosophical_comparison.png')
        plt.close()
        
        print("Results plotted to philosophical_comparison.png")

if __name__ == "__main__":
    comparison = ComparisonRunner()
    comparison.run_comparison(iterations=50)