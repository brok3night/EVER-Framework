"""
Self-Modification System - Enables EVER to adapt its processing over time
"""
from typing import Dict, List, Any, Callable
import numpy as np
import copy
import time
import json

class SelfModificationSystem:
    """Enables EVER to modify its own processing capabilities"""
    
    def __init__(self, config_path=None):
        # Core components that can be modified
        self.modifiable_components = {}
        
        # Modification history
        self.modification_history = []
        
        # Performance tracking
        self.performance_metrics = {}
        
        # Safety constraints
        self.safety_constraints = {
            'max_modification_rate': 0.05,  # Max 5% change per iteration
            'modification_cooldown': 60,    # Seconds between modifications
            'stability_threshold': 0.8,     # Performance must be above threshold
            'reversibility': True           # Modifications must be reversible
        }
        
        # Last modification timestamp
        self.last_modification_time = 0
        
        # Load configuration if provided
        if config_path:
            self._load_configuration(config_path)
    
    def register_component(self, component_id: str, component: Any,
                          modifiable_attributes: Dict[str, Dict],
                          modification_handler: Callable = None) -> None:
        """
        Register a component for potential modification
        
        Args:
            component_id: Identifier for the component
            component: The component object
            modifiable_attributes: Dict mapping attribute names to constraints
            modification_handler: Optional function to handle modifications
        """
        self.modifiable_components[component_id] = {
            'component': component,
            'attributes': modifiable_attributes,
            'handler': modification_handler,
            'original_state': self._capture_state(component, modifiable_attributes),
            'current_state': self._capture_state(component, modifiable_attributes),
            'modifications': []
        }
    
    def record_performance(self, metric_id: str, value: float) -> None:
        """
        Record performance metric
        
        Args:
            metric_id: Identifier for the metric
            value: Performance value
        """
        if metric_id not in self.performance_metrics:
            self.performance_metrics[metric_id] = {
                'values': [],
                'timestamps': []
            }
        
        self.performance_metrics[metric_id]['values'].append(value)
        self.performance_metrics[metric_id]['timestamps'].append(time.time())
    
    def evaluate_modification_needs(self) -> Dict:
        """Evaluate whether modifications are needed based on performance"""
        # Check if we're in cooldown period
        current_time = time.time()
        if current_time - self.last_modification_time < self.safety_constraints['modification_cooldown']:
            return {
                'modification_needed': False,
                'reason': 'Cooldown period active'
            }
        
        # Check if we have enough performance data
        if not self._has_sufficient_performance_data():
            return {
                'modification_needed': False,
                'reason': 'Insufficient performance data'
            }
        
        # Analyze performance trends
        declining_metrics = []
        improving_metrics = []
        
        for metric_id, data in self.performance_metrics.items():
            trend = self._analyze_performance_trend(data['values'])
            
            if trend < -0.1:  # Declining performance
                declining_metrics.append(metric_id)
            elif trend > 0.1:  # Improving performance
                improving_metrics.append(metric_id)
        
        # Determine if modification is needed
        if declining_metrics:
            return {
                'modification_needed': True,
                'reason': 'Declining performance',
                'metrics': declining_metrics,
                'trend': 'declining'
            }
        elif not improving_metrics and self.modification_history:
            # No improvement after previous modification
            return {
                'modification_needed': True,
                'reason': 'Stagnant performance',
                'metrics': list(self.performance_metrics.keys()),
                'trend': 'stagnant'
            }
        
        return {
            'modification_needed': False,
            'reason': 'Performance satisfactory',
            'improving_metrics': improving_metrics
        }
    
    def generate_modification_plan(self, evaluation: Dict) -> Dict:
        """Generate a plan for system modification"""
        if not evaluation.get('modification_needed', False):
            return {'plan': None}
        
        # Identify components to modify based on metrics
        target_components = self._identify_components_for_metrics(evaluation.get('metrics', []))
        
        if not target_components:
            return {'plan': None, 'reason': 'No suitable components identified'}
        
        # Generate modification plan
        plan = {
            'components': {},
            'expected_impact': {},
            'reversion_plan': {}
        }
        
        for component_id in target_components:
            if component_id in self.modifiable_components:
                component_info = self.modifiable_components[component_id]
                
                # Identify attributes to modify
                attributes_to_modify = self._select_attributes_to_modify(
                    component_info, evaluation.get('trend', 'declining'))
                
                if attributes_to_modify:
                    plan['components'][component_id] = attributes_to_modify
                    
                    # Generate reversion plan
                    plan['reversion_plan'][component_id] = {
                        attr: component_info['current_state'][attr]
                        for attr in attributes_to_modify
                    }
                    
                    # Estimate impact
                    plan['expected_impact'][component_id] = self._estimate_modification_impact(
                        component_id, attributes_to_modify)
        
        return {'plan': plan if plan['components'] else None}
    
    def apply_modification(self, plan: Dict) -> Dict:
        """
        Apply a modification plan
        
        Args:
            plan: Modification plan to apply
        """
        if not plan or 'components' not in plan:
            return {'success': False, 'reason': 'Invalid plan'}
        
        # Check safety constraints
        if not self._validate_safety_constraints(plan):
            return {
                'success': False,
                'reason': 'Safety constraints violated'
            }
        
        # Apply modifications
        results = {}
        successful_mods = []
        failed_mods = []
        
        for component_id, attributes in plan['components'].items():
            if component_id in self.modifiable_components:
                component_info = self.modifiable_components[component_id]
                component = component_info['component']
                
                # Apply each attribute modification
                component_results = {}
                
                for attr_name, new_value in attributes.items():
                    try:
                        # Check if attribute is modifiable
                        if attr_name in component_info['attributes']:
                            # Get current value
                            current_value = self._get_attribute_value(component, attr_name)
                            
                            # Apply modification
                            if component_info['handler']:
                                # Use custom handler
                                success = component_info['handler'](attr_name, new_value)
                            else:
                                # Direct modification
                                self._set_attribute_value(component, attr_name, new_value)
                                success = True
                            
                            if success:
                                component_results[attr_name] = {
                                    'success': True,
                                    'old_value': current_value,
                                    'new_value': new_value
                                }
                                
                                # Update current state
                                component_info['current_state'][attr_name] = new_value
                                
                                # Record modification
                                mod_record = {
                                    'timestamp': time.time(),
                                    'component': component_id,
                                    'attribute': attr_name,
                                    'old_value': current_value,
                                    'new_value': new_value
                                }
                                component_info['modifications'].append(mod_record)
                                self.modification_history.append(mod_record)
                                
                                successful_mods.append(f"{component_id}.{attr_name}")
                            else:
                                component_results[attr_name] = {
                                    'success': False,
                                    'reason': 'Modification handler failed'
                                }
                                failed_mods.append(f"{component_id}.{attr_name}")
                        else:
                            component_results[attr_name] = {
                                'success': False,
                                'reason': 'Attribute not modifiable'
                            }
                            failed_mods.append(f"{component_id}.{attr_name}")
                    except Exception as e:
                        component_results[attr_name] = {
                            'success': False,
                            'reason': str(e)
                        }
                        failed_mods.append(f"{component_id}.{attr_name}")
                
                results[component_id] = component_results
        
        # Update last modification time
        if successful_mods:
            self.last_modification_time = time.time()
        
        return {
            'success': len(successful_mods) > 0,
            'modified': successful_mods,
            'failed': failed_mods,
            'results': results
        }
    
    def revert_modification(self, modification_id: int = None) -> Dict:
        """
        Revert a previous modification
        
        Args:
            modification_id: ID of modification to revert (None = most recent)
        """
        if not self.modification_history:
            return {'success': False, 'reason': 'No modifications to revert'}
        
        # Determine which modification to revert
        if modification_id is None:
            # Revert most recent
            mod_to_revert = self.modification_history[-1]
            mod_index = len(self.modification_history) - 1
        else:
            # Find specific modification
            if 0 <= modification_id < len(self.modification_history):
                mod_to_revert = self.modification_history[modification_id]
                mod_index = modification_id
            else:
                return {'success': False, 'reason': 'Invalid modification ID'}
        
        # Extract modification details
        component_id = mod_to_revert['component']
        attribute = mod_to_revert['attribute']
        old_value = mod_to_revert['old_value']
        
        # Check if component still exists
        if component_id not in self.modifiable_components:
            return {
                'success': False,
                'reason': f"Component {component_id} no longer exists"
            }
        
        # Get component info
        component_info = self.modifiable_components[component_id]
        component = component_info['component']
        
        # Apply reversion
        try:
            if component_info['handler']:
                # Use custom handler
                success = component_info['handler'](attribute, old_value)
            else:
                # Direct modification
                self._set_attribute_value(component, attribute, old_value)
                success = True
            
            if success:
                # Update current state
                component_info['current_state'][attribute] = old_value
                
                # Record reversion
                reversion_record = {
                    'timestamp': time.time(),
                    'component': component_id,
                    'attribute': attribute,
                    'reverted_from': mod_to_revert['new_value'],
                    'reverted_to': old_value,
                    'reverts_modification': mod_index
                }
                
                self.modification_history.append(reversion_record)
                
                # Remove original modification from history
                if self.safety_constraints.get('reversibility', True):
                    self.modification_history.pop(mod_index)
                
                return {
                    'success': True,
                    'reverted': f"{component_id}.{attribute}",
                    'record': reversion_record
                }
            else:
                return {
                    'success': False,
                    'reason': 'Reversion handler failed'
                }
        except Exception as e:
            return {
                'success': False,
                'reason': str(e)
            }
    
    def get_modification_history(self) -> List[Dict]:
        """Get modification history"""
        return self.modification_history.copy()
    
    def _capture_state(self, component: Any, attributes: Dict) -> Dict:
        """Capture current state of modifiable attributes"""
        state = {}
        
        for attr_name in attributes:
            try:
                state[attr_name] = self._get_attribute_value(component, attr_name)
            except:
                state[attr_name] = None
        
        return state
    
    def _get_attribute_value(self, component: Any, attr_name: str) -> Any:
        """Get attribute value, handling nested attributes"""
        if '.' in attr_name:
            # Handle nested attributes
            parts = attr_name.split('.')
            value = component
            
            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                elif isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    raise AttributeError(f"Cannot access {part} in {attr_name}")
            
            return value
        else:
            # Simple attribute
            if hasattr(component, attr_name):
                return getattr(component, attr_name)
            elif isinstance(component, dict) and attr_name in component:
                return component[attr_name]
            else:
                raise AttributeError(f"Cannot access {attr_name}")
    
    def _set_attribute_value(self, component: Any, attr_name: str, value: Any) -> None:
        """Set attribute value, handling nested attributes"""
        if '.' in attr_name:
            # Handle nested attributes
            parts = attr_name.split('.')
            target = component
            
            # Navigate to parent object
            for part in parts[:-1]:
                if hasattr(target, part):
                    target = getattr(target, part)
                elif isinstance(target, dict) and part in target:
                    target = target[part]
                else:
                    raise AttributeError(f"Cannot access {part} in {attr_name}")
            
            # Set value on parent
            final_part = parts[-1]
            if hasattr(target, final_part):
                setattr(target, final_part, value)
            elif isinstance(target, dict):
                target[final_part] = value
            else:
                raise AttributeError(f"Cannot set {final_part} on {type(target)}")
        else:
            # Simple attribute
            if hasattr(component, attr_name):
                setattr(component, attr_name, value)
            elif isinstance(component, dict):
                component[attr_name] = value
            else:
                raise AttributeError(f"Cannot set {attr_name} on {type(component)}")
    
    def _has_sufficient_performance_data(self) -> bool:
        """Check if we have sufficient performance data for analysis"""
        # Need at least some metrics with sufficient history
        for metric_id, data in self.performance_metrics.items():
            if len(data['values']) >= 5:  # At least 5 data points
                return True
        
        return False
    
    def _analyze_performance_trend(self, values: List[float]) -> float:
        """Analyze trend in performance values"""
        if len(values) < 3:
            return 0.0
        
        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0
            
        slope = numerator / denominator
        
        # Normalize slope by mean value to get relative trend
        if y_mean != 0:
            return slope / abs(y_mean)
        else:
            return slope
    
    def _identify_components_for_metrics(self, metrics: List[str]) -> List[str]:
        """Identify components related to specific metrics"""
        # In a real system, this would use a mapping of metrics to components
        # For this example, return all components
        return list(self.modifiable_components.keys())
    
    def _select_attributes_to_modify(self, component_info: Dict, trend: str) -> Dict:
        """Select attributes to modify based on performance trend"""
        # Get modifiable attributes
        attributes = component_info['attributes']
        
        # Select subset of attributes to modify
        selected = {}
        
        for attr_name, constraints in attributes.items():
            # Check if attribute should be modified based on trend
            if self._should_modify_attribute(attr_name, constraints, trend):
                # Generate new value within constraints
                current_value = component_info['current_state'].get(attr_name)
                
                if current_value is not None:
                    new_value = self._generate_new_value(current_value, constraints, trend)
                    selected[attr_name] = new_value
        
        return selected
    
    def _should_modify_attribute(self, attr_name: str, constraints: Dict, trend: str) -> bool:
        """Determine if an attribute should be modified"""
        # Check modification probability
        if 'modification_probability' in constraints:
            if np.random.random() > constraints['modification_probability']:
                return False
        
        # Check trend-specific constraints
        if trend == 'declining' and constraints.get('modify_on_decline', True):
            return True
        elif trend == 'stagnant' and constraints.get('modify_on_stagnation', True):
            return True
        
        return False
    
    def _generate_new_value(self, current_value: Any, constraints: Dict, trend: str) -> Any:
        """Generate a new value for an attribute within constraints"""
        if isinstance(current_value, (int, float)):
            # Numeric value
            min_val = constraints.get('min', current_value * 0.5)
            max_val = constraints.get('max', current_value * 1.5)
            
            # Max modification rate
            max_rate = self.safety_constraints['max_modification_rate']
            
            # Calculate allowed range
            lower_bound = max(min_val, current_value * (1 - max_rate))
            upper_bound = min(max_val, current_value * (1 + max_rate))
            
            # Modify based on trend
            if trend == 'declining':
                # Stronger modification for declining performance
                if np.random.random() < 0.7:
                    # Usually try increasing value for declining performance
                    return current_value + np.random.uniform(0, upper_bound - current_value)
                else:
                    # Sometimes try decreasing
                    return current_value - np.random.uniform(0, current_value - lower_bound)
            else:
                # Random modification for stagnant performance
                return np.random.uniform(lower_bound, upper_bound)
        
        elif isinstance(current_value, bool):
            # Boolean value - toggle
            return not current_value
        
        elif isinstance(current_value, list) and all(isinstance(x, (int, float)) for x in current_value):
            # List of numbers (e.g., vector)
            new_value = current_value.copy()
            
            # Modify a random element
            if new_value:
                idx = np.random.randint(0, len(new_value))
                
                # Get constraints for this element
                min_val = constraints.get('min', new_value[idx] * 0.5)
                max_val = constraints.get('max', new_value[idx] * 1.5)
                
                # Max modification rate
                max_rate = self.safety_constraints['max_modification_rate']
                
                # Calculate allowed range
                lower_bound = max(min_val, new_value[idx] * (1 - max_rate))
                upper_bound = min(max_val, new_value[idx] * (1 + max_rate))
                
                # Modify the element
                new_value[idx] = np.random.uniform(lower_bound, upper_bound)
            
            return new_value
        
        # For other types, return current value
        return current_value
    
    def _estimate_modification_impact(self, component_id: str, attributes: Dict) -> Dict:
        """Estimate the impact of a modification"""
        # Simple impact estimation
        impact = {}
        
        for attr_name, new_value in attributes.items():
            current_value = self.modifiable_components[component_id]['current_state'].get(attr_name)
            
            if isinstance(current_value, (int, float)) and isinstance(new_value, (int, float)):
                # Calculate relative change
                if current_value != 0:
                    relative_change = (new_value - current_value) / abs(current_value)
                else:
                    relative_change = 1.0 if new_value != 0 else 0.0
                
                impact[attr_name] = {
                    'relative_change': relative_change,
                    'significance': 'high' if abs(relative_change) > 0.2 else 'medium' if abs(relative_change) > 0.05 else 'low'
                }
            else:
                impact[attr_name] = {
                    'changed': new_value != current_value,
                    'significance': 'unknown'
                }
        
        return impact
    
    def _validate_safety_constraints(self, plan: Dict) -> bool:
        """Validate that a modification plan meets safety constraints"""
        # Check cooldown period
        current_time = time.time()
        if current_time - self.last_modification_time < self.safety_constraints['modification_cooldown']:
            return False
        
        # Check system stability
        if self.safety_constraints.get('stability_threshold'):
            system_stability = self._calculate_system_stability()
            if system_stability < self.safety_constraints['stability_threshold']:
                return False
        
        # Check modification rates
        for component_id, attributes in plan['components'].items():
            if component_id in self.modifiable_components:
                component_info = self.modifiable_components[component_id]
                
                for attr_name, new_value in attributes.items():
                    current_value = component_info['current_state'].get(attr_name)
                    
                    if isinstance(current_value, (int, float)) and isinstance(new_value, (int, float)):
                        # Calculate relative change
                        if current_value != 0:
                            relative_change = abs((new_value - current_value) / current_value)
                            
                            # Check against max modification rate
                            if relative_change > self.safety_constraints['max_modification_rate']:
                                return False
        
        return True
    
    def _calculate_system_stability(self) -> float:
        """Calculate overall system stability"""
        # Average recent performance metrics
        if not self.performance_metrics:
            return 1.0  # Assume stable if no metrics
        
        recent_values = []
        
        for metric_id, data in self.performance_metrics.items():
            values = data['values']
            if values:
                # Get recent values
                recent = values[-5:] if len(values) >= 5 else values
                
                # Calculate coefficient of variation
                if len(recent) >= 3:
                    mean = np.mean(recent)
                    if mean != 0:
                        std = np.std(recent)
                        cv = std / abs(mean)  # Coefficient of variation
                        
                        # Convert to stability score (0-1)
                        stability = max(0, 1 - min(1, cv * 2))
                        recent_values.append(stability)
        
        # Calculate average stability
        if recent_values:
            return np.mean(recent_values)
        else:
            return 1.0  # Assume stable if no recent values
    
    def _load_configuration(self, config_path: str) -> None:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Update safety constraints
            if 'safety_constraints' in config:
                for key, value in config['safety_constraints'].items():
                    self.safety_constraints[key] = value
        except Exception as e:
            print(f"Error loading configuration: {e}")