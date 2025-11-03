import numpy as np

"""
CoefficientReconstructor is a class that reconstructs coefficients of a dynamical system using CombOpNet weights.

Currently, the reconstructor only supports the case where the width of the CombOpNet is the same for all layers.
"""

class CoefficientReconstructor:
    def __init__(self, rhs_label_generator, lhs_label_generator):
        """
        Initialize the CoefficientReconstructor class.

        Args:
            rhs_label_generator (callable): Function to generate RHS labels
            lhs_label_generator (callable): Function to generate LHS labels
        """
        self.rhs_label_generator = rhs_label_generator
        self.lhs_label_generator = lhs_label_generator

    def reconstruct_coefficients(self, 
                                dim_index,
                                weights, 
                                minimum_coefficient_threshold=1e-5,
                                coefficient_output_digits=10, 
                                enable_coefficient_sorting=False):
        """
        Reconstruct coefficients from CombOpNet weights.
        
        Args:
            dim_index (int): Index of the dimension
            weights (list): List of neural network weights
            minimum_coefficient_threshold (float): Minimum threshold for coefficients
            coefficient_output_digits (int): Number of digits for coefficient output
            enable_coefficient_sorting (bool): Whether to enable coefficient sorting
            
        Returns:
            list: List of reconstructed coefficients with equation formatting
        """

        max_neurons = max(max(len(layer) for layer in combopnet) for combopnet in weights)
        labels = self.rhs_label_generator(max_neurons - 1)  # number of states = number of neurons - 1
        LHS_labels = self.lhs_label_generator(max_neurons - 1)

        def explore_path(combopnet_index, layer, neuron, weight_product, label_counts, non_zero_paths):
            """
            Recursively explores non-zero paths in a neural network.
            """
            if layer == len(weights[combopnet_index]) - 1:
                non_zero_paths.append((label_counts, weight_product))
                return

            for next_neuron, weight in enumerate(weights[combopnet_index][layer][neuron]):
                if abs(weight) <= 1e-15:
                    continue  
                new_weight_product = weight_product * weight
                next_label_counts = label_counts.copy()
                next_label_counts[labels[next_neuron]] = next_label_counts.get(labels[next_neuron], 0) + 1
                explore_path(combopnet_index, layer + 1, next_neuron, new_weight_product, next_label_counts, non_zero_paths)
        
        all_combopnet_products = []
        for combopnet_index, _ in enumerate(weights):
            non_zero_paths = []
            for neuron_index in range(len(weights[combopnet_index][0])):
                label_counts = {labels[neuron_index]: 1}
                explore_path(combopnet_index, 0, neuron_index, 1, label_counts, non_zero_paths)
            
            combopnet_products = {}
            for label_counts, weight_product in non_zero_paths:
                label_counts_key = frozenset(label_counts.items())
                if label_counts_key in combopnet_products:
                    combopnet_products[label_counts_key] += weight_product
                else:
                    combopnet_products[label_counts_key] = weight_product
            
            combopnet_products = {k: v for k, v in combopnet_products.items() if abs(v) >= minimum_coefficient_threshold}
            
            if enable_coefficient_sorting:
                sorted_combopnet_products = dict(sorted(combopnet_products.items(), key=lambda item: abs(item[1]), reverse=True))
            else:
                sorted_combopnet_products = combopnet_products

            all_combopnet_products.append(sorted_combopnet_products)

        labeled_all_combopnet_products = []
        for combopnet_index, sorted_combopnet_products in enumerate(all_combopnet_products):
            combopnet_product_strings = []
            for key, value in sorted_combopnet_products.items():
                labels_list = []
                only_constant = len(key) == 1 and any(label == 'Constant' for label, _ in key)
                
                for label, count in sorted(key):
                    if label == 'Constant' and len(key) > 1:
                        continue
                    if label != 'Constant' and count > 1:
                        labels_list.append(f"{label}^{count}")
                    elif label == 'Constant' or count == 1:
                        labels_list.append(label if label != 'Constant' else '')
                
                label_str = '*'.join(filter(None, labels_list))
                            
                formatted_value = f"{abs(value):.{coefficient_output_digits}g}"

                term_str = label_str if formatted_value == '1' else f"{formatted_value}*{label_str}" if not only_constant else formatted_value if formatted_value != '1' else ''
                sign = '+' if value >= 0 else '-'
                if combopnet_product_strings or value < 0:
                    term_str = f" {sign} {term_str}"
                else:
                    term_str = term_str
                combopnet_product_strings.append(term_str)

            formatted_output = ''.join(combopnet_product_strings)
            layer_prefix = LHS_labels[combopnet_index+dim_index] + " = "  
            full_output = layer_prefix + formatted_output
            labeled_all_combopnet_products.append(full_output)

        return labeled_all_combopnet_products