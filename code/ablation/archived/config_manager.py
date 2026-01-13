from config import ExperimentConfig, BaseConfig, PruningBaseConfig, NoPruningBaseConfig
from itertools import product


class ConfigManager:

    def __init__(self):
        pass

    def generate_all_configs(self):

        configs = []

        configs.extend(self.generate_pruning_configs())
        configs.extend(self.generate_no_pruning_configs())

        return configs

    def generate_pruning_configs(self):

        configs = []
        pruning_methods = ['PPCP', 'STLSQ-V']


        for pruning_method in pruning_methods:
            
            pruning_base = PruningBaseConfig(pruning_method=pruning_method)

            config = ExperimentConfig(
                config_id = f"SREINet-Pruning-{pruning_method}",
                loss_type = PruningBaseConfig.loss_type,
                use_pruning = True,
                base_config = pruning_base
            )
            configs.append(config)

        return configs

    def generate_no_pruning_configs(self):

        configs = []

        loss_types = ['lasso', 'ridge']
        regularization_strengths = [1e-4, 1e-3, 1e-2]
        for loss_type, regularization_strength in product(loss_types, regularization_strengths):
            no_pruning_base = NoPruningBaseConfig(loss_type=loss_type, lambda_reg=regularization_strength)
            config = ExperimentConfig(
                config_id = f"SREINet-NoPruning-{loss_type}-{regularization_strength}",
                loss_type = loss_type,
                use_pruning = False,
                base_config = no_pruning_base
            )
            configs.append(config)

        return configs


#unit test
if __name__ == "__main__":
    config_manager = ConfigManager()
    configs = config_manager.generate_all_configs()
    for config in configs:
        print(f"Config ID: {config.config_id}")
        print(f"Loss Type: {config.loss_type}")
        print(f"Use Pruning: {config.use_pruning}")
        print(f"Base Config: {config.base_config}")
        print("-"*60)


