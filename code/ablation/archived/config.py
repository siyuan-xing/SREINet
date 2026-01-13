from dataclasses import dataclass, field


@dataclass
class BaseConfig:
    LAYER_DEPTH = 2
    DATA_TYPE = 'float32'
    layer_neuron_nums_per_act_fn: int
    my_activation = [lambda x:x]
    WEIGHT_INIT_STDDEV = 0.001
    
    LEARNING_RATE = 0.001
    MINI_BATCH_SIZE = 128
    VAL_SPLIT = 0.2
    RANDOM_SEED = 42

    VALIDATION_SPLIT = 0.2
    PATIENCE_PERIODS = 15





@dataclass 
class PruningBaseConfig(BaseConfig):
    pruning_method: str

    #this configuration may be changed for STLSQ-V
    MAX_WEIGHT_PRUNING_THRESHOLD = 0.30
    N_PERIODS = 50
    PRUNING_EPOCHS_PER_PERIOD = 40 # total epochs = pruning_period * pruning_epochs_per_period    
    NUM_EPOCHS = N_PERIODS * PRUNING_EPOCHS_PER_PERIOD 

    EARLY_STOPPING_LOSS_THRESHOLD = 1e-10
    EARLY_STOPPING_PRUNING_THRESHOLD = 0.05


@dataclass
class NoPruningBaseConfig(BaseConfig):
    EPOCHS = 2000 # equal to pruning_period * pruning_epochs_per_period
    EARLY_STOPPING_LOSS_THRESHOLD = 1e-10
    lambda_reg : float

@dataclass
class ExperimentConfig:
    config_id: str
    loss_type: str
    use_pruning: bool

    base_config: BaseConfig = field(default_factory=BaseConfig)

