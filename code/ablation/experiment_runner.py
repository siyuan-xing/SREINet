from SREINet import SREINet
from config_manager import ConfigManager
import tensorflow as tf
from loss import myLoss
from SREINetTrainer import SREINetTrainer


#generate data
def generate_data(seed=42):
    pass



def run_single_experiment(config, dataset):
    train_data, val_data = dataset

    model = SREINet(
        layer_neuron_nums_per_act_fn = config.base_config.layer_neuron_nums_per_act_fn,
        activations = config.base_config.my_activation,
        data_type = config.base_config.DATA_TYPE,
        weight_initializer = tf.initializers.random_normal(stddev=config.base_config.WEIGHT_INIT_STDDEV, seed=config.base_config.RANDOM_SEED),
        name = 'SREINet'
    )

    if config.loss_type == 'mse':
        loss_fn = myLoss(model=model, loss='mse')
    elif config.loss_type == 'lasso':
        loss_fn = myLoss(model=model, loss='lasso', l1_regularization_strength=config.base_config.lambda_reg)
    elif config.loss_type == 'ridge':
        loss_fn = myLoss(model=model, loss='ridge', l2_regularization_strength=config.base_config.lambda_reg)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.base_config.LEARNING_RATE)

    trainer = SREINetTrainer(model=model, 
                optimizer=optimizer, 
                loss_fn=loss_fn, 
                data_type=config.base_config.DATA_TYPE)

    if config.use_pruning:
        if config.pruning_method == 'PPCP':
            # with pruning
            pruning_scheduler = PruningScheduler(
                steps_per_period=40,
                num_periods=50,
                maximum_threshold=0.30
            )
            history = trainer.fit(
                training_input=train_data['x'],
                training_output=train_data['dx'],
                batch_size=128,
                validation_split=0.2,
                epochs=2000,
                pruning_scheduler=pruning_scheduler,
                verbose=0
            )
        elif config.pruning_method == 'STLSQ-V':
            pass 
            #update the method for STLSQ-V
      
    else:
        # Without pruning (0 maximum threshold)
        pruning_scheduler = PruningScheduler(
            steps_per_period=1,
            num_periods=1,
            maximum_threshold=0.0
        )
        history = trainer.fit(
            training_input=train_data['x'],
            training_output=train_data['dx'],
            batch_size=128,
            validation_split=0.2,
            epochs=2000,
            pruning_scheduler=pruning_scheduler,
            verbose=0
        )



#save results


def main():
    '''main program: ablation experiments'''

    print("=" * 60)
    print("SREINet Ablation Study")
    print("=" * 60)


    print("Generating data...")
    data = generate_data()

    print("Generating configs...")
    config_manager = ConfigManager()
    configs = config_manager.generate_all_configs()


    print("Running experiments...")

    result = []
    for i, config in enumerate(configs):
        print(f"Running experiment {i+1} of {len(configs)}: {config.config_id}")
        results = run_single_experiment(config, dataset)
        result.append(results)

    print("Saving results...")
    save_results(result)

    