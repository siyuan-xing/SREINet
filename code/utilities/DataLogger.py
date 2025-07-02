import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

class DataLoggingHelper:
    def __init__(self, output_filename="training_data.xlsx", fig_file_directory="figs", plotting_fn=None):
        self.hyper_frame = {}
        self.loss_history = []
        self.output_filename = output_filename
        self.fig_file_directory = fig_file_directory
        self.epochs = 0
        self.lbfgs_starting_epoch = 0
        self.plotting_fn = plotting_fn

    def set_hyper_parameters(self, hyperparameters):
        domain_lb, domain_ub, grid_size, training_size, mini_batch_size, boundary_batch_size = hyperparameters

        self.hyper_frame = pd.DataFrame(
            {
                "domain upper bound": domain_ub,
                "domain lower bound": domain_lb,
                "grid size": grid_size,
                "training size": training_size,
                "mini batch size": mini_batch_size,
                "boundary_batch_size": boundary_batch_size,
                "Neurons": sum(grid_size),
            }
        )

    def set_final_epoch(self, epoch):
        self.epochs = epoch

    def append_loss(self, loss):
        self.loss_history.append(loss)

    def record_gradient_blowup(self, epoch):
        self.training_logs.append(f"Gradient blow up at epoch {epoch}")

    def save_data(self):
        import statistics
        
        result_frame = pd.DataFrame(
            {   
                "Final epochs": [self.epochs],
                "LBFGS starting epoch": [self.lbfgs_starting_epoch],
                "Final Loss ": [self.loss_history[-1].numpy()],
                "min loss": [min([elem.numpy() for elem in self.loss_history])],
            }
        )
        config_frame = self.hyper_frame

        loss_frame = pd.DataFrame(
            {
            "Loss History":[elem.numpy() for elem in self.loss_history],
            }
        )

        with pd.ExcelWriter(self.output_filename) as writer:
            # Write each dataframe to a different worksheet.
            config_frame.to_excel(writer,sheet_name="config",index=False)
            result_frame.to_excel(writer, sheet_name='result', index=False)
            loss_frame.to_excel(writer,sheet_name="loss_history",index=False)
        

    def save_plot(self, epoch):
        if self.plotting_fn is not None:
            self.plotting_fn(self.loss_history, self.fig_file_directory, epoch)
    
    def save_weights(self, model):
        import os
        model.save_weights(os.path.join(self.fig_file_directory, "model_final_epoch.h5"))

def setupOutputDirectory(run_folder_name):
    import os

    base_directory = os.path.join(os.getcwd(), run_folder_name)
    if not os.path.exists(base_directory):
    # Create the directory
        os.makedirs(base_directory)
        print("Directory created successfully.")

    existing_folders = [folder for folder in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, folder)) and "run" in folder]
    num_runs = len(existing_folders)
    run_folder = f"run{num_runs+1}"
    
    run_directory = os.path.join(base_directory, run_folder)
    os.makedirs(run_directory, exist_ok=True)

    return run_directory

    
  