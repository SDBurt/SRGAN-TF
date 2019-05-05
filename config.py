import argparse

# ----------------------------------------
# Global variables
arg_lists = []
parser = argparse.ArgumentParser()

# ----------------------------------------
# Macro for arg parse
def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# ----------------------------------------
# Arguments for preprocessing
preprocess_arg = add_argument_group("Preprocess")

preprocess_arg.add_argument("--data_dir",  type=str,
                            default="/data",
                            help="Path to image data")

preprocess_arg.add_argument("--package_data",
                            default=False,
                            help="Whether or not to invoke preprocessing.py")

preprocess_arg.add_argument("--hr_resolution",
                            default=(1536,1536),
                            help="Resolution of original images")

preprocess_arg.add_argument("--crop_resolution",
                            default=(96,96),
                            help="Resolution of cropped images")

preprocess_arg.add_argument("--lr_resolution",
                            default=(24,24),
                            help="Resolution of downsampled images")

# ----------------------------------------
# Arguments for training
train_arg = add_argument_group("Training")

train_arg.add_argument("--learning_rate", type=float,
                       default=1e-3,
                       help="Learning rate (gradient step size)")

train_arg.add_argument("--batch_size", type=int,
                       default=16,
                       help="Number of experiences to sample from memory during training")

train_arg.add_argument("--epochs", type=int,
                       default=100,
                       help="Number of epochs for training")

train_arg.add_argument("--log_dir", type=str,
                       default="./logs/",
                       help="Directory to save logs")

train_arg.add_argument("--log_freq", type=int,
                       default=1,
                       help="Number of steps before logging weights")

train_arg.add_argument("--save_dir", type=str,
                       default="./saves/",
                       help="Directory to save current model")

train_arg.add_argument("--save_freq", type=int,
                       default=20,
                       help="Number of episodes before saving model")

train_arg.add_argument("-f", "--extension", type=str,
                       default="best",
                       help="Specific name to save training session or restore from")

# ----------------------------------------
# Arguments for model
model_arg = add_argument_group("Model")

model_arg.add_argument("--activ", type=str,
                       default="relu",
                       choices=["relu", "elu", "selu", "tanh", "sigmoid"],
                       help="Activation function to use")

model_arg.add_argument("--init", type=str,
                       default="glorot_normal",
                       choices=["glorot_normal", "glorot_uniform", "random_normal", "random_uniform", "truncated_normal"],
                       help="Initialization function to use")

model_arg.add_argument("--num_channels",
                        default=3,
                        help="Number of colour channels")

# ----------------------------------------
# Function to be called externally
def get_config():
    config, unparsed = parser.parse_known_args()

    # If there are unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        parser.print_usage()
        exit(1)

    return config

