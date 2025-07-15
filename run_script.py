import random
from gymnasium import Env
import os
from osim.osim import OsimModel
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import SubprocVecEnv, make_vec_env
from stable_baselines3.common.logger import configure
from argparse import ArgumentParser, Namespace



def linear_learning(initial_value: float) -> callable:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining: Float value from 1 to 0
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def make_callable_env(model_path: str, data_path: str, visualize: bool, continuous_action_space: bool = False,
                      prosthetic: bool = False,
                      random_init: bool = False) -> callable:
    """
    Utility function for multiprocessed env.

    :param model_path: path of .osim model either healthy (22) or prosthetic (17= 15musc + 2act).
    :param data_path: path of imitation data (175-fix.csv for the course).
    :param visualize: boolean value to visualize simulation (makes it slower but more fun!).
    :param continuous_action_space: boolean value to train with continious action space, makes it very hard to learn!.
    :param prosthetic: boolean value to activate prosthetic model (weight etc.).
    :param random_init: randomly init model after reset, could help prevent overfitting to first 1/2 steps.

    :return: A function to create an OpenSim environment
    """

    def _init(rank: int = -1) -> Env:
        """
        Initialises the model and resets it to a clean start. The thread rank can be used to visual the environment.

        :param rank: The rank of the thread running the environment
        :return: The OpenSim environment (OsimModel).
        """
        env = OsimModel(
            model_path=model_path,
            data_path=data_path,
            integrator_accuracy=0.01,
            visualize=visualize and rank == 0,
            random_init=random_init,  # <-- can be useful to train more quickly
            step_size=0.01,
            continious_action_space=continuous_action_space,
            prosthetic=prosthetic
        )
        env.reset()
        return env

    return _init


def save_specs(args: Namespace, folder: str) -> None:
    """
    Method to save all arguments/run specs to a txt file such that you do not lose track of what you are doing :-)

    :param args: the arguments to save.
    :param folder: the location folder
    """

    txt_path = os.path.join(folder, "setup.txt")

    with open(txt_path, "w") as file:
        for arg, value in vars(args).items():
            file.write(f"{arg}: {value}\n")


def get_cmd_arguments() -> Namespace:
    """
    Method to retrieve cmd arguments. If no arguments are specified the default arguments are used.
       To change a parameter simply add the argument and the desired value.
       e.g. your command should look like: 'python run_script.py -b 1024 -lr 0.0001'
       this changes the batch size to 1024 and learning rate to 0.0001!
       Note: if action='store_true' only the flag is necessary.
    """
    arg_parser = ArgumentParser()
    # run name
    arg_parser.add_argument("-n", "--run_name", type=str, default="standard_run",
                            help="the name of the run, give a good descriptive name!")

    # run parameters
    arg_parser.add_argument("-t", "--total_timesteps", type=int, default=30_000_000,
                            help="the total number of timesteps within the simulation")
    arg_parser.add_argument("-n_envs", "--n_envs", type=int, default=16, help="the number of parallel environments")
    arg_parser.add_argument("-st", "--steps_per_iteration", type=int, default=6144,
                            help="the number of steps per iteration")
    arg_parser.add_argument("-b", "--batch_size", type=int, default=512, help="the batch size")
    arg_parser.add_argument("-e", "--n_epochs", type=int, default=4,
                            help="the number of epochs (run over gather data for updates)")

    # ppo params
    arg_parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="the learning rate")
    arg_parser.add_argument("-s", "--size", type=int, default=128, help="the size of the hidden MLP layers")
    arg_parser.add_argument("-c", "--clip_range", type=float, default=0.25,
                            help="the clip range for the new found policy (such that it doesnt change too drastically)")
    arg_parser.add_argument("-g", "--gamma", type=float, default=0.99, help="discount factor")
    arg_parser.add_argument("-l", "--gae_lambda", type=float, default=0.95, help="")
    # more params can be added here if found necessary!

    # model type
    arg_parser.add_argument("-pr", "--prosthetic_model", action="store_true", default=False,
                            help="Flag in order to use the prosthetic model instead of healthy model")

    arg_parser.add_argument("-v", "--visualize", action="store_true", default=False,
                            help="visualize 1 of the environments! (slows down training depending on pc specs)")
    arg_parser.add_argument("-ca", "--continuous_action_space", action="store_true", default=False,
                            help="control muscles/actuators continiously instead of binary actuation")

    arg_parser.add_argument("-r", "--random_init", action="store_false", default=True,
                            help="activaging STOPS the randomly init model after reset")

    # logging
    return arg_parser.parse_args()


def retrieve_model_and_data_paths(prosthetic_model: bool) -> tuple:
    """
    Method to retrieve the model and data paths.

    :param prosthetic_model: boolean value to activate prosthetic model (weight etc.).

    :returns: tuple containing path to model and path to data
    """
    prosthetic_path = os.path.join(os.getcwd(), "osim", "models", "OS4_gait14dof15musc_2act_LTFP_VR_DynAct.osim")
    healthy_path = os.path.join(os.getcwd(), "osim", "models", "Model_Contact3.osim")
    data_path = os.path.join(os.getcwd(), "training_data", "175", "175-FIX_vel.csv")

    # new_simplified_model_FPL
    # FPL_model_efficient_maybe

    print(f"Prosthetic model path: {prosthetic_path}")

    if not os.path.exists(prosthetic_path) or not os.path.exists(healthy_path):
        raise "Model path does not exist, file missing, or working from wrong directory!"
    if not os.path.exists(data_path):
        raise "Data path does not exist, file missing, or working from wrong directory!"

    if prosthetic_model:
        return prosthetic_path, data_path
    else:
        return healthy_path, data_path


if __name__ == "__main__":
    """
    Main method to train the OpenSim framework in combination with stable baselines3.
    Parameters can be kept default or use the command line interface to change the parameters (necessary for prosthetic implementation!).
    """

    """retrieve cmd arguments (if any)"""
    args = get_cmd_arguments()

    """setup environment and model"""
    print('setting up environment and model...')
    model_path, data_path = retrieve_model_and_data_paths(args.prosthetic_model)

    n_steps = args.steps_per_iteration // args.n_envs
    envs = make_vec_env(env_id=make_callable_env(model_path=model_path,
                                                 data_path=data_path,
                                                 visualize=args.visualize,
                                                 prosthetic=args.prosthetic_model,
                                                 continuous_action_space=args.continuous_action_space,
                                                 random_init=args.random_init),
                        n_envs=args.n_envs,
                        vec_env_cls=SubprocVecEnv,
                        seed=random.randint(1, 9999))

    """setup MLP policy and Stable Baelines3 PPO algorithm"""
    print('Setting up MLP policy and Stable Baelines3 PPO algorithm...')
    size = args.size
    policy_kwargs = {
        "net_arch": {
            "pi": [size, size],
            "vf": [size, size],
        },
    }

    """Organize the run and setting up logger"""
    print('Organizing the run and setting up logger...')
    folder = f"./models/{args.run_name}"
    os.makedirs(folder, exist_ok=True)
    save_specs(args, folder)

    print('initializing model...')
    model = PPO(
        policy="MlpPolicy",
        env=envs,
        learning_rate=linear_learning(initial_value=args.learning_rate),
        n_steps=n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=0.01,
        clip_range=args.clip_range,
        policy_kwargs=policy_kwargs,
        max_grad_norm=1,
        seed=999,
        tensorboard_log=folder
    )


    log = configure(folder=folder, format_strings=["stdout", "csv", "tensorboard"])
    model.set_logger(log)

    """Start training the model!"""
    print('Starting to train the model...')
    model.learn(total_timesteps=args.total_timesteps, progress_bar=True, log_interval=2)
