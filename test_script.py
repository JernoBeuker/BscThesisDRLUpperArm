import os

from osim.osim import OsimModel
from stable_baselines3 import PPO
from argparse import ArgumentParser


def test_trained_model(prosthetic: bool, save_dir: int, iterations: int, random_positions: bool = False,
         continious: bool = False, ) -> None:
    """
    Method to observe (and write results to file?!) results.
    :param prosthetic: boolean value to activate prosthetic model (weight etc.).
    :param save_dir: directory where to find trained models.
    :param iterations: iteration to load the zip file e.g. PPO_20.zip.
    :param random_positions: boolean value to reset model at random positions.
    :param continious: boolean value to activate continious action space.
    """


    model_name = "OS4_gait14dof15musc_2act_LTFP_VR_DynAct.osim" if prosthetic else "OS4_gait14dof22musc.osim"
    env = OsimModel(
        model_path=f"osim/models/{model_name}",
        data_path="training_data/u_limb/IKresult.csv",
        integrator_accuracy=0.01,
        visualize=False,
        random_init=random_positions,
        step_size=0.01,  # make sure this corresponds to your dataset
        continious_action_space=continious,
        prosthetic=prosthetic,

    )
    observations, _ = env.reset()
    # load your ppo model directory, from where you saved the model zip (PPO_xx.zip)
    if not os.path.exists(f"./models/{save_dir}/PPO_{iterations}.zip"):
        print(f"Trained model not found in directory: /models/{save_dir}, with iterations: {iterations}")
        raise FileNotFoundError
    model_pros = PPO.load(path=f"./models/{save_dir}/PPO_{iterations}.zip", env=env)

    while True:
        actions_pros, _ = model_pros.predict(observation=observations)
        observations, reward, terminated, done, _ = env.step(actions=actions_pros)
        current_model_state = env.state

        # for plotting you could write the observation space to a csv file and do some plotting!
        # Or extract information from current_model_state (similar as in osim.py)
        if terminated or done:
            observations, _ = env.reset()


def retrieve_cmd_args():
    """Retrieve specifics based on your trained model! (similarly as run_script.py)"""
    parser = ArgumentParser()
    parser.add_argument("-p", "--prosthetic", action="store_true", default=False)
    parser.add_argument("-d", "--model_dir", type=str, default="standard_run", help="the name of the run: models/xxx")
    parser.add_argument("-i", "--iterations", type=int, default=20, help="the number of iterations to load from")
    parser.add_argument("-r", "--random_positions", action="store_true", default=False, help="reset model at random positions?")
    parser.add_argument("-c", "--continious", action="store_true", default=False)
    return parser.parse_args()


if __name__ == '__main__':
    """
    This file can/should be used to see the performance of the model, you can write your results to a csv or txt file to plot them later!
    """
    test_args = retrieve_cmd_args()
    test_trained_model(prosthetic=test_args.prosthetic,
                       save_dir=test_args.model_dir,
                       iterations=test_args.iterations,
                       random_positions=test_args.random_positions,
                       continious=test_args.continious)
