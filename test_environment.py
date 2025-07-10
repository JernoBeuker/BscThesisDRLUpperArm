from osim.osim import OsimModel

"""Small script to see if everything is setup correctly"""

env = OsimModel(
    model_path="osim/models/Model_Contact3.osim",
    data_path="training_data/175/175-FIX_vel.csv",
    integrator_accuracy=0.01,
    visualize=True,
    random_init=False,
    step_size=0.01,
)

observations, _ = env.reset()

while True:
    action = env.action_space.sample()
    observations, reward, terminated, done, _ = env.step(actions=action)

    # for plotting you could write the observation space to a csv file and do some plotting!
    if terminated or done:
        observations, _ = env.reset()
