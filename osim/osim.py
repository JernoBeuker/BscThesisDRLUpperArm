import numpy as np
import pandas as pd
from gymnasium import Env, spaces
from opensim import ActivationCoordinateActuator, Constant, Logger, Manager, Model, PrescribedController, \
    ScalarActuator, CoordinateActuator
from typing import Optional

"""Opensim environment for musculoskeletal movement"""


class OsimModel(Env):

    def __init__(self, model_path: str, data_path: str, integrator_accuracy: float, visualize: bool, random_init: bool,
                 step_size: float, continious_action_space: bool = False, prosthetic=False) -> None:
        """
        OpenSim Env class that is used as an interface to the OpenSim environments.

        The required methods for this class are:
         - step()
         - reset()
         - get_action_space()
         - get_observation_space()
         and their return values must match those of the Env super class.

        The other methods are defined for the use cases of DRL algorithms.
        They can be edited to suit the need of the training data

        :param model_path: Path to the model file.
        :param data_path: Path to the data file.
        :param integrator_accuracy: Accuracy of the integrator.
        :param visualize: Whether to visualize.
        :param random_init: Whether to reset the position of the model to a random point in the training data.
        :param step_size: timestep size of your imitation
        :param continious_action_space: flag to enable continious action space
        :param prosthetic: flag to enable prosthetic model and settings
        Keep this false, except if you want to explore if you can predict floating values from your policy (instead of integers)
        """
        Logger.setLevelString("off")  # Leave to off to stop the terminal getting spammed.
        self.integrator_accuracy = integrator_accuracy
        self.data = pd.read_csv(filepath_or_buffer=data_path, index_col=False)
        self.random_init = random_init
        self.model = Model(model_path)
        self.model.setUseVisualizer(visualize=visualize)
        self.step_size = step_size
        self.pros = prosthetic
        self.continious_action_space = continious_action_space

        # Set up future parameters
        self.i_step: int = 0 #keep track of imitation dataset step
        self.model_mass:float = 67.39 if self.pros else 75.16
        self.gravity = 9.80665
        self.manager: Optional[Manager] = None

        #  Set up everything that is observable in the environment
        self.muscleSet = self.model.getMuscles()
        self.forceSet = self.model.getForceSet()
        self.bodySet = self.model.getBodySet()
        self.jointSet = self.model.getJointSet()
        self.markerSet = self.model.getMarkerSet()
        self.contactGeometrySet = self.model.getContactGeometrySet()
        self.actuatorSet = self.model.getActuators()

        #  Set up the brain and its functions for each actuator
        self.brain = PrescribedController()
        for i, actuator in enumerate(self.actuatorSet):
            func = Constant(1.0)
            self.brain.addActuator(actuator)
            self.brain.prescribeControlForActuator(i, func)
        self.model.addController(self.brain)
        self.state = self.model.initSystem()
        self.model.realizeAcceleration(self.state)

        #  Set up the action/observation
        self.action_space: spaces = self.get_action_space()
        self.observation_space: spaces = self.get_observation_space()

    def get_action_space(self) -> spaces:
        """
        :return: The action space of the environment
        Note: actuators (knee and ankle) range from [-1,1] while muscles [0,1]
        """

        size = self.actuatorSet.getSize()
        if self.continious_action_space and self.pros:  # prosthetic action space (15 muscles 2 actuators)
            action_space = spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1], dtype=np.int32),
                high=np.ones(size, dtype=np.int32),
                shape=(size,),
                dtype=np.int32,
            )
        elif self.continious_action_space and not self.pros:
            print("continious!!")
            action_space = spaces.Box(
                low=np.zeros(size, dtype=np.int32),
                high=np.ones(size, dtype=np.int32),
                shape=(size,),
                dtype=np.int32,
            )
        else:  # healthy action space (22 muscles)
            action_space = spaces.MultiBinary(n=size)
        return action_space

    def get_observation_space(self) -> spaces:
        """
        Method to get the observation space of the osim environment

        :return: The observation space of the environment
        """
        obs_list, _ = self.get_states()
        size = len(obs_list)
        return spaces.Box(low=-2, high=2, shape=(size,))


    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """
        Resets the environment to its initial state.
        - Sets the angles/skeleton to default position
        - Equilibrates the muscles
        - Sets time = 0
        - Resets the Manager
          - Initialises new Manager
          - Sets method and accuracy

        Called at the start of every training/optimization/testing, and if the environment is truncated/terminated

        :param seed: (unused) The seed to set for any random operations
        :param options: (unused) Any additional information important for a reset.
        :return: The state as an array for the network, the state as a dictionary for human-readable data
        """
        self.i_step = 0
        self.state = self.model.initializeState()

        if self.random_init:
            while True:  # loop to only continue after equilibratemuscles
                self.i_step = np.random.randint(0, self.data.shape[0] - 400)  # <- select random timestep from dataset
                try:
                    self.set_state(self.i_step)
                    self.model.equilibrateMuscles(self.state)
                    break
                except:  # exception can happen when using random init, when no activation can get the desired position
                    pass
        else:
            self.set_state(self.i_step)
            self.model.equilibrateMuscles(self.state)

        self.state.setTime(self.i_step * self.step_size)

        self.manager = Manager(self.model)
        self.manager.setIntegratorMethod(Manager.IntegratorMethod_SemiExplicitEuler2)
        self.manager.setIntegratorAccuracy(self.integrator_accuracy)
        self.manager.initialize(self.state)
        self.model.realizeAcceleration(self.state)

        observation, _ = self.get_states()

        return observation, {}

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Simulates what happens in the environment for 1 timestep (step_size)

        Called at every step for each independent environment.

        :param actions: An array representing the actions for each actuator
        :return: The observation as a list, the reward, if terminated, if truncated, the observation as a dictionary
        """
        self.i_step += 1

        self.actuate(actions=actions)

        self.state = self.manager.integrate(self.step_size * self.i_step)
        self.model.realizeAcceleration(self.state)

        obs_list, obs_dict = self.get_states()
        reward = self.get_reward(obs_dict=obs_dict)
        truncated = self.is_truncated(obs_dict=obs_dict)
        terminated = self.is_terminated()
        return obs_list, reward, terminated, truncated, {}

    def actuate(self, actions: np.ndarray) -> None:
        """
        Takes the actions provided by the algorithm and applies them the brain's functions.
        Actions can be seen as action_space.sample(). They should be in the range [0, 1]

        Called by self.step().

        :param actions: An array representing the actions for each actuator
        :return:
        """
        function_set = self.brain.get_ControlFunctions()

        for i, function in enumerate(function_set):
            func = Constant.safeDownCast(function)
            func.setValue(float(actions[i]))

    def is_terminated(self) -> bool:
        """
        Determines if the environment is terminated.

        The terminated state is defined as when the environment has reached the end of the training data.
        You should not redefine this.

        :return: True if the environment is terminated, false otherwise
        """
        return self.i_step >= self.data.shape[0] - 1

    @staticmethod
    def is_truncated(obs_dict: dict) -> bool:
        """
        Determines if the environment is truncated.

        The truncated state is when the environment has become "unstable", the model has fallen over or gone of track.
        You can define this how you like, but keep the conditions within the bounds of the training data.

        :param obs_dict: The observation of the environment
        :return: True if the environment is truncated, false otherwise
        """
        return obs_dict["pelvis_tx"] < -0.10 or obs_dict["pelvis_ty"] < 0.60 or abs(obs_dict["pelvis_tz"]) > 0.4


    def get_reward(self, obs_dict: dict) -> float:
        """
        Calculates the rewards by comparing the observed state to the training data.

        Example:
         pelvis_tx_error = (0.98 - 0.90)^2 = 0.0064
         reward = e^(-8 x 0.0064) = 0.9500

        The error is the square of the difference, and the reward is in the form:
          reward = e^(-c * sum(errors))
        c determines how aggressive the reward is. Higher = more aggressive.

        Make sure that the reward calculated is a value in the range [0, 1]

        :param obs_dict: The current state of the environment
        :return: The reward of environment at the current observation, range [0, 1]
        """
        t = self.i_step
        d = self.data

        # GOAL REWARD
        # Goal mean squared errors
        pelvis_x = (obs_dict["pelvis_tx"] - d["pelvis_tx"][t]) ** 2
        pelvis_y = (obs_dict["pelvis_ty"] - d["pelvis_ty"][t]) ** 2
        pelvis_z = (obs_dict["pelvis_tz"] - d["pelvis_tz"][t]) ** 2

        goal_reward = np.exp(-8 * (pelvis_x + pelvis_y + pelvis_z))

        # IMITATION REWARD
        # POSITION mean squared errors
        pelvis_tilt = (obs_dict["pelvis_tilt"] - d["pelvis_tilt"][t]) ** 2
        pelvis_list = (obs_dict["pelvis_list"] - d["pelvis_list"][t]) ** 2
        pelvis_rotation = (obs_dict["pelvis_rotation"] - d["pelvis_rotation"][t]) ** 2

        ankle_l = (obs_dict["ankle_angle_l"] - d["ankle_angle_l"][t]) ** 2
        knee_l = (obs_dict["knee_angle_l"] - d["knee_angle_l"][t]) ** 2
        hip_l_flex = (obs_dict["hip_flexion_l"] - d["hip_flexion_l"][t]) ** 2
        hip_l_add = (obs_dict["hip_adduction_l"] - d["hip_adduction_l"][t]) ** 2

        ankle_r = (obs_dict["ankle_angle_r"] - d["ankle_angle_r"][t]) ** 2
        knee_r = (obs_dict["knee_angle_r"] - d["knee_angle_r"][t]) ** 2
        hip_r_flex = (obs_dict["hip_flexion_r"] - d["hip_flexion_r"][t]) ** 2
        hip_r_add = (obs_dict["hip_adduction_r"] - d["hip_adduction_r"][t]) ** 2

        p_loss = (
                pelvis_tilt
                + pelvis_list
                + pelvis_rotation
                + ankle_l
                + knee_l
                + hip_l_flex
                + hip_l_add
                + ankle_r
                + knee_r
                + hip_r_flex
                + hip_r_add
        )
        position_reward = np.exp(-4 * p_loss)

        # VELOCITY mean squared errors
        ankle_l_vel = (obs_dict["ankle_angle_l_vel"] - d["ankle_angle_l_vel"][t]) ** 2
        knee_l_vel = (obs_dict["knee_angle_l_vel"] - d["knee_angle_l_vel"][t]) ** 2
        hip_l_flex_vel = (obs_dict["hip_flexion_l_vel"] - d["hip_flexion_l_vel"][t]) ** 2
        hip_l_add_vel = (obs_dict["hip_adduction_l_vel"] - d["hip_adduction_l_vel"][t]) ** 2

        ankle_r_vel = (obs_dict["ankle_angle_r_vel"] - d["ankle_angle_r_vel"][t]) ** 2
        knee_r_vel = (obs_dict["knee_angle_r_vel"] - d["knee_angle_r_vel"][t]) ** 2
        hip_r_flex_vel = (obs_dict["hip_flexion_r_vel"] - d["hip_flexion_r_vel"][t]) ** 2
        hip_r_add_vel = (obs_dict["hip_adduction_r_vel"] - d["hip_adduction_r_vel"][t]) ** 2

        v_loss = (
                ankle_l_vel
                + knee_l_vel
                + hip_l_flex_vel
                + hip_l_add_vel
                + ankle_r_vel
                + knee_r_vel
                + hip_r_flex_vel
                + hip_r_add_vel
        )
        velocity_reward = np.exp(-0.1 * v_loss)

        imitation_reward = 0.9 * position_reward + 0.1 * velocity_reward

        return 0.9 * imitation_reward + 0.1 * goal_reward

    def get_states(self) -> tuple[np.ndarray, dict]:
        """
        Gets the states that are needed for training. The states are measured from the environment.

        When creating the observation it is important to NOT nest the dictionary, this can help create the observation
        as a list with np.asarray(list(obs_dict.values()))

        :return: The state as an array for the network, the state as dictionary for human-readable data
        """
        obs_dict = {}
        obs_list = []
        for joint in self.jointSet:
            for i in range(joint.numCoordinates()):
                coord = joint.get_coordinates(i)
                name = coord.getName()
                if name in ["hip_rotation_r", "hip_rotation_l", "lumbar_extension"]:
                    continue
                obs_dict[f"{name}"] = coord.getValue(self.state)
                obs_dict[f"{name}_vel"] = coord.getSpeedValue(self.state)
                obs_list.append(coord.getValue(self.state))
                obs_list.append(coord.getSpeedValue(self.state))

        return np.asarray(obs_list), obs_dict

    def set_state(self, t: int) -> None:
        """
        Sets the state (joint positions and velocities) of the environment taken from the training data at index t.

        :param t: The index of where to take the values from the training data.
        """
        for joint in self.jointSet:
            for i in range(joint.numCoordinates()):
                coord = joint.get_coordinates(i)
                name = coord.getName()
                if name in ["hip_rotation_r", "hip_rotation_l", "lumbar_extension"]:
                    coord.setValue(self.state, 0.0)
                    coord.setSpeedValue(self.state, 0.0)
                    continue
                coord.setValue(self.state, self.data[name][t])
                coord.setSpeedValue(self.state, self.data[f"{name}_vel"][t])

    def get_observation_example(self) -> tuple[np.ndarray, dict]:
        """
        observation space of Brown for prosthetic model; size=91.
        From the paper:
        Learning to Walk With Deep Reinforcement Learning:
        Forward Dynamic Simulation of a Physics-Based Musculoskeletal Model of an Osseointegrated Transfemoral Amputee.

        This is how the model was able to walk in the previous architecture, that architecture was fit to this observation space.
        You could take inspiration from this (and see how some values are extracted from self.state, this is a bit odd as you first get the body/joint/muscle and then put in the state, to get the corresponding value)
        """

        obs_dict = {}
        obs_list = []

        """joints: 8 (hip) + 4(knee) + 4(ankle) + 12 (pelvis) = +28"""
        for joint in self.jointSet:
            for i in range(joint.numCoordinates()):
                coord = joint.get_coordinates(i)
                name = coord.getName()
                if name in [
                    "hip_rotation_r",
                    "hip_rotation_l",
                    "lumbar_extension",
                ]:
                    continue
                obs_dict[f"{name}"] = coord.getValue(self.state)
                obs_dict[f"{name}_vel"] = coord.getSpeedValue(self.state)
                obs_list.extend([coord.getValue(self.state), coord.getSpeedValue(self.state)])

        """muscle: 15 * 3 = +45"""

        for muscle in self.muscleSet:
            name = muscle.getName()
            obs_dict[f"{name}_fiber_length"] = muscle.getFiberLength(self.state) / muscle.getOptimalFiberLength()
            obs_dict[f"{name}_fiber_velocity"] = muscle.getFiberVelocity(self.state) / muscle.getOptimalFiberLength()
            obs_dict[f"{name}_fiber_force"] = muscle.getFiberForce(self.state) / muscle.getMaxIsometricForce()
            obs_list.extend(
                [obs_dict[f"{name}_fiber_length"], obs_dict[f"{name}_fiber_velocity"], obs_dict[f"{name}_fiber_force"]])

        """ground reaction forces: 2 * 3 (xyz) = +6"""
        for grf_name in ["foot_r", "foot_l"]:
            force = self.forceSet.get(grf_name)
            forces = force.getRecordValues(self.state)
            for i in range(3):  # xyz
                obs_dict[f"force_{grf_name}_{i}"] = forces.get(i) / (self.model_mass * self.gravity)
                obs_list.append(obs_dict[f"force_{grf_name}_{i}"])

        """actuator: 6 * 2 = +12"""
        for act_name in ["knee_actuator", "ankle_actuator"]:
            actuator = self.actuatorSet.get(act_name)
            act_scalar = ScalarActuator.safeDownCast(actuator)
            act_dyn = ActivationCoordinateActuator.safeDownCast(actuator)

            obs_dict[f"{act_name}_speed"] = act_scalar.getSpeed(self.state)
            obs_dict[f"{act_name}_control"] = act_scalar.getControl(self.state)
            obs_dict[f"{act_name}_actuation"] = act_scalar.getActuation(self.state)
            obs_dict[f"{act_name}_power"] = act_scalar.getPower(self.state)

            obs_dict[f"{act_name}_activation"] = act_dyn.getStateVariableValue(self.state,
                                                                               f'/forceset/{act_name}/activation')
            obs_dict[f"{act_name}_force"] = self.forceSet.get(act_name).getRecordValues(self.state).get(
                0) / CoordinateActuator.safeDownCast(actuator).getOptimalForce()

            obs_list.extend(
                [obs_dict[f"{act_name}_speed"], obs_dict[f"{act_name}_control"], obs_dict[f"{act_name}_actuation"],
                 obs_dict[f"{act_name}_power"], obs_dict[f"{act_name}_activation"], obs_dict[f"{act_name}_force"]])

        if len(obs_list) != 91:
            print(f"{len(obs_list)} != 91")
            exit()

        return np.asarray(obs_list), obs_dict
