import numpy as np
import pandas as pd
from gymnasium import Env, spaces
from opensim import ActivationCoordinateActuator, Constant, Logger, Manager, Model, Muscle, PrescribedController, \
    ScalarActuator, CoordinateActuator
from typing import Optional

"""Opensim environment for musculoskeletal movement"""

COORDS_TO_SKIP = {
    "sternoclavicular_r2",
    "sternoclavicular_r3",
    "unrotscap_r3",
    "unrotscap_r2",
    "acromioclavicular_r2",
    "acromioclavicular_r3",
    "acromioclavicular_r1",
    "unrothum_r1",
    "unrothum_r3",
    "unrothum_r2",
    "shoulder1_r2",
    "proximal_distal_r1",
    "proximal_distal_r3",
    "cmc_flexion",
    "mp_flexion",
    "ip_flexion",
    "2mcp_abduction",
    "2pm_flexion",
    "2md_flexion",
    "APLpt_tx",
    "APLpt_tx",
    "APLpt_ty",
    "APLpt_tz",
    "FPLpt_tx",
    "FPLpt_ty",
    "FPLpt_tz",
    "FPLpt2_tx",
    "FPLpt2_ty",
    "FPLpt2_tz",
    "rx",
    "ry",
    "rz",
    "tx",
    "ty",
    "tz"
}

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
        print("Resetting environment...")
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

        try:
            self.state = self.manager.integrate(self.step_size * self.i_step)
            self.model.realizeAcceleration(self.state)
        except Exception as e:
            print(f"[ERROR] Simulation error at step {self.i_step}: {e}")
            obs, _ = self.get_states()

            if np.isnan(obs).any():
                print(f"[FATAL] NaNs detected in observation after integration failure at step {self.i_step}")

            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
            return obs, 0.0, False, True, {}

        obs_list, obs_dict = self.get_states()

        if np.isnan(obs_list).any():
            print(f"[FATAL] NaNs detected in valid state at step {self.i_step}")
            obs_list = np.nan_to_num(obs_list, nan=0.0, posinf=1.0, neginf=-1.0)
            return obs_list, 0.0, False, True, {}

        if self.is_truncated(obs_dict):
            print(f"[INFO] Truncating due to invalid muscle state at step {self.i_step}")
            return obs_list, 0.0, False, True, {}

        reward = self.get_reward(obs_dict)
        terminated = self.is_terminated()
        truncated = self.is_truncated(obs_dict)

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

    def is_truncated(self, obs_dict: dict) -> bool:
        """
        Determines if the environment is truncated.

        The truncated state is when the environment has become "unstable", the model has fallen over or gone of track.
        You can define this how you like, but keep the conditions within the bounds of the training data.

        :param obs_dict: The observation of the environment
        :return: True if the environment is truncated, false otherwise
        """
        for i in range(self.muscleSet.getSize()):
            force = self.muscleSet.get(i)
            muscle = Muscle.safeDownCast(force)
            if muscle is None:
                continue  # Not a Muscle object, skip it
            if muscle.getFiberLength(self.state) <= 0.001:
                print("TRUNCATED: Muscle fiber length is too short.")
                return True
        return False


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

        p_loss = 0.0
        v_loss = 0.0

        for name in obs_dict:
            base_name = name.replace("_vel", "")
            if base_name in COORDS_TO_SKIP:
                continue  # skip locked DoFs

            if name.endswith("_vel") and name in d.columns:
                v_loss += (obs_dict[name] - d[name][t]) ** 2
            elif name in d.columns:
                p_loss += (obs_dict[name] - d[name][t]) ** 2

        position_reward = np.exp(-4 * p_loss)
        velocity_reward = np.exp(-0.1 * v_loss)
        imitation_reward = 0.9 * position_reward + 0.1 * velocity_reward

        return imitation_reward

    def get_states(self) -> tuple[np.ndarray, dict]:
        """
        Gets the states that are needed for training. The states are measured from the environment.

        When creating the observation it is important to NOT nest the dictionary, this can help create the observation
        as a list with np.asarray(list(obs_dict.values()))

        :return: The state as an array for the network, the state as dictionary for human-readable data
        """
        obs_dict = {}
        obs_list = []

        try:
            for joint in self.jointSet:
                for i in range(joint.numCoordinates()):
                    coord = joint.get_coordinates(i)
                    name = coord.getName()

                    if name in COORDS_TO_SKIP:
                        continue

                    try:
                        pos = coord.getValue(self.state)
                        vel = coord.getSpeedValue(self.state)

                        if np.isnan(pos) or np.isnan(vel):
                            print(f"[NaN WARNING] NaN in joint state: {name} at step {self.i_step}")
                            pos, vel = 0.0, 0.0

                    except Exception as e:
                        print(f"[WARN] Failed to get state for coordinate {name} at step {self.i_step}: {e}")
                        pos, vel = 0.0, 0.0

                    obs_dict[f"{name}"] = pos
                    obs_dict[f"{name}_vel"] = vel
                    obs_list.extend([pos, vel])

        except Exception as e:
            print(f"[ERROR] Failed in get_states loop at step {self.i_step}: {e}")
            # Fallback: Create safe dummy observation of same length
            obs_list = [0.0] * (len(self.jointSet) * 2)  # fallback estimate

        obs_array = np.nan_to_num(np.asarray(obs_list, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        return obs_array, obs_dict

    def set_state(self, t: int) -> None:
        """
        Sets the state (joint positions and velocities) of the environment taken from the training data at index t.

        :param t: The index of where to take the values from the training data.
        """
        for joint in self.jointSet:
            for i in range(joint.numCoordinates()):
                coord = joint.get_coordinates(i)
                name = coord.getName()
                if name in COORDS_TO_SKIP:
                    coord.setValue(self.state, 0.0)
                    coord.setSpeedValue(self.state, 0.0)
                    continue
                coord.setValue(self.state, self.data[name][t])
                coord.setSpeedValue(self.state, self.data[f"{name}_vel"][t])

    '''def get_observation_example(self) -> tuple[np.ndarray, dict]:
        This is currently not relevant as it is from the legs.
    
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

        return np.asarray(obs_list), obs_dict'''