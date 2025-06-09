import numpy as np
import pandas as pd
from gymnasium import Env, spaces
from opensim import ActivationCoordinateActuator, Constant, Logger, Manager, Model, PrescribedController, \
    ScalarActuator, CoordinateActuator
from typing import Optional
import math
import os

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

 ## Not used rn, but very useful to have for now
LOCKED_DOFS = {
    "cmc_flexion", 
    "cmc_abduction", 
    "mp_flexion", 
    "ip_flexion",
    "2mcp_flexion", 
    "2mcp_abduction", 
    "2pm_flexion", 
    "2md_flexion"
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

        os.makedirs("model_dump", exist_ok=True)
        self.data.to_csv("model_dump/loaded_data.csv", index=False)
        print(f"Saved loaded imitation data to model_dump/loaded_data.csv with shape {self.data.shape}")

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
        Resets the environment to a valid initial state.
        Attempts to randomly initialize from imitation data, skipping invalid states.
        """
        self.i_step = 0
        self.state = self.model.initializeState()

        def is_valid_state() -> bool:
            try:
                # Check all muscles have nonzero fiber length
                for i in range(self.muscleSet.getSize()):
                    muscle = self.muscleSet.get(i)
                    fiber_len = muscle.getFiberLength(self.state)
                    if fiber_len <= 0.001:  # Tiny tolerance
                        return False
                return True
            except Exception:
                return False

        if self.random_init and self.data.shape[0] > 400:
            max_attempts = 50
            for attempt in range(max_attempts):
                self.i_step = np.random.randint(0, self.data.shape[0] - 400)
                try:
                    self.set_state(self.i_step)
                    self.model.equilibrateMuscles(self.state)
                    if is_valid_state():
                        break  # success
                except Exception as e:
                    print(f"[reset] Skipping i_step={self.i_step} due to error: {e}")
            else:
                print("[reset] Failed to find a valid random state. Falling back to i_step=0")
                self.i_step = 0
                self.set_state(self.i_step)
                self.model.equilibrateMuscles(self.state)
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
        actions = np.clip(actions, 0.01, 0.99)

        self.actuate(actions=actions)

        try:
            self.state = self.manager.integrate(self.step_size * self.i_step)
            self.model.realizePosition(self.state)

            self.critical_muscles = {"DELT1", "EIP", "EPB"}
            for name in self.critical_muscles:
                muscle = self.forceSet.get(name)
                if muscle.getFiberLength(self.state) <= 0.001:
                    raise ValueError(f"Fiber length too short for {name}")
        except Exception as e:
            print(f"[step] Truncating due to simulation error at step {self.i_step}: {e}")
            # Save observation for debugging
            obs_list, obs_dict = self.get_states()
            pd.DataFrame(obs_dict, index=[0]).to_csv("model_dump/failing_obs.csv")
            return obs_list, 0.0, False, True, {}

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
        actions = np.clip(actions, 0.01, 0.99)  # safe bounds
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
    def is_truncated(self, obs_dict: dict) -> bool:
        """
        Determines if the environment is truncated.

        The truncated state is when the environment has become "unstable", the model has fallen over or gone of track.
        You can define this how you like, but keep the conditions within the bounds of the training data.

        :param obs_dict: The observation of the environment
        :return: True if the environment is truncated, false otherwise
        """
        try:
            for i in range(self.muscleSet.getSize()):
                muscle = self.muscleSet.get(i)
                if muscle.getFiberLength(self.state) <= 0.001:
                    return True
        except Exception:
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

        # === GOAL REWARD ===
        # try: ## This is a temporary goal reward funtion, Robert made a proper one.
        #     hand_y = obs_dict["ty"]
        #     target_y = d["ty"][t]
        #     dist_squared = (hand_y - target_y) ** 2
        #     goal_reward = np.exp(-10 * dist_squared)
        # except KeyError:
        #     print("Hand/target markers are missing")
        #     goal_reward = 1.0

        cmc_abduction = self.get_joint_angle("CMC1b", "cmc_abduction")
        two_mcp_flexion = self.get_joint_angle("2MCP", "2mcp_flexion")

        # Calculate for finger contact
        fingers_touching = two_mcp_flexion < math.radians(0.5 * (math.degrees(cmc_abduction) - 5) + 57)

        # Get muscle forces
        eip_force = self.get_muscle_force("EIP")
        epb_force = self.get_muscle_force("EPB")

        # Grasp reward: fingers not touching, but muscles are active
        if not fingers_touching and (eip_force > 0.0 or epb_force > 0.0):
            goal_reward = 1.0
        else:
            goal_reward = 0.0

        # === IMITATION REWARD ===

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
                if name in COORDS_TO_SKIP:
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

    def get_joint_angle(self, joint_name: str, coordinate_name: str) -> float:
        """
        Retrieves the current angle/value of a specific coordinate in a joint.

        :param joint_name: The name of the joint
        :param coordinate_name: The name of the coordinate
        :return: The current angle/value of the coordinate
        :raises: ValueError if joint or coordinate is not found
        """
        if not self.jointSet.contains(joint_name):
            raise ValueError(f"Joint '{joint_name}' not found in JointSet.")
        joint = self.jointSet.get(joint_name)
        for i in range(joint.numCoordinates()):
            coord = joint.get_coordinates(i)
            if coord.getName() == coordinate_name:
                return coord.getValue(self.state)
        raise ValueError(f"Coordinate '{coordinate_name}' not found in joint '{joint_name}'.")
    
    def get_muscle_force(self, muscle_name: str) -> float:
        """
        Safely get the current force exerted by a named muscle.

        :param muscle_name: The name of the muscle in the ForceSet
        :return: The force in Newtons
        :raises: ValueError if the muscle name is invalid
        """
        if not self.forceSet.contains(muscle_name):
            raise ValueError(f"Muscle '{muscle_name}' not found in ForceSet.")
        return self.forceSet.get(muscle_name).getRecordValues(self.state).get(0)