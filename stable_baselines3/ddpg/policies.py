# DDPG can be view as a special case of TD3
from stable_baselines3.td3.policies import (  # noqa:F401
    CnnPolicy,
    MlpPolicy,
    MultiInputPolicy,
)
