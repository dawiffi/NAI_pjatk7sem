import gymnasium as gym
import ale_py
import os
import torch
import time
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

"""
by Kacper Pach s27112 & Dawid Frontczak s29608
Ustawienie środowiska oraz przykład wykonania programu w readme (https://github.com/dawiffi/NAI_pjatk7sem/blob/main/reinforced_learning/README.md)

Moduł do trenowania i testowania agenta Reinforcement Learning (DQN) w grze Frogger.

Skrypt wykorzystuje bibliotekę Stable Baselines3 do implementacji algorytmu Deep Q-Network (DQN)
oraz Gymnasium z paczką ALE (Arcade Learning Environment) do emulacji gry Atari Frogger.
Obsługuje automatyczne wczytywanie zapisanego modelu oraz bezpieczne zapisywanie po przerwaniu treningu.
"""

# Konfiguracja urządzenia i środowiska
gym.register_envs(ale_py)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "dqn_frogger_model.zip"

env = make_atari_env("ALE/Frogger-v5", n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)

# Logika wczytywania lub tworzenia modelu
if os.path.exists(model_path):
    print(f"Znaleziono zapisany model ({model_path}). Wczytywanie...")
    model = DQN.load(model_path, env=env, device=device)
else:
    print("Nie znaleziono modelu. Tworzenie nowej sieci...")
    model = DQN(
        "CnnPolicy", 
        env, 
        verbose=1, 
        device=device,
        buffer_size=50000, 
        learning_rate=1e-4,
        optimize_memory_usage=False
    )

# Trening
try:
    print("Rozpoczynam naukę... (Naciśnij Ctrl+C, aby przerwać i zapisać)")
    model.learn(total_timesteps=100000, reset_num_timesteps=False)
except KeyboardInterrupt:
    print("\nPrzerwano ręcznie.")

# Zapisanie postępów
print("Zapisywanie modelu...")
model.save(model_path)

# Testowanie bota
print("Testowanie bota...")
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render("human")
    time.sleep(0.05)