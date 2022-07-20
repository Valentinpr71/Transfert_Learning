import cv2
import gym
import numpy as np
import torch


def ReplayBuffer(state_dim, atari_preprocessing, batch_size, buffer_size, device):
    return StandardBuffer(state_dim, batch_size, buffer_size, device)



# Generic replay buffer for standard gym tasks
class StandardBuffer(object):
    def __init__(self, state_dim, batch_size, buffer_size, device):
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.device = device

        self.ptr = 0
        self.crt_size = 0

        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, 1))
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

    def add(self, state, action, next_state, reward, done, episode_done, episode_start):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)

    def sample(self):
        ind = np.random.randint(0, self.crt_size, size=self.batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def save(self, save_folder):
        np.save(f"{save_folder}_state.npy", self.state[:self.crt_size])
        np.save(f"{save_folder}_action.npy", self.action[:self.crt_size])
        np.save(f"{save_folder}_next_state.npy", self.next_state[:self.crt_size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.crt_size])
        np.save(f"{save_folder}_not_done.npy", self.not_done[:self.crt_size])
        np.save(f"{save_folder}_ptr.npy", self.ptr)

    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}_reward.npy")

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.crt_size = min(reward_buffer.shape[0], size)

        self.state[:self.crt_size] = np.load(f"{save_folder}_state.npy")[:self.crt_size]
        self.action[:self.crt_size] = np.load(f"{save_folder}_action.npy")[:self.crt_size]
        self.next_state[:self.crt_size] = np.load(f"{save_folder}_next_state.npy")[:self.crt_size]
        self.reward[:self.crt_size] = reward_buffer[:self.crt_size]
        self.not_done[:self.crt_size] = np.load(f"{save_folder}_not_done.npy")[:self.crt_size]

        print(f"Replay Buffer loaded with {self.crt_size} elements.")


# Atari Preprocessing
# Code is based on https://github.com/openai/gym/blob/master/gym/wrappers/atari_preprocessing.py
class AtariPreprocessing(object):
    def __init__(
            self,
            env,
            frame_skip=4,
            frame_size=84,
            state_history=4,
            done_on_life_loss=False,
            reward_clipping=True,  # Clips to a range of -1,1
            max_episode_timesteps=27000
    ):
        self.env = env.env
        self.done_on_life_loss = done_on_life_loss
        self.frame_skip = frame_skip
        self.frame_size = frame_size
        self.reward_clipping = reward_clipping
        self._max_episode_steps = max_episode_timesteps
        self.observation_space = np.zeros((frame_size, frame_size))
        self.action_space = self.env.action_space

        self.lives = 0
        self.episode_length = 0

        # Tracks previous 2 frames
        self.frame_buffer = np.zeros(
            (2,
             self.env.observation_space.shape[0],
             self.env.observation_space.shape[1]),
            dtype=np.uint8
        )
        # Tracks previous 4 states
        self.state_buffer = np.zeros((state_history, frame_size, frame_size), dtype=np.uint8)

    def reset(self):
        self.env.reset()
        self.lives = self.env.ale.lives()
        self.episode_length = 0
        self.env.ale.getScreenGrayscale(self.frame_buffer[0])
        self.frame_buffer[1] = 0

        self.state_buffer[0] = self.adjust_frame()
        self.state_buffer[1:] = 0
        return self.state_buffer

    # Takes single action is repeated for frame_skip frames (usually 4)
    # Reward is accumulated over those frames
    def step(self, action):
        total_reward = 0.
        self.episode_length += 1

        for frame in range(self.frame_skip):
            _, reward, done, _ = self.env.step(action)
            total_reward += reward

            if self.done_on_life_loss:
                crt_lives = self.env.ale.lives()
                done = True if crt_lives < self.lives else done
                self.lives = crt_lives

            if done:
                break

            # Second last and last frame
            f = frame + 2 - self.frame_skip
            if f >= 0:
                self.env.ale.getScreenGrayscale(self.frame_buffer[f])

        self.state_buffer[1:] = self.state_buffer[:-1]
        self.state_buffer[0] = self.adjust_frame()

        done_float = float(done)
        if self.episode_length >= self._max_episode_steps:
            done = True

        return self.state_buffer, total_reward, done, [np.clip(total_reward, -1, 1), done_float]

    def adjust_frame(self):
        # Take maximum over last two frames
        np.maximum(
            self.frame_buffer[0],
            self.frame_buffer[1],
            out=self.frame_buffer[0]
        )

        # Resize
        image = cv2.resize(
            self.frame_buffer[0],
            (self.frame_size, self.frame_size),
            interpolation=cv2.INTER_AREA
        )
        return np.array(image, dtype=np.uint8)

    def seed(self, seed):
        self.env.seed(seed)


# Create environment, add wrapper if necessary and create env_properties
def make_env(env_name, manager):
    env = gym.make(env_name, dim=manager.dim, data=manager.data)

    env = env

    state_dim = env.observation_space.shape[0]

    return (
        env,
        state_dim,
        env.action_space.n
    )