import streamlit as st
import cv2
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributions as distributions
from torch.distributions import Categorical
import time
import base64
from io import BytesIO
from PIL import Image
import sys
import subprocess
import os
import asyncio

# Fix for asyncio event loop issues
try:
    asyncio.get_running_loop()
except RuntimeError:
    # No running event loop
    asyncio.set_event_loop(asyncio.new_event_loop())

# If on Windows, set compatible event loop policy
if sys.platform == 'win32' and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Set page title
st.set_page_config(page_title="A3C Reinforcement Learning - Kung Fu Master", layout="wide")

# Title and description
st.title("A3C Reinforcement Learning - Kung Fu Master")
st.write("""
This application demonstrates A3C (Asynchronous Advantage Actor-Critic) reinforcement learning
on the Kung Fu Master Atari game. Watch the AI agent learn to play the game!
""")

# Check and install required packages
missing_packages = []
try:
    import gymnasium as gym
    from gymnasium.spaces import Box
    from gymnasium import ObservationWrapper
except ImportError:
    missing_packages.append("gymnasium")

try:
    import ale_py
except ImportError:
    missing_packages.append("ale-py")

if missing_packages:
    st.warning(f"Missing required packages: {', '.join(missing_packages)}")
    st.info("Installing missing packages...")
    
    # Install packages
    for package in missing_packages:
        if package == "gymnasium":
            st.code("pip install gymnasium")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gymnasium"])
            st.code('pip install "gymnasium[atari, accept-rom-license]"')
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gymnasium[atari, accept-rom-license]"])
        elif package == "ale-py":
            st.code("pip install ale-py")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ale-py"])
    
    st.success("Packages installed. Please restart the application.")
    st.stop()

# Import gymnasium after ensuring it's installed
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium import ObservationWrapper

# Display installation verification info
st.header("Environment Setup")

# Check for ROM availability
try:
    import ale_py
    from ale_py import ALEInterface
    ale = ALEInterface()
    
    # List available Atari environments
    all_envs = gym.envs.registry.keys()
    atari_envs = [env for env in all_envs if 'ALE' in env or 'KungFu' in env]
    
    if atari_envs:
        st.success(f"Found {len(atari_envs)} Atari environments!")
        # Find KungFu-related environments
        kungfu_envs = [env for env in atari_envs if 'KungFu' in env or 'Kung-Fu' in env or 'kungfu' in env.lower()]
        if kungfu_envs:
            st.success(f"Found Kung Fu environments: {', '.join(kungfu_envs)}")
            env_id = kungfu_envs[0]  # Use the first available KungFu environment
        else:
            st.warning("No Kung Fu environments found. Using another Atari environment.")
            env_id = atari_envs[0]
    else:
        st.error("No Atari environments found. Manual configuration required.")
        
        # Provide options for common environment IDs
        env_options = [
            "ALE/KungFuMaster-v5",
            "KungFuMasterDeterministic-v0",
            "KungFuMaster-v0",
            "ALE/KungFuMaster-v0"
        ]
        env_id = st.selectbox("Select an environment ID to try:", env_options)
        
        # Option to enter custom environment ID
        custom_env = st.text_input("Or enter a custom environment ID:", "")
        if custom_env:
            env_id = custom_env

    # Display selected environment
    st.info(f"Using environment: **{env_id}**")
    
except Exception as e:
    st.error(f"Error detecting environments: {e}")
    st.error("Please ensure you have properly installed gymnasium with Atari support:")
    st.code("""
    pip install gymnasium
    pip install "gymnasium[atari, accept-rom-license]"
    pip install ale-py
    """)
    
    # Provide manual environment selection as fallback
    env_id = st.text_input("Enter Atari environment ID to use:", "ALE/KungFuMaster-v5")

# Define the neural network
class Network(nn.Module):
    def __init__(self, action_size):
        super(Network, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(3,3), stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=2)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(512, 128)
        self.fc2a = torch.nn.Linear(128, action_size)
        self.fc2s = torch.nn.Linear(128, 1)

    def forward(self, state):
        x = self.conv1(state)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        action_values = self.fc2a(x)
        state_value = self.fc2s(x)[0]
        return action_values, state_value

# Preprocess Atari environments
class PreprocessAtari(ObservationWrapper):
    def __init__(self, env, height=42, width=42, crop=lambda img: img, dim_order='pytorch', color=False, n_frames=4):
        super(PreprocessAtari, self).__init__(env)
        self.img_size = (height, width)
        self.crop = crop
        self.dim_order = dim_order
        self.color = color
        self.frame_stack = n_frames
        n_channels = 3 * n_frames if color else n_frames
        obs_shape = {'tensorflow': (height, width, n_channels), 'pytorch': (n_channels, height, width)}[dim_order]
        self.observation_space = Box(0.0, 1.0, obs_shape)
        self.frames = np.zeros(obs_shape, dtype=np.float32)

    def reset(self, **kwargs):
        self.frames = np.zeros_like(self.frames)
        obs, info = self.env.reset(**kwargs)
        self.update_buffer(obs)
        return self.frames, info

    def observation(self, img):
        img = self.crop(img)
        img = cv2.resize(img, self.img_size)
        if not self.color:
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype('float32') / 255.
        if self.color:
            self.frames = np.roll(self.frames, shift=-3, axis=0)
        else:
            self.frames = np.roll(self.frames, shift=-1, axis=0)
        if self.color:
            self.frames[-3:] = img
        else:
            self.frames[-1] = img
        return self.frames

    def update_buffer(self, obs):
        self.frames = self.observation(obs)

# Create environment
def make_env(env_id):
    try:
        env = gym.make(env_id, render_mode='rgb_array')
        env = PreprocessAtari(env, height=42, width=42, crop=lambda img: img, dim_order='pytorch', color=False, n_frames=4)
        return env
    except Exception as e:
        st.error(f"Error creating environment {env_id}: {e}")
        return None

# Agent class
class Agent():
    def __init__(self, action_size, learning_rate=1e-4):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.network = Network(action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

    def act(self, state):
        if state.ndim == 3:
            state = [state]
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action_values, _ = self.network(state)
        policy = F.softmax(action_values, dim=-1)
        return np.array([np.random.choice(len(p), p=p) for p in policy.detach().cpu().numpy()])

    def step(self, state, action, reward, next_state, done, discount_factor=0.99):
        batch_size = state.shape[0]
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.bool, device=self.device).to(dtype=torch.float32)
        action_values, state_value = self.network(state)
        _, next_state_value = self.network(next_state)
        target_state_value = reward + discount_factor * next_state_value * (1 - done)
        advantage = target_state_value - state_value
        probs = F.softmax(action_values, dim=-1)
        logprobs = F.log_softmax(action_values, dim=-1)
        entropy = -torch.sum(probs * logprobs, axis=-1)
        batch_idx = np.arange(batch_size)
        logp_actions = logprobs[batch_idx, action]
        actor_loss = -(logp_actions * advantage.detach()).mean() - 0.001 * entropy.mean()
        critic_loss = F.mse_loss(target_state_value.detach(), state_value)
        total_loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    # def save(self, path):
    #     torch.save(self.network.state_dict(), path)
        
    # def load(self, path):
    #     self.network.load_state_dict(torch.load(path, map_location=self.device))
    #     self.network.eval()

# Environment batch for training
class EnvBatch:
    def __init__(self, env_id, n_envs=10):
        self.envs = [make_env(env_id) for _ in range(n_envs)]
        self.valid_envs = [env for env in self.envs if env is not None]
        if len(self.valid_envs) < n_envs:
            st.warning(f"Only {len(self.valid_envs)} of {n_envs} environments were successfully created.")

    def reset(self):
        _states = []
        for env in self.valid_envs:
            _states.append(env.reset()[0])
        return np.array(_states)

    def step(self, actions):
        next_states = []
        rewards = []
        dones = []
        infos = []

        for env, action in zip(self.valid_envs, actions[:len(self.valid_envs)]):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_states.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return np.array(next_states), np.array(rewards), np.array(dones), infos

# Function to evaluate agent
def evaluate(agent, env, n_episodes=1):
    episodes_rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        while True:
            action = agent.act(state)
            state, reward, terminated, truncated, info = env.step(action[0])
            done = terminated or truncated
            total_reward += reward
            if done:
                break
        episodes_rewards.append(total_reward)
    return episodes_rewards

# Function to convert frames to video display in Streamlit
def get_img_as_base64(img):
    buffered = BytesIO()
    img = Image.fromarray(img)
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Sidebar controls
with st.sidebar:
    st.header("Training Parameters")
    learning_rate = st.slider("Learning Rate", min_value=0.00001, max_value=0.001, value=0.0001, step=0.00001, format="%.5f")
    discount_factor = st.slider("Discount Factor", min_value=0.8, max_value=0.999, value=0.99, step=0.001, format="%.3f")
    num_environments = st.slider("Number of Parallel Environments", min_value=1, max_value=20, value=10)
    
    st.header("Testing")
    test_agent = st.button("Test Agent")
    
    # st.header("Save/Load Model")
    # save_model = st.button("Save Model")
    # load_model = st.button("Load Model")

# Initialize environment and agent
try:
    # Try to create environment
    env = make_env(env_id)
    
    if env is not None:
        state_shape = env.observation_space.shape
        number_actions = env.action_space.n
        
        # Display environment info
        st.sidebar.header("Environment Info")
        st.sidebar.write(f"State shape: {state_shape}")
        st.sidebar.write(f"Number of actions: {number_actions}")
        try:
            st.sidebar.write(f"Action meanings: {env.env.env.env.get_action_meanings()}")
        except:
            st.sidebar.write("Could not retrieve action meanings.")
        
        # Initialize agent
        if 'agent' not in st.session_state:
            st.session_state.agent = Agent(number_actions, learning_rate)
        
        # Main training section
        st.header("Training")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            train_button = st.button("Start Training")
            stop_button = st.button("Stop Training")
            training_progress = st.empty()
            training_metrics = st.empty()
        
        with col2:
            episodes = st.number_input("Training Episodes", min_value=1, max_value=10000, value=1000)
            eval_interval = st.number_input("Evaluation Interval", min_value=1, max_value=1000, value=100)
        
        # Create a placeholder for game display
        game_display = st.empty()
        
        # Training logic
        if train_button:
            st.session_state.training = True
            
            # Create environment batch for training
            env_batch = EnvBatch(env_id, num_environments)
            if not env_batch.valid_envs:
                st.error("No valid environments could be created for training.")
                st.session_state.training = False
            else:
                batch_states = env_batch.reset()
                
                progress_bar = training_progress.progress(0)
                
                for i in range(episodes):
                    if not getattr(st.session_state, 'training', True) or stop_button:
                        st.session_state.training = False
                        break
                        
                    batch_actions = st.session_state.agent.act(batch_states)
                    batch_next_states, batch_rewards, batch_dones, _ = env_batch.step(batch_actions)
                    batch_rewards *= 0.01  # Scale rewards
                    
                    st.session_state.agent.step(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, discount_factor)
                    batch_states = batch_next_states
                    
                    # Update progress
                    progress_bar.progress((i + 1) / episodes)
                    
                    # Evaluate periodically
                    if (i + 1) % eval_interval == 0:
                        rewards = evaluate(st.session_state.agent, env, n_episodes=3)
                        avg_reward = np.mean(rewards)
                        training_metrics.write(f"Episode {i+1}/{episodes}, Average Reward: {avg_reward:.2f}")
        
        if stop_button:
            st.session_state.training = False
            st.write("Training stopped!")
        
        # Testing logic
        if test_agent:
            st.header("Agent Testing")
            test_env = make_env(env_id)
            state, _ = test_env.reset()
            done = False
            total_reward = 0
            
            # Create columns for display
            frame_col, info_col = st.columns([3, 1])
            
            with info_col:
                reward_placeholder = st.empty()
                action_placeholder = st.empty()
            
            with frame_col:
                frame_placeholder = st.empty()
            
            frames_buffer = []
            
            while not done:
                # Get raw frame for display
                # frame = test_env.render()
                frames_buffer.append(frame)
                
                # Display the current frame
                frame_placeholder.image(frame, caption="Game Screen", use_container_width=True)  # Fixed deprecated parameter
                
                # Get action from agent
                action = st.session_state.agent.act(state)
                action_idx = action[0]
                
                # Update info display
                try:
                    action_name = test_env.env.env.env.get_action_meanings()[action_idx]
                    action_placeholder.write(f"Action: {action_name} ({action_idx})")
                except:
                    action_placeholder.write(f"Action: {action_idx}")
                
                # Take step in environment
                state, reward, terminated, truncated, info = test_env.step(action_idx)
                done = terminated or truncated
                total_reward += reward
                
                # Update reward display
                reward_placeholder.write(f"Current Reward: {reward:.2f}\nTotal Reward: {total_reward:.2f}")
                
                # Slow down visualization
                time.sleep(0.05)
            
            # Final results
            st.write(f"Game finished with total reward: {total_reward:.2f}")
            
            # Create and display video
            if frames_buffer:
                st.header("Replay")
                frames_placeholder = st.empty()
                
                # Convert frames to video display
                video_html = f"""
                <div style="display: flex; justify-content: center;">
                    <video autoplay loop controls style="max-width: 100%; height: auto;">
                """
                
                for i, frame in enumerate(frames_buffer):
                    img_str = get_img_as_base64(frame)
                    video_html += f'<img src="data:image/png;base64,{img_str}" style="display: none;">'
                
                video_html += """
                    </video>
                </div>
                <script>
                    const video = document.querySelector('video');
                    const imgs = video.querySelectorAll('img');
                    let imgIndex = 0;
                    
                    function updateFrame() {
                        for (let i = 0; i < imgs.length; i++) {
                            imgs[i].style.display = 'none';
                        }
                        imgs[imgIndex].style.display = 'block';
                        imgIndex = (imgIndex + 1) % imgs.length;
                    }
                    
                    setInterval(updateFrame, 50);
                    updateFrame();
                </script>
                """
                
                frames_placeholder.markdown(video_html, unsafe_allow_html=True)
        
        # Save/Load model logic
        # if save_model:
        #     try:
        #         model_filename = "kungfu_master_agent.pth"
        #         st.session_state.agent.save(model_filename)
        #         st.sidebar.success(f"Model saved as {model_filename}!")
        #     except Exception as e:
        #         st.sidebar.error(f"Error saving model: {e}")
        
        # if load_model:
        #     try:
        #         model_filename = "kungfu_master_agent.pth"
        #         st.session_state.agent.load(model_filename)
        #         st.sidebar.success(f"Model {model_filename} loaded successfully!")
        #     except Exception as e:
        #         st.sidebar.error(f"Error loading model: {e}")
    else:
        st.error(f"Failed to create environment: {env_id}")
        st.error("Most likely causes:")
        st.write("1. Missing Atari ROMs")
        st.write("2. Environment ID has changed in the newer Gymnasium versions")
        st.write("3. Required packages are not properly installed")
        
        st.info("Manual installation instructions:")
        st.code("""
        # Install required packages
        pip install streamlit
        pip install torch torchvision
        pip install opencv-python
        pip install Pillow
        
        # Install Atari-specific packages
        pip install ale-py
        pip install gymnasium
        pip install "gymnasium[atari, accept-rom-license]"
        """)

except Exception as e:
    st.error(f"Unexpected error: {e}")
    st.error("Make sure you have installed all required packages and dependencies!")