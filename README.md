# 🧠 A3C Reinforcement Learning Agent with Streamlit Interface
This project showcases an Asynchronous Advantage Actor-Critic (A3C) reinforcement learning agent trained to play the KungFuMaster Atari environment using the Gymnasium API. The project leverages PyTorch for deep reinforcement learning and integrates a sleek, interactive Streamlit frontend for visualizing agent performance and making it accessible via the web.

🚀 Features
🧮 A3C-based deep reinforcement learning with parallel training

🕹️ Trained on the KungFuMasterDeterministic-v0 environment (ALE backend)

🎞️ Real-time rendering of agent actions via Streamlit

📉 Display of episode rewards, frame-by-frame visualizations, and inference stats

☁️ Hosted on Streamlit Cloud with a lightweight, GPU-free deployment

⚠️ Deployment Notes
  Due to Streamlit Cloud's headless and resource-limited environment:

  GUI-dependent libraries (e.g., cv2.imshow) are not used

  Performance may vary compared to local machines with GPU support

📦 Dependencies

   streamlit

  opencv-python

  numpy

  torch

  gymnasium[atari]

  ale-py

Install them using : pip install -r requirements.txt

🌐 Live Demo
Check out the live version: https://a3c-reinforcement-learning.streamlit.app
