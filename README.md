# Eye Tracking OBS Scene Switcher

An **eye-tracking powered OBS scene switcher** that dynamically changes your OBS scenes based on where you are looking on the screen.  
Built with **Python**, **OpenCV**, **MediaPipe**, and **OBS WebSocket API**, this project enables creators, streamers, and presenters to create hands-free scene transitions driven by eye gaze.

---

## ✨ Features
- Real-time **eye tracking** using a webcam.  
- Detects gaze direction: **left, right, up, down, center**.  
- Maps gaze regions to **OBS scenes** automatically.  
- Smoothens gaze signal using a **rolling average filter** to reduce jitter.  
- Integrates seamlessly with OBS via **OBS WebSocket API**.  
- Debug mode with on-screen visualization of gaze detection.

## Note

 - You may have to play around with the parameters for your own setup.
---

## 🖥️ How It Works
1. The script uses **MediaPipe Face Mesh** to detect eye landmarks.  
2. Calculates relative pupil/gaze position on the screen.  
3. Applies **smoothing (rolling average)** to reduce noise and avoid flickering.  
   - This is done by keeping a buffer of the last N gaze predictions.  
   - A majority vote or averaged value stabilizes the gaze before mapping to an OBS scene.  
4. Depending on where you look (left, right, top, bottom, center), it sends a request to OBS to switch scenes.

---

## 📦 Requirements
Install the following Python packages:

```bash
pip install opencv-python mediapipe obs-websocket-py numpy

