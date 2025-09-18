# Gesture Control System 🎥🖱️✋🎮🎨

A **multi-functional gesture recognition system** that uses computer vision, hand-tracking, and AI-powered tools to control media, presentations, mouse, games, and even generate AI art — **all with your hands**.  

The project integrates **OpenCV, MediaPipe, PyQt6, PyAutoGUI, Pycaw, Pygame, and Hugging Face APIs** to provide an interactive gesture-controlled experience.

---

## 🚀 Features

- **Movie & Presentation Mode**
  - Play/Pause media with a fist ✊  
  - Control volume with a pinch 🤏  
  - Slide navigation with left/right index finger 👉👈  

- **Mouse Control Mode**
  - Control the cursor with your index finger  
  - Left click (Thumb–Index pinch)  
  - Right click (Thumb–Middle pinch)  

- **Testing Modes**
  - **Hand Detection** → Visualize hand landmarks  
  - **Gesture Detection** → Finger counting, peace ✌️, thumbs up 👍, etc.  

- **Air Drawing & Games**
  - **Air Canvas** → Draw in the air, fist to clear  
  - **Flappy Bird Game** → Control the bird with an open palm ✋  
  - **AI Air Canvas** → Draw → Pinch → Convert sketch to **AI-generated art**  

---

## 📂 Project Structure

```
Gesture-Control-System/
│── project.py            # Main application script
│── Images/               # Instruction images (e.g., S5.png, S6.png, etc.)
│── README.md             # Documentation (this file)
```

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/Gesture-Control-System.git
   cd Gesture-Control-System
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install opencv-python mediapipe pyqt6 pyautogui pygame numpy pillow transformers huggingface_hub pycaw comtypes
   ```

   > ⚠️ `pycaw` works only on Windows for volume control.

---

## ▶️ Usage

1. **Launch GUI**
   ```bash
   python project.py
   ```

   A window will appear with all available modes.

2. **Run specific mode directly**
   ```bash
   python project.py --mode session5       # Play/Pause control
   python project.py --mode session6       # Volume control
   python project.py --mode session7       # Slide/Video navigation
   python project.py --mode session8       # Mouse control
   python project.py --mode session9       # Air Canvas
   python project.py --mode session10      # Flappy Bird
   python project.py --mode ai_air_canvas  # AI-powered Air Canvas
   ```

---

## 📸 Screenshots

Screenshots of each mode are available inside the **Images/** folder.  
The GUI also displays these images alongside instructions when you launch a mode.  

---

## 🔑 Requirements

- Python **3.8+**
- Webcam for gesture recognition  
- Windows (for full feature support, esp. `pycaw` volume control)  
- GPU recommended for faster AI image generation  

---

## 🧠 AI-Powered Features

- Uses **Hugging Face API** + **Stable Diffusion XL** for converting air sketches into AI-generated art.  
- Requires a Hugging Face API key — replace the placeholder key in `project.py` with your own.  

```python
client = InferenceClient(provider="nscale", api_key="your_api_key_here")
```

Get a free key from: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)  

---

## 🤝 Contributing

1. Fork the project  
2. Create a new branch (`feature-newmode`)  
3. Commit your changes  
4. Push to the branch  
5. Open a Pull Request  

---

## 📜 License

This project is licensed under the **MIT License** – feel free to use, modify, and share.  
