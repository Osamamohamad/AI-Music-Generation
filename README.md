# 🎵 AI Music Composer Chat App

An AI-powered desktop application that generates original piano melodies using a **Long Short-Term Memory (LSTM)** deep learning model. The application features a modern chat-style interface built with **CustomTkinter**, allowing users to generate MIDI compositions in an intuitive and interactive way.

---

##  Project Structure

| File | Description |
|------|-------------|
| **app.py** | Launches the desktop application, manages the graphical user interface (GUI), and performs real-time music generation using the trained model. |
| **preprocess.py** | Extracts musical notes and chords from MIDI files, then preprocesses them into sequences suitable for model training. |
| **train.py** | Builds, trains, and saves the LSTM neural network for music generation. |
| **notes** | Serialized dataset containing the processed musical notes and chord sequences. |
| **music_model.h5** | Pre-trained LSTM model used for generating new melodies. |


---

##  Technologies Used

- Python 3.10+
- TensorFlow / Keras
- Music21
- CustomTkinter
- NumPy
- Pickle

---
## 📊 Dataset
The model was trained on a custom selection of piano MIDI files. You can download the dataset used for training from the link below:
* [Download Classical Music MIDI Dataset](https://www.kaggle.com/datasets/soumikrakshit/classical-music-midi)

---
##  Demo
<img width="1030" height="604" alt="Screenshot 2026-07-30 022200" src="https://github.com/user-attachments/assets/d4844d27-78c5-4756-bf91-e7121065718c" />



---

##  Installation


Install the required dependencies:

```bash
    pip install -r requirements.txt
```

Or install them manually:

```bash
    pip install tensorflow music21 customtkinter numpy
```

---

## ▶ Running the Application

Make sure the following files are located in the project directory:

- `music_model.h5`
- `notes`

Then start the application:

```bash
    python app.py
```

---


