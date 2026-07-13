import os
import threading
import pickle
import random
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
from music21 import instrument, note, stream, chord

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class MusicChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Music Generator Chat")
        self.geometry("550x650")
        self.resizable(False, False)

        self.top_label = ctk.CTkLabel(self, text="🎹 AI Music Composer", font=ctk.CTkFont(size=20, weight="bold"))
        self.top_label.pack(pady=15)

        self.chat_frame = ctk.CTkScrollableFrame(self, width=500, height=450, corner_radius=15)
        self.chat_frame.pack(pady=10, padx=20, fill="both", expand=True)

        self.control_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.control_frame.pack(pady=20, fill="x", padx=30)

        self.generate_btn = ctk.CTkButton(self.control_frame, text="✨ Creating new music",
                                          font=ctk.CTkFont(size=15, weight="bold"),
                                          height=45, corner_radius=10, command=self.start_generation_thread)
        self.generate_btn.pack(fill="x", side="left", expand=True, padx=5)

        self.download_btn = ctk.CTkButton(self.control_frame, text="📥 Download the melody (ready)",
                                          fg_color="#2ecc71", hover_color="#27ae60",
                                          font=ctk.CTkFont(size=15, weight="bold"),
                                          height=45, corner_radius=10, command=self.download_midi)

        self.add_chat_bubble("AI Composer","Welcome! The model is now connected to the interface and ready to play a live, real tune. Click below to begin. 🎼",
                             is_user=False)

    def add_chat_bubble(self, sender, text, is_user=False):
        bubble_bg = "#1f538d" if is_user else "#333333"
        anchor_side = "e" if is_user else "w"

        frame = ctk.CTkFrame(self.chat_frame, fg_color=bubble_bg, corner_radius=12)
        frame.pack(pady=8, padx=10, anchor=anchor_side)

        label = ctk.CTkLabel(frame, text=f"💬 {sender}:\n{text}", justify="left",
                             font=ctk.CTkFont(size=13), wraplength=350)
        label.pack(pady=8, padx=12)
        self.chat_frame._parent_canvas.yview_moveto(1.0)

    def start_generation_thread(self):
        self.generate_btn.configure(state="disabled", text="⏳ Calling...")
        self.download_btn.pack_forget()

        self.add_chat_bubble("You", "Start creating a music track.", is_user=True)
        self.add_chat_bubble("AI Composer",
                             "The smart model is loading... Please wait a few seconds. 🧠🎹",is_user=False)

        threading.Thread(target=self.generate_with_real_model, daemon=True).start()

    def generate_with_real_model(self):
        try:
            notes_path = 'data/notes' if os.path.exists('data/notes') else 'notes'
            if not os.path.exists(notes_path):
                raise FileNotFoundError("The notes file is missing! Make sure it's in the project folder.")

            with open(notes_path, 'rb') as filepath:
                notes = pickle.load(filepath)

            pitchnames = sorted(set(item for item in notes))

            while len(pitchnames) < 184:
                pitchnames.append(f"PAD_{len(pitchnames)}")

            n_vocab = 184
            sequence_length = 100

            if not os.path.exists('music_model.h5'):
                raise FileNotFoundError("The file 'music_model.h5' is not in the main folder!")

            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation

            model = Sequential()
            model.add(LSTM(256, input_shape=(sequence_length, 1), return_sequences=True))
            model.add(Dropout(0.3))
            model.add(LSTM(256))
            model.add(Dropout(0.3))
            model.add(Dense(184))
            model.add(Activation('softmax'))

            print("Loading model weights locally...")
            model.load_weights('music_model.h5')

            note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
            int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

            network_input = []

            for i in range(0, len(notes) - sequence_length, 1):
                sequence_in = notes[i:i + sequence_length]
                network_input.append([note_to_int[char] for char in sequence_in])

            if not network_input:
                raise ValueError("The current notes file is too short, increase the length of the notes file or try a longer file.")

            start = random.randint(0, len(network_input) - 1)
            pattern = network_input[start]
            prediction_output = []

            for note_index in range(200):
                prediction_input = np.reshape(pattern, (1, len(pattern), 1))
                prediction_input = prediction_input / float(n_vocab)

                prediction = model.predict(prediction_input, verbose=0)
                index = np.argmax(prediction)
                result = int_to_note[index]

                if "PAD_" in str(result):
                    result = random.choice([n for n in pitchnames if "PAD_" not in str(n)])

                prediction_output.append(result)

                pattern.append(index)
                pattern = pattern[1:len(pattern)]

            offset = 0
            output_notes = []

            for pattern_str in prediction_output:
                if ('.' in pattern_str) or pattern_str.isdigit():
                    notes_in_chord = pattern_str.split('.')
                    chord_notes = []
                    for current_note in notes_in_chord:
                        new_note = note.Note(int(current_note))
                        new_note.storedInstrument = instrument.Piano()
                        chord_notes.append(new_note)
                    new_chord = chord.Chord(chord_notes)
                    new_chord.offset = offset
                    output_notes.append(new_chord)
                else:
                    new_note = note.Note(pattern_str)
                    new_note.offset = offset
                    new_note.storedInstrument = instrument.Piano()
                    output_notes.append(new_note)
                offset += 0.5

            midi_stream = stream.Stream(output_notes)
            midi_stream.write('midi', fp='temp_output.mid')

            self.after(0, self.on_generation_complete)

        except Exception as e:
            self.after(0, lambda error=e: messagebox.showerror("A real mistake", f"A problem occurred while running the model:{error}"))
            self.after(0, lambda: self.generate_btn.configure(state="normal", text="✨ Creating new music"))

    def on_generation_complete(self):
        self.generate_btn.configure(state="normal", text="✨ Creating new music")
        self.add_chat_bubble("AI Composer",
                             "🎉 That's fantastic! Press the green button to save and play the real MIDI file!",
                             is_user=False)
        self.download_btn.pack(fill="x", side="right", expand=True, padx=5)

    def download_midi(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".mid",
                                                 filetypes=[("MIDI files", "*.mid")],
                                                 title="Save the tune of real AI")
        if file_path:
            if os.path.exists("temp_output.mid"):
                import shutil
                shutil.copy("temp_output.mid", file_path)
                messagebox.showinfo("Saved", "The clever melody has been successfully saved! 🎧")
                self.download_btn.pack_forget()
            else:
                messagebox.showerror("Error", "The temporary file does not exist.")


if __name__ == "__main__":
    app = MusicChatApp()
    app.mainloop()