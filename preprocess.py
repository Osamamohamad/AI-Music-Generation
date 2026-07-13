import os
import pickle
from music21 import converter, instrument, note, chord


def get_notes():

    notes = []
    songs_folder = "midi_songs"

    if not os.path.exists(songs_folder):
        print(f"Error: Folder '{songs_folder}' not found! Please create it and add MIDI files.")
        return []

    files = [f for f in os.listdir(songs_folder) if f.endswith(".mid") or f.endswith(".midi")]
    print(f"Found {len(files)} MIDI files to process.")

    for file in files:
        print(f"Parsing {file}...")
        try:
            midi = converter.parse(os.path.join(songs_folder, file))
            notes_to_parse = None

            s2 = instrument.partitionByInstrument(midi)
            if s2:
                notes_to_parse = s2.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        except Exception as e:
            print(f"Skipping {file} due to error: {e}")

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


if __name__ == "__main__":
    if not os.path.exists('data'):
        os.makedirs('data')

    all_notes = get_notes()
    print(f"\nDone preprocessing! Total extracted notes: {len(all_notes)}")
    print("Notes successfully saved to data/notes")