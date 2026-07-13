import pickle
import numpy as np
from tensorflow.keras.models import load_model
from music21 import instrument, note, stream, chord


def generate():
    """ تحميل الموديل وتوليد نغمات جديدة وحفظها كملف MIDI """
    # 1. تحميل النوتات والقاموس الموسيقي
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(pitchnames)

    # 2. تحميل الموديل اللي أنت لسه منزله من كولاب
    print("Loading trained model...")
    model = load_model('music_model.h5')

    # تحويلات النصوص لأرقام والعكس
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])

    # اختيار نقطة بداية عشوائية (لحن عشوائي يبدأ الموديل يكمل عليه)
    start = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start]
    prediction_output = []

    print("Generating notes... Please wait.")
    # توليد 500 نوتة موسيقية ورا بعض
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    # 3. تحويل النوتات المتوقعة لملف ميدي حقيقي
    create_midi(prediction_output)


def create_midi(prediction_output):
    """ تحويل النوتات إلى كائنات مسارات ميدي وحفظها """
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        # لو النمط عبارة عن كورد (مجموعة نوتات مع بعض يفصل بينها نقطة)
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # لو النمط نوتة منفردة
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # زيادة المسافة بين النوتات عشان اللحن ما يبقاش متداخل
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='test_output.mid')
    print("\n🎉 Success! Music generated and saved as 'test_output.mid'")


if __name__ == '__main__':
    generate()