import os
import numpy as np
from music21 import converter, instrument, note, chord, stream

def get_midi_files(path):
    midi_files = [f for f in os.listdir(path) if f.endswith(".mid")]
    return [os.path.join(path, midi) for midi in midi_files]

def extract_notes(midi_paths):
    notes = []
    for midi_path in midi_paths:
        midi = converter.parse(midi_path)
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)
        if parts:  
            notes_to_parse = parts.parts[0].recurse()  
        else:
            notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

midi_folder = r"C:\Users\g.surya reddy\midi_folder\Datasets\beeth"  # Corrected the path
if not os.path.exists(midi_folder):
    os.makedirs(midi_folder)
midi_files = get_midi_files(midi_folder)
print(f"MIDI files found: {midi_files}")
notes = extract_notes(midi_files)
print(f"Total notes and chords extracted: {len(notes)}")

if not notes:
    raise ValueError("No notes extracted. Please ensure the MIDI files are correctly formatted and not empty.")

note_names = sorted(set(notes))
note_to_int = {note: number for number, note in enumerate(note_names)}
int_to_note = {number: note for note, number in note_to_int.items()}

sequence_length = 100  # Number of notes per sequence
network_input = []
network_output = []
for i in range(len(notes) - sequence_length):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append([note_to_int[note] for note in sequence_in])
    network_output.append(note_to_int[sequence_out])

if not network_input or not network_output:
    raise ValueError("Insufficient data for training. Please check the note sequences.")

X = np.reshape(network_input, (len(network_input), sequence_length, 1))
X = X / float(len(note_names))  

y = np.zeros((len(network_output), len(note_names)))
for i, output in enumerate(network_output):
    y[i][output] = 1

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation

model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(len(note_names), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X, y, epochs=200, batch_size=64)

def generate_music(model, sequence_length, note_to_int, int_to_note, seed=None, num_notes=500):
    if seed is None:
        start_index = np.random.randint(0, len(network_input)-1)
        prediction_input = network_input[start_index]
    else:
        prediction_input = [note_to_int[n] for n in seed]
    prediction_output = []
    for _ in range(num_notes):  
        prediction_input_reshaped = np.reshape(prediction_input, (1, len(prediction_input), 1))
        prediction_input_reshaped = prediction_input_reshaped / float(len(note_to_int))
        predicted_probs = model.predict(prediction_input_reshaped, verbose=0)
        index = np.argmax(predicted_probs)
        result = int_to_note[index]
        prediction_output.append(result)
        prediction_input.append(index)
        prediction_input = prediction_input[1:]
    return prediction_output

generated_notes = generate_music(model, sequence_length, note_to_int, int_to_note)
print("Generated Notes:", generated_notes)

def create_midi(notes, file_name="generated_music.mid"):
    midi_stream = stream.Stream()
    for pattern in notes:
        if '.' in pattern or pattern.isdigit(): 
            chord_notes = pattern.split('.')
            chord_notes = [note.Note(int(n)) for n in chord_notes]
            midi_stream.append(chord.Chord(chord_notes))
        else:
            midi_stream.append(note.Note(pattern))
    midi_stream.write('midi', fp=file_name)

create_midi(generated_notes, file_name="generated_lofi_music.mid")
print("MIDI file generated.")

print("Current working directory:", os.getcwd())
