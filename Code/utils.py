import numpy as np

class_mapping = {
    0: 'Alarm',
    1: 'Brötchen',
    2: 'Fernseher',
    3: 'Haus',
    4: 'Heizung',
    5: 'Leitung',
    6: 'Licht',
    7: 'Lüftung',
    8: 'Ofen',
    9: 'Radio',
    10: 'Schraube',
    11: 'Spiegel',
    12: 'Staubsauger',
    13: 'an',
    14: 'aus',
    15: 'kann',
    16: 'nicht',
    17: 'offen',
    18: 'warm',
    19: 'wunderbar'
}


def segment_features(features, segment_frames=44, hop_frames=1):
    # Number of feature frames in the file
    num_frames = features.shape[1]

    # Calculate the number of segments we can extract
    num_segments = 1 + (num_frames - segment_frames) // hop_frames

    # Prepare an array to store segments
    segments = []

    # Extract segments using a sliding window
    for i in range(num_segments):
        start_frame = i * hop_frames
        end_frame = start_frame + segment_frames
        segment = features[:, start_frame:end_frame]
        segments.append(segment)

    return np.array(segments)


def predictions(results):
    threshold_60 = [13, 14]
    threshold_80 = [6]
    threshold_99 = [1, 2, 3, 12, 19]

    device_keywords = ['Alarm', 'Fernseher', 'Heizung', 'Licht', 'Lüftung', 'Ofen', 'Radio', 'Staubsauger']
    activation_keywords = ['an', 'aus']

    prediction_counts = {name: 0 for name in class_mapping.values()}
    words_position = []
    for i, result in enumerate(results):
        probabilities = np.array(result[0])
        threshold = 0.9
        predicted_index = np.argmax(probabilities)
        if predicted_index in threshold_60:
            threshold = 0.7
        elif predicted_index in threshold_99:
            threshold = 0.99
        elif predicted_index in threshold_80:
            threshold = 0.8
        if max(probabilities) > threshold:
            predicted_class = class_mapping[predicted_index]
            prediction_counts[predicted_class] += 1
            words_position.append([i, predicted_class])

    unique_predictions = []
    last_word = None
    range_limit = 68  # Define a range to skip close duplicate words - equal to 2s

    for idx, word in words_position:
        if last_word is None or word != last_word or (
                unique_predictions and idx - unique_predictions[-1][0] > range_limit):
            unique_predictions.append((idx, word))
            last_word = word

    last_word = None
    last_idx = 0
    final_predictions = []
    for idx, word in unique_predictions:
        if last_word is None:
            last_idx = idx
            last_word = word
            continue
        elif word in activation_keywords and last_word in device_keywords:
            command = last_word + " " + word
            final_predictions.append((command, last_idx, idx))
            last_word = word
            last_idx = idx
        else:
            last_word = word
            last_idx = idx

    return final_predictions
