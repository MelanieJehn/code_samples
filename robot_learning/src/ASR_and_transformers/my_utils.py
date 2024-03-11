import speech_recognition as sr
from transformers import AutoTokenizer
import os
import sys
import transformers
file_path = os.path.realpath(__file__)
file_path = os.path.abspath(os.path.dirname(__file__))
print(file_path)
sys.path.append(os.path.abspath(f"{file_path}/wav2vec2-live"))
from live_asr import LiveWav2Vec2
from wav2vec2_inference import Wave2Vec2Inference
import pandas as pd
import pydub
from tqdm import tqdm
import numpy as np
import random


def asr_mic():
    #''' 
    #    Listens to the microphone, performs ASR
    #    @returns: ASR text output
    #'''

    # choose model: https: // huggingface.co / models?pipeline_tag = automatic - speech - recognition & search = wav2vec2
    english_model = "facebook/wav2vec2-large-960h-lv60-self"

    # mic_name = 'KLIM Talk: USB Audio(hw: 1, 0)'
    # print(sr.Microphone.list_microphone_names())
    asr = LiveWav2Vec2(english_model, device_name='default')
    total_text = []
    asr.start()

    try:
        while True:
            text, sample_length, inference_time = asr.get_last_text()
            print(f"{sample_length:.3f}s"
                  + f"\t{inference_time:.3f}s"
                  + f"\t{text}")
            total_text.append(text)

    except KeyboardInterrupt:
        asr.stop()

    return total_text


def asr_wav(path):
    #'''
    #    Performs ASR
    #    @param path: path to a .wav audio file
    #    @returns: ASR text output
    #'''

    # choose model: https: // huggingface.co / models?pipeline_tag = automatic - speech - recognition & search = wav2vec2
    english_model = "facebook/wav2vec2-large-960h-lv60-self"

    asr = Wave2Vec2Inference(english_model)
    text = asr.file_to_text(path)
    return text


def to_token_vec(text, tokenizer_name):
    #'''
    #    Converts text to tokens
    #    @param text: the text to be converted
    #    @param tokenizer_name: name of the tokenizer to be used
    #    @returns: text tokens
    #'''

    # Bert tokenizer name: 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    tokens = tokenizer.encode(text)
    return tokens


def rec_to_file():
    #'''
    #    Record microphone input to .wav audio file
    #    Waits for maximal 5 sec before recording if no one speaks
    #    Records for 10 sec
    #'''

    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        r.adjust_for_ambient_noise(source, duration=1)
        print("Talk now!")
        audio = r.listen(source, timeout=5, phrase_time_limit=10)
    with open(f"system.wav", "wb") as f:
        f.write(audio.get_wav_data(convert_rate=16000))


def set_samplerate(path):
    #'''
    #    Change the sample rate to 16000 for wav files
    #    @param path: path to wav files which need resampling
    #'''
    wav_paths = []
    for fname in os.listdir(path):
        if not fname.endswith('.txt'):
            for (dirpath, dirnames, filenames) in os.walk(os.path.join(path, fname)):
                wav_paths += [os.path.join(dirpath, file) for file in filenames if filenames]
    for audio_file in tqdm(wav_paths):
        # change the samplerate to 16000
        sound = pydub.AudioSegment.from_file(audio_file, format='wav')
        sound = sound.set_frame_rate(16000)
        sound.export(audio_file, format='wav')


def create_token_dataset(path, tok_class):
    """
	Tokenizes text and audio data in path and stores them in a txt file.
	Expects positive examples in a txt file starting with pos and negative examples in wav or txt format.
	Writes resulting tokens with labels to file
	:param path: Path to txt file and wav folders
	:return:
	"""
    token_dicts = []

    wav_paths = []
    # tokenizer = tok_class.from_pretrained('distilbert-base-uncased')
    tokenizer = tok_class.from_pretrained('bert-base-uncased')
    file_path = os.path.abspath(os.path.dirname(__file__))
    path = f"{file_path}{path}"
    print(file_path, path)
    for fname in os.listdir(path):
        if fname.endswith('.txt'):
            # Read in text data and tokenize
            with open(os.path.join(path, fname), 'r') as file:
                pos = (fname[:3] == 'pos')
                lines = file.readlines()
                for line in tqdm(lines):
                    # Add key label: 1 for positive data examples
                    token_dict = {}
                    l = line.split()
                    if pos:
                        if l[-1] == "1":
                            token_dict['label'] = 1
                        else:
                            token_dict['label'] = 2
                    else:
                        token_dict['label'] = 0
                    line = line[:-2]
                    token_dict['tokens'] = tokenizer.encode(line.strip(), add_special_tokens=True)
                    token_dicts.append(token_dict)

        else:
            for (dirpath, dirnames, filenames) in os.walk(os.path.join(path, fname)):
                wav_paths += [os.path.join(dirpath, file) for file in filenames if filenames]
            for audio_file in tqdm(wav_paths):
                if audio_file.endswith('.wav'):
                    # samplerate has to be 16000
                    text = asr_wav(audio_file)
                    token_dict = {}
                    token_dict['tokens'] = tokenizer.encode(text.strip(), add_special_tokens=True)
                    # label: 0 for negative data examples
                    token_dict['label'] = 0
                    token_dicts.append(token_dict)

    # write data to csv file
    df = pd.DataFrame(token_dicts)
    df.to_csv(os.path.join(path, f'classifier_dataset_small2.csv'))


def create_context_dataset(path, text_file, yolo_file, tok_class):
    """
	Tokenizes the yolo data and combines it with the text tokens
	:param path: Path for csv output
	:param text_file: Path to tokenized text dataset
	:param yolo_file: Path to the yolo detections
	:return:
	"""

    # Read yolo data
    file_path = os.path.abspath(os.path.dirname(__file__))
    path = f"{file_path}{path}"
    yolo_df = pd.read_csv(os.path.join(path, yolo_file))
    N_detections = yolo_df.size

    # Create positive/negative examples
    # Read in the positive text data
    text_df = pd.read_csv(os.path.join(path, text_file))
    text_df = text_df.loc[text_df['label'] != 0]
    N_text = text_df.size

    #tokenizer = tok_class.from_pretrained('distilbert-base-uncased')
    tokenizer = tok_class.from_pretrained('bert-base-uncased')

    # Randomly choose 1 of each N times
    dataset = []
    for i in range(400):
        data = {}
        text_row = text_df.sample(replace=True)
        yolo_row = yolo_df.sample(3, replace=True)
        object1 = yolo_row.iloc[0]
        object2 = yolo_row.iloc[1]
        object3 = yolo_row.iloc[2]
        objects = [object1['name'], object2['name'], object3['name']]

        # object can be bottle, vase, cup or dining table ->
        # text object has: cup, mug, object, box, sphere, cross, bottle, salt/pepper shaker/sprinkler

        # bottle -> bottle 5835, salt 5474 /pepper 11565 shaker(6073, 2099)/sprinkler(11867, 6657, 19099, 2099)
        # bottle -> box 3482, object 4874
        # cup -> cup 2452, mug 14757, sphere 10336
        # no match for cross or dining table -> object couldn't be found

        # Match text and yolo object
        tokens = text_row.iloc[0]['tokens']
        string_list = tokens[1:-1].split(',')
        tokens = [int(s) for s in tokens[1:-1].split(', ')]
        data['out'] = [-1.0, -1.0, -1.0, -1.0]
        unknown = False
        print(tokenizer.decode(tokens))

        # left: from camera left
        # pair with left or right outputs: make sure bboxes don't overlap!
        # right: [2157], left: [2187]
        # The text examples asks for left/right

        # User wants the right or left object
        if 2157 in tokens or 2187 in tokens:
            if 5835 in tokens or 5474 in tokens or 11565 in tokens or 3482 in tokens or 4874 in tokens:
                print('Asked for left or right bottle')
                type_df = yolo_df.loc[yolo_df['name'] == 'bottle']
                # check xmax of left object is smaller than xmin of right object!
                overlap = True
                while overlap:
                    type_row = type_df.sample(2)
                    object1 = type_row.iloc[0]
                    object2 = type_row.iloc[1]
                    left, overlap = check_overlap(object1, object2)
                
                print(f'Is the first object left? {left}')
                if 2157 in tokens and left:
                    obj1 = object2
                    obj2 = object1
                elif 2157 in tokens and not left:
                    obj1 = object1
                    obj2 = object2
                elif 2187 in tokens and left:
                    obj1 = object1
                    obj2 = object2
                else:
                   obj1 = object2
                   obj2 = object1
                n1 = obj1['name']
                n2 = obj2['name']
                print(f'Objects chosen: {n1}; {n2}')
                xmin = round(obj1['xmin'], 1)
                xmax = round(obj1['xmax'], 1)
                ymin = round(obj1['ymin'], 1)
                ymax = round(obj1['ymax'], 1)
                x = round(obj1['3D_x'], 1)
                y = round(obj1['3D_y'], 1)
                z = round(obj1['3D_z'], 1)
                xmin2 = round(obj2['xmin'], 1)
                xmax2 = round(obj2['xmax'], 1)
                ymin2 = round(obj2['ymin'], 1)
                ymax2 = round(obj2['ymax'], 1)
                x2 = round(obj2['3D_x'], 1)
                y2 = round(obj2['3D_y'], 1)
                z2 = round(obj2['3D_z'], 1)
                yolo1 = f'{obj1} {xmin} {xmax} {ymin} {ymax} {x} {y} {z}'
                yolo1 = tokenizer.encode(yolo1, add_special_tokens=True)
                yolo2 = f'{obj2} {xmin2} {xmax2} {ymin2} {ymax2} {x2} {y2} {z2}'
                yolo2 = tokenizer.encode(yolo2, add_special_tokens=True)
                data['out'] = [obj1['Trans_x'], obj1['Trans_y'], obj1['Trans_z'], obj1['Q_z']]
                print(f'{x}; {x2}')
                print(obj1['Trans_x'])

            elif 2452 in tokens or 14757 in tokens:
                print('Asked for left or right cup')
                type_df = yolo_df.loc[yolo_df['name'] == 'cup']
                # check xmax of left object is smaller than xmin of right object!
                overlap = True
                while overlap:
                    type_row = type_df.sample(2)
                    object1 = type_row.iloc[0]
                    object2 = type_row.iloc[1]
                    left, overlap = check_overlap(object1, object2)
                print(f'Is the first object left? {left}')
                if 2157 in tokens and left:
                    obj1 = object2
                    obj2 = object1
                elif 2157 in tokens and not left:
                    obj1 = object1
                    obj2 = object2
                elif 2187 in tokens and left:
                    obj1 = object1
                    obj2 = object2
                else:
                   obj1 = object2
                   obj2 = object1
                n1 = obj1['name']
                n2 = obj2['name']
                print(f'Objects chosen: {n1}; {n2}')
                xmin = round(obj1['xmin'], 1)
                xmax = round(obj1['xmax'], 1)
                ymin = round(obj1['ymin'], 1)
                ymax = round(obj1['ymax'], 1)
                x = round(obj1['3D_x'], 1)
                y = round(obj1['3D_y'], 1)
                z = round(obj1['3D_z'], 1)
                xmin2 = round(obj2['xmin'], 1)
                xmax2 = round(obj2['xmax'], 1)
                ymin2 = round(obj2['ymin'], 1)
                ymax2 = round(obj2['ymax'], 1)
                x2 = round(obj2['3D_x'], 1)
                y2 = round(obj2['3D_y'], 1)
                z2 = round(obj2['3D_z'], 1)
                yolo1 = f'{obj1} {xmin} {xmax} {ymin} {ymax} {x} {y} {z}'
                yolo1 = tokenizer.encode(yolo1, add_special_tokens=True)
                yolo2 = f'{obj2} {xmin2} {xmax2} {ymin2} {ymax2} {x2} {y2} {z2}'
                yolo2 = tokenizer.encode(yolo2, add_special_tokens=True)
                data['out'] = [obj1['Trans_x'], obj1['Trans_y'], obj1['Trans_z'], obj1['Q_z']]
                print(f'{x}; {x2}')
                print(obj1['Trans_x'])

            else:
                print('Asked for sth else')
                # Unknown object requested
                unknown = True

        # The text example doesn't ask for left vs right
        # So take 1 bottle and 1 cup! if there are two it should be specified!
        else:
            if 5835 in tokens or 5474 in tokens or 11565 in tokens or 3482 in tokens or 4874 in tokens:
                print('Asked for bottle')
                if 'bottle' in objects:
                    idx = objects.index('bottle')
                    obj1 = yolo_row.iloc[idx]
                    xmin = round(yolo_row.iloc[idx]['xmin'], 1)
                    xmax = round(yolo_row.iloc[idx]['xmax'], 1)
                    ymin = round(yolo_row.iloc[idx]['ymin'], 1)
                    ymax = round(yolo_row.iloc[idx]['ymax'], 1)
                    x = round(yolo_row.iloc[idx]['3D_x'], 1)
                    y = round(yolo_row.iloc[idx]['3D_y'], 1)
                    z = round(yolo_row.iloc[idx]['3D_z'], 1)
                    yolo1 = f'bottle {xmin} {xmax} {ymin} {ymax} {x} {y} {z}'
                    yolo1 = tokenizer.encode(yolo1, add_special_tokens=True)

                    type_df = yolo_df.loc[yolo_df['name'] == 'cup']
                    # check xmax of left object is smaller than xmin of right object!
                    overlap = True
                    while overlap:
                        type_row = type_df.sample(1)
                        obj2 = type_row.iloc[0]
                        _, overlap = check_overlap(obj1, obj2)

                    xmin2 = round(obj2['xmin'], 1)
                    xmax2 = round(obj2['xmax'], 1)
                    ymin2 = round(obj2['ymin'], 1)
                    ymax2 = round(obj2['ymax'], 1)
                    x2 = round(obj2['3D_x'], 1)
                    y2 = round(obj2['3D_y'], 1)
                    z2 = round(obj2['3D_z'], 1)
                    n2 = obj2['name']
                    n1 = obj1['name']
                    print(f'Objects chosen: {n1}; {n2}')
                    yolo2 = f'{n2} {xmin2} {xmax2} {ymin2} {ymax2} {x2} {y2} {z2}'
                    yolo2 = tokenizer.encode(yolo2, add_special_tokens=True)
                    data['out'] = [obj1['Trans_x'], obj1['Trans_y'], obj1['Trans_z'], obj1['Q_z']]
                    print(f'{x}; {x2}')
                    print(obj1['Trans_x'])


                else:
                    # Bottle wasn't detected!
                    unknwon = True

            elif 2452 in tokens or 14757 in tokens:
                print('Asked for cup')
                if 'cup' in objects:
                    idx = objects.index('cup')
                    obj1 = yolo_row.iloc[idx]
                    xmin = round(yolo_row.iloc[idx]['xmin'], 1)
                    xmax = round(yolo_row.iloc[idx]['xmax'], 1)
                    ymin = round(yolo_row.iloc[idx]['ymin'], 1)
                    ymax = round(yolo_row.iloc[idx]['ymax'], 1)
                    x = round(yolo_row.iloc[idx]['3D_x'], 1)
                    y = round(yolo_row.iloc[idx]['3D_y'], 1)
                    z = round(yolo_row.iloc[idx]['3D_z'], 1)
                    yolo1 = f'cup {xmin} {xmax} {ymin} {ymax} {x} {y} {z}'
                    yolo1 = tokenizer.encode(yolo1, add_special_tokens=True)

                    type_df = yolo_df.loc[yolo_df['name'] == 'bottle']
                    # check xmax of left object is smaller than xmin of right object!
                    overlap = True
                    while overlap:
                        type_row = type_df.sample(1)
                        obj2 = type_row.iloc[0]
                        _, overlap = check_overlap(obj1, obj2)

                    xmin2 = round(obj2['xmin'], 1)
                    xmax2 = round(obj2['xmax'], 1)
                    ymin2 = round(obj2['ymin'], 1)
                    ymax2 = round(obj2['ymax'], 1)
                    x2 = round(obj2['3D_x'], 1)
                    y2 = round(obj2['3D_y'], 1)
                    z2 = round(obj2['3D_z'], 1)
                    n2 = obj2['name']
                    n1 = obj1['name']
                    print(f'Objects chosen: {n1}; {n2}')
                    yolo2 = f'{n2} {xmin2} {xmax2} {ymin2} {ymax2} {x2} {y2} {z2}'
                    yolo2 = tokenizer.encode(yolo2, add_special_tokens=True)
                    data['out'] = [obj1['Trans_x'], obj1['Trans_y'], obj1['Trans_z'], obj1['Q_z']]
                    print(f'{x}; {x2}')
                    print(obj1['Trans_x'])


                else:
                    # Cup wasn't detected!
                    unknown = True
            else:
                print('Asked for sth else')
                # Unknown object requested
                unknown = True

        if unknown:
            xmin = round(yolo_row.iloc[0]['xmin'], 1)
            xmax = round(yolo_row.iloc[0]['xmax'], 1)
            ymin = round(yolo_row.iloc[0]['ymin'], 1)
            ymax = round(yolo_row.iloc[0]['ymax'], 1)
            x = round(yolo_row.iloc[0]['3D_x'], 1)
            y = round(yolo_row.iloc[0]['3D_y'], 1)
            z = round(yolo_row.iloc[0]['3D_z'], 1)
            n1 = object1['name']
            yolo1 = f'{n1} {xmin} {xmax} {ymin} {ymax} {x} {y} {z}'
            yolo1 = tokenizer.encode(yolo1, add_special_tokens=True)
            xmin2 = round(yolo_row.iloc[1]['xmin'], 1)
            xmax2 = round(yolo_row.iloc[1]['xmax'], 1)
            ymin2 = round(yolo_row.iloc[1]['ymin'], 1)
            ymax2 = round(yolo_row.iloc[1]['ymax'], 1)
            x2 = round(yolo_row.iloc[1]['3D_x'], 1)
            y2 = round(yolo_row.iloc[1]['3D_y'], 1)
            z2 = round(yolo_row.iloc[1]['3D_z'], 1)
            obj2 = yolo_row.iloc[1]['name']
            yolo2 = f'{object2} {xmin2} {xmax2} {ymin2} {ymax2} {x2} {y2} {z2}'
            yolo2 = tokenizer.encode(yolo2, add_special_tokens=True)
            print(f'Objects chosen: {n1}; {obj2}')
            print('Problems caused by unknown!')


        # Tokenize yolo_data
        # Shuffle human part with 2 yolo token lists and then combine!
        human = tokenizer.encode("human", add_special_tokens=False)
        det = tokenizer.encode("detector", add_special_tokens=False)
        human_sentence = human + tokens[1:-1]
        det_1 = det + yolo1[1:-1]
        det_2 = det + yolo2[1:-1]
        order = [human_sentence, det_1, det_2]
        random.shuffle(order)
        # combine text and yolo tokens with human and detector keywords in front of those parts!
        data['tokens'] = [tokens[0]] + order[0] + order[1] + order[2] + [tokens[-1]]
        dataset.append(data)
        res = data['out']
        print(res)

    df = pd.DataFrame(dataset)
    df.to_csv(os.path.join(path, f'context_dataset_2det_big_s.csv'))


def check_overlap(o1, o2):
    #'''
    #    Check if 2 objects overlap
    #    @param o1: object 1
    #    @param o2: object 2
    #    @returns: boolean left if o1 is left of o2 and boolean overlap of the 2 objects
    #'''
    left = o1['3D_x'] < o2['3D_x']
    overlap = np.abs(o1['3D_x'] - o2['3D_x']) < 30
    return left, overlap