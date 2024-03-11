import my_utils as utils
import transformers
import dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pickle
import rospy
import actionlib
import actionlib_tutorials.msg
from std_msgs.msg import String
from ASR_tests.msg import Tokens
from ASR_tests.srv import *
import os
import sys

file_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(f"{file_path}/wav2vec2-live"))

global bert_model
global clf_model

# define Transforms
class ToTensor(object):
    """Convert sample to Tensors."""

    def __call__(self, sample):
        padded, mask, label = sample
        return torch.tensor(padded), torch.tensor(mask), label


def create_new_classifier(model_class, tokenizer_class, pretrained_weights):
    #'''
    #    Trains a Classifier, evaluates it's performance and saves the model to models/classifier.sav
    #    @param model_class: model class for the BERT model
    #    @param tokenizer_class: tokenizer class for initializing the BERT tokenizer
    #    @param pretrained_weights: weight for the BERT model
    #'''

    # create a Dataset
    df = pd.read_csv(f'{file_path}/data/classifier_dataset.csv')
    train_df, val_df = train_test_split(df, test_size=0.2)

    # Initialize Datasets
    tensor_transform = transforms.Compose([ToTensor()])
    train_dataset = dataset.ClassifierDataset(train_df, transform=tensor_transform)
    val_dataset = dataset.ClassifierDataset(val_df, transform=tensor_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=0)

    # prepare the (distil)BERT model
    model = model_class.from_pretrained(pretrained_weights)

    # train the classifier
    classifier = train_lr_clf(model, train_dataloader)

    # evaluate the classifier
    eval_lr_clf(classifier, val_dataloader, model)

    # Test the classifier on some examples
    predict_values(classifier, tokenizer_class, model)

    # save the model to disk
    filename = 'classifier.sav'
    pickle.dump(classifier, open(f'{file_path}/models/{filename}', 'wb'))
    print('Classifier created successfully!')


def train_lr_clf(model, train_dl):
    #'''
    #    Trains a DecisionTreeClassifier
    #    @param model: the pretrained BERT model
    #    @param train_dl: training Dataloader
    #    @return: classifier object
    #'''
    for batch in train_dl:
        padded_batch = batch[0]
        mask_batch = batch[1]
        labels = batch[2]

        # feed the batch through (Distil)BERT
        with torch.no_grad():
            last_hidden_states = model(padded_batch, attention_mask=mask_batch)

        # convert (Distil)Bert output to classifier input!
        features = last_hidden_states[0][:, 0, :].numpy()

        clf = DecisionTreeClassifier()
        clf.fit(features, labels)

    return clf


def eval_lr_clf(clf, val_dl, model):
    #'''
    #    Evaluates a lassifier
    #    @param clf: the trained classifier object
    #    @param val_dl: validation Dataloader
    #    @param model: the pretrained BERT model
    #'''
    for batch in val_dl:
        padded_batch = batch[0]
        mask_batch = batch[1]
        labels = batch[2]
        with torch.no_grad():
            last_hidden_states = model(padded_batch, attention_mask=mask_batch)

        # convert (Distil)Bert output to classifier input!
        features = last_hidden_states[0][:, 0, :].numpy()

        score = clf.score(features, labels)
        print(f'Accuracy: {score}')


def predict_values(clf, tok_class, model):
    #'''
    #    Tests a classifier
    #    @param clf: the classifier to be tested
    #    @param tok_class: tokenizer class for BERT input
    #    @param model: the pretrained BERT model
    #'''

    # Uncomment to test on microphone input
    # pos_ex = utils.asr_mic()
    # speech_ex = utils.asr_mic()
    pos_ex = 'Pass me the pepper shaker please'
    pos_ex2 = 'Can you please grasp the object'
    neg_ex = 'This is a negative example'
    speech_ex = 'That is a pretty cool cup, I like it'
    

    max_len = 0
    #tokenizer = tok_class.from_pretrained('distilbert-base-uncased')
    tokenizer = tok_class.from_pretrained('bert-base-uncased')
    pos_tok = tokenizer.encode(pos_ex, add_special_tokens=True)
    pos2_tok = tokenizer.encode(pos_ex2, add_special_tokens=True)
    pos3_tok = tokenizer.encode(neg_ex, add_special_tokens=True)
    pos4_tok = tokenizer.encode(speech_ex, add_special_tokens=True)

    for i in [pos_tok, pos2_tok, pos3_tok, pos4_tok]:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0] * (max_len - len(i)) for i in [pos_tok, pos2_tok, pos3_tok, pos3_tok]])

    padded = np.array(padded)

    # Attention mask
    attention_mask = np.where(padded != 0, 1, 0)

    with torch.no_grad():
        last_hidden_states = model(torch.tensor(padded), attention_mask=torch.tensor(attention_mask))

    # convert (Distil)Bert output to classifier input!
    features = last_hidden_states[0][:, 0, :].numpy()
    print(clf.predict(features))


def callback(data: object) -> object:
    #'''
    #    ROS subscriber callback, receives text tokens and classifies them
    #'''
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    a_list = data.data[1:-1].split(', ')
    map_object = map(int, a_list)
    tokens = list(map_object)
    assert len(tokens) <= 100
    max_len = 100
    padded = np.array(tokens + [0] * (max_len - len(tokens)))
    padded = np.array(padded)

    # Attention mask
    attention_mask = np.where(padded != 0, 1, 0)

    with torch.no_grad():
        last_hidden_states = bert_model(torch.tensor(padded)[None, :],
                                        attention_mask=torch.tensor(attention_mask)[None, :])

    # convert (Distil)Bert output to classifier input!
    features = last_hidden_states[0][:, 0, :].numpy()
    res = clf_model.predict(features)
    print(f'Class: {res}')

    # save the latest tokens and result
    global last_tokens
    last_tokens = data.data
    global last_result
    last_result = res


def retrieve_result(req):
    #'''
    #    ROS service callback, responds with the last classification result and corresponding token list
    #'''
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", req.token_ids)
    print(last_result)
    out = ' '.join(str(x) for x in last_result)
    return ClassifyResponse(out, last_tokens)


if __name__ == '__main__':
    global last_tokens
    global last_result
    last_result = [4, 4]
    last_tokens = 'no tokens'
    # For DistilBERT:
    #model_class, tokenizer_class, pretrained_weights = (
    #transformers.DistilBertModel, transformers.DistilBertTokenizer, 'distilbert-base-uncased')

    # For Bert:
    model_class, tokenizer_class, pretrained_weights = (transformers.BertModel, transformers.BertTokenizer, 'bert-base-uncased')

    #create_new_classifier(model_class, tokenizer_class, pretrained_weights)

    # prepare the (distil)BERT model
    bert_model = model_class.from_pretrained(pretrained_weights)
    # load the model from disk
    filename = 'lr_clf_2a_dataset.sav'
    clf_model = pickle.load(open(f'{file_path}/models/{filename}', 'rb'))

    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("tokens", String, callback)
    server = rospy.Service('classify', Classify, retrieve_result)
    rospy.spin()
