import my_utils as utils
import transformers
import dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
import pickle
import rospy
from std_msgs.msg import String
from ASR_tests.msg import Tokens
from inference.msg import YOLO
from ASR_tests.srv import *
import os
import sys
from sklearn.tree import DecisionTreeRegressor
import random

file_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(f"{file_path}/wav2vec2-live"))

global bert_model
global reg_model
global last_detection
global tokenizer
global human
global det


class ContextNet(nn.Module):

    def __init__(self, Ni, Nh1, Nh2, No, p):
        """
        Ni - Input size
        Nh1 - Neurons in the 1st hidden layer
        Nh2 - Neurons in the 2nd hidden layer
        No - Output size
        p - dropout probability
        """
        super().__init__()

        print('Network initialized')
        self.fc1 = nn.Linear(in_features=Ni, out_features=Nh1)
        self.do = nn.Dropout(p=p)
        self.fc2 = nn.Linear(in_features=Nh1, out_features=Nh2)
        self.out = nn.Linear(in_features=Nh2, out_features=No)
        self.act = nn.ReLU()

    def forward(self, x, additional_out=False):
        x = self.act(self.do(self.fc1(x)))
        x1 = self.act(self.do(self.fc2(x)))
        x2 = self.out(x1)
        return x2


# define Transforms
class ToTensor(object):
    """Convert sample to Tensors."""

    def __call__(self, sample):
        padded, mask, label = sample
        return torch.tensor(padded), torch.tensor(mask), torch.tensor(label)


def create_model(model_class, tokenizer_class, pretrained_weights):
    """
        Creates and saves a Context model with a Regression head on BERT
        model_class: class for the transformer model
        tokenizer_class: class for the tokenizer
        pretrained_weights: pretrained weights for the model
    """

    # Read a Dataset
    df = pd.read_csv(f'{file_path}/data/context_dataset_200_shuffled.csv')
    train_df, val_df = train_test_split(df, test_size=0.2)

    # Initialize Datasets
    tensor_transform = transforms.Compose([ToTensor()])
    train_dataset = dataset.ContextDataset(train_df, transform=tensor_transform)
    val_dataset = dataset.ContextDataset(val_df, transform=tensor_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=0)

    # prepare the (distil)BERT model
    model = model_class.from_pretrained(pretrained_weights)

    # train Regression head
    context_net = train_cnet(model, train_dataloader, val_dataloader)

    # save the model to disk
    filename = 'context_net_test3.torch'
    torch.save(context_net.state_dict(), f'{file_path}/models/{filename}')
    # Uncomment for train_reg instead of train_cnet
    # pickle.dump(context_net, open(f'{file_path}/models/{filename}', 'wb'))


def train_reg(model, train_dl, val_dl):
    """
        Creates a DecisionTreeRegressor head for the model
        model: the pretrained transformer model
        train_dl: the training DataLoader
        val_dl: the validation DataLoader
        returns: the trained regressor
    """
    for batch in train_dl:
        padded_batch = batch[0]
        mask_batch = batch[1]
        labels = batch[2]

        # feed the batch through (Distil)BERT
        with torch.no_grad():
            last_hidden_states = model(padded_batch, attention_mask=mask_batch)

        # convert (Distil)Bert output to regressor input!
        features = last_hidden_states[0][:, 0, :].numpy()

        # define model
        reg = DecisionTreeRegressor()
        reg.fit(features, labels)
        print(features.shape)

        for batch in val_dl:
            padded_batch = batch[0]
            mask_batch = batch[1]
            labels = batch[2]
            with torch.no_grad():
                last_hidden_states = model(padded_batch, attention_mask=mask_batch)

            # convert (Distil)Bert output to regressor input!
            features = last_hidden_states[0][:, 0, :].numpy()

            score = reg.score(features, labels)
            print(f'Loss: {score}')

            print(f'Prediction: {reg.predict(features)[:3]}')
            print(f'Real labels: {labels[:3]}')

    return reg


def train_cnet(model, train_dl, val_dl):
    """
        Creates a FCN head for the model
        model: the pretrained transformer model
        train_dl: the training DataLoader
        val_dl: the validation DataLoader
        returns: the trained FCN regressor
    """
    torch.manual_seed(0)
    Ni = 768
    Nh1 = 256
    Nh2 = 128
    No = 4
    net = ContextNet(Ni, Nh1, Nh2, No, p=0)
    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    net.to(device)

    # Set hyperparameters
    # Adagrad add weight decay weight_decay=0 because it overfits
    # optimizer = optim.Adagrad(net.parameters(), lr=0.01, weight_decay=0.005)
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    # try lr between 0.001 and 0.005 -> 0.003 better
    # try different optimizer -> Adagrad promising, but overfits without weight decay -> needs careful tuning
    # try one more layer! -> learned more slowly; stagnates quickly at 0.025
    # ReLu not Sigmoid -> slightly better after 50 epochs
    # TODO: run best version with many epochs: 200? 300? 400? overnight
    num_epochs = 200
    loss_fn = nn.MSELoss()

    train_loss_log = []
    val_loss_log = []
    for epoch_num in range(num_epochs):
        print('#################')
        print(f'# EPOCH {epoch_num}')
        print('#################')

        ### TRAIN
        train_loss = []
        net.train()  # Training mode (e.g. enable dropout)
        for batch in train_dl:
            padded_batch = batch[0]
            mask_batch = batch[1]
            labels = batch[2].to(device)

            # feed the batch through (Distil)BERT
            with torch.no_grad():
                last_hidden_states = model(padded_batch, attention_mask=mask_batch)

            # convert (Distil)Bert output to classifier input!
            features = last_hidden_states.to_tuple()[0][:, 0, :]
            features.to(device)

            # Forward pass
            out = net(features)

            # Compute loss
            loss = loss_fn(out, labels)

            # Backpropagation
            net.zero_grad()
            loss.backward()

            # Update the weights
            optimizer.step()

            # Save train loss for this batch
            loss_batch = loss.detach().cpu().numpy()
            train_loss.append(loss_batch)

        # Save average train loss
        train_loss = np.mean(train_loss)
        print(f"AVERAGE TRAIN LOSS: {train_loss}")
        train_loss_log.append(train_loss)

        ### VALIDATION
        val_loss = []
        net.eval()  # Evaluation mode (e.g. disable dropout)
        with torch.no_grad():  # Disable gradient tracking
            for batch in val_dl:
                padded_batch = batch[0]
                mask_batch = batch[1]
                labels = batch[2].to(device)

                # feed the batch through (Distil)BERT
                last_hidden_states = model(padded_batch, attention_mask=mask_batch)
                # convert (Distil)Bert output to classifier input!
                features = last_hidden_states.to_tuple()[0][:, 0, :]
                features.to(device)

                # Forward pass
                out = net(features)
                print(f'Prediction: {out[:3]}')
                print(f'Real labels: {labels[:3]}')

                # Compute loss
                loss = loss_fn(out, labels)

                # Save val loss for this batch
                loss_batch = loss.detach().cpu().numpy()
                val_loss.append(loss_batch)

            # Save average validation loss
            val_loss = np.mean(val_loss)
            print(f"AVERAGE VAL LOSS: {np.mean(val_loss)}")
            val_loss_log.append(val_loss)

    inputs, masks, labels = next(iter(val_dl))
    with torch.no_grad():
        last_hidden_states = model(inputs, attention_mask=masks)
        features = last_hidden_states.to_tuple()[0][:, 0, :]
        features.to(device)
        out = net(features)

    print(f'real labels: {labels[:3]}')
    print(f'predicted labels: {out[:3]}')

    return net



def callback(data: object) -> object:
    """
        ROS subscriber callback 
        saves last yolo detections
    """
    # Saves detections from the YOLO topic
    global last_detection
    last_detection = data
    #print(f'Data from YOLO: {data}')



def call_cnet(req):
    """
        ROS service callback 
        receives text tokens in req
        combines them with last yolo detections
        extracts the context
        response: 3D object pose (x, y, z, alpha)
    """
    # Extract text tokens from the request
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", req.token_ids)
    a_list = req.token_ids[1:-1].split(', ')
    map_object = map(int, a_list)
    tokens = list(map_object)
    # Extract the yolo tokens
    yolo_detections = []
    global last_detection
    global tokenizer
    global human
    global det
    if last_detection is None:
        print(f'No detection available!')
        return ContextResponse(str(-1.), str(-1.), str(-1.), str(-1.))
    for i in range(len(last_detection.xmin)):
        if last_detection.name[i] == 'bottle' or last_detection.name[i] == 'cup':
            print(f'Detected: {last_detection.name}')
            yolo = f'{last_detection.name[i]} {last_detection.xmin[i]} {last_detection.xmax[i]} {last_detection.ymin[i]} {last_detection.ymax[i]}'
            yolo = tokenizer.encode(yolo, add_special_tokens=True)
            yolo_detections.append(yolo)

    # Combine text with 2 detections
    if (len(yolo_detections) == 0):
        print(f'No objects recognized!')
        return ContextResponse(str(-1.), str(-1.), str(-1.), str(-1.))
    net_inputs = []
    padded_batch = []
    mask_batch = []
    sentence_list = [[human] + tokens[1:-1], [det] + yolo_detections[0][1:-1]]
    if (len(yolo_detections) == 1):
        print(f'1 object recognized!')
        random.shuffle(sentence_list)
        in_tokens = [101] + sentence_list[0] + sentence_list[1] + [102]
    else:
        print(f'2 objects recognized!')
        sentence_list.append([det] + yolo_detections[0][1:-1])
        random.shuffle(sentence_list)
        in_tokens = [101] + sentence_list[0] + sentence_list[1] + sentence_list[2] + [102]
    
    
    assert len(in_tokens) <= 100
    max_len = 100
    padded = np.array(in_tokens + [0] * (max_len - len(in_tokens)))
    padded_batch.append(np.array(padded))
    attention_mask = np.where(np.array(padded) != 0, 1, 0)
    mask_batch.append(attention_mask)

    # Run through the net and return the result
    with torch.no_grad():
        last_hidden_states = bert_model(torch.tensor(padded_batch), attention_mask=torch.tensor(mask_batch))
    features = last_hidden_states[0][:, 0, :].numpy()
    #res = reg_model.predict(features)[0]
    #device = torch.device("cpu")
    #reg_model.to(device)
    #features.to(device)
    res = reg_model(torch.tensor(features))[0].detach().numpy()
    print(res)
    return ContextResponse(res[0], res[1], res[2], res[3])


if __name__ == '__main__':
    global last_detection
    global tokenizer
    last_detection = None
    global human
    human = 0
    global det
    det = 0
    # For DistilBERT:
    #model_class, tokenizer_class, pretrained_weights = (
    #    transformers.DistilBertModel, transformers.DistilBertTokenizer, 'distilbert-base-uncased')
    model_class, tokenizer_class, pretrained_weights = (
        transformers.BertModel, transformers.BertTokenizer, 'bert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained('bert-base-uncased')


    #create_model(model_class, tokenizer_class, pretrained_weights)

    # prepare the model
    bert_model = model_class.from_pretrained(pretrained_weights)
    #filename = 'regression_tree_2det_big.sav'
    #reg_model = pickle.load(open(f'{file_path}/models/{filename}', 'rb'))
    
    Ni = 768
    Nh1 = 256
    Nh2 = 128
    No = 4
    filename = 'context_net.torch'
    reg_model = ContextNet(Ni, Nh1, Nh2, No, p=0)
    reg_model.load_state_dict(torch.load(f'{file_path}/models/{filename}'))

    rospy.init_node('context', anonymous=True)
    rospy.Subscriber("model/coordinates", YOLO, callback)
    server = rospy.Service('get_context', Context, call_cnet)
    rospy.spin()