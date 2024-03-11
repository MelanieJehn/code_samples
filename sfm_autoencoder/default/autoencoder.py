import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import nn
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(ROOT_DIR))

from dataset import DescriptorDataset


class Autoencoder:
    """
    An Autoencoder consists of an Encoder and Decoder part. It takes a list of descriptors
    and reconstructs them.
    """
    # TODO: change architecture for analysis
    def __init__(self):
        self.descriptors = None
        self.encoder = self.Encoder(16, 64, 32)
        self.decoder = self.Decoder(16, 64, 32)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.loss_fn = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.num_epochs = 50
        self.test_outputs = []

    class Encoder(nn.Module):
        """
        The Encoder part of the network.
        Has 128 input neurons to match the 128 features per descriptor.
        """

        def __init__(self, encoded_space_dim, h1, h2):
            super().__init__()

            self.fc1 = nn.Linear(128, h1)
            self.fc2 = nn.Linear(h1, h2)
            self.out = nn.Linear(h2, encoded_space_dim)
            self.act = nn.ReLU()

        def forward(self, x):
            x = self.act(self.fc1(x))
            x = self.act(self.fc2(x))
            x = self.out(x)
            return x

    class Decoder(nn.Module):
        """
        The Decoder part of the network. Mirrors the Encoder architecture.
        """

        def __init__(self, encoded_space_dim, h1, h2):
            super().__init__()

            self.fc1 = nn.Linear(encoded_space_dim, h2)
            self.fc2 = nn.Linear(h2, h1)
            self.out = nn.Linear(h1, 128)
            self.act = nn.ReLU()

        def forward(self, x):
            x = self.act(self.fc1(x))
            x = self.act(self.fc2(x))
            x = self.out(x)
            return x

    def prepare_data(self, descriptors, train):
        """
        Create DataLoaders for the descriptor data
        :param descriptors: list of lists of surf descriptors
        :param train: boolean it True split data into training and validation, else use for testing
        :return: void
        """
        data = [item for sublist in descriptors for item in sublist]
        # want to split the data to train and val if train
        if train:
            train_data, val_data = train_test_split(data, test_size=0.2, random_state=0)
            train_dataset = DescriptorDataset(train_data)
            val_dataset = DescriptorDataset(val_data)
            self.train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            self.val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True)

        else:
            test_dataset = DescriptorDataset(data)
            # This dataset must not be shuffled!
            self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    def initialize_autoencoder(self):
        """
        Define loss function and optimizer for the autoencoder, move network to GPU if possible
        :return: void
        """
        ### Set the random seed for reproducible results
        torch.manual_seed(0)
        ### Define the loss function
        #self.loss_fn = torch.nn.MSELoss()
        self.loss_fn = torch.nn.L1Loss()
        ### Define an optimizer (both for the encoder and the decoder!)
        lr = 1e-3
        params_to_optimize = [
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()}
        ]
        self.optimizer = torch.optim.Adam(params_to_optimize, lr=lr)

        # Move both the encoder and the decoder to the selected device
        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def train_epoch(self):
        """
        Train the network once on the data
        :return: a list of train losses per batch
        """
        train_losses = []
        # Set train mode for both the encoder and the decoder
        self.encoder.train()
        self.decoder.train()
        # Iterate the dataloader
        for descriptor_batch in tqdm(self.train_loader):
            # Move tensor to the proper device
            descriptor_batch = descriptor_batch.to(self.device)
            # Encode data
            encoded_data = self.encoder(descriptor_batch)
            # Decode data
            decoded_data = self.decoder(encoded_data)
            # Evaluate loss
            loss = self.loss_fn(decoded_data, descriptor_batch)
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss.data.cpu())
        return train_losses

    def test_epoch(self, dataloader):
        """
        Test the network once on the data
        :param dataloader: the validation or test dataloader
        :return: list of validation/test loss per batch
        """
        val_losses = []
        self.test_outputs = []
        # Set evaluation mode for encoder and decoder
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():  # No need to track the gradients
            for descriptor_batch in tqdm(dataloader):
                # Move tensor to the proper device
                descriptor_batch = descriptor_batch.to(self.device)
                # Encode data
                encoded_data = self.encoder(descriptor_batch)
                # Decode data
                decoded_data = self.decoder(encoded_data)
                # Evaluate loss
                val_loss = self.loss_fn(decoded_data, descriptor_batch)
                val_losses.append(val_loss.data.cpu())
                # Remember the outputs
                self.test_outputs.append(decoded_data.cpu().numpy())
        return val_losses

    def training_loop(self):
        """
        train the autoencoder over various epochs
        :return: lists of average training and validation losses per epoch
        """
        avg_train_losses = []
        avg_val_losses = []
        for epoch in range(self.num_epochs):
            print('EPOCH %d/%d' % (epoch + 1, self.num_epochs))
            ### Training (use the training function)
            train_losses = self.train_epoch()
            avg_train_losses.append(np.mean(train_losses))
            print('\n\n\t TRAINING - EPOCH %d/%d - loss: %f\n\n' % (epoch + 1, self.num_epochs, np.mean(train_losses)))
            ### Validation  (use the testing function)
            val_losses = self.test_epoch(self.val_loader)
            # Print Validationloss
            print('\n\n\t VALIDATION - EPOCH %d/%d - loss: %f\n\n' % (epoch + 1, self.num_epochs, np.mean(val_losses)))
            avg_val_losses.append(np.mean(val_losses))
        return avg_train_losses, avg_val_losses

    def plot_losses(self, train, val):
        """
        Plot the training and validation losses
        :param train: list of training losses
        :param val: list of validation losses
        :return: void
        """
        plt.figure(figsize=(12, 8))
        plt.semilogy(train, label='Train loss')
        plt.semilogy(val, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()
        plt.show()

    def test(self):
        """
        Test the network on only one epoch
        :return: a list of outputs and the mean test loss
        """
        self.test_outputs = []
        test_losses = self.test_epoch(self.test_loader)
        return self.test_outputs, np.mean(test_losses)

    def save_net(self):
        """
        Save the network state to a file
        :return: void
        """
        ### Save the network states
        # The state dictionary includes all the parameters of the network
        enc_state_dict = self.encoder.state_dict()
        dec_state_dict = self.decoder.state_dict()
        # Save the state dict to a file
        torch.save(enc_state_dict, os.path.join(ROOT_DIR, 'enc_parameters_4l.torch'))
        torch.save(dec_state_dict, os.path.join(ROOT_DIR, 'dec_parameters_4l.torch'))

    def load_net(self):
        """
        Load an existing network state from a file
        :return: void
        """
        ### Reload the network state
        # Load the state dict previously saved
        enc_state_dict = torch.load(os.path.join(ROOT_DIR, 'enc_parameters_4l.torch'))
        dec_state_dict = torch.load(os.path.join(ROOT_DIR, 'dec_parameters_4l.torch'))
        # Update the network parameters
        self.encoder.load_state_dict(enc_state_dict)
        self.decoder.load_state_dict(dec_state_dict)
        self.encoder.to(self.device)
        self.decoder.to(self.device)
