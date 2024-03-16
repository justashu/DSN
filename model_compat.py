import torch
import torch.nn as nn
from functions import ReverseLayerF


class DSN(nn.Module):
    def __init__(self, code_size=100, n_class=10):
        super(DSN, self).__init__()
        self.code_size = code_size

        ##########################################
        # private source encoder
        ##########################################

        # self.source_encoder_conv = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        
        # self.source_encoder_fc = nn.Sequential(
        #     nn.Linear(in_features=64 * 32 * 64, out_features=code_size),
        #     nn.ReLU(True)
        # )

        #########################################
        # private target encoder
        #########################################

        self.target_encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.target_encoder_fc = nn.Sequential(
            nn.Linear(in_features=64 * 32 * 64, out_features=code_size),
            nn.ReLU(True)
        )

        ################################
        # shared encoder (dann_mnist)
        ################################
        self.shared_encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.shared_encoder_fc = nn.Sequential(
            nn.Linear(in_features=64*32*48, out_features=code_size),
            nn.ReLU(True)
        )

          
        # classify 10 numbers
        # self.shared_encoder_pred_class = nn.Sequential()
        # self.shared_encoder_pred_class.add_module('fc_se4', nn.Linear(in_features=code_size, out_features=100))
        # self.shared_encoder_pred_class.add_module('relu_se4', nn.ReLU(True))
        # self.shared_encoder_pred_class.add_module('fc_se5', nn.Linear(in_features=100, out_features=n_class))

        self.shared_encoder_pred_domain = nn.Sequential(
            nn.Linear(in_features=code_size, out_features=100),
            nn.ReLU(True),
            nn.Linear(in_features=100, out_features=2) # classify two domain
        )

        
        
        ######################################
        # shared decoder (small decoder)
        ######################################

        self.shared_decoder_fc = nn.Sequential(
            nn.Linear(in_features=code_size, out_features=588),
            nn.ReLU(True)
        )

        self.shared_decoder_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        )

    def forward(self, input_data, mode, rec_scheme, p=0.0):
        result = []

        if mode == 'source':
            private_feat = self.source_encoder_conv(input_data)
            private_feat = private_feat.view(-1, 64 * 32 * 64)
            private_code = self.source_encoder_fc(private_feat)
        elif mode == 'target':
            private_feat = self.target_encoder_conv(input_data)
            private_feat = private_feat.view(-1, 64 * 32 * 64)
            private_code = self.target_encoder_fc(private_feat)

        result.append(private_code)

        shared_feat = self.shared_encoder_conv(input_data)
        shared_feat = shared_feat.view(-1, 64 * 32 * 48)
        shared_code = self.shared_encoder_fc(shared_feat)
        result.append(shared_code)

        reversed_shared_code = ReverseLayerF.apply(shared_code, p)
        domain_label = self.shared_encoder_pred_domain(reversed_shared_code)
        result.append(domain_label)

        if mode == 'source':
            class_label = self.shared_encoder_pred_class(shared_code)
            result.append(class_label)

        if rec_scheme == 'share':
            union_code = shared_code
        elif rec_scheme == 'all':
            union_code = private_code + shared_code
        elif rec_scheme == 'private':
            union_code = private_code

        rec_vec = self.shared_decoder_fc(union_code)
        rec_vec = rec_vec.view(-1, 3, 128, 64)
        rec_code = self.shared_decoder_conv(rec_vec)
        result.append(rec_code)

        return result
