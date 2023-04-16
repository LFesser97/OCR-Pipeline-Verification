import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils import get_default_device
import R2
import time


class VGG_FeatureExtractor(nn.Module):
    """ FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf) """

    def __init__(self, input_channel, output_channel):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [8, 16, 32, 64]
        # self.ConvNet = nn.Sequential( #1x20x100
        #     nn.Conv2d(input_channel, self.output_channel[1], kernel_size=5, stride=1, padding=0), #16x16x96
        #     nn.MaxPool2d(2,2),  # 16x8x48
        #     nn.ReLU(True),
        #     # nn.Dropout(p=0),
        #     nn.Conv2d(self.output_channel[1], self.output_channel[2], 5, 1, 2), #32x8x48
        #     nn.MaxPool2d(2,2),  # 32x4x24
        #     nn.ReLU(True),
        #     # nn.BatchNorm2d(self.output_channel[2], affine=False),
        #     nn.Dropout(p=0),
        #     nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1), #64x4x24
        #     nn.MaxPool2d(2, 2),  # 64x2x12
        #     nn.ReLU(True),
        #     nn.Dropout(p=0),
        #     # nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1),
        #     # nn.ReLU(True),  # 64x2x12
        #     # nn.Dropout(p=0.2),
        #     nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0),
        #     nn.ReLU(True),  # hid*1x11
        #     nn.BatchNorm2d(self.output_channel[3],affine=False),
        # )


        self.ConvNet = nn.Sequential(  # 1x20x100
            nn.Conv2d(input_channel, self.output_channel[1], kernel_size=5, stride=1, padding=0),  # 16x16x96
            nn.MaxPool2d(2, 2),  # 16x8x48
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 5, 1, 2),  # 32x8x48
            nn.MaxPool2d(2, 2),  # 32x4x24
            nn.ReLU(True),
            nn.BatchNorm2d(self.output_channel[2], affine=False),
            nn.Dropout(p=0.2),
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1),  # 64x4x24
            nn.MaxPool2d(2, 2),  # 64x2x12
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            # nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1),
            # nn.ReLU(True),  # 64x2x12
            # nn.Dropout(p=0.2),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0),
            nn.ReLU(True),  # hid*1x11
            nn.BatchNorm2d(self.output_channel[3], affine=False),
            nn.Dropout(p=0.2),
        )
    def forward(self, input):
        return self.ConvNet(input)


class LSTM(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super(LSTM,self).__init__()
        self.rnn=nn.LSTM(input_size,hidden_size,bidirectional=False,batch_first=True)
        # self.linear=nn.Linear(hidden_size, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        # self.rnn.flatten_parameters()
        output, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x hidden_size
        # output=self.dropout(recurrent)
        # output = self.linear(output)  # batch_size x T x output_size
        return output

class OCRModel(nn.Module):
    def __init__(self,num_class,input_channel=1,output_channel=32,hidden_size=64):
        super().__init__()

        dev = get_default_device()

        self.FeatureExtraction=VGG_FeatureExtractor(input_channel,output_channel).to(dev)

        # self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1
        self.SequenceModeling=LSTM(output_channel,hidden_size,hidden_size)


        """ Prediction """
        # if opt.Prediction == 'CTC':
        self.Prediction = nn.Linear(hidden_size, num_class)
        # elif opt.Prediction == 'Attn':
        #     self.Prediction = Attention(hidden_size, opt.hidden_size, opt.num_class)

    def forward(self, input):


        visual_feature = self.FeatureExtraction(input)
        # print(visual_feature.squeeze(0).permute(1,2,0).flatten())
        visual_feature=visual_feature.permute(0, 3, 1, 2)
        # visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)
        """ Sequence modeling stage """

        # self.rnn.flatten_parameters()
        contextual_feature= self.SequenceModeling(visual_feature)  # batch_size x T x input_size -> batch_size x T x hidden_size
        # recurrent = self.dropout(recurrent)

        """ Prediction stage """
        # if self.stages['Pred'] == 'CTC':
        prediction = self.Prediction(contextual_feature.contiguous())
        # else:
        #     prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction




class OCRModelDP(OCRModel):
    def __init__(self, *args, **kwargs):
        super(OCRModelDP, self).__init__(*args, **kwargs)

    def set_bound_method(self, bound_method):
        self.bound_method = bound_method

    def certify(self, input, gt,image_tensors,max_length, max_iter=100,verbose=False):
        layers = []
        lstm_pack = []
        dev = input.device
        _,c,h,w=image_tensors.shape

        # feed=input
        out=input
        last_layer=None
        for layer in self.FeatureExtraction.ConvNet:
            if isinstance(layer, nn.Conv2d):
                print('Conv layer')
                R2conv,h,w=R2.Convolution_Layer.convert(layer,h,w,prev_layer=last_layer,device=dev)
                out=R2conv(out)
                layers.append(R2conv)
                last_layer=R2conv
                print(h,w)
            elif isinstance(layer, nn.MaxPool2d):
                print('MaxPool layer')
                R2maxPool = R2.MaxPool.convert(layer, int(out.dim/(h*w)), h, w, prev_layer=last_layer,device=dev)
                out = R2maxPool(out)
                h,w,_=R2maxPool.out_shape
                layers.append(R2maxPool)
                last_layer=R2maxPool
                print(h,w)
            elif isinstance(layer, nn.BatchNorm2d):
                print('BatchNorm layer')
                R2bn = R2.BatchNormalization.convert(layer, prev_layer=last_layer,device=dev)
                out = R2bn(out)
                layers.append(R2bn)
                last_layer=R2bn
            elif isinstance(layer,nn.ReLU):
                print('ReLU')
                R2ReLU = R2.ReLU(prev_layer=last_layer)
                out = R2ReLU(out)
                layers.append(R2ReLU)
                last_layer=R2ReLU

        # print(out.lb,out.ub) # h(1)*w(11)*c(out_channel) flattened


        out_channel=int(out.dim/max_length)
        assert out_channel*max_length==out.dim
        lin1 = R2.Linear(out.dim, out.dim)
        lin1.assign(torch.eye(out.dim), device=dev)
        out = lin1(out)
        last_layer=lin1

        frames = []
        for frame_idx in range(max_length):
            select=R2.Selection([i for i in range(frame_idx*out_channel,(frame_idx+1)*out_channel)],prev_layer=last_layer)
            lstm_in=select(out)

            R2lstm = R2.LSTMCell.convert(
                self.SequenceModeling.rnn,
                prev_layer=select,
                prev_cell=None if frame_idx == 0 else lstm_pack[-1],
                method=self.bound_method,
            )
            lstm_pack.append(R2lstm)
            lstm_out = R2lstm(lstm_in)
            frames.append(lstm_out)

            lin = R2.Linear.convert(self.Prediction, prev_layer=R2lstm, device=dev)
            lin_out = lin(lstm_out)
            # print(lin_out.lb,lin_out.ub)

            layers.append(lin)
            out_dim = lin.out_features
            lin_compare = R2.Linear(out_dim, 1, prev_layer=layers[-1])
            layers.append(lin_compare)
            # idx=lin_out.lb.argmax()

            # chain.append(lstm)

            # layers.append(chain)
            lp_proven = True
            possible_labels=[]
            frame_label=gt[frame_idx]
            for fl in range(out_dim):  # false label
                if fl == frame_label:
                    continue
                if verbose:
                    print(f"Testing label {fl} | ground truth {frame_label}")

                comp_mtx = torch.zeros(out_dim, 1)
                comp_mtx[frame_label, 0] = 1
                comp_mtx[fl, 0] = -1
                lin_compare.assign(comp_mtx, device=dev)
                lp_res = lin_compare(lin_out)

                if lp_res.lb[0] > 0:
                    if verbose:
                        print("\tProven.")
                    continue
                elif self.bound_method != "opt":
                    if verbose:
                        print("\tChange of class.")
                    possible_labels.append(fl)

            print("correct: ",frame_label,"miss:", possible_labels)
        #
        #     lp_proven = False
        #     st_time = time.time()
        #     lmbs = []
        #     for chain in layers[:-2]:
        #         for layer in chain:
        #             if hasattr(layer, "lmb"):
        #                 lmbs.append(layer.set_lambda(dev))
        #
        #     optim = torch.optim.Adam(lmbs)
        #     lr_fn = lambda e: 100 * 0.98 ** e
        #     scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_fn)
        #     success = False
        #     for epoch in range(max_iter):
        #         for chain, inp in zip(layers[:-2], feed):
        #             out = inp
        #             for layer in chain:
        #                 out = layer(out)
        #         out = layers[-2](out)
        #         out = layers[-1](out)
        #
        #         if verbose:
        #             print(f"\tEpoch {epoch}: min(LB_gt - UB_fl) = {out.lb[0]}")
        #         if out.lb[0] > 0:
        #             if verbose:
        #                 print("\tROBUSTNESS PROVED")
        #             success = True
        #             break
        #
        #         loss = -out.lb[0]
        #         loss.backward(retain_graph=True)
        #         optim.step()
        #         optim.zero_grad()
        #         scheduler.step()
        #
        #     if not success:
        #         return False
        #
        # return True