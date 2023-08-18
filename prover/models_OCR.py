import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils import get_default_device
import R2_new as R2
# import R2
import time

class VGG_FeatureExtractor_short(nn.Module):
    """ FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf) """

    def __init__(self, input_channel, output_channel=512):
        super(VGG_FeatureExtractor_short, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [8, 16, 32, 64]
        self.ConvNet = nn.Sequential( #1x20x100
            nn.Conv2d(input_channel, self.output_channel[1], kernel_size=5, stride=1, padding=0), #16x16x96
            nn.MaxPool2d(2,2),  # 16x8x48
            nn.ReLU(True),
            nn.Dropout(p=0.1),
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 5, 1, 2), #32x8x48
            nn.MaxPool2d((1,2),(1,2)),  # 32x8x24
            nn.ReLU(True),
            nn.BatchNorm2d(self.output_channel[2], affine=False),
            nn.Dropout(p=0.2),
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 0), #64x6x22
            nn.MaxPool2d((2, 2), (2, 2)),  # 64x3x11
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1),# 64x3x11
            nn.ReLU(True),  # 64x3x11
            nn.Dropout(p=0.2),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 0),# 64x1x9
            nn.ReLU(True),  # hid*1x9
            nn.BatchNorm2d(self.output_channel[3],affine=False),
            nn.Dropout(p=0.2),
        )

    def forward(self, input):
        return self.ConvNet(input)


class VGG_FeatureExtractor_CTC(nn.Module):
    def __init__(self, input_channel, output_channel=512):
        super(VGG_FeatureExtractor_CTC, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [8, 16, 32, 64]
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

            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1),  # 64x2x12
            nn.ReLU(True),
            nn.Dropout(p=0.2),

            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1),  # 64x2x12
            nn.ReLU(True),
            nn.Dropout(p=0.2),

            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0),
            nn.ReLU(True),  # hid*1x11
            nn.BatchNorm2d(self.output_channel[3], affine=False),
            nn.Dropout(p=0.2),
        )

    def forward(self, input):
        return self.ConvNet(input)

class VGG_FeatureExtractor_9frames(nn.Module):
    """ FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf) """

    def __init__(self, input_channel, output_channel=512):
        super(VGG_FeatureExtractor_9frames, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [8, 16, 32, 64]
        self.ConvNet = nn.Sequential( #1x20x100
            nn.Conv2d(input_channel, self.output_channel[1], kernel_size=6, stride=2, padding=0), #16x8x48
            # nn.MaxPool2d(2,2),  # 16x8x48
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 4, 2, 1), #32x4x24
            nn.MaxPool2d((1,2),(1,2)),  # 32x4x12
            nn.ReLU(True),
            nn.BatchNorm2d(self.output_channel[2], affine=False),
            nn.Dropout(p=0.1),
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 0), #64x2x10
            # nn.MaxPool2d((2, 2), (2, 2)),  # 64x3x11
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1),# 64x2x10
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0),# 64x1x9
            nn.ReLU(True),  # hid*1x9
            nn.BatchNorm2d(self.output_channel[3],affine=False),
            nn.Dropout(p=0.2),
        )

    def forward(self, input):
        return self.ConvNet(input)

class VGG_FeatureExtractor(nn.Module):
    """ FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf) """

    def __init__(self, input_channel, output_channel):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [8, 16, 32, 64]

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

        self.FeatureExtraction=VGG_FeatureExtractor_short(input_channel,output_channel).to(dev)
        # self.FeatureExtraction=VGG_FeatureExtractor_9frames(input_channel,output_channel).to(dev)

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

    def certify(self, input, gt,image_tensors,max_length, converter, dev,args,max_iter=15):
        layers = []
        lstm_pack = []
        _,c,h,w=image_tensors.shape

        verbose=args.verbose
        char=['[blank]'] + list(args.character)

        # feed=input
        out=input
        last_layer=None
        alphas=[]
        for layer in self.FeatureExtraction.ConvNet:
            if isinstance(layer, nn.Conv2d):
                R2conv,h,w=R2.Convolution_Layer.convert(layer,h,w,prev_layer=last_layer,device=dev)
                out=R2conv(out)
                layers.append(R2conv)
                last_layer=R2conv
                if verbose:
                    print('Conv layer')
                    print(h,w)
            elif isinstance(layer, nn.MaxPool2d):
                R2maxPool = R2.MaxPool.convert(layer, int(out[0].size()[0]/(h*w)), h, w, prev_layer=last_layer,device=dev)
                out = R2maxPool(out)
                h,w,_=R2maxPool.out_shape
                layers.append(R2maxPool)
                last_layer=R2maxPool
                if verbose:
                    print('MaxPool layer')
                    print(h,w)
            elif isinstance(layer, nn.BatchNorm2d):
                R2bn = R2.BatchNormalization.convert(layer, prev_layer=last_layer,device=dev)
                out = R2bn(out)
                layers.append(R2bn)
                last_layer=R2bn
                if verbose:
                    print('BatchNorm layer')
            elif isinstance(layer,nn.ReLU):
                R2ReLU = R2.ReLU(prev_layer=last_layer,device=dev)
                alphas.append(R2ReLU.set_alpha(out,dev))
                out = R2ReLU(out)
                layers.append(R2ReLU)
                last_layer=R2ReLU
                if verbose:
                    print('ReLU')

        optim = torch.optim.RMSprop(alphas,lr=1)
        # optim = torch.optim.Adam(alphas,lr=0.1)

        # lr_fn = lambda e: 100 * 0.98 ** e
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_fn)
        loss_fn = nn.L1Loss()
        for epoch in range(0):
            optim.zero_grad()
            loss = loss_fn(out[1]-out[0],torch.zeros(out[0].size()[0]).to(dev))
            loss.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(alphas, 5)

            optim.step()
            # scheduler.step()
            for alpha in alphas:
                a=alpha.data
                a.clamp_(0,1)

            if verbose:
                print(f"\tEpoch {epoch}: loss = {loss.item()}")
            out = input
            for layer in layers:
                out = layer(out)
        optim.zero_grad()

        for alpha in alphas:
            alpha.requires_grad_(requires_grad=False)

        out_channel=int(out[0].size()[0]/max_length)
        assert out_channel*max_length==out[0].size()[0]


        # lin1 = R2.Linear(out[0].size()[0], out[0].size()[0],prev_layer=last_layer)
        # lin1.assign(torch.eye(out[0].size()[0]), device=dev)
        # out = lin1(out)
        # last_layer=lin1
        # layers.append(lin1)

        # print(loss_fn(out[1]-out[0],torch.zeros(out[0].size()[0]).to(dev)).item())
        # print(out[0],out[1]) # h(1)*w(11)*c(out_channel) flattened


        # frames = []
        feed = []
        lstm_layers=[]
        all_possible_labels=[]
        lmbs=[]
        for frame_idx in range(max_length):
            chain = []
            select=R2.Selection([i for i in range(frame_idx*out_channel,(frame_idx+1)*out_channel)],prev_layer=last_layer)
            lstm_in=select(out)
            # chain.append(select)
            feed.append(lstm_in)

            R2lstm = R2.LSTMCell.convert(
                self.SequenceModeling.rnn,
                prev_layer=select,
                prev_cell=None if frame_idx == 0 else lstm_pack[-1],
                method=self.bound_method,
                device=dev
            )
            lstm_pack.append(R2lstm)
            lmbs.append(R2lstm.set_lambda(dev))

            lstm_out = R2lstm(lstm_in)
            # frames.append(lstm_out)
            chain.append(R2lstm)


            lin = R2.Linear.convert(self.Prediction, prev_layer=R2lstm, device=dev)
            lin_out = lin(lstm_out)
            # print(lin_out.lb,lin_out.ub)
            chain.append(lin)

            out_dim = lin.out_features
            lin_compare = R2.Linear(out_dim, 1, prev_layer=chain[-1])
            chain.append(lin_compare)
            # idx=lin_out.lb.argmax()

            lstm_layers.append(chain)

            possible_labels=[]
            frame_label=gt[frame_idx]
            for fl in range(out_dim):  # false label
                if len(possible_labels)>2:
                    return False

                if fl == frame_label:
                    continue
                # if verbose:
                #     print(f"Testing label {fl} | ground truth {frame_label}")

                comp_mtx = torch.zeros(out_dim, 1)
                comp_mtx[frame_label, 0] = 1
                comp_mtx[fl, 0] = -1
                lin_compare.assign(comp_mtx, device=dev)
                lp_res = lin_compare(lin_out)

                # print(lp_res[0])
                if lp_res[0][0] > 0:
                    # if verbose:
                    #     print("\tProven.")
                    continue
                elif self.bound_method != "opt":
                    if verbose:
                        print("\tChange of class",char[fl])
                    possible_labels.append(fl)
                elif self.bound_method == "opt":
                    # if max_length-frame_idx>3:
                    #     return False
                    if lp_res[0][0]<-7 and max_length-frame_idx>2:
                        possible_labels.append(fl)
                        continue
                    elif lp_res[0][0]<-10:
                        possible_labels.append(fl)
                        continue
                    st_time = time.time()
                    lmbs = []
                    for layer in lstm_pack:
                        if hasattr(layer, "lmb"):
                            lmbs.append(layer.set_lambda(dev))

                    # optim = torch.optim.RMSprop(alphas, lr=0.1)
                    optim = torch.optim.Adam(lmbs,lr=0.5)
                    # optim = torch.optim.Adagrad(lmbs,lr=0.001)
                    # lr_fn = lambda e: 100 * 0.98 ** e
                    # scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_fn)
                    success = False

                    for epoch in range(max_iter):
                        for chain, inp in zip(lstm_layers, feed):
                            opt_out = inp
                            for layer in chain:
                                opt_out = layer(opt_out)

                        if verbose:
                            print(char[fl],f"\tEpoch {epoch} {opt_out[0][0]}")
                        if opt_out[0][0] > 0:
                            if verbose:
                                print("\tROBUSTNESS PROVED")
                            success = True
                            break

                        loss = -opt_out[0][0]
                        loss.backward(retain_graph=True)
                        optim.step()
                        optim.zero_grad()
                        # scheduler.step()

                    if not success:
                        possible_labels.append(fl)
            print("correct: ",char[frame_label.item()],"miss:", [char[i] for i in possible_labels])
            if len(possible_labels)>2:
                return False
            else:
                all_possible_labels.append(possible_labels)

        print('all possible labels',all_possible_labels)
        size=torch.IntTensor([len(gt)] * 1)
        manipulated=[gt]
        true_word=converter.decode(gt.unsqueeze(0),size)
        for i,labels in enumerate(all_possible_labels):
            temp=[]
            for l in labels:
                for preds in manipulated:
                    new=preds.detach().clone()
                    new[i]=l
                    preds_word = converter.decode(new.unsqueeze(0), size)
                    # print(preds_word,true_word)
                    if preds_word!=true_word:
                        return False
                    temp.append(new)
            manipulated+=temp
        return True