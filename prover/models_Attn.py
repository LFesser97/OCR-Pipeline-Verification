import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils import get_default_device
import R2
import time


class VGG_FeatureExtractor_Attn(nn.Module):
    def __init__(self, input_channel, output_channel=512):
        super(VGG_FeatureExtractor_Attn, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [8, 16, 32, 64]
        self.ConvNet = nn.Sequential(  # 1x20x100
            nn.Conv2d(input_channel, self.output_channel[1], kernel_size=5, stride=1, padding=0),  # 16x16x96
            nn.MaxPool2d(2, 2),  # 16x8x48
            nn.ReLU(True),
            nn.Dropout(p=0.1),

            nn.Conv2d(self.output_channel[1], self.output_channel[2], 5, 1, 2),  # 32x8x48
            nn.MaxPool2d(2, 2),  # 32x4x24
            nn.ReLU(True),
            nn.BatchNorm2d(self.output_channel[2], affine=False),
            nn.Dropout(p=0.1),

            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1),  # 64x4x24
            nn.MaxPool2d(2, 2),  # 64x2x12
            nn.ReLU(True),
            nn.Dropout(p=0.1),

            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1),  # 64x2x12
            nn.ReLU(True),
            nn.Dropout(p=0.1),

            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1),  # 64x2x12
            nn.ReLU(True),
            nn.Dropout(p=0.1),

            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1),  # 64x2x12
            nn.ReLU(True),
            nn.Dropout(p=0.1),

            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0),
            nn.ReLU(True),  # hid*1x11
            nn.BatchNorm2d(self.output_channel[3], affine=False),
            nn.Dropout(p=0.1),
        )

    def forward(self, input):
        return self.ConvNet(input)

class VGG_FeatureExtractor(nn.Module):
    """ FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf) """

    def __init__(self, input_channel, output_channel):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [8, 16, 32, 64]
        self.ConvNet = nn.Sequential( #1x20x100
            nn.Conv2d(input_channel, self.output_channel[1], kernel_size=5, stride=1, padding=0), #16x16x96
            nn.MaxPool2d(2,2),  # 16x8x48
            nn.ReLU(True),
            # nn.Dropout(p=0.1),
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 5, 1, 2), #32x8x48
            nn.MaxPool2d(2,2),  # 32x4x24
            nn.ReLU(True),
            nn.BatchNorm2d(self.output_channel[2], affine=False),
            nn.Dropout(p=0.2),
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1), #64x4x24
            nn.MaxPool2d(2, 2),  # 64x2x12
            nn.ReLU(True),
            nn.Dropout(p=0.3),
            # nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1),
            # nn.ReLU(True),  # 64x2x12
            # nn.Dropout(p=0.3),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0),
            nn.ReLU(True),  # hid*1x11
            nn.BatchNorm2d(self.output_channel[3],affine=False),
            nn.Dropout(p=0.3),
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

class Attention(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes,args):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)
        self.dropout=nn.Dropout(p=0.3)
        self.dev=args.dev
        self.args=args


    def _char_to_onehot(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(self.dev)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, batch_H, text, is_train=False, batch_max_length=20):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.

        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(self.dev)
        hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(self.dev),
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(self.dev))

        if is_train:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)

            output_hiddens=self.dropout(output_hiddens)
            probs = self.generator(output_hiddens)

        else:
            targets = torch.LongTensor(batch_size).fill_(0).to(self.dev)  # [GO] token
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(self.dev)

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes


class AttentionCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size
        self.dropout=nn.Dropout(p=0.3)


    def forward(self, prev_hidden, batch_H, char_onehots):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # batch_size x num_channel
        concat_context = torch.cat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
        concat_context=self.dropout(concat_context)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha


class AttnModel(nn.Module):
    def __init__(self,num_class,input_channel=1,output_channel=32,hidden_size=64,args=None):
        super().__init__()

        self.args=args
        self.FeatureExtraction=VGG_FeatureExtractor_Attn(input_channel,output_channel).to(args.dev)

        # self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1
        self.SequenceModeling=LSTM(output_channel,hidden_size,hidden_size)


        """ Prediction """
        # if opt.Prediction == 'CTC':
        # self.Prediction = nn.Linear(hidden_size, num_class)
        # elif opt.Prediction == 'Attn':
        self.Prediction = Attention(hidden_size, hidden_size, num_class,args)

    def forward(self, input,text, is_train=False):


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
        #     prediction = self.Prediction(contextual_feature.contiguous())
        # else:
        prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.args.batch_max_length)

        return prediction




class AttnModelDP(AttnModel):
    def __init__(self, *args, **kwargs):
        super(AttnModelDP, self).__init__(*args, **kwargs)

    def set_bound_method(self, bound_method):
        self.bound_method = bound_method

    def certify(self, input, gt,image_tensors,max_length, converter, max_iter=100,verbose=False):
        layers = []
        lstm_pack = []
        dev = input.device
        _,c,h,w=image_tensors.shape

        # feed=input
        out=input
        last_layer=None
        # for layer in self.FeatureExtraction.ConvNet:
        #     if isinstance(layer, nn.Conv2d):
        #         R2conv,h,w=R2.Convolution_Layer.convert(layer,h,w,prev_layer=last_layer,device=dev)
        #         out=R2conv(out)
        #         layers.append(R2conv)
        #         last_layer=R2conv
        #         if verbose:
        #             print('Conv layer')
        #             print(h,w)
        #     elif isinstance(layer, nn.MaxPool2d):
        #         R2maxPool = R2.MaxPool.convert(layer, int(out.dim/(h*w)), h, w, prev_layer=last_layer,device=dev)
        #         out = R2maxPool(out)
        #         h,w,_=R2maxPool.out_shape
        #         layers.append(R2maxPool)
        #         last_layer=R2maxPool
        #         if verbose:
        #             print('MaxPool layer')
        #             print(h,w)
        #     elif isinstance(layer, nn.BatchNorm2d):
        #         R2bn = R2.BatchNormalization.convert(layer, prev_layer=last_layer,device=dev)
        #         out = R2bn(out)
        #         layers.append(R2bn)
        #         last_layer=R2bn
        #         if verbose:
        #             print('BatchNorm layer')
        #     elif isinstance(layer,nn.ReLU):
        #         R2ReLU = R2.ReLU(prev_layer=last_layer)
        #         out = R2ReLU(out)
        #         layers.append(R2ReLU)
        #         last_layer=R2ReLU
        #         if verbose:
        #             print('ReLU')

        # print(out.lb,out.ub) # h(1)*w(11)*c(out_channel) flattened


        out_channel=int(out.dim/max_length)
        assert out_channel*max_length==out.dim
        lin1 = R2.Linear(out.dim, out.dim)
        lin1.assign(torch.eye(out.dim), device=dev)
        out = lin1(out)
        last_layer=lin1

        frames = []
        feed = []
        lstm_layers=[]
        all_possible_labels=[]
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
            )
            lstm_pack.append(R2lstm)
            lstm_out = R2lstm(lstm_in)
            frames.append(lstm_out)
            chain.append(R2lstm)

            lstm_layers.append(chain)
        # append all lstms

        num_steps = self.argsbatch_max_length + 1  # +1 for [s] at end of sentence.

        prev_hidden = (torch.FloatTensor(1, self.hidden_size).fill_(0).to(dev),
                  torch.FloatTensor(1, self.hidden_size).fill_(0).to(dev))

        targets = torch.LongTensor(1).fill_(0).to(dev)  # [GO] token
        probs = torch.FloatTensor(1, num_steps, self.num_classes).fill_(0).to(dev)

        attn_lb=[]
        attn_ub=[]
        for lstm_out in frames:
            attn_lb.append(lstm_out.lb)
            attn_ub.append(lstm_out.ub)
        batch_H=R2.DeepPoly(torch.tensor(attn_lb), torch.tensor(attn_ub), None, None)

        attn_layers=[]
        frames = []
        feed = []
        lstm_layers=[]
        for step in range(num_steps):
            char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)

            i2h = R2.Linear.convert(self.Prediction.attention_cell.i2h, prev_layer=None, device=dev)
            i2h_out = i2h(batch_H)
            attn_layers.append(i2h)
            # batch_H_proj = self.i2h(batch_H)

            # h2h_prev=None
            # h2h = R2.Linear.convert(self.Prediction.attention_cell.h2h, prev_layer=h2h_prev, device=dev)
            # h2h_out = h2h(prev_hidden[0]).unsqueeze(1)
            h2h_out=self.Prediction.attention_cell.h2h(prev_hidden[0]).unsqueeze(1)
            # attn_layers.append(h2h)

            addition=R2.Add(prev_layer=i2h)
            add_out=addition(i2h_out,h2h_out)

            tanh=R2.Sigmoidal("tanh",prev_layer=addition)
            tanh_out=tanh(add_out)

            score=R2.Linear.convert(self.Prediction.attention_cell.score,prev_layer=tanh,device=dev)
            e=score(tanh_out)


            alpha = F.softmax(e, dim=1)




            # mat mul
            context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # batch_size x num_channel

            concat_context = torch.cat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)



            R2lstm = R2.LSTMCell.convert(
                self.SequenceModeling.rnn,
                prev_layer=select,
                prev_cell=None if step == 0 else lstm_pack[-1],
                method=self.bound_method,
            )
            lstm_pack.append(R2lstm)
            lstm_out = R2lstm(concat_context)
            # frames.append(lstm_out)
            chain.append(R2lstm)
            # lstm_out is hidden state value

            cur_hidden = self.rnn(concat_context, prev_hidden)
            prev_hidden=cur_hidden

            generator=R2.linear.convert(self.Prediction.generator,prev_layer=...,device=dev)
            probs_step=generator(lstm_out)

            _, next_input = probs_step.max(1)
            targets = next_input



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

                if lp_res.lb[0] > 0:
                    # if verbose:
                    #     print("\tProven.")
                    continue
                elif self.bound_method != "opt":
                    if verbose:
                        print("\tChange of class",lp_res.lb[0])
                    possible_labels.append(fl)
                elif self.bound_method == "opt":
                    # if lp_res.lb[0]<-15:
                    #     if verbose:
                    #         print("\tChange of class", lp_res.lb[0])
                    #     possible_labels.append(fl)
                    #     continue
                    cuda_dev='cuda'
                    st_time = time.time()
                    lmbs = []
                    for layer in lstm_pack:
                        if hasattr(layer, "lmb"):
                            lmbs.append(layer.set_lambda(dev))

                    optim = torch.optim.Adam(lmbs,lr=0.005)
                    lr_fn = lambda e: 100 * 0.98 ** e
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_fn)
                    success = False

                    # for inp in feed:
                    #     inp.device=cuda_dev
                    # for chain in lstm_layers:
                    #     for layer in chain:

                    for epoch in range(max_iter):
                        for chain, inp in zip(lstm_layers, feed):
                            opt_out = inp
                            for layer in chain:
                                opt_out = layer(opt_out)

                        if verbose:
                            print(f"\tEpoch {epoch}: min(LB_gt - UB_fl) = {opt_out.lb[0]}")
                        if opt_out.lb[0] > 0:
                            if verbose:
                                print("\tROBUSTNESS PROVED")
                            success = True
                            break

                        loss = -opt_out.lb[0]
                        loss.backward(retain_graph=True)
                        optim.step()
                        optim.zero_grad()
                        scheduler.step()

                    if not success:
                        possible_labels.append(fl)

            print("correct: ",frame_label,"miss:", possible_labels)
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