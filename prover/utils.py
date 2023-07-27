import torch, math
import numpy as np


# General utility functions
def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def negative_only(w):
    return -torch.relu(-w)


def positive_only(w):
    return torch.relu(w)


# Below methods are for the speech preprocessing stages. Most of the things are
# for algebraic functions and definitions.
def mel2hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


def hz2mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)


def get_filterbanks(nfilt=26, nfft=512, samplerate=16000, lowfreq=0, highfreq=8000):
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft + 1) * mel2hz(melpoints) / samplerate)

    fbank = np.zeros([nfilt, nfft // 2 + 1])
    for j in range(0, nfilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    return fbank


class M:
    def __init__(
        self,
        frame_size=256,
        frame_step=200,
        n_filt=10,
        samprate=8000,
        dev=torch.device("cpu"),
    ):
        N = frame_size
        s = frame_step
        n = N // 2 + 1

        self.preemph = torch.Tensor(np.eye(N) - np.eye(N, k=1) * 0.97).to(device=dev)
        self.hamming = torch.hamming_window(N).to(device=dev)
        self.W = np.array(
            [
                [
                    [math.cos(i * k * -2.0 * math.pi / N) for k in range(n)]
                    for i in range(N)
                ],
                [
                    [math.sin(i * k * -2.0 * math.pi / N) for k in range(n)]
                    for i in range(N)
                ],
            ],
            dtype=np.float32,
        )  # W[0, ]: real, W[1, ]: imaginary
        self.W_comb = torch.Tensor(
            np.concatenate([self.W[0], self.W[1]], axis=1) / np.sqrt(N)
        ).to(device=dev)

        self.real_add_img = torch.Tensor(np.concatenate([np.eye(n), np.eye(n)], 0)).to(
            device=dev
        )
        self.fb = torch.Tensor(
            get_filterbanks(n_filt, N, samprate, 0, samprate // 2).T.astype(np.float32)
        ).to(device=dev)


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        dev=get_default_device()
        length = [len(s) for s in text]
        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return batch_text.to(dev), torch.IntTensor(length).to(dev)

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts

class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        dev=get_default_device()

        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return batch_text.to(dev), torch.IntTensor(length).to(dev)

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts