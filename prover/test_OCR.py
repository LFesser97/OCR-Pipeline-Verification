import R2_new as R2
# import R2
import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
import os, sys, random
from dataset import RawDataset,AlignCollate
from models_OCR import OCRModel, OCRModelDP
from utils import get_default_device,CTCLabelConverter
import numpy as np
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description="OCR verification")
parser.add_argument(
    "--nframes",
    type=int,
    default=15,
    choices=[15],
    help="the number of fixed frames. choose 15.",
)
parser.add_argument(
    "--out_channels", type=int, default=64, help="the output dimension of feature extractor."
)

parser.add_argument(
    "--nhidden", type=int, default=32, help="the hidden dimension of each LSTM cell."
)
parser.add_argument("--nlayers", type=int, default=1, help="the number of LSTM layers.")
parser.add_argument("--eps", type=float, default=0.01, help="perturbation epsilon.")
parser.add_argument(
    "--bound_method",
    type=str,
    default="lp",
    choices=["lp", "opt"],
    help="bounding method, either lp or opt (optimize gradient).",
)
parser.add_argument("--model_dir", type=str, default="", help="target model directory.")
parser.add_argument(
    "--seed", type=int, default=1000, help="random seed for reproducibility."
)
parser.add_argument(
    "--verbose", action="store_true", help="print debug information during verifiation."
)
parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')

args = parser.parse_args()


# args.model_dir='saved/ctc/best_accuracy.pt'
# args.model_dir='saved/ctc_6432_70acc.pt'

args.model_dir='saved/9_frames.pt'
# args.model_dir='saved/low_maxpool_9f.pt'

nframes = args.nframes
nhidden = args.nhidden
out_channel = args.out_channels
input_channel=1
eps = args.eps
bound_method = "lp"
seed = args.seed
model_name = args.model_dir

args.rgb=False
args.imgH=20
args.imgW=100
args.valid_data='crnndata/ICDAR_C1_testing'
args.character = "0123456789abcdefghijklmnopqrstuvwxyz"
args.batch_max_length=9
args.prediction='CTC'
args.verbose=True
args.sensitive=False

if args.sensitive:
    args.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

dev = get_default_device()
dev="cpu"
print(dev)
# devtxt = "cuda" if dev == torch.device("cuda") else "cpu"

# stt_dict = torch.load(model_name, map_location=devtxt)

num_class=len(args.character)+1

model= OCRModel(num_class,input_channel,out_channel,nhidden).to('cpu')
model.load_state_dict(torch.load(model_name, map_location='cpu'))
print(model)
filtered_parameters = []
params_num = []
for p in filter(lambda p: p.requires_grad, model.parameters()):
    filtered_parameters.append(p)
    params_num.append(np.prod(p.size()))
print('Trainable params num : ', sum(params_num))
model.eval()

r2model = OCRModelDP(num_class,input_channel,out_channel,nhidden).to('cpu')
r2model.load_state_dict(torch.load(model_name, map_location='cpu'))
r2model.eval()

r2model.set_bound_method(bound_method)

AlignCollate_valid = AlignCollate(imgH=args.imgH, imgW=args.imgW, keep_ratio_with_pad=False)
valid_dataset = RawDataset(root=args.valid_data, opt=args)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=1,
    shuffle=False,  # 'True' to check training progress with validation function.
    collate_fn=AlignCollate_valid, pin_memory=True)

proven_dp = 0
correct = 0
running_time = 0.0
converter=CTCLabelConverter(args.character)
for i, (image_tensors, labels) in enumerate(valid_loader):
# for i in tqdm(range(120)):
    sys.stdout.flush()
    print(f"[Testing Input #{i:03d} ({proven_dp} proven / {correct} correct)]")
    batch_size = image_tensors.size(0)
    assert batch_size==1
    image = image_tensors.to('cpu')
    # For max length prediction
    if 'CTC' in args.prediction:
        preds = model(image)
        # preds (batch,frame,classes)
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, preds_index = preds_prob.detach().max(dim=2)
        preds_size = torch.IntTensor([preds.size(1)] * 1)
        confidence=preds_max_prob[0].cumprod(dim=0)[-1]
        preds_str = converter.decode(preds_index.data,preds_size.data)
        print(confidence, preds_str, labels)

    if preds_str!=labels:
        print("- prediction failed")
        continue
    # if confidence<0.9:
    #     print("- confidence low")
    #     continue
    print(preds_str)

    correct+=1

    # from (c,h,w) to (h,w,c)
    input=image_tensors.squeeze(0).permute(1,2,0).flatten() # image input
    # print(input)
    # input=model.FeatureExtraction(image_tensors).squeeze(0).permute(1,2,0).flatten() # feature extractor output
    eps=0.001
    input_dp = R2.deeppoly_from_perturbation(input.to(dev), eps,truncate=(-1,1)) # truncate=(-1, 1)
    st = time.time()
    proven = r2model.certify(input_dp, preds_index.squeeze(0), image_tensors,args.batch_max_length,converter,dev,args)
    # proven = True
    ed = time.time()
    # print(ed-st)
    running_time += ed - st

    print(f" - took {ed-st} sec to verify")
    if proven:
        print("\t[PROVEN]")
        proven_dp += 1
    else:
        print("\t[FAILED]")
    if correct == 100:
        break

print(f"provability: {proven_dp/correct*100}%")
print(f"avg running time: {running_time/correct}")
#
# os.makedirs("results", exist_ok=True)
# res_name = f"results/exp_mnist_{nframes}f_{nhidden}h_{nlayers}l.csv"
# with open(res_name, "a") as f:
#     f.write(f"{bound_method},{eps},{proven_dp/correct*100},{running_time/correct}\n")
