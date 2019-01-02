import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from gif_writer import write_gif

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.ref = nn.ReflectionPad2d(4) 
        self.c1  = nn.Conv2d(1,  32, 3, 1, padding=0, dilation=1)
        self.c2  = nn.Conv2d(32, 32, 2, 1, padding=0, dilation=2)
        self.c3  = nn.Conv2d(32,  4, 2, 1, padding=0, dilation=1)

    def forward(self, x):
        x = F.elu(self.c1(self.ref(x)))
        x = F.elu(self.c2(self.ref(x)))
        x = F.elu(self.c3(x))
        x = self.avg(x).view(-1)
        return x

def main():
    # grab cli argument
    positive_example = False
    for arg in sys.argv:
        if "pos" in arg:
            positive_example = True

    # build faux agent and optimizer
    agent     = Agent()
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)
    kl_loss   = nn.KLDivLoss(reduction='batchmean')

    # construct random input for example
    x = torch.randn([9,9])
    x = x.view(1,1,9,9)

    # construct the target dist. based on cli
    if positive_example:
        p = torch.FloatTensor([0,0,1 - 1e-4,0]) 
    else:
        p    = torch.zeros(4) + (1/3) + 1e-4
        p[2] = 1e-4

    # optimize and collect dist. snapshots
    ims = []
    for i in tqdm(range(100)):
        q = agent(x)
        q_l = F.log_softmax(q, dim=0)
        q_s = F.softmax(q, dim=0)
        l = kl_loss(q_l, p)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        dim = 200
        w   = dim // 4
        im  = np.zeros([2*dim, dim, 3], dtype=np.uint8) + 255

        # optimizing histogram
        for i, val in enumerate(q_s.detach().numpy()):
            im[i*w:(i+1)*w, dim-int(dim*val):dim, 0] = min(int(255 * (1-val)), 255)
            im[i*w:(i+1)*w, dim-int(dim*val):dim, 1] = min(int((64 * (1-val))), 255)
            im[i*w:(i+1)*w, dim-int(dim*val):dim, 2] = min(int(((255 * val) + (i*64))), 255)

        # bar separator
        im[dim-1:dim+1, ...] = 0

        # target histogram
        for i, val in enumerate(p.numpy()):
            im[(i+4)*w:(i+5)*w, dim-int(dim*val):dim, 0] = min(int(255 * (1-val)), 255)
            im[(i+4)*w:(i+5)*w, dim-int(dim*val):dim, 1] = min(int((64 * (1-val))), 255)
            im[(i+4)*w:(i+5)*w, dim-int(dim*val):dim, 2] = min(int(((255 * val) + (i*64))), 255)

        ims.append(im)
    
    # save as gif
    filename = "res/"
    if positive_example:
        filename += "pos_div_optim.gif"
    else:
        filename += "neg_div_optim.gif"
    write_gif(ims, filename, fps=12)

if __name__ == "__main__":
    main()