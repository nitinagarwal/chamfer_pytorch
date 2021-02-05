import torch
import dist_chamfer_idx as ext
distChamfer = ext.chamferDist()
from torch.autograd import Variable
import time


def timings():
    distChamfer = ext.chamferDist()
    p1 = torch.rand(2, 5, 3).cuda()
    p2 = torch.rand(2, 10, 3).cuda()
    print("Start CUDA version")
    start = time.time()
    for i in range(1000):
        points1 = Variable(p1, requires_grad=True)
        points2 = Variable(p2)
        mydist1, mydist2, idx1, idx2 = distChamfer(points1, points2)
        print(mydist1.size(), mydist2.size(), idx1.size(), idx2.size())
        print(idx1)
        loss = torch.sum(mydist1)
        loss.backward()
    print("Ellapsed time is", {time.time() - start}, "seconds.")

timings()
