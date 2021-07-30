import torch
from .Utils import IOU_2D, Largest_box2D, Center_Diff2D

def DIOU_loss2D(PD, GT):
    
    ious = IOU_2D(PD, GT)
    minR,minC,maxR,maxC = Largest_box2D(PD, GT)
    C_squared = (maxR-minR)**2+(maxC-minC)**2
    d = Center_Diff2D(PD, GT)
    
    DIOU = 1.0-ious+d/C_squared
    
    return DIOU