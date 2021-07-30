import torch
import torch.nn.functional as F

def IOU_2D(PD, GT):
    """
    PD -> prediction; GT -> ground truth
    Both are of shape (batch, 4),  the four
    entries are the box's center row (y), center column (x),
    height (h), and width (w)
    """
    
    y_p,x_p,h_p,w_p = PD[:,0], PD[:,1], PD[:,2], PD[:,3]
    y_g,x_g,h_g,w_g = GT[:,0], GT[:,1], GT[:,2], GT[:,3]

    minr_p, minc_p, maxr_p, maxc_p = y_p-0.5*h_p,x_p-0.5*w_p,\
    y_p+0.5*h_p,x_p+0.5*w_p
    minr_g, minc_g, maxr_g, maxc_g = y_g-0.5*h_g,x_g-0.5*w_g,\
    y_g+0.5*h_g,x_g+0.5*w_g

    overlap_minr = torch.max(minr_p, minr_g)
    overlap_minc = torch.max(minc_p, minc_g)
    overlap_maxr = torch.min(maxr_p, maxr_g)
    overlap_maxc = torch.min(maxc_p, maxc_g)

    overlap_r = F.relu(overlap_maxr-overlap_minr)
    overlap_w = F.relu(overlap_maxc-overlap_minc)
    overlap_area = overlap_r*overlap_w

    area_p, area_g = h_p*w_p, h_g*w_g
    union_area = area_p+area_g-overlap_area
    
    return overlap_area/union_area


def Largest_box2D(PD, GT):
    
    
    y_p,x_p,h_p,w_p = PD[:,0], PD[:,1], PD[:,2], PD[:,3]
    y_g,x_g,h_g,w_g = GT[:,0], GT[:,1], GT[:,2], GT[:,3]

    minr_p, minc_p, maxr_p, maxc_p = y_p-0.5*h_p,x_p-0.5*w_p,\
    y_p+0.5*h_p,x_p+0.5*w_p
    minr_g, minc_g, maxr_g, maxc_g = y_g-0.5*h_g,x_g-0.5*w_g,\
    y_g+0.5*h_g,x_g+0.5*w_g
    
    minr = torch.min(minr_p, minr_g)
    minc = torch.min(minc_p, minc_g)
    maxr = torch.max(maxr_p, maxr_g)
    maxc = torch.max(maxc_p, maxc_g)
    
    return minr,minc,maxr,maxc


def Center_Diff2D(PD, GT):
    
    
    y_p,x_p = PD[:,0], PD[:,1]
    y_g,x_g = GT[:,0], GT[:,1]
    
    return torch.sqrt((y_p-y_g)**2+(x_p-x_g)**2)