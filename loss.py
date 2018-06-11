import torch

# three pixel error
def px3_loss(softmax_scores, gt, weights):    
    loss = 0
    for i in range(softmax_scores.size(0)):
        # Select the predicted scores for the GT disparities and apply 3 pixel error weighting
        gt_disparity = gt[i]
        pred_scores = softmax_scores[i, gt_disparity[0]-2:gt_disparity[0]+2+1]
        sl = torch.mul(pred_scores, weights).sum()
        loss -= sl
        
    return loss