import torch

def multi_scale_grad_loss(prediction, target, mask):
    total = 0
    for scale in range(4):
        step = pow(2, scale)
        total += grad_loss(prediction[:, ::step, ::step], 
            target[:, ::step, ::step],
            mask[:, ::step, ::step])
    return total

def grad_loss(prediction, target, mask):
    M = torch.sum(mask, (1, 2))
    diff = prediction - target
    diff = torch.mul(mask, diff)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)
    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return torch.sum(image_loss) / torch.sum(M)