import sys
import torch
from tqdm import tqdm

# the function to test a neural network model using a test data loader
def testNN(model, testLoader):
    """
    Args:
        model: a Pytorch model
        testLoader: a PyTorch dataLoader object, including (input, output) pairs for model
    """
    device = next(model.parameters()).device
    # set up testing mode
    model.eval()
    # check if total prediction is correct
    correct = total = 0
    # check if each single prediction is correct
    singleCorrect = singleTotal = 0
    with torch.no_grad():
        for data, target in testLoader:
            output = model(data.to(device))
            if isinstance(output, tuple):
                output = output[0]
            if target.shape == output.shape[:-1]:
                pred = output.argmax(dim=-1) # get the index of the max value
            elif target.shape == output.shape:
                pred = (output >= 0).int()
            else:
                print(f'Error: none considered case for output with shape {output.shape} v.s. label with shape {target.shape}')
                sys.exit()
            target = target.to(device).view_as(pred)
            correctionMatrix = torch.logical_or(target.int() == pred.int(), target < 0).view(target.shape[0], -1)
            correct += correctionMatrix.all(1).sum().item()
            total += target.shape[0]
            singleCorrect += correctionMatrix[target >= 0].sum().item()
            singleTotal += (target >= 0).sum().item()
    return correct, total, singleCorrect, singleTotal

# Test the accuracy of RRN
def test_RRN(model, dataloader, nn, limit=None):
    model.eval()
    test_loss = []
    test_res = []
    device = next(model.parameters()).device
    for idx, g in enumerate(dataloader):
        if limit and idx == limit:
            break
        g = g.to(device)
        target = g.ndata['a']
        target = target.view([-1, 81])
        with torch.no_grad():
            if nn == 'rrn':
                preds, loss, _ = model(g, is_training=False)
            else:
                inp = g.ndata['q'].view(-1, 81).to(device) # (batch_size, 81)
                loss, logits = model(inp, target)
                preds = torch.argmax(logits, -1)
            preds = preds.view([-1, 81])
            for i in range(preds.size(0)):
                test_res.append(int(torch.equal(preds[i, :], target[i, :])))
            test_loss.append(loss.cpu().detach().data)
    avg_acc = sum(test_res) / len(test_res) * 100
    avg_loss = sum(test_loss) / len(test_loss)
    return avg_acc, avg_loss

def inference_trick_RRN(model, g):
    """
    Args:
        model: a Pytorch model
        X: a tensor of shape (batchSize, 81), denoting the input to the NN
    """
    model.eval()
    pred = g.ndata['q'].view(-1, 81) # (batch_size, 81)

    # tt = torch.cuda.get_device_properties(0).total_memory
    # r_memory = torch.cuda.memory_reserved(0)
    # a_memory = torch.cuda.memory_allocated(0)
    # free_memory = r_memory - a_memory
    
    while 0 in pred:
        g_clone = g.clone()
        _, _, logits = model(g_clone, is_training=False)
        logits = logits[:,1:].view(-1, 81, 9) # (batch_size, 81, 9)
        probs = torch.nn.functional.softmax(logits, dim=-1) # (batch_size, 81, 9)
        values, indices = probs.max(dim=-1) # (batch_size, 81), (batch_size, 81)
        values[pred != 0] = 0
        cell_indices = values.argmax(dim=-1) # (batch_size)
        for batch_idx, cell_idx in enumerate(cell_indices.tolist()):            
            if pred[batch_idx,cell_idx] == 0:
                # pred contains number 0-9, where 1-9 are labels
                pred[batch_idx,cell_idx] = indices[batch_idx,cell_idx] + 1
        g.ndata['q'] = pred.view(-1)
    return pred

# def testNN_trick(model, test_dataloader, device):
#     # check if total prediction is correct
#     correct = total = 0
#     # check if each single prediction is correct
#     singleCorrect = singleTotal = 0
#     # start evaluation
#     pbar = tqdm(test_dataloader)
#     for g in pbar:
#         g = g.to(device)
#         target = g.ndata['a']
#         # target = target.view([-1, 81])
#         pred = inference_trick_RRN(model, g)
#         breakpoint()
#         target = target.to(device).view_as(pred)
#         correctionMatrix = torch.logical_or(target.int() == pred.int(), target < 0).view(target.shape[0], -1)
#         correct += correctionMatrix.all(1).sum().item()
#         total += target.shape[0]
#         singleCorrect += correctionMatrix[target >= 0].sum().item()
#         singleTotal += (target >= 0).sum().item()
#         # report progress
#         pbar.set_description(f'inference trick: board acc {correct/total:0.4f}')
#     return correct, total, singleCorrect, singleTotal

def testNN_trick(model, dataloader):
    model.eval()
    test_res = []
    device = next(model.parameters()).device
    for g in dataloader:
        g = g.to(device)
        target = g.ndata['a']
        target = target.view([-1, 81])
        with torch.no_grad():
            # preds, _, _ = model(g, is_training=False)
            preds = inference_trick_RRN(model, g)
            preds = preds.view([-1, 81])
            for i in range(preds.size(0)):
                test_res.append(int(torch.equal(preds[i, :], target[i, :])))
    avg_acc = sum(test_res) / len(test_res) * 100
    return avg_acc