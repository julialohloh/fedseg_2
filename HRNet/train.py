import os
import numpy as np
from tqdm import tqdm
from torch import optim
import torch
import torch.nn as nn
from src.modelling import _hrnet
# from utils.auxillary import to_Excel
from utils.metrics import get_all_metrics, get_iou_score
import time
import copy

##################
# TRAIN FUNCTION #
##################

def train_hrnet(model, device, trainloader, optimizer, loss_function, epoch):
    #train
    model.train()
    running_loss = 0
    mask_list, iou,iou_new = [], [],[]
    accuracy,precision,recall,f1_score,dsc = [],[],[],[],[]
    for i, (input, mask) in enumerate(tqdm(trainloader)):
        # load data into cuda
        input, mask = input.to(device), mask.to(device)
        # input, mask = input.float()/255,mask.float()/255
        input=torch.unsqueeze(input, dim=0)
        mask = torch.unsqueeze(mask,dim=0)
        mask = torch.where(mask>0,torch.ones_like(mask),mask)
        
        # forward
        predict = model(input)
        loss = loss_function(predict, mask)
        # zero the gradient + backprpagation + step
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        predict = torch.where(predict>0.5,torch.ones_like(predict),predict)
        predict = torch.where(predict<0.5,torch.zeros_like(predict),predict)
        iou.append(get_iou_score(predict, mask).mean())
        results = []
        predict_np = predict.cpu().detach().numpy()
        mask_np = mask.cpu().detach().numpy()
        results = get_all_metrics(predict_np,mask_np)        
        accuracy.append(results[0]) 
        precision.append(results[1]) 
        recall.append(results[2]) 
        f1_score.append(results[3])
        dsc.append(results[4])
        # hm.append(results[5])   
        
        running_loss += loss.item()
        # if torch.cuda.is_available():
        # predict_ts = torch.from_numpy(predict_np)
        # mask_ts = torch.from_numpy(mask_np)
        # log the first image of the batch
        # if ((i + 1) % 10) == 0:
        #     # predict_ts = torch.from_numpy(predict_np)
        #     pred = normtensor(predict_ts[0])
        #     img, pred, mak = tensor2np(input[0]), tensor2np(pred), tensor2np(mask_ts[0])

    
    mean_iou = np.mean(iou)
    total_loss = running_loss/len(trainloader)
    # mean_iou_new = np.mean(iou_new)
    mean_accuracy = np.mean(accuracy)
    mean_precision = np.mean(precision)
    mean_recall = np.mean(recall)
    mean_f1_score = np.mean(f1_score)
    mean_dsc = np.mean(dsc)
    # mean_hm = np.mean(hm)
    
    return total_loss, mean_iou,mean_accuracy,mean_precision,mean_recall,mean_f1_score,mean_dsc,model

def test_hrnet(model, device, testloader, loss_function, best_iou, epoch, SAVE_PATH, RUN_NAME):
    model.eval()
    running_loss = 0
    mask_list, iou,iou_new  = [], [],[]
    accuracy,precision,recall,f1_score,dsc = [],[],[],[],[]
    with torch.no_grad():
        for i, (input, mask) in enumerate(tqdm(testloader)):
            input, mask = input.to(device), mask.to(device)
            # input, mask = input.float()/255,mask.float()/255
            input=torch.unsqueeze(input, dim=0)
            mask = torch.unsqueeze(mask,dim=0)
            mask = torch.where(mask>0,torch.ones_like(mask),mask)
            #forward
            predict = model(input)
            print(predict.shape)
            loss = loss_function(predict, mask)
            
            running_loss += loss.item()
            iou.append(get_iou_score(predict, mask).mean())
            
            results = []
            predict_np = predict.cpu().detach().numpy()
            mask_np = mask.cpu().detach().numpy()
            results = get_all_metrics(predict_np,mask_np)
            accuracy.append(results[0])
            precision.append(results[1])
            recall.append(results[2])
            f1_score.append(results[3])
            dsc.append(results[4])
            # hm.append(results[5])
            # iou_new.append(results[6])
            # if torch.cuda.is_available():
            # predict_ts = torch.from_numpy(predict_np)
            # mask_ts = torch.from_numpy(mask_np)
            
            # # log the first image of the batch
            
            # if ((i + 1) % 1) == 0:
            #     pred = normtensor(predict[0])
            #     img, pred, mak = tensor2np(input[0]), tensor2np(pred), tensor2np(mask[0])

    test_loss = running_loss/len(testloader)
    mean_iou = np.mean(iou)
    if mean_iou>best_iou:
    # export to pt
        try:
            torch.save(model, SAVE_PATH + RUN_NAME+'.pth')
        except:
            print('Can export weights')

    return test_loss, mean_iou


def model_pipeline(modelcfg, lr, epochs, device, train_dataset, valid_dataset,  SAVE_PATH,RUN_NAME,prev_model = None):
    best_model = None
        
    # make the model, data, and optimization 
    model, criterion, optimizer = make_model(modelcfg, lr, prev_model = prev_model)
    metrics = []
    best_iou = -1        
    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_iou,train_accuracy,train_precision,train_recall,train_f1_score,train_dsc,val_model = train_hrnet(model, device, train_dataset, optimizer, criterion, epoch)
        t1 = time.time()
        metrics.append(train_loss)
        # metrics.append(train_new_iou)
        metrics.append(train_accuracy)
        metrics.append(train_precision)
        metrics.append(train_recall)
        metrics.append(train_f1_score)
        metrics.append(train_dsc)
        # metrics.append(train_fm)
        print(f'Epoch: {epoch} | Train loss: {train_loss:.3f} | Train IoU: {train_iou:.3f} | Time: {(t1-t0):.1f}s')
        # print(f'Train New Iou:{train_new_iou:.3f}.')
        print(f'Train Acc: {train_accuracy:.3f} | Train Precision:{train_precision:.3f}| Train Recall :{train_recall:.3f} | Train F1 Score:{train_f1_score:.3f}.')
        print(f' Train DSC :{train_dsc:.3f}.')
        test_loss, test_iou = test_hrnet(val_model, device, valid_dataset, criterion, best_iou, epoch, SAVE_PATH, RUN_NAME)
        print(f'Epoch: {epoch} | Valid loss: {test_loss:.3f} | Valid IoU: {test_iou:.3f} | Time: {(t1-t0):.1f}s')
        torch.save(model.state_dict(), os.path.join(SAVE_PATH,'saved_model.pth'))
        # to_Excel(EXCEL_PATH, SAVE_PATH,metrics, epochs)
        if best_iou < test_iou:
            best_iou = test_iou
            best_model = copy.deepcopy(model)

    return best_model, best_iou


def make_model(modelcfg, lr, prev_model = None):
    # Make the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if prev_model == None:
        # model = _hrnet("hrnet18").to(device)
        model = _hrnet(modelcfg).to(device)
        print(model)
    else:
        model = prev_model

    # print('Number of parameter:', count_params(model))

    # Make the loss and optimizer
    #     criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer