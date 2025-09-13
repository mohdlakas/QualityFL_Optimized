import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
     
class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, embedding_gen=None):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = None
        self.criterion = None
        self.embedding_gen = embedding_gen

    def _ensure_device_and_criterion(self, model):
        if self.device is None:
            self.device = next(model.parameters()).device
        if self.criterion is None:
            self.criterion = nn.NLLLoss().to(self.device)

    def update_weights(self, model, global_round):
        """Enhanced training with embedding generation"""
        self._ensure_device_and_criterion(model)
        
        # Store initial state
        initial_state = {name: param.clone().detach() for name, param in model.named_parameters()}
        
        model.train()
        epoch_loss = []

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        final_loss = sum(epoch_loss) / len(epoch_loss)
        
        # Calculate parameter update
        param_update = {name: model.state_dict()[name] - initial_state[name]
                       for name in initial_state}
        
        # Generate embedding if generator available
        update_embedding = None
        if self.embedding_gen is not None:
            try:
                update_embedding = self.embedding_gen.generate_embedding(param_update)
            except Exception as e:
                print(f"Warning: Embedding generation failed: {e}")

        return {
            'model_state': model.state_dict(),
            'embedding': update_embedding,
            'data_size': len(self.trainloader.dataset),
            'avg_loss': final_loss
        }

    def train_val_test(self, dataset, idxs):
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]
        
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                batch_size=max(1, min(len(idxs_val), self.args.local_bs)), 
                                shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=max(1, min(len(idxs_test), self.args.local_bs)), 
                                shuffle=False)
        return trainloader, validloader, testloader
    
    def inference(self, model, eval_type='train'):
        """Returns the inference accuracy and loss"""
        self._ensure_device_and_criterion(model)
        
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        # Choose correct dataloader
        if eval_type == 'train':
            dataloader = self.trainloader
        elif eval_type == 'val':
            dataloader = self.validloader
        else:
            dataloader = self.testloader

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

        accuracy = correct/total if total > 0 else 0.0
        return accuracy, loss

def test_inference(args, model, test_dataset):
    """Returns the test accuracy and loss"""
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = next(model.parameters()).device
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

    accuracy = correct/total
    return accuracy, loss