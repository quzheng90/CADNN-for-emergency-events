"""Pre-train encoder and classifier for source dataset."""
import torch
import torch.nn as nn
import torch.optim as optim
import param_da
from utils import save_model
from sklearn import metrics
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('always')


def train_src(args, encoder, classifier, data_loader, data_loader_eval):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################
    # instantiate EarlyStop
    earlystop = EarlyStop(args.patience)

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=param_da.c_learning_rate,
        betas=(param_da.beta1, param_da.beta2))
    criterion = nn.CrossEntropyLoss()

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    ####################
    # 2. train network #
    ####################

    for epoch in range(args.num_epochs_pre):
        for step, (reviews, labels) in enumerate(data_loader):

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = classifier(encoder(reviews))
            loss = criterion(preds, labels)


            # with torch.set_grad_enabled(True):
            #     # Forward
            #     logits = self.model(inputs, lens, mask, labels)
            #     _loss = self.criterion(logits, labels)
            #     loss += _loss.item()
            #     y_pred = logits.argmax(dim=1).cpu().numpy()

            #     if y_pred_all is None:
            #         y_pred_all = y_pred
            #     else:
            #         y_pred_all = np.concatenate((y_pred_all, y_pred))

            #     # Backward
            #     _loss.backward()
            #     if self.clip:
            #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            #     self.optimizer.step()
            #     if self.scheduler is not None:
            #         self.scheduler.step()


            # optimize source classifier
            loss.backward()
            optimizer.step()





            # print step info
            if (step + 1) % args.log_step_pre == 0:
                print("Epoch [%.3d/%.3d] Step [%.2d/%.2d]: loss=%.4f, "
                      % (epoch + 1,
                         args.num_epochs_pre,
                         step + 1,
                         len(data_loader),
                         loss.item()))

                # preds=preds.detach().numpy()
                preds=preds.argmax(dim=1).detach().numpy()
                # loss=loss.detach().numpy()
                cr=classification_report(labels, preds, zero_division=1)
                print(cr)
                # print('metrics: AUC=%.3f, PRE=%.3f, REC=%.3f, F1=%.3f' %(roc_auc_score(labels, preds, average='macro'), 
                                                                        # precision_score(labels, preds, average='macro'),
                                                                        # recall_score(labels, preds, average='macro'),
                                                                        # f1_score(labels, preds, average='macro') 
                                                                        # ))
        # eval model on test set
        if (epoch + 1) % args.eval_step_pre == 0:
            # print('Epoch [{}/{}]'.format(epoch + 1, param.num_epochs_pre))
            eval_src(encoder, classifier, data_loader)
            earlystop.update(eval_src(encoder, classifier, data_loader_eval))
            print()

        # save model parameters
        if (epoch + 1) % args.save_step_pre == 0:
            save_model(encoder, "CADNN-source-encoder-{}.pt".format(epoch + 1))
            save_model(classifier, "CADNN-source-classifier-{}.pt".format(epoch + 1))

        if earlystop.stop:
            break

    # # save final model
    save_model(encoder, "CADNN-source-encoder-final.pt")
    save_model(classifier, "CADNN-source-classifier-final.pt")

    return encoder, classifier


def eval_src(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (reviews, labels) in data_loader:

        preds = classifier(encoder(reviews))
        loss += criterion(preds, labels).item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum().item()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = %.4f, Avg Accuracy = %.4f" % (loss, acc))

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    return loss


class EarlyStop:
    def __init__(self, patience):
        self.count = 0
        self.maxAcc = 0
        self.patience = patience
        self.stop = False

    def update(self, acc):
        if acc < self.maxAcc:
            self.count += 1
        else:
            self.count = 0
            self.maxAcc = acc

        if self.count > self.patience:
            self.stop = True
