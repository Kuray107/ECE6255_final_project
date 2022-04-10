import os
import time
import argparse
import torch
from torch.utils.data import DataLoader


from utils import get_acc_and_confusion_matrix
from x_vector import X_vector
from GSC_hparams import create_hparams
from data_utils import MelDataset_SpeechCommand, MelLabelCollate
from train_utils import warm_start_model, load_checkpoint, save_checkpoint

GSC_labels = ['nine', 'yes',
    'no', 'up', 'down', 'left',
    'right', 'on', 'off', 'stop',
    'go', 'zero', 'one', 'two',
    'three', 'four', 'five', 'six',
    'seven', 'eight', 'backward', 'bed',
    'bird', 'cat', 'dog', 'follow',
    'forward', 'happy', 'house', 'learn',
    'marvin', 'sheila', 'tree', 'visual','wow'
]


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = MelDataset_SpeechCommand(hparams.training_files, hparams)
    valset = MelDataset_SpeechCommand(hparams.validation_files, hparams)

    #labels = sorted(list(set(datapoint[1] for datapoint in trainset)))
    print(GSC_labels)
    collate_fn = MelLabelCollate(hparams.n_frames_per_utt, GSC_labels)

    train_loader = DataLoader(
                        trainset, 
                        num_workers=4,  
                        shuffle=True,
                        sampler=None,
                        batch_size=hparams.batch_size, 
                        pin_memory=False,
                        drop_last=True, 
                        collate_fn=collate_fn
    )

    return train_loader, valset, collate_fn


def validate(model, criterion, valset, iteration, batch_size,
             collate_fn, device):
    """Handles all the validation scoring and printing"""
    labels = []
    predictions = []
    model.eval()
    with torch.no_grad():
        val_loader = DataLoader(valset, sampler=None, num_workers=4,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch, device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
            labels.append(y)
            predictions.append(y_pred)
        val_loss = val_loss / (i + 1)

        ### calculate average accuracy & confusion matrixi
        labels = torch.cat(labels, dim=0).cpu().numpy()
        predictions = torch.cat(predictions, dim=0).argmax(dim=1).cpu().numpy()
        acc, confusion_matrix = get_acc_and_confusion_matrix(labels, predictions)

    model.train()
    print("Validation loss {}: {:6f}  ".format(iteration, val_loss))
    print("Validation accuracy {}: {:4f}  ".format(iteration, acc))


def train(output_directory, checkpoint_path, warm_start, hparams, device):

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = X_vector(hparams).to(device)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    criterion = torch.nn.NLLLoss()

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch, device)
            y_pred = model(x)

            loss = criterion(y_pred, y)
            reduced_loss = loss.item()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()

            duration = time.perf_counter() - start
            print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, reduced_loss, grad_norm, duration))

            if (iteration % hparams.iters_per_checkpoint == 0):
                validate(model, criterion, valset, iteration,
                         hparams.batch_size, collate_fn, device)
                checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)

            iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')

    args = parser.parse_args()
    hparams = create_hparams()

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)
    print("Use {} as training device".format(device))

    train(args.output_directory, args.checkpoint_path, args.warm_start, hparams, device)
