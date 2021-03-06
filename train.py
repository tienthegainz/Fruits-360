import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath('model'))  # nopep8
from model.simple import FruitNet  # nopep8
from model.resnet import ResFruitNet  # nopep8

parser = argparse.ArgumentParser(
    description='Simple training script ')

parser.add_argument('--epochs', help='Number of epochs',
                    type=int, default=20)


parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')

parser.add_argument('--train', type=str,
                    help='Rootdir of train data')

parser.add_argument('--val', type=str,
                    help='Rootdir of val data')

parser.add_argument('--freeze', type=bool, default=True,
                    help='Freeze Resnet')

parser.add_argument('--save', type=str,
                    help='Save path')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Training on {}'.format(device))


def main():
    # flatten dataset
    tfms = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = ImageFolder(args.train, transform=tfms)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = ImageFolder(args.val, transform=tfms)
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True)

    # param
    # model = FruitNet(len(train_dataset.classes)).to(device)
    model = ResFruitNet(len(train_dataset.classes),
                        freeze=args.freeze).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0015)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=4)
    best_loss = 100
    # Train
    print('Start training')
    t = time.time()
    for epoch in range(args.epochs):
        print('Epoch [{}/{}]:'.format(epoch+1, args.epochs))
        model.train()
        with torch.enable_grad():
            j = 0
            running_loss = 0.0
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(train_dataloader):
                images = images.to(device)
                labels = labels.to(device)
                # print('Image: {} - label: {}'.format(images.shape, labels))
                # Forward pass
                outputs = model(images)
                # print('Output:{}'.format(outputs))
                loss = criterion(outputs, labels)

                # Loss and Acc
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # .item() won't save full architecture
                correct += (predicted == labels).sum().item()
                running_loss += loss.item() * images.size(0)
                j = j+1

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print('\nTrain Loss: {:.4f}, Train Acc: {:.2f}'.format(running_loss /
                                                               (j*args.batch_size),
                                                               correct/total))
        if not args.val:
            # skip validation
            continue

        # Eval mode
        model.eval()
        with torch.no_grad():
            j = 0
            running_loss = 0.0
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(val_dataloader):
                images = images.to(device)
                labels = labels.to(device)
                # print('Val:')
                # print('Image: {} - label: {}'.format(images.shape, labels))
                # Forward pass
                outputs = model(images)
                # print('Output:{}'.format(outputs))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item() * images.size(0)
                j = j+1

        print('\nVal Loss: {:.4f}, Val Acc: {:.2f}'.format(running_loss / (j*args.batch_size),
                                                           correct/total))
        if args.save:
            if running_loss / (j*args.batch_size) < best_loss:
                best_loss = running_loss / (j*args.batch_size)
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict()
                }
                torch.save(
                    state,
                    os.path.join(
                        args.save,
                        "{}_epoch_{}.pth".format(model.name, epoch)
                    )
                )
        if (running_loss / (j*args.batch_size)) == 0.00:
            print('Val loss == 0. Exitting!')
            break
        # Check val loss
        scheduler.step(running_loss)
    print('Training finnish. Took {} mins'.format((time.time()-t)/60))


if __name__ == '__main__':
    main()
