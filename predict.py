import torch
import datetime
import torch.utils.data as utilsData
import argparse
from model import *
from utils import *
from constants import *

parser = argparse.ArgumentParser()
parser.add_argument('--predict_option', default=0, type=int, choices=[0, 1],
                    help='0: predict with best acc model -- 1: predict with convergence model')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size - default: 256')
parser.add_argument('--depth', default=18, choices = [18, 50, 152], type=int, help='depth of model')
args = parser.parse_args()

if not os.path.isdir('./results'):
    os.mkdir('./results')

RESULT_FILE = os.path.join(PROJECT_DIR, 'results', 'test_results')
print(torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print('Available device is:{}'.format(device))
print(torch.cuda.current_device)

###########
print('DataLoader ....')
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# input_size = 224
input_size = 224

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}


def predict(model, test_loader):
    torch.set_grad_enabled(False)
    model.eval()
    test_correct = 0
    total = 0

    f = open(RESULT_FILE + str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')) + '.csv', 'w+')
    f.write('correct,predicted\n')

    for idx, (images, labels) in enumerate(test_loader):
        images, labels = cvt_to_gpu(images), cvt_to_gpu(labels)
        outputs = model(images)
        _, predicted = outputs.max(1)
        test_correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        outputs = outputs.data.cpu().numpy()
        outputs = np.argsort(outputs, axis=1)[:, -3:][:, ::-1]
        for i, image_id in enumerate(labels):
            tmp = gen_outputline(image_id.data.cpu().numpy(), list(outputs[i]))
            f.write(tmp)

        if idx % 2000 and idx > 1:
            print("Processing {}/{}".format(idx + 1, len(test_loader)))

    print('Accuracy on test set: {:.6}'.format(test_correct/total))


if __name__ == '__main__':
    model = MyResNet(depth=args.depth, num_classes=4)
    test_set = MyTestDataset(HDF5_TEST_PATH, root_dir='./data', transform=data_transforms['val'])
    test_loader = utilsData.DataLoader(dataset=test_set, batch_size=args.batch_size, sampler=None, shuffle=False,
                                       batch_sampler=None)
    assert os.path.isdir('./checkpoint'), 'Error: model is not availabel!'
    if args.predict_option == 1:
        ckpt = torch.load('./checkpoint/convergence.t7', map_location=lambda storage, loc: storage)
        model = ckpt['model']
        model = unparallelize_model(model)
        model = parallelize_model(model)
        loss = ckpt['loss']
        epoch = ckpt['epoch']
        print('Model used to predict converges at epoch {} and loss {:.6}'.format(epoch, loss))
    else:
        ckpt = torch.load('./checkpoint/best_acc_model.t7', map_location=lambda storage, loc: storage)
        model = ckpt['model']
        model = unparallelize_model(model)
        model = parallelize_model(model)
        acc = ckpt['acc']
        epoch = ckpt['epoch']
        print('Model used to predict has best acc {:.6} on validate set at epoch {}'.format(acc, epoch))
    predict(model, test_loader)
