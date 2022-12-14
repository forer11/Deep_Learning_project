import torch
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import argparse
import utils
from tqdm import tqdm
import torch.nn.functional as F
import gc
from PIL import Image
import numpy as np

from prepare_data import get_files_list, get_extracted_objects_dict


def contrastive_loss(out_1, out_2):
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    bs = out_1.size(0)
    temp = 0.25
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temp)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=sim_matrix.device)).bool()
    # [2B, 2B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temp)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


def train_model(model, train_loader, test_loader, train_loader_1, device, args):
    transform = utils.transform_color if args.backbone == 152 else utils.transform_resnet18
    print('getting train objects...')
    train_file_name = 'saved_training_data'
    train_files_list = get_files_list(train_file_name, 31)
    all_objects_train = get_extracted_objects_dict(train_files_list)

    print('getting test objects...')
    test_file_name = 'saved_test_data'
    test_files_list = get_files_list(test_file_name, 37)
    all_objects_test = get_extracted_objects_dict(test_files_list)

    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    auc, feature_space = get_score(model, device, train_loader, test_loader, all_objects_train, all_objects_test,
                                   transform)
    print('Epoch: {}, AUROC is: {}'.format(0, auc))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005)
    center = torch.FloatTensor(feature_space).mean(dim=0)
    if args.angular:
        center = F.normalize(center, dim=-1)
    center = center.to(device)
    for epoch in range(args.epochs):
        gc.collect()
        torch.cuda.empty_cache()
        running_loss = run_epoch(model, train_loader_1, optimizer, center, device, args.angular)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        auc, _ = get_score(model, device, train_loader, test_loader, all_objects_train, all_objects_test, transform)
        print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))


def run_epoch(model, train_loader, optimizer, center, device, is_angular):
    total_loss, total_num = 0.0, 0
    for ((img1, img2), _, img_names) in tqdm(train_loader, desc='Train...'):

        img1, img2 = img1.to(device), img2.to(device)

        optimizer.zero_grad()

        out_1 = model(img1)
        out_2 = model(img2)
        out_1 = out_1 - center
        out_2 = out_2 - center

        loss = contrastive_loss(out_1, out_2)

        if is_angular:
            loss += ((out_1 ** 2).sum(dim=1).mean() + (out_2 ** 2).sum(dim=1).mean())

        loss.backward()

        optimizer.step()

        total_num += img1.size(0)
        total_loss += loss.item() * img1.size(0)

    return total_loss / (total_num)


def get_objects(img_name, objects_dict, transform):
    objects_list = objects_dict[img_name] if img_name in objects_dict else []
    object_samples = []
    for object in objects_list:
        obj_image = Image.fromarray(object)
        if transform:
            obj_image = transform(obj_image)
        object_samples.append(obj_image)
    return object_samples


def get_score(model, device, train_loader, test_loader, all_objects_train, all_objects_test, transform):
    train_feature_space = []
    with torch.no_grad():
        for (imgs, _, img_names) in tqdm(train_loader, desc='Train set feature extracting'):
            all_imgs = imgs.to(device)
            images_objects = []
            for img_name in img_names:
                objects = get_objects(img_name, all_objects_train, transform)
                images_objects += objects
            if images_objects:
                images_objects = torch.stack(images_objects).to(device)
                all_imgs = torch.cat((images_objects, all_imgs))
            features = model(all_imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    test_labels = []
    total_distances = np.array([])
    with torch.no_grad():
        for (imgs, labels, img_names) in tqdm(test_loader, desc='Test set feature extracting'):
            for index, img_name in enumerate(img_names):
                objects = get_objects(img_name, all_objects_test, transform)
                if objects:
                    objects = torch.stack(objects).to(device)
                    features = model(objects)
                    test_feature_space = features.contiguous().cpu().numpy()
                    test_labels.append(labels[index])
                    distances = utils.knn_score(train_feature_space, test_feature_space)
                    total_distances = np.append(total_distances, distances.max())
        # test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        # test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

    auc = roc_auc_score(test_labels, total_distances)

    return auc, train_feature_space


def main(args):
    print('Dataset: {}, Normal Label: {}, LR: {}'.format(args.dataset, args.label, args.lr))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = utils.Model(args.backbone)
    model = model.to(device)

    train_loader, test_loader, train_loader_1 = utils.get_loaders(dataset=args.dataset, label_class=args.label,
                                                                  batch_size=args.batch_size, backbone=args.backbone)
    train_model(model, train_loader, test_loader, train_loader_1, device, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='custom')
    parser.add_argument('--epochs', default=20, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--label', default=5, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-5, help='The initial learning rate.')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--backbone', default=152, type=int, help='ResNet 18/152')
    parser.add_argument('--angular', action='store_true', help='Train with angular center loss')
    args = parser.parse_args()
    main(args)
