
import torch
import argparse
import numpy as np

from PIL import Image

from RegDanbooru2019_8G import RegDanbooru2019

parser = argparse.ArgumentParser(description='Test RegDeepDanbooru')
parser.add_argument('--model', default='', type=str, help='trained model')
parser.add_argument('--image', default='', type=str, help='image to test')
parser.add_argument('--size', default=768, type=int, help='canvas size')
parser.add_argument('--threshold', default=0.5, type=float, help='threshold')
args = parser.parse_args()

DANBOORU_LABEL_MAP = {}

def load_danbooru_label_map() :
    print(' -- Loading danbooru2019 labels')
    global DANBOORU_LABEL_MAP
    with open('danbooru_labels.txt', 'r') as fp :
        for l in fp :
            l = l.strip()
            if l :
                idx, tag = l.split(' ')
                DANBOORU_LABEL_MAP[int(idx)] = tag

def test(model, image_resized) :
    print(' -- Running model on GPU')
    image_resized_torch = torch.from_numpy(image_resized).float() / 127.5 - 1.0
    if len(image_resized_torch.shape) == 3 :
        image_resized_torch = image_resized_torch.unsqueeze(0).permute(0, 3, 1, 2)
    elif len(image_resized_torch.shape) == 4 :
        image_resized_torch = image_resized_torch.permute(0, 3, 1, 2)
    image_resized_torch = image_resized_torch.cuda()
    with torch.no_grad() :
        danbooru_logits = model(image_resized_torch)
        danbooru = danbooru_logits.sigmoid().cpu()
    return danbooru

def load_and_resize_image(img_path, canvas_size = 512) :
    img = Image.open(img_path).convert('RGB')
    old_size = img.size
    w, h = old_size
    w, h = float(w), float(h)
    ratio = float(canvas_size) / max(old_size)
    new_size = tuple([int(round(x * ratio)) for x in old_size])
    print(f'Test image size: {new_size}')
    return np.array(img.resize(new_size, Image.ANTIALIAS))

def translate_danbooru_labels(probs, threshold = 0.8) :
    global DANBOORU_LABEL_MAP
    choosen_indices = (probs > threshold).nonzero()
    result = []
    for i in range(probs.size(0)) :
        prob_single = probs[0].numpy()
        indices_single = choosen_indices[choosen_indices[:, 0] == i][:, 1].numpy()
        tag_prob_map = {DANBOORU_LABEL_MAP[idx]: prob_single[idx] for idx in indices_single}
        result.append(tag_prob_map)
    return result

def main() :
    model = RegDanbooru2019().cuda()
    model.load_state_dict(torch.load(args.model)['model'])
    model.eval()
    torch.save(model, 'RegNetY-8G.pth',)

    test_img = load_and_resize_image(args.image, args.size)

    danbooru = test(model, test_img)

    tags = translate_danbooru_labels(danbooru, args.threshold)
    print(tags)

if __name__ == "__main__":
    load_danbooru_label_map()
    main()
