import torch
import numpy as np
import utils
import model
import argparse
import json

parser.add_argument('image_path',
                    help='Image directory path')
parser.add_argument('checkpoint',
                    help='Checkpoint of the model')
parser.add_argument('--top_k', action='store',
                    dest='top_k',
                    help='Top number of most likely class')
parser.add_argument('--category_names', action='store',
                    dest='category_names',
                    default=None
                    help='Maping of categories')
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Set training to gpu')

results = parser.parse_args()

model = model.load_checkpoint("model_checkpoint.pth")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)

image = utils.process_image(results.image_path).to(device)
np_image = image.unsqueeze_(0)

model.eval()
with torch.no_grad():
    log_ps = model.forward(np_image)
    
ps = torch.exp(log_ps)
    
top_probs, top_idx_probs = results.topk(topk, dim=1)
top_probs, top_idx_probs = np.array(top_probs.to('cpu')[0]), np.array(top_idx_probs.to('cpu')[0])
    
idx_to_class = defaultdict(list)
{idx_to_class[v].append(k) for k, v in checkpoint['class_to_idx'].items()}
    
top_classes = []
for idx in top_idx_probs:
    top_classes.append(idx_to_class[idx])
        
if results.category_names != None:
    with open(results.category_names, 'r') as f:
        cat_to_name = json.load(f)
        top_class_names = [cat_to_name[top_class] for top_class in list(top_classes)]
        print(f'Top {results.topk} probabilities: {list(top_k)}')
        print(f'Top {results.topk} classes: {top_class_names}')
else:
    print(f'Top {results.topk} probabilities: {list(top_k)}')
    print(f'Top {results.topk} classes: {list(top_classes)}')