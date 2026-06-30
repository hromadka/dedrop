import argparse
import cv2
import numpy as np
import torch
from models import DedropNet

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DedropNet().to(device)
checkpoint = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

image = cv2.imread(args.input)
if image is None:
    raise FileNotFoundError(args.input)

image = cv2.resize(image, (704, 448))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = torch.from_numpy(np.float32(image) / 255).permute(2, 0, 1)
image = image.unsqueeze(0).to(device)

# Split 448×704 into 7×11 non-overlapping 64×64 patches.
patches = (
    image.unfold(2, 64, 64)
         .unfold(3, 64, 64)
         .permute(0, 2, 3, 1, 4, 5)
         .reshape(-1, 3, 64, 64)
)

with torch.inference_mode():
    output, _, _ = model(patches)

# Reassemble the 77 output patches.
output = (
    output.reshape(7, 11, 3, 64, 64)
          .permute(2, 0, 3, 1, 4)
          .reshape(3, 448, 704)
          .clamp(0, 1)
)

output = output.permute(1, 2, 0).cpu().numpy()
output = cv2.cvtColor(np.uint8(output * 255), cv2.COLOR_RGB2BGR)
cv2.imwrite(args.output, output)