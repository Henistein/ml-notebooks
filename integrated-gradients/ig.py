import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import requests
from io import BytesIO
from torchvision import transforms

def interpolate_images(baseline, image, alphas):
  # Interpolate the images with alphas
  return [Image.blend(baseline, image, alpha) for alpha in alphas]

def compute_gradients(images, target_class_idx):
  if images.requires_grad is False: images.requires_grad = True
  logits = model(images)
  probs = torch.nn.functional.softmax(logits, dim=-1)[:, target_class_idx]
  grads = torch.autograd.grad(outputs=probs, inputs=images, grad_outputs=torch.ones_like(probs), retain_graph=True, create_graph=True)[0]
  return grads

def integral_approximation(gradients):
  # riemann_trapezoidal
  grads = (gradients[:-1] + gradients[1:]) / torch.tensor(2.0)
  integrated_gradients = torch.mean(grads, dim=0)
  return integrated_gradients

def integrated_gradients(baseline, image, target_class_idx, m_steps=50, batch_size=10):

  normalize = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
  ])

  # 1. Generate alphas
  alphas = torch.linspace(0.0, 1.0, m_steps+1)

  # Collect gradients
  gradient_batches = []

  # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
  for alpha in range(0, len(alphas), batch_size):
    from_ = alpha
    to = min(from_ + batch_size, len(alphas))
    alpha_batch = alphas[from_:to]

    # 2. Generate interpolated inputs between baseline and input
    interpolated_imgs = interpolate_images(Image.fromarray(baseline), image, alpha_batch)
    # convert pil images to torch.tensors and stack them
    interpolated_imgs = torch.stack([normalize(pil_img) for pil_img in interpolated_imgs]).cuda()
    print(interpolated_imgs.shape)

    # 3. Compute gradients between model outputs and interpolated inputs
    gradient_batch = compute_gradients(images=interpolated_imgs,
                                      target_class_idx=target_class_idx)
    gradient_batches.append(gradient_batch.cpu())

  # Stack path gradients together row-wise into single tensor.
  total_gradients = torch.cat(gradient_batches, dim=0).cuda()

  # Integral approximation through averaging gradients.
  avg_gradients = integral_approximation(gradients=total_gradients)

  # permute to (H, W, C) and move to cpu
  avg_gradients = avg_gradients.permute(1, 2, 0).detach().cpu().numpy()

  # Scale integrated gradients with respect to input.
  ig = (np.array(image) - baseline) * avg_gradients

  return ig

if __name__ == '__main__':
  # load model
  model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).cuda()
  model.eval()
  
  # load image
  url = 'http://storage.googleapis.com/download.tensorflow.org/example_images/Giant_Panda_2.jpeg'
  #url = 'https://hips.hearstapps.com/hmg-prod/images/2016-bmw-m2-202-1585760824.jpg'
  response = requests.get(url)
  raw_img = Image.open(BytesIO(response.content))

  # resize raw_image to (224, 224) and convert it to tensor
  resize = transforms.Compose([
      transforms.Resize(224),
      transforms.CenterCrop(224),
  ])
  image = resize(raw_img)

  # vars
  m_steps = 20
  batch_size = 5
  target_class_idx = 388 # giant panda
  #target_class_idx = 817 # sports car
  baseline = np.zeros((224, 224, 3), dtype=np.uint8)

  # compute integrated gradients
  ig = integrated_gradients(baseline=baseline, image=image, m_steps=m_steps, batch_size=batch_size, target_class_idx=target_class_idx)
  attribution_mask = np.sum(np.abs(ig), axis=-1)

  # show results
  plt.imshow(np.uint8(attribution_mask))
  plt.show()