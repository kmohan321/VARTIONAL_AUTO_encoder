import torch
import torch.nn as nn
import torch.optim as optim
from model import VAE
from data import data_loader
from torchvision.utils import save_image
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()
from tqdm import tqdm
from torch.cuda.amp import GradScaler
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#hyperparameters

epochs = 100
lr = 1e-4
beta = 1e-4
load_model = False
path = 'images\image_{}.png'

vae = VAE(64,device).to(device)
if load_model==True:
  vae.load_state_dict(torch.load('model2.pth'))

optimizer = optim.Adam(vae.parameters(),lr=lr)
loss_fn = nn.MSELoss(reduction='none')

def train():
  def save_img(images,epoch):
    vae.eval()
    images= next(iter(data_loader))
    images = images.to(device)
    with torch.no_grad():
      images,_,_ = vae.forward_decoder(images)
    save_image(images,path.format(epoch))


  scaler = GradScaler()
  global_steps = 0
  for epoch in range(epochs):
    vae.train()
    for i,images in tqdm(enumerate(data_loader)):
      global_steps += 1
      images = images.to(device)
      with torch.autocast(device_type='cuda', dtype=torch.float16):
        constructed_img,mu,log_var = vae.forward_decoder(images)

        kl_div = -0.5*torch.sum(1 + log_var - mu**2 - torch.exp(log_var),dim=1)
        kl_div = kl_div.mean()
        kl_div = beta*kl_div
          # writer.add_scalar('loss/kl',kl_div.item(),global_steps)
          
        cons_loss = loss_fn(constructed_img,images)
        cons_loss = cons_loss.reshape(images.shape[0],-1).sum(dim=1)
        cons_loss = cons_loss.mean()
          # writer.add_scalar('loss/cons',cons_loss.item(),global_steps)

        overall_loss = cons_loss + kl_div
        # writer.add_scalar('loss',overall_loss.item(),global_steps)

      optimizer.zero_grad()
      sacler.scale(overall_loss).backward()
      scaler.step(optimizer)
      scaler.update()

    print(f'epoch {epoch} | kl_loss {kl_div:.4f} | cons_loss {cons_loss:.4f}')
    save_img(images,epoch)
    torch.save(vae.state_dict(),'model.pth')

if __name__ == '__main__':
    train()