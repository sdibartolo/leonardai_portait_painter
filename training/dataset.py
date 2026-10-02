from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
class PortraitFolder(Dataset):
 def __init__(self,root,size=128):
  self.files=sum((list(Path(root).rglob(e)) for e in ("*.jpg","*.jpeg","*.png","*.webp")),[]);self.tf=T.Compose([T.Resize((size,size)),T.ToTensor()])
 def __len__(self):return len(self.files)
 def __getitem__(self,i):return self.tf(Image.open(self.files[i]).convert("RGB"))
