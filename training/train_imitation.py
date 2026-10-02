"""Phase-2 scaffold. First generate real teacher state/action pairs; do not train on dummy labels."""
import torch
from training.dataset import PortraitFolder
def main():
 d=PortraitFolder("data/portraits",128)
 print("Device:","cuda" if torch.cuda.is_available() else "cpu","Portraits:",len(d))
 if not len(d):print("Put training portraits in data/portraits.")
 else:print("Dataset is visible. Next step: cached best-of-N teacher rollouts -> behaviour cloning.")
if __name__=="__main__":main()
