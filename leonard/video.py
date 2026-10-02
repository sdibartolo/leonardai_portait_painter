import cv2
class VideoWriter:
    def __init__(self,path,size,fps=30):self.size=size;self.w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),fps,size)
    def add(self,rgb,label=""):
        f=cv2.resize(rgb,self.size)
        if label:cv2.putText(f,label,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.75,(245,245,245),2,cv2.LINE_AA)
        self.w.write(cv2.cvtColor(f,cv2.COLOR_RGB2BGR))
    def close(self):self.w.release()
