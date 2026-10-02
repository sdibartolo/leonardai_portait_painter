import tkinter as tk, threading, queue, time, cv2, numpy as np
from tkinter import ttk,filedialog,messagebox
from pathlib import Path
from PIL import Image,ImageTk
from leonard.config import PaintConfig
from leonard.painter import LeonardPainter
from leonard.video import VideoWriter
class App(tk.Tk):
 def __init__(self):
  super().__init__();self.title("Leonard Portrait Lab");self.geometry("1160x760");self.path=None;self.q=queue.Queue();self.stop_event=threading.Event();self.worker=None;self.pic=None
  self.strokes=tk.IntVar(value=2600);self.cands=tk.IntVar(value=28);self.size=tk.IntVar(value=384);self.cycles=tk.IntVar(value=3);self.fcands=tk.IntVar(value=52);self.status=tk.StringVar(value="Upload a portrait.")
  l=ttk.Frame(self,padding=12);l.pack(side="left",fill="y");m=ttk.Frame(self,padding=12);m.pack(fill="both",expand=True)
  ttk.Label(l,text="LEONARD PORTRAIT LAB",font=("Segoe UI",15,"bold")).pack(anchor="w");ttk.Label(l,text="Phase 1.6A — Dense Facial Vision + Brush Upgrade").pack(anchor="w",pady=(0,12))
  ttk.Button(l,text="Upload portrait",command=self.upload).pack(fill="x")
  for name,var,lo,hi,inc in [("Painting strokes",self.strokes,400,10000,200),("Candidates / stroke",self.cands,4,96,2),("Working resolution",self.size,192,768,32),("Feature cycles",self.cycles,1,8,1),("Feature candidates",self.fcands,12,128,4)]:
   r=ttk.Frame(l);r.pack(fill="x",pady=4);ttk.Label(r,text=name).pack(side="left");ttk.Spinbox(r,textvariable=var,from_=lo,to=hi,increment=inc,width=8).pack(side="right")
  ttk.Label(l,text="Fixed palette: red / yellow / blue / black / white\n3 brushes: round / filbert / flat\nConstruction sketch → block-in → form → features → detail",wraplength=270).pack(anchor="w",pady=12)
  ttk.Button(l,text="GO — Paint portrait",command=self.go).pack(fill="x");ttk.Button(l,text="Stop",command=lambda:self.stop_event.set()).pack(fill="x",pady=4);ttk.Label(l,textvariable=self.status,wraplength=270).pack(anchor="w",pady=10)
  self.view=ttk.Label(m,anchor="center");self.view.pack(fill="both",expand=True);self.after(60,self.poll)
 def upload(self):
  p=filedialog.askopenfilename(filetypes=[("Images","*.png *.jpg *.jpeg *.webp *.bmp")])
  if p:self.path=Path(p);self.show(cv2.cvtColor(cv2.imread(p),cv2.COLOR_BGR2RGB));self.status.set(self.path.name)
 def show(self,a):
  h,w=a.shape[:2];s=min(820/w,680/h,1);im=Image.fromarray(a).resize((max(1,int(w*s)),max(1,int(h*s))));self.pic=ImageTk.PhotoImage(im);self.view.configure(image=self.pic)
 def go(self):
  if not self.path:messagebox.showinfo("Leonard","Upload a portrait first.");return
  if self.worker and self.worker.is_alive():return
  self.stop_event.clear();self.worker=threading.Thread(target=self.work,daemon=True);self.worker.start()
 def work(self):
  try:
   rgb=cv2.cvtColor(cv2.imread(str(self.path)),cv2.COLOR_BGR2RGB);cfg=PaintConfig(work_size=self.size.get(),paint_strokes=self.strokes.get(),candidates=self.cands.get(),feature_cycles=self.cycles.get(),feature_candidates=self.fcands.get())
   out=Path("output")/time.strftime("%Y%m%d_%H%M%S");out.mkdir(parents=True,exist_ok=True);vw=[None]
   def prog(p,s):
    if p.accepted%10==0:self.q.put(("status",f"{s.title()} — mark {p.accepted:,} — loss {p.loss():.5f}"))
   painter=LeonardPainter(rgb,cfg,prog,self.stop_event)
   def frame(im,stage):
    if vw[0] is None:vw[0]=VideoWriter(out/"painting.mp4",(im.shape[1],im.shape[0]),cfg.fps)
    vw[0].add(im,stage);self.q.put(("image",im.copy()))
   final=painter.run(frame)
   if vw[0]:vw[0].close()
   cv2.imwrite(str(out/"final.png"),cv2.cvtColor(final,cv2.COLOR_RGB2BGR));ov=painter.analyzer.overlay(painter.target,painter.info);cv2.imwrite(str(out/"portrait_analysis.png"),cv2.cvtColor(ov,cv2.COLOR_RGB2BGR));np.savetxt(out/"loss.csv",painter.loss_history,delimiter=",");self.q.put(("done",str(out.resolve())))
  except Exception as e:self.q.put(("error",repr(e)))
 def poll(self):
  try:
   while True:
    a=self.q.get_nowait()
    if a[0]=="image":self.show(a[1])
    elif a[0]=="status":self.status.set(a[1])
    elif a[0]=="done":self.status.set("Finished: "+a[1])
    else:messagebox.showerror("Leonard",a[1])
  except queue.Empty:pass
  self.after(60,self.poll)
if __name__=="__main__":App().mainloop()
