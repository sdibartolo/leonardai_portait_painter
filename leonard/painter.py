import cv2, numpy as np, math



from .vision import PortraitAnalyzer



from .brushes import Stroke,render_stroke,mix_palette,render_stroke_local







class LeonardPainter:



    """



    Phase 1.5 Structured Teacher.







    Broad stages still use best-of-N global painting. During the feature stage,



    Leonard switches to explicit left-eye/right-eye/nose/mouth/jaw sessions.



    Candidate strokes are generated from the target feature's local edge field



    and evaluated primarily inside that feature crop.



    """



    def __init__(self,target,cfg,progress=None,stop_event=None):



        self.cfg=cfg; self.progress=progress; self.stop_event=stop_event



        self.rng=np.random.default_rng(cfg.seed)



        h,w=target.shape[:2]; sc=cfg.work_size/max(h,w)



        self.target=cv2.resize(target,(max(1,int(w*sc)),max(1,int(h*sc))),interpolation=cv2.INTER_AREA)



        self.h,self.w=self.target.shape[:2]



        self.canvas=np.full_like(self.target,238)



        self.analyzer=PortraitAnalyzer()



        self.info=self.analyzer.analyze(self.target)



        self.accepted=0; self.loss_history=[]; self.current_feature=None



        gray=cv2.cvtColor(self.target,cv2.COLOR_RGB2GRAY)



        self.gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)



        self.gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)



        self.edge_mag=cv2.magnitude(self.gx,self.gy)



        if self.edge_mag.max()>0:self.edge_mag/=self.edge_mag.max()

        # Fast Teacher: cache immutable target transforms once.

        self.target_gray=gray

        self.target_edges=cv2.Canny(gray,50,130).astype(np.float32)/255.

        self._feature_cache={}

        for _name,(x0,y0,x1,y1) in self.info["regions"].items():

            if x1>x0 and y1>y0:

                self._feature_cache[_name]={

                    "target":self.target[y0:y1,x0:x1].astype(np.float32)/255.,

                    "gray":self.target_gray[y0:y1,x0:x1],

                    "edges":self.target_edges[y0:y1,x0:x1],

                }







    def loss_map_for(self,canvas=None):



        c=self.canvas if canvas is None else canvas



        d=((c.astype(np.float32)-self.target.astype(np.float32))/255.)**2



        return d.mean(2)*(1+1.8*self.info["geometry"]+.65*self.info["edges"]+.35*self.info["face_mask"])







    def loss_map(self): return self.loss_map_for()



    def loss(self): return float(self.loss_map().mean())







    def region_loss(self,canvas,name):

        x0,y0,x1,y1=self.info["regions"][name]

        if x1<=x0 or y1<=y0:return 1e9

        c=canvas[y0:y1,x0:x1].astype(np.float32)/255.

        cache=self._feature_cache[name]

        pix=np.mean((c-cache["target"])**2)

        cg=cv2.cvtColor((c*255).astype(np.uint8),cv2.COLOR_RGB2GRAY)

        ce=cv2.Canny(cg,50,130).astype(np.float32)/255.

        edge=np.mean(np.abs(ce-cache["edges"]))

        value=np.mean(np.abs(cg.astype(np.float32)-cache["gray"].astype(np.float32)))/255.

        return float(.48*pix+.34*edge+.18*value)



    def point(self,stage):



        err=self.loss_map()



        if stage=="block": att=.6+self.info["face_mask"]*.5



        elif stage=="form": att=.35+.65*self.info["face_mask"]+.45*self.info["edges"]



        else: att=.35+1.2*self.info["geometry"]+1.2*self.info["edges"]



        p=(err*att+1e-7).ravel(); p/=p.sum()



        i=self.rng.choice(p.size,p=p); return i%self.w,i//self.w







    def feature_point(self,name):



        x0,y0,x1,y1=self.info["regions"][name]



        err=self.loss_map()[y0:y1,x0:x1]



        edge=self.edge_mag[y0:y1,x0:x1]



        # Feature candidates deliberately favour important target contours.



        p=(err*(.25+2.4*edge)+1e-8).ravel()



        if not np.isfinite(p).all() or p.sum()<=0:



            return (self.rng.integers(x0,x1),self.rng.integers(y0,y1))



        p/=p.sum(); i=self.rng.choice(p.size,p=p)



        ww=x1-x0



        return x0+(i%ww), y0+(i//ww)







    def color(self,x,y):



        c=self.target[int(y),int(x)].astype(np.float32)



        P=np.asarray(self.cfg.palette,np.float32).T



        q=np.clip(np.linalg.lstsq(P,c,rcond=None)[0]+self.rng.normal(0,.018,5),0,None)



        return mix_palette(q,self.cfg.palette)







    def candidate(self,stage):



        x,y=self.point(stage)



        if stage=="block":



            brush="flat";width=self.cfg.brush_sizes[2];length=self.rng.uniform(width*1.8,width*4)



        elif stage=="form":



            brush="filbert";width=self.cfg.brush_sizes[1];length=self.rng.uniform(width*1.6,width*3.5)



        else:



            brush="round" if self.rng.random()<.72 else "filbert"



            width=self.cfg.brush_sizes[0] if brush=="round" else self.cfg.brush_sizes[1]



            length=self.rng.uniform(width*1.1,width*2.8)



        ang=math.atan2(float(self.gy[y,x]),float(self.gx[y,x]))+math.pi/2+self.rng.normal(0,.18)



        return Stroke(x,y,length,ang,width,self.rng.normal(0,.045),self.rng.uniform(.72,.94),self.color(x,y),brush)







    def feature_candidate(self,name):



        x,y=self.feature_point(name)



        # Small controlled marks for eyes/mouth; somewhat broader for nose/jaw.



        if name in ("left_eye","right_eye","mouth"):



            brush="round" if self.rng.random()<.78 else "filbert"



            width=self.cfg.brush_sizes[0] if brush=="round" else max(self.cfg.brush_sizes[0]+2,self.cfg.brush_sizes[1]-3)



            length=self.rng.uniform(width*1.1,width*3.4)



        elif name=="nose":



            brush="filbert" if self.rng.random()<.68 else "round"



            width=max(self.cfg.brush_sizes[0]+1,self.cfg.brush_sizes[1]-3) if brush=="filbert" else self.cfg.brush_sizes[0]



            length=self.rng.uniform(width*1.4,width*3.8)



        else:



            brush="filbert";width=max(6,self.cfg.brush_sizes[1])



            length=self.rng.uniform(width*1.8,width*4.2)







        # Follow the actual local reference contour, with painterly error.



        ang=math.atan2(float(self.gy[y,x]),float(self.gx[y,x]))+math.pi/2+self.rng.normal(0,.10)



        return Stroke(x,y,length,ang,width,self.rng.normal(0,.028),



                      self.rng.uniform(.72,.94),self.color(x,y),brush)







    def _cheap_score(self,s,loss_map):

        """Fast parameter-only score; no rendering."""

        x=int(np.clip(round(s.x),0,self.w-1))

        y=int(np.clip(round(s.y),0,self.h-1))

        err=float(loss_map[y,x])

        edge=float(self.edge_mag[y,x])

        target=self.target[y,x].astype(np.float32)

        col=np.asarray(s.color,np.float32)

        colour=float(np.mean(((col-target)/255.)**2))

        return colour-.70*err-.12*edge



    def _local_pixel_score(self,painted,bounds):

        x0,y0,x1,y1=bounds

        if painted is None or x1<=x0 or y1<=y0:return 1e9

        target=self.target[y0:y1,x0:x1].astype(np.float32)

        c=painted.astype(np.float32)

        return float(np.mean(((c-target)/255.)**2))



    def _local_feature_score(self,painted,bounds,name):

        """

        Feature finalist score using only the intersection of the stroke crop

        and the current semantic feature region.

        """

        if painted is None:return 1e9

        bx0,by0,bx1,by1=bounds

        fx0,fy0,fx1,fy1=self.info["regions"][name]

        x0=max(bx0,fx0); y0=max(by0,fy0)

        x1=min(bx1,fx1); y1=min(by1,fy1)

        if x1<=x0 or y1<=y0:return 1e9



        pc=painted[y0-by0:y1-by0,x0-bx0:x1-bx0]

        tc=self.target[y0:y1,x0:x1]

        cf=pc.astype(np.float32)/255.

        tf=tc.astype(np.float32)/255.

        pix=float(np.mean((cf-tf)**2))



        cg=cv2.cvtColor(pc,cv2.COLOR_RGB2GRAY)

        ce=cv2.Canny(cg,50,130).astype(np.float32)/255.

        te=self.target_edges[y0:y1,x0:x1]

        edge=float(np.mean(np.abs(ce-te)))

        value=float(np.mean(np.abs(

            cg.astype(np.float32)-self.target_gray[y0:y1,x0:x1].astype(np.float32)

        ))/255.)

        return .48*pix+.34*edge+.18*value



    def _commit_crop(self,painted,bounds):

        if painted is None:return

        x0,y0,x1,y1=bounds

        self.canvas[y0:y1,x0:x1]=painted



    def best(self,stage):

        # Calculate the expensive full loss map ONCE for this accepted mark.

        lm=self.loss_map()

        proposal_n=max(self.cfg.candidates,32)

        full_n=min(8,self.cfg.candidates)



        props=[self.candidate(stage) for _ in range(proposal_n)]

        props.sort(key=lambda s:self._cheap_score(s,lm))



        best_score=1e9

        best_crop=None

        best_bounds=None



        # Only finalists are physically rendered, and only on tiny crops.

        for stroke in props[:full_n]:

            # Independent RNG prevents one finalist from changing another's

            # random brush texture sequence.

            seed=int(self.rng.integers(0,2**32-1))

            painted,bounds=render_stroke_local(

                self.canvas,stroke,np.random.default_rng(seed)

            )

            score=self._local_pixel_score(painted,bounds)

            if score<best_score:

                best_score=score

                best_crop=painted

                best_bounds=bounds



        if best_crop is not None:

            self._commit_crop(best_crop,best_bounds)

        return self.canvas



    def best_feature(self,name):

        # One loss map per accepted feature mark, not one per proposal.

        lm=self.loss_map()

        old_local=self.region_loss(self.canvas,name)



        proposal_n=max(self.cfg.feature_candidates,48)

        full_n=min(10,max(4,self.cfg.feature_candidates//5))

        props=[self.feature_candidate(name) for _ in range(proposal_n)]

        props.sort(key=lambda s:self._cheap_score(s,lm))



        best_score=old_local

        best_crop=None

        best_bounds=None



        for stroke in props[:full_n]:

            seed=int(self.rng.integers(0,2**32-1))

            painted,bounds=render_stroke_local(

                self.canvas,stroke,np.random.default_rng(seed)

            )

            score=self._local_feature_score(painted,bounds,name)

            if score<best_score:

                best_score=score

                best_crop=painted

                best_bounds=bounds



        if best_crop is not None:

            self._commit_crop(best_crop,best_bounds)

        return self.canvas,old_local,best_score



    def precision_candidate(self,name):
        """Very small, contour-led marks used only in the final facial precision pass."""
        x,y=self.feature_point(name)
        base=float(self.cfg.brush_sizes[0])
        if name in ("left_eye","right_eye","mouth"):
            width=max(2.0,base*0.52)
            length=self.rng.uniform(max(2.5,width*.75),max(4.0,width*2.15))
        elif name=="nose":
            width=max(2.0,base*0.62)
            length=self.rng.uniform(width*.9,width*2.5)
        else:
            width=max(2.5,base*0.72)
            length=self.rng.uniform(width*1.0,width*2.8)
        ang=math.atan2(float(self.gy[y,x]),float(self.gx[y,x]))+math.pi/2+self.rng.normal(0,.055)
        return Stroke(x,y,length,ang,width,self.rng.normal(0,.014),
                      self.rng.uniform(.68,.90),self.color(x,y),"round")

    def _precision_feature_score(self,painted,bounds,name):
        """Score a candidate on the WHOLE feature, not just the stroke footprint.
        This keeps the fast local renderer but rewards coherent eyelids, lips,
        nose contours and jaw structure. 2x scoring makes small edge errors matter.
        """
        if painted is None:return 1e9
        fx0,fy0,fx1,fy1=self.info["regions"][name]
        if fx1<=fx0 or fy1<=fy0:return 1e9
        region=self.canvas[fy0:fy1,fx0:fx1].copy()
        bx0,by0,bx1,by1=bounds
        x0=max(bx0,fx0); y0=max(by0,fy0); x1=min(bx1,fx1); y1=min(by1,fy1)
        if x1<=x0 or y1<=y0:return 1e9
        region[y0-fy0:y1-fy0,x0-fx0:x1-fx0]=painted[y0-by0:y1-by0,x0-bx0:x1-bx0]
        target=self.target[fy0:fy1,fx0:fx1]
        # Foveated 2x evaluation: cheap because feature crops are tiny.
        size=(max(2,(fx1-fx0)*2),max(2,(fy1-fy0)*2))
        rc=cv2.resize(region,size,interpolation=cv2.INTER_CUBIC)
        rt=cv2.resize(target,size,interpolation=cv2.INTER_CUBIC)
        rf=rc.astype(np.float32)/255.; tf=rt.astype(np.float32)/255.
        pix=float(np.mean((rf-tf)**2))
        rg=cv2.cvtColor(rc,cv2.COLOR_RGB2GRAY); tg=cv2.cvtColor(rt,cv2.COLOR_RGB2GRAY)
        re=cv2.Canny(rg,42,115).astype(np.float32)/255.
        te=cv2.Canny(tg,42,115).astype(np.float32)/255.
        edge=float(np.mean(np.abs(re-te)))
        value=float(np.mean(np.abs(rg.astype(np.float32)-tg.astype(np.float32)))/255.)
        # Contrast/shape term penalises the washed-out facial features seen in V2.
        rl=cv2.Laplacian(rg,cv2.CV_32F); tl=cv2.Laplacian(tg,cv2.CV_32F)
        structure=float(np.mean(np.abs(rl-tl))/255.)
        return .25*pix+.43*edge+.17*value+.15*structure

    def best_precision(self,name):
        lm=self.loss_map()
        old=self.region_loss(self.canvas,name)
        # More ideas than normal feature painting, but still only render a small finalist set.
        proposal_n=max(56,min(80,int(self.cfg.feature_candidates*1.15)))
        full_n=12
        props=[self.precision_candidate(name) for _ in range(proposal_n)]
        props.sort(key=lambda st:self._cheap_score(st,lm))
        best_score=1e9; best_crop=None; best_bounds=None
        for stroke in props[:full_n]:
            seed=int(self.rng.integers(0,2**32-1))
            painted,bounds=render_stroke_local(self.canvas,stroke,np.random.default_rng(seed))
            score=self._precision_feature_score(painted,bounds,name)
            if score<best_score:
                best_score=score; best_crop=painted; best_bounds=bounds
        if best_crop is not None:self._commit_crop(best_crop,best_bounds)
        return self.canvas,old,best_score

    def precision_pass(self,frame=None):
        """One compact final facial pass (~225 marks) before training is frozen."""
        sequence=[("left_eye",45),("right_eye",45),("nose",50),("mouth",50),("jaw",35)]
        for name,marks in sequence:
            self.current_feature=name
            for _ in range(marks):
                if self.stop_event and self.stop_event.is_set():return
                self.canvas,old,new=self.best_precision(name)
                self.accepted+=1; self.loss_history.append(self.loss())
                if self.progress:self.progress(self,"Precision: "+name.replace("_"," "))
                if frame:frame(self.canvas,"Precision: "+name.replace("_"," ").title())
        self.current_feature=None

    def sketch(self,frame=None):



        lm=self.info["landmarks"];x,y,w,h=self.info["face_box"];pencil=(92,82,72)



        # Construction lines + explicit feature axes.



        lines=[



            ((x+w*.5,y+h*.03),(x+w*.5,y+h*.96)),



            ((lm["left_eye"][0]-w*.14,lm["left_eye"][1]),(lm["right_eye"][0]+w*.14,lm["right_eye"][1])),



            ((x+w*.38,lm["nose"][1]),(x+w*.62,lm["nose"][1])),



            ((x+w*.30,lm["mouth"][1]),(x+w*.70,lm["mouth"][1])),



        ]



        per=max(1,self.cfg.sketch_strokes//8)



        for a,b in lines:



            for _ in range(per):



                if self.stop_event and self.stop_event.is_set():return



                s=Stroke((a[0]+b[0])/2+self.rng.normal(0,1.6),(a[1]+b[1])/2+self.rng.normal(0,1.6),



                         math.dist(a,b)*self.rng.uniform(.55,1),math.atan2(b[1]-a[1],b[0]-a[0])+self.rng.normal(0,.035),



                         2,self.rng.normal(0,.012),.25,pencil,"round")



                self.canvas=render_stroke(self.canvas,s,self.rng);self.accepted+=1



                if frame:frame(self.canvas,"Sketch")







        # Sketch local reference contours for the actual features.



        for name in ("left_eye","right_eye","nose","mouth","jaw"):



            x0,y0,x1,y1=self.info["regions"][name]



            local_edges=self.info["edges"][y0:y1,x0:x1]



            ys,xs=np.where(local_edges>.5)



            if len(xs)==0:continue



            count=max(4,per//2)



            for _ in range(count):



                j=self.rng.integers(0,len(xs)); px=x0+int(xs[j]);py=y0+int(ys[j])



                ang=math.atan2(float(self.gy[py,px]),float(self.gx[py,px]))+math.pi/2+self.rng.normal(0,.04)



                s=Stroke(px,py,self.rng.uniform(8,22),ang,2,self.rng.normal(0,.018),.22,pencil,"round")



                self.canvas=render_stroke(self.canvas,s,self.rng);self.accepted+=1



                if frame:frame(self.canvas,"Sketch: "+name.replace("_"," "))







    def feature_passes(self,frame=None):



        sequence=[



            ("left_eye",self.cfg.eye_marks),("right_eye",self.cfg.eye_marks),



            ("nose",self.cfg.nose_marks),("mouth",self.cfg.mouth_marks),



            ("jaw",self.cfg.jaw_marks)



        ]



        for cycle in range(self.cfg.feature_cycles):



            for name,marks in sequence:



                self.current_feature=name



                for _ in range(marks):



                    if self.stop_event and self.stop_event.is_set():return



                    trial,old,new=self.best_feature(name)



                    self.canvas=trial;self.accepted+=1;self.loss_history.append(self.loss())



                    if self.progress:self.progress(self,"Feature: "+name.replace("_"," "))



                    if frame:



                        frame(self.canvas,f"Feature {cycle+1}/{self.cfg.feature_cycles}: {name.replace('_',' ').title()}")



        self.current_feature=None







    def run(self,frame=None):



        self.sketch(frame)



        # Reserve the old feature portion for explicit sessions instead.



        broad=int(self.cfg.paint_strokes*.55)



        detail=max(1,self.cfg.paint_strokes-broad)



        for i in range(broad):



            if self.stop_event and self.stop_event.is_set():break



            stage="block" if i<int(broad*.36) else "form"



            self.canvas=self.best(stage);self.accepted+=1;self.loss_history.append(self.loss())



            if self.progress:self.progress(self,stage)



            if frame:frame(self.canvas,stage.title())







        self.feature_passes(frame)







        # Final detail returns to global/local error, but facial geometry remains weighted.



        for i in range(detail):



            if self.stop_event and self.stop_event.is_set():break



            self.canvas=self.best("detail");self.accepted+=1;self.loss_history.append(self.loss())



            if self.progress:self.progress(self,"detail")



            if frame:frame(self.canvas,"Detail")



        # Final foveated facial cleanup: small, structure-first marks.
        if not (self.stop_event and self.stop_event.is_set()):
            self.precision_pass(frame)

        return self.canvas
