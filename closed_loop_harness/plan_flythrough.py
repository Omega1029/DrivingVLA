import json, numpy as np, requests, io, os, cv2, time
from PIL import Image
SC="/tmp/claude-1012/-home-justin-williams1-OpenDriveVLA/daebd0c0-5f2e-4b21-9ce2-529f0f6f3334/scratchpad"
ROOT="/home/justin_williams1/neuroncap/neuro-ncap/output/rerecord"
OUT="/home/justin_williams1/OpenDriveVLA/videos"
cam2ego=np.load(f"{SC}/cam2ego.npy",allow_pickle=True).item()
gt=np.load(f"{SC}/gt_start.npy",allow_pickle=True).item()
S=np.array(gt[8])                       # sane road-aligned start ego2world
T0=1533151609048020                     # start timestamp (us)
HORIZON_US=3_000_000                     # 3s plan horizon
CAMS=["CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT"]
NPTS=60; FPS=20
URL="http://localhost:8000/render_image"

def Rz(phi):
    c,s=np.cos(phi),np.sin(phi); M=np.eye(4); M[0,0]=c;M[0,1]=-s;M[1,0]=s;M[1,1]=c; return M

def local_path(plan):
    # planner output p=(lateral, forward). ego-matrix frame: +x=forward, +y=left.
    # map to (X=forward=p[1], Y=lateral=-p[0]). prepend origin.
    pts=np.array([[0.0,0.0]]+[[p[1],-p[0]] for p in plan])   # (X=fwd, Y=lat)
    seg=np.linalg.norm(np.diff(pts,axis=0),axis=1); L=seg.sum()
    if L<0.5:   # frozen: stay put
        return [(np.eye(4),0.0)]*NPTS
    cum=np.concatenate([[0],np.cumsum(seg)])
    out=[]
    for k in range(NPTS):
        d=L*k/(NPTS-1)
        i=np.searchsorted(cum,d)-1; i=max(0,min(i,len(seg)-1))
        f=(d-cum[i])/max(seg[i],1e-6)
        X=pts[i,0]*(1-f)+pts[i+1,0]*f; Y=pts[i,1]*(1-f)+pts[i+1,1]*f
        dX,dY=pts[i+1]-pts[i]; phi=np.arctan2(dY,dX)            # +x -> tangent
        Lm=np.eye(4); Lm[:3,:3]=Rz(phi)[:3,:3]; Lm[0,3]=X; Lm[1,3]=Y
        out.append((Lm, d/L))
    return out

def render(ego2world,timestamp,cam):
    pose=ego2world@cam2ego[cam]
    r=requests.post(URL,json={"pose":pose.tolist(),"timestamp":int(timestamp),
                              "camera_name":cam,"image_format":"png"},timeout=120)
    return cv2.cvtColor(np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB")),cv2.COLOR_RGB2BGR)

def panel(imgs):
    fr=imgs["CAM_FRONT"]; h,w=fr.shape[:2]; TW=1200; fr=cv2.resize(fr,(TW,int(h*TW/w)))
    bw=TW//2
    rs=lambda x:cv2.resize(x,(bw,int(x.shape[0]*bw/x.shape[1])))
    l,r=rs(imgs["CAM_FRONT_LEFT"]),rs(imgs["CAM_FRONT_RIGHT"])
    bh=max(l.shape[0],r.shape[0]); bottom=np.zeros((bh,TW,3),np.uint8)
    bottom[:l.shape[0],:bw]=l; bottom[:r.shape[0],bw:bw+r.shape[1]]=r
    return np.vstack([fr,bottom])

def banner(img,text,color,h=64):
    w=img.shape[1]; bar=np.zeros((h,w,3),np.uint8); cv2.rectangle(bar,(0,0),(w,h),(25,25,25),-1)
    cv2.rectangle(bar,(0,0),(12,h),color,-1)
    cv2.putText(bar,text,(28,int(h*0.64)),cv2.FONT_HERSHEY_SIMPLEX,0.72,(255,255,255),2,cv2.LINE_AA)
    return np.vstack([bar,img])

CONFIGS=[("fp16","FP16 - planned trajectory: drives forward (79% coherent)",(40,180,40)),
         ("naive_w4","naive per-channel W4 - planned trajectory: FROZEN (0% coherent)",(40,40,220)),
         ("group_w4_g128","group-wise W4 g128 - planned trajectory: drives forward (71%) [FIX]",(40,180,40))]
allf={}
for cfg,label,color in CONFIGS:
    d=json.load(open(f"{ROOT}/{cfg}/trajectories.json"))
    plan=d[sorted(d.keys(),key=lambda x:int(x))[0]]
    path=local_path(plan); print(f"[{cfg}] {len(path)} pts, moving={np.linalg.norm(path[-1][0][:2,3])>0.5}",flush=True)
    frames=[]; t0=time.time()
    for Lm,frac in path:
        ego=S@Lm; ts=int(T0+HORIZON_US*frac)
        frames.append(panel({c:render(ego,ts,c) for c in CAMS}))
    print(f"[{cfg}] rendered {len(frames)} in {time.time()-t0:.0f}s",flush=True)
    allf[cfg]=(frames,label,color)
    framed=[banner(f,label,color) for f in frames]; h,w=framed[0].shape[:2]
    p=f"{OUT}/0103_seed0_{cfg}_PLAN.mp4"
    vw=cv2.VideoWriter(p,cv2.VideoWriter_fourcc(*"mp4v"),FPS,(w,h))
    for f in framed: vw.write(f)
    vw.release(); print(f"wrote {os.path.basename(p)} {os.path.getsize(p)//1024}KB",flush=True)
# side-by-side
TH=420
def col(c):
    return [cv2.resize(f,(int(f.shape[1]*TH/f.shape[0]),TH)) for f in allf[c][0]]
cols={c:col(c) for c,_,_ in CONFIGS}; n=min(len(v) for v in cols.values())
short={"fp16":"FP16","naive_w4":"naive W4 (FROZEN)","group_w4_g128":"group W4 (FIX)"}
rows=[np.hstack([banner(cols[c][i],short[c],cl,h=44) for c,_,cl in CONFIGS]) for i in range(n)]
h,w=rows[0].shape[:2]; p=f"{OUT}/0103_seed0_sidebyside_PLAN.mp4"
vw=cv2.VideoWriter(p,cv2.VideoWriter_fourcc(*"mp4v"),FPS,(w,h))
for f in rows: vw.write(f)
vw.release(); print(f"wrote {os.path.basename(p)} {os.path.getsize(p)//1024}KB {w}x{h}",flush=True)
print("ALL DONE",flush=True)
