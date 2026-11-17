from pathlib import Path
import json
import numpy as np
from sklearn.neural_network import MLPRegressor

def fit(output):
    OUT=Path(output)
    read=lambda p:json.loads(Path(p).read_text())
    write=lambda n,d:(OUT/n).write_text(json.dumps(d,indent=2))
    rows=read(OUT/'training.json')
    groups=[]
    for r in rows:
        a=np.load(OUT/r['transition_path'])['measurements']
        keep=a[:-1,0]>.3
        X=a[:-1,1:][keep];Y=np.diff(a[:,1:4],axis=0)[keep]
        groups.append((X,Y))
    # Whole trajectories, not overlapping random time samples, form validation.
    val_ids=[2,9]
    X=np.concatenate([a[0] for i,a in enumerate(groups) if i not in val_ids]).astype(np.float32)
    Y=np.concatenate([a[1] for i,a in enumerate(groups) if i not in val_ids]).astype(np.float32)
    VX=np.concatenate([groups[i][0] for i in val_ids]).astype(np.float32)
    VY=np.concatenate([groups[i][1] for i in val_ids]).astype(np.float32)
    xm=X.mean(0);xs=np.maximum(X.std(0),1e-6)
    ym=Y.mean(0);ys=np.maximum(Y.std(0),1e-6)
    xx=(X-xm)/xs;yy=(Y-ym)/ys;vx=(VX-xm)/xs
    pack=dict(x_mean=xm,x_scale=xs,y_mean=ym,y_scale=ys)
    reports=[];predictions=[]
    for j in range(3):
        nn=MLPRegressor(hidden_layer_sizes=(200,200,200),activation='relu',solver='adam',
            learning_rate_init=.001,alpha=0.,batch_size=1024,random_state=75100+j)
        best=float('inf');history=[];weights=None;patience=0
        for epoch in range(80):
            nn.partial_fit(xx,yy)
            prediction=nn.predict(vx)*ys+ym
            score=float(np.mean(((prediction-VY)/ys)**2))
            history.append(score)
            if score < best:
                best=score;weights=([w.copy() for w in nn.coefs_],[b.copy() for b in nn.intercepts_]);patience=0
            else:patience+=1
            if epoch%10==0:print('network',j,'epoch',epoch,'validation',score,flush=True)
            if patience>=12:break
        nn.coefs_,nn.intercepts_=weights
        predictions.append(nn.predict(vx)*ys+ym)
        for i,(w,b) in enumerate(zip(*weights)):
            pack[f'w{j}_{i}']=w;pack[f'b{j}_{i}']=b
        reports.append(dict(network=j,epochs=len(history),best_standardized_mse=best,history=history))
    pred=np.mean(predictions,axis=0)
    write('model_validation.json',dict(training_samples=len(X),validation_samples=len(VX),
        heldout_training_trajectory_indices=val_ids,networks=reports,
        delta_rmse=np.sqrt(np.mean((pred-VY)**2,axis=0)).tolist(),
        persistence_delta_rmse=np.sqrt(np.mean(VY**2,axis=0)).tolist()))
    np.savez_compressed(OUT/'oracle_ensemble.npz',**pack)
