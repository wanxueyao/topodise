"""Patient-level frozen probes; ALL preprocessing is refit inside inner folds."""
from pathlib import Path
import argparse,json
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from topodise.training.runtime.protocol import write,file_hash


def evaluate(features,labels,patient_ids,fold_ids,*,seed=42,pca_dim=None,mlp=False):
    X,y=np.asarray(features),np.asarray(labels,dtype=int)
    ids=np.asarray(patient_ids).astype(str);fold_ids=np.atleast_2d(fold_ids)
    if len(set(ids))!=len(ids) or len(X)!=len(y) or fold_ids.shape[1]!=len(y): raise ValueError('Require one vector and label per unique patient')
    if not np.isfinite(X).all(): raise ValueError('Non-finite frozen features')
    classes=np.unique(y)
    if not np.array_equal(classes,np.arange(len(classes))): raise ValueError('Labels must follow the fixed task class order')
    rows=[];searches=[];fold_count=len(np.unique(fold_ids[0]))
    for repeat,assignment in enumerate(fold_ids):
        if set(assignment)!=set(range(1,fold_count+1)): raise ValueError('Incomplete outer fold assignments')
        for fold in range(1,fold_count+1):
            train=np.flatnonzero(assignment!=fold);test=np.flatnonzero(assignment==fold)
            counts=np.bincount(y[train],minlength=len(classes))
            inner_n=min(3,int(counts.min()))
            if inner_n<2: raise ValueError('Too few training patients for nested selection')
            inner=list(StratifiedKFold(inner_n,shuffle=True,random_state=seed+repeat*1000+fold).split(X[train],y[train]))
            steps=[('scale',StandardScaler())]
            if pca_dim is not None:
                cap=min(X.shape[1],min(len(a)-1 for a,b in inner))
                if pca_dim>cap: raise ValueError(f'PCA dimension {pca_dim} exceeds common feasible rank {cap}')
                steps.append(('pca',PCA(pca_dim,svd_solver='full')))
            if mlp:
                steps.append(('head',MLPClassifier(hidden_layer_sizes=(64,),activation='relu',max_iter=1000,early_stopping=False,random_state=seed)))
                grid={'head__alpha':[.01,.1,1.]}
            else:
                steps.append(('head',LogisticRegression(max_iter=2000,solver='lbfgs',random_state=seed)))
                grid={'head__C':[.01,.1,1.,10.,100.]}
            model=GridSearchCV(Pipeline(steps),grid,scoring='balanced_accuracy',cv=inner,refit=True,error_score='raise')
            model.fit(X[train],y[train]);probs=model.predict_proba(X[test])
            searches.append(dict(repeat=repeat,fold=fold,selected=model.best_params_,inner_score=float(model.best_score_),train_patients=ids[train].tolist(),test_patients=ids[test].tolist()))
            for index,p in zip(test,probs): rows.append(dict(patient_id=ids[index],y_true=int(y[index]),y_pred=int(p.argmax()),patient_probs=p.tolist(),repeat=repeat,fold=repeat*fold_count+fold))
    from utils.confidence_intervals import report
    return dict(protocol_id='REP-NESTED-v1',evaluation_unit='LINEAR_PROBE' if not mlp else 'MLP_PROBE',predictions=rows,searches=searches,
                feature_dim=X.shape[1],linear_parameters=(pca_dim or X.shape[1])*len(classes)+len(classes),statistics=report(rows,len(classes),fold_count))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--pca-dim',type=int);p.add_argument('--mlp',action='store_true');a=p.parse_args(argv)
    data=np.load(a.features,allow_pickle=False)
    result=evaluate(data['features'],data['labels'],data['patient_ids'],data['fold_ids'],pca_dim=a.pca_dim,mlp=a.mlp)
    metadata=json.loads(a.features.with_suffix('.json').read_text())
    result.update(feature_sha256=file_hash(a.features),encoder=metadata['backend'],task=metadata['task'],feature_metadata=metadata)
    write(a.output,result)

if __name__=='__main__': main()
