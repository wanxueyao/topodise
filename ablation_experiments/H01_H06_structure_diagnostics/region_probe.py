"""H06 fixed PCA visualization and group-independent linear region probe."""
from pathlib import Path
import argparse
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from topodise.training.runtime.protocol import write,file_hash
from .probes import split_indices

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv);d=np.load(a.features,allow_pickle=False);tr,va,te=split_indices(d);y=d['region_labels'];X=d['features']
    candidates=[]
    for c in (.01,.1,1.,10.):
        model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=2000)).fit(X[tr],y[tr]);candidates.append((balanced_accuracy_score(y[va],model.predict(X[va])),c,model))
    _,c,model=max(candidates,key=lambda x:(x[0],-x[1]));pca=make_pipeline(StandardScaler(),PCA(2)).fit(X[tr]);coordinates=pca.transform(X)
    a.output.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output/'projection.npz',coordinates=coordinates,region_labels=y,groups=d['groups'],splits=d['splits'])
    write(a.output/'report.json',dict(selected_C=c,test_bacc=balanced_accuracy_score(y[te],model.predict(X[te])),source_sha256=file_hash(a.features),warning='PCA projection is descriptive, not evidence of disease generalization'))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,5))
    for label in np.unique(y):
        ix=y==label;ax.scatter(coordinates[ix,0],coordinates[ix,1],s=15,label=str(label))
    ax.legend(title='region');ax.set_xlabel('train-fitted PC1');ax.set_ylabel('train-fitted PC2');fig.tight_layout();fig.savefig(a.output/'region_pca.png',dpi=200);plt.close(fig)


if __name__=="__main__": main()
