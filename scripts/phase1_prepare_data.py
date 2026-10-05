from pathlib import Path
import sys, json, yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ids.data.load_data import load_csv_folder
from ids.data.clean import clean_dataframe
from ids.data.labels import harmonize_labels
from ids.data.split import split_train_val_test
from ids.data.partition import make_iid_partitions,make_mild_non_iid_partitions,make_extreme_non_iid_partitions
from ids.data.validate import assert_no_split_overlap
from ids.logging_utils import setup_logging

def save_parts(base,name,parts):
    d=base/name; d.mkdir(parents=True,exist_ok=True)
    for c,df in parts.items(): df.to_csv(d/f'{c}.csv',index=False)

def main():
    cfg=yaml.safe_load((ROOT/'configs/config.yaml').read_text(encoding='utf-8'))
    log=setup_logging(str(ROOT/cfg['paths']['logs_dir']))
    df=clean_dataframe(load_csv_folder(ROOT/cfg['paths']['raw_dir']))
    label=next((c for c in cfg['dataset']['label_candidates'] if c in df.columns),None)
    if not label: raise KeyError('Label column not found')
    df=harmonize_labels(df,label)
    tr,va,te=split_train_val_test(df,random_seed=cfg['project']['random_seed'])
    assert_no_split_overlap(tr,va,te)
    out=ROOT/cfg['paths']['processed_dir']; out.mkdir(parents=True,exist_ok=True)
    tr.to_csv(out/'train.csv',index=False); va.to_csv(out/'validation.csv',index=False); te.to_csv(out/'test_LOCKED.csv',index=False)
    p=ROOT/cfg['paths']['partitions_dir']; seed=cfg['project']['random_seed']
    iid=make_iid_partitions(tr,seed); mild=make_mild_non_iid_partitions(tr,dominant_fraction=cfg['partitioning']['mild_non_iid']['dominant_fraction'],seed=seed); ext=make_extreme_non_iid_partitions(tr,seed=seed)
    save_parts(p,'iid',iid); save_parts(p,'mild_non_iid',mild); save_parts(p,'extreme_non_iid',ext)
    report={'rows':len(df),'splits':{'train':len(tr),'validation':len(va),'test_locked':len(te)},'partition_sizes':{'iid':{k:len(v) for k,v in iid.items()},'mild_non_iid':{k:len(v) for k,v in mild.items()},'extreme_non_iid':{k:len(v) for k,v in ext.items()}}}
    r=ROOT/cfg['paths']['reports_dir']; r.mkdir(parents=True,exist_ok=True); (r/'phase1_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    log.info('Phase 1 complete. Locked test set created.')
if __name__=='__main__': main()
