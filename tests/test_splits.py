from pathlib import Path
import sys, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from ids.data.validate import assert_no_split_overlap

def test_no_overlap():
    a=pd.DataFrame({'x':[1,2]}); b=pd.DataFrame({'x':[3]}); c=pd.DataFrame({'x':[4]})
    assert_no_split_overlap(a,b,c)
