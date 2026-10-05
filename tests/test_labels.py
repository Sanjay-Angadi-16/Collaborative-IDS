from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from ids.data.labels import map_to_family

def test_mapping():
    assert map_to_family('BENIGN')=='Normal'
    assert map_to_family('Web Attack - XSS')=='Web Attack'
    assert map_to_family('PortScan')=='Scan/Probe'
    assert map_to_family('DDoS')=='DDoS/DoS'
    assert map_to_family('Bot')=='Botnet/Malware'
