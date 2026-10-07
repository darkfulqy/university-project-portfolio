from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.solver import run

def main():
    run(1,100,1,1800,tag='q1',record_interval_s=1)

if __name__=='__main__':main()
