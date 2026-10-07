from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.solver import run

def main():
    run(2,100,.5,10800,tag='q2',record_interval_s=1)

if __name__=='__main__':main()
