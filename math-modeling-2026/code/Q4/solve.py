from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.solver import run

def main():
    run(4,100,1,72*3600,tag='q4_mass')
    # Evaluate interface D at the mean of the adjacent C and T values.
    # Keep the existing arithmetic average of nodal k for heat conduction.
    run(5,100,1,72*3600,face_mode=0,heat_face_mode=1,tag='q4_uniform')

if __name__=='__main__':main()
