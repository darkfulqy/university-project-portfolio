"""Parameter sensitivity, grid refinement and end-effect calculations."""
from common.solver import run
from Q1.end_effects import run as end_effects

def main():
    end_effects(100,32,.05)
    for name,key,val in [('hm08','hms',.8),('hm12','hms',1.2),('D08','ds',.8),
        ('D12','ds',1.2),('h08','hs',.8),('h12','hs',1.2),('Tminus2','tshift',-2.),('Tplus2','tshift',2.)]:
        run(2,100,.5,96*3600,tag='sensitivity_'+name,**{key:val})
    run(2,200,.125,72*3600,tag='q3_grid200')
    run(4,200,.25,72*3600,tag='q4_mass_grid200')
    # Use the same interface D and k definitions as Q4/solve.py.
    run(5,200,.25,72*3600,face_mode=0,heat_face_mode=1,tag='q4_uniform_grid200')

if __name__=='__main__':main()
