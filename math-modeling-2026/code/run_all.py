"""Run all four problems; official Excel attachments are external inputs."""
import argparse
from common.data import prepare
from Q1.solve import main as q1
from Q2.solve import main as q2
from Q3.solve import main as q3
from Q4.solve import main as q4

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',required=True)
    p.add_argument('--checks',action='store_true',help='Also run sensitivity and N=200 studies')
    p.add_argument('--plots',action='store_true',help='Generate 27 figures with the visualization module')
    a=p.parse_args();prepare(a.data_dir)
    for function in [q1,q2,q3,q4]:function()
    if a.checks:
        from checks import main as checks
        checks()
    from export_results import main as export
    from verify_results import main as verify
    export();verify()
    if a.plots:
        from plot_paper_figures import main as plot
        plot()

if __name__=='__main__':main()
