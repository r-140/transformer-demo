"""Exact reference calculator for mathematical revision; NOT neural inference."""
import argparse,json
from data import validate,reference_cell,reference

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--left',required=True);p.add_argument('--right',required=True);a=p.parse_args()
    try:left=json.loads(a.left);right=json.loads(a.right);m,k,n=validate(left,right)
    except (ValueError,TypeError) as exc:p.error(str(exc))
    cells=[]
    for i in range(m):
        for j in range(n):
            cells.append(dict(cell=[i,j],row=left[i],column=[right[t][j] for t in range(k)],
                expansion=' + '.join(f'{left[i][t]}*{right[t][j]}' for t in range(k)),**reference_cell(left,right,i,j)))
    print(json.dumps(dict(computation='exact Python reference, not model output',left_shape=[m,k],right_shape=[k,n],
        result_shape=[m,n],cells=cells,result=reference(left,right)),indent=2))
if __name__=='__main__':main()
