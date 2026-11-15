"""Deterministic placement of model predictions; no numerical repair."""
from data import pair_key,validate,reference

def assemble(cells):
    """Only place generated cell values; never fill failures with reference values."""
    groups={}
    for cell in cells:
        key=pair_key(cell['left'],cell['right']);groups.setdefault(key,[]).append(cell)
    matrices=[]
    for values in groups.values():
        left,right=values[0]['left'],values[0]['right'];m,k,n=validate(left,right)
        slots={tuple(v['cell']):v for v in values}
        if len(slots)!=len(values):raise ValueError('Repeated cell in evaluation')
        complete=len(slots)==m*n
        result=[[None]*n for _ in range(m)]
        for (i,j),value in slots.items():
            if value['prediction'] is not None:result[i][j]=value['prediction']['result']
        valid=complete and all(value is not None for row in result for value in row)
        matrices.append(dict(left=left,right=right,prediction=result,complete=complete,valid=valid,
            correct=valid and result==reference(left,right),expected=reference(left,right),
            all_traces_exact=complete and all(v['exact'] for v in values)))
    return matrices

