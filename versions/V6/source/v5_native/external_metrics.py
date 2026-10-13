"""External-label canonicalization and pinned official task metrics."""
import re


def formula(text,require_outer=False):
    from src.core.preprocess.data_preprocess import strip_formula_delimiters,strip_formula_tags
    value=strip_formula_tags(strip_formula_delimiters(str(text or '').strip()))
    start=re.match(r'\\begin\{([^}]+)\}',value)
    if require_outer and (not start or not value.endswith('\\end{'+start[1]+'}')):
        raise ValueError('Mixed prose or no single complete outer math environment')
    if start:
        env=start[1];end='\\end{'+env+'}'
        if not value.endswith(end):raise ValueError('Incomplete outer math environment')
        body=value[start.end():-len(end)].strip()
        if env in ('equation','equation*','displaymath'):value=body
        elif env in ('align','align*','gather','gather*','multline','multline*'):
            target='gathered' if env.startswith('gather') else 'aligned'
            value='\\begin{'+target+'}'+body+'\\end{'+target+'}'
        elif env not in ('aligned','gathered','split','cases','matrix','pmatrix','bmatrix','Bmatrix','vmatrix','Vmatrix','array'):
            raise ValueError('Unsupported outer math environment: '+env)
    return value


def score(kind,truth,prediction,output_root,identifier):
    if kind=='equation':
        from src.metrics.cdm_metric import CDM
        result=CDM(output_root=str(output_root)).evaluate(truth,formula(prediction),identifier)
        return {'quality':float(result['F1_score']),'quality_valid':'cdm_eval_error' not in result,
                'details':result}
    from src.core.preprocess.data_preprocess import normalized_table
    from src.metrics.table_metric import TEDS
    gt=normalized_table(truth,'html');pred=normalized_table(prediction,'html')
    return {'quality':float(TEDS(n_jobs=1).evaluate(pred,gt)),'quality_valid':True}
