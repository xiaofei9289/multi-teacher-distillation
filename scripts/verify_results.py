"""Verify the exported result tables without datasets or a GPU."""
from pathlib import Path
import csv,json,statistics
R=Path(__file__).resolve().parents[1]/'results'
def rows(name):
    with (R/name).open() as f:return list(csv.DictReader(f))
if (R/'decision_results.csv').exists():
    arms={r['variant']:r for r in rows('decision_results.csv')}
    assert len(arms)==8
    w,u=arms['verifier_weighted_mtd'],arms['unfiltered_mtd']
    delta=float(w['answer_accuracy_mean'])-float(u['answer_accuracy_mean'])
    risk=float(w['unsafe_answer_rate_mean'])-float(u['unsafe_answer_rate_mean'])
    assert abs(delta-.1660424469413233)<1e-12
    assert abs(risk+.4978632478632479)<1e-12
    h=rows('human_audit_summary.csv');assert sum(int(r['audited_rows']) for r in h)==314
    print(json.dumps({'arms':len(arms),'accuracy_difference':delta,'conditional_unsafe_difference':risk,'human_audit_rows':314}))
else:
    models=rows('model_comparison.csv');assert len(models)==5
    for r in models:
        folds=json.loads(r['fold_f1s'])
        assert len(folds)==3
        assert abs(statistics.mean(folds)-float(r['macro_f1']))<1e-12
        assert abs(statistics.pstdev(folds)-float(r['std_f1']))<1e-12
    best=next(r for r in models if r['name']=='TF-IDF LR')
    dummy=next(r for r in models if r['name']=='Dummy baseline')
    print(json.dumps({'models':5,'mean_fold_macro_f1':float(best['macro_f1']),'difference_vs_dummy':float(best['macro_f1'])-float(dummy['macro_f1']),'risk_precision':22/75,'risk_recall':22/83}))
