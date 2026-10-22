"""Logistic membership probabilities with shared stratified-CV selection."""
from logistic_weight import fit_membership_logistic


def propensityscore(target_x, source_x, penalty="l2", C=1.0, max_iter=1000,
                    cv=True, C_grid=None, cv_folds=5, diagnostics=None):
    model = fit_membership_logistic(target_x, source_x, cv=cv, C=C,
                                    penalty=penalty, C_grid=C_grid,
                                    cv_folds=cv_folds, max_iter=max_iter)
    if diagnostics is not None:
        diagnostics.update(model.weight_selection_)
    return (model.predict_proba(target_x)[:, 1],
            model.predict_proba(source_x)[:, 1])
