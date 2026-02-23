% ---------------------------
% ProbLog model v2.0 (softened, graded)
% Target: commercial_success_pred(B)
% All probabilistic clauses grounded via block(B)
% ---------------------------

% --- Trap possibility from seismic (graded)
0.75::trap_possible(B) :- block(B), seismic_good(B).
0.55::trap_possible(B) :- block(B), seismic_medium(B).
0.35::trap_possible(B) :- block(B), seismic_poor(B).

% --- Reservoir quality (graded)
0.70::reservoir_ok(B)  :- block(B), reservoir_strong(B).
0.50::reservoir_ok(B)  :- block(B), reservoir_weak(B).
0.25::reservoir_ok(B)  :- block(B), reservoir_none(B).

% --- Seal quality (graded)
0.65::seal_ok(B) :- block(B), seal_strong(B).
0.45::seal_ok(B) :- block(B), seal_weak(B).
0.20::seal_ok(B) :- block(B), seal_none(B).

% --- Dry well penalty as soft factor (not hard negation)
0.80::dry_penalty_ok(B) :- block(B), \+ many_dry_wells(B).
0.35::dry_penalty_ok(B) :- block(B), many_dry_wells(B).

% --- Working petroleum system (WPS) prior (temporary)
% Make it mildly higher in underexplored blocks to avoid over-penalizing frontier
0.50::wps(B) :- block(B), poorly_explored(B).
0.42::wps(B) :- block(B), \+ poorly_explored(B).

% --- Geological discovery: graded sufficiency rules (instead of strict AND only)
% Strong evidence case
0.90::geol_discovery(B) :-
    block(B),
    wps(B),
    trap_possible(B),
    reservoir_ok(B),
    seal_ok(B),
    dry_penalty_ok(B).

% Medium evidence: no explicit seal OR weaker reservoir/structure still can succeed
0.55::geol_discovery(B) :-
    block(B),
    wps(B),
    trap_possible(B),
    reservoir_ok(B),
    dry_penalty_ok(B).

% Weak evidence: WPS + trap only (frontier / sparse data)
0.25::geol_discovery(B) :-
    block(B),
    wps(B),
    trap_possible(B).

% --- Economic viability (keep simple but soften deepwater)
0.70::econ_viable(B) :- block(B), near_field_true(B), \+ deepwater_penalty_true(B).
0.40::econ_viable(B) :- block(B), near_field_true(B), deepwater_penalty_true(B).
0.20::econ_viable(B) :- block(B), \+ near_field_true(B).

% --- Target predicate
commercial_success_pred(B) :-
    block(B),
    geol_discovery(B),
    econ_viable(B).