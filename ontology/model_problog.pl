% ===== ProbLog model (no seismic) =====

% baseline prior
0.01::commercial_success_pred(B) :- block(B).

% positive evidence (proximity + activity)
0.25::commercial_success_pred(B) :- very_close_to_field(B).
0.12::commercial_success_pred(B) :- close_to_field(B).
0.06::commercial_success_pred(B) :- near_to_field(B).

0.10::commercial_success_pred(B) :- near_field(B).
0.08::commercial_success_pred(B) :- active_area(B).
0.06::commercial_success_pred(B) :- many_wells(B).

% frontier: мало данных не значит плохо (мягкий prior)
0.04::commercial_success_pred(B) :- few_wells(B).

% negative evidence
0.20::risk(B) :- many_dry_wells(B).
0.10::risk(B) :- deep_water(B).
0.06::risk(B) :- deepwater_penalty(B).

% apply risk as inhibitor (simple gate)
commercial_success_pred(B) :- commercial_success_raw(B), \+ risk(B).
commercial_success_raw(B) :- commercial_success_pred(B).