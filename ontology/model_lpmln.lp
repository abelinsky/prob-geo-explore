% ===== LPMLN model (no seismic) for plingo 1.1.0 =====
% We assume evidence.lp contains facts:
% block/1, near_field/1, close_to_field/1, very_close_to_field/1, near_to_field/1,
% few_wells/1, many_wells/1, many_dry_wells/1,
% shallow_water/1, deep_water/1, deepwater_penalty/1, active_area/1.

% optional domain declarations (no semantics change)
#defined very_close_to_field/1.
#defined close_to_field/1.
#defined near_to_field/1.
#defined near_field/1.
#defined many_wells/1.
#defined many_dry_wells/1.
#defined deep_water/1.
#defined deepwater_penalty/1.
#defined few_wells/1.
#defined active_area/1.
#defined shallow_water/1.

% --- Positive evidence ---
success(B) :- very_close_to_field(B), &weight("2.5").
success(B) :- close_to_field(B),      &weight("1.2").
success(B) :- near_to_field(B),       &weight("0.6").

success(B) :- near_field(B),          &weight("0.8").
success(B) :- active_area(B),         &weight("0.8").
success(B) :- many_wells(B),          &weight("0.6").

% frontier prior: few wells is not negative
success(B) :- few_wells(B),           &weight("0.4").

% --- Negative evidence ---
success(B) :- many_dry_wells(B),      &weight("-2.0").
success(B) :- deep_water(B),          &weight("-1.0").
success(B) :- deepwater_penalty(B),   &weight("-0.6").

% Optional: shallow water small plus
success(B) :- shallow_water(B),       &weight("0.3").