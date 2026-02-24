% ---------------------------
% Derived (threshold) facts v2.0
% Adds medium bands to avoid hard cut-offs
% ---------------------------

% Seismic quality bands
seismic_good(B)   :- seismic_ratio(B,R), R >= 0.50.
seismic_medium(B) :- seismic_ratio(B,R), R >= 0.25, R < 0.50.
seismic_poor(B)   :- seismic_ratio(B,R), R < 0.25.

% Exploration state
poorly_explored(B) :- well_count(B,N), N < 2.
many_dry_wells(B)  :- dry_well_count(B,N), N >= 3.

% Strat/lith evidence bands (if present; defaults 0.0 still ok)
reservoir_strong(B) :- reservoir_score(B,S), S >= 0.30.
reservoir_weak(B)   :- reservoir_score(B,S), S >= 0.10, S < 0.30.
reservoir_none(B)   :- reservoir_score(B,S), S < 0.10.

seal_strong(B) :- seal_score(B,S), S >= 0.20.
seal_weak(B)   :- seal_score(B,S), S >= 0.10, S < 0.20.
seal_none(B)   :- seal_score(B,S), S < 0.10.

% Proximity
near_field_true(B) :- near_field(B,1).

% Deepwater penalty flag
deepwater_penalty_true(B) :- deepwater_penalty(B,1).