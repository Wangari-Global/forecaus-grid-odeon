.PHONY: setup ingest forecast causal benchmark ingest-ss ingest-ss-real aggregate-ss forecast-ss forecast-ss-agg causal-ss flex flex-run fl-demo fl-train figures-real test all
setup:       ; pip install -e '.[notebooks,fl]'
ingest:      ; python -m forecaus_grid_odeon.cli ingest
forecast:    ; python -m forecaus_grid_odeon.cli forecast
causal:      ; python -m forecaus_grid_odeon.cli causal
benchmark:   ; python -m forecaus_grid_odeon.cli benchmark
ingest-ss:   ; python -m forecaus_grid_odeon.cli ingest-ss
# REAL LV-feeder demand -> data/raw/ss (needs UKPN_API_KEY or the local token file).
ingest-ss-real: ; python -m forecaus_grid_odeon.cli ingest-ss-real
# Roll per-feeder load_kw up to SS-totals per substation -> data/raw/ss_agg.
aggregate-ss: ; python -m forecaus_grid_odeon.cli aggregate-ss
forecast-ss: ; python -m forecaus_grid_odeon.cli forecast-ss
# Day-ahead benchmark on the SS-TOTAL series (Challenge-4 target level).
forecast-ss-agg: ; python -m forecaus_grid_odeon.cli forecast-ss-agg
causal-ss:   ; python -m forecaus_grid_odeon.cli causal-ss
flex:        ; python -m forecaus_grid_odeon.cli flex
flex-run:    ; python -m forecaus_grid_odeon.cli flex-run
fl-demo:     ; python -m forecaus_grid_odeon.cli fl-demo
fl-train:    ; python -m forecaus_grid_odeon.cli fl-train
# REAL-data figures (federated convergence, forecast->flex, edge); needs data/raw/ss.
figures-real: ; python scripts/regenerate_real_figures.py
test:        ; pytest -q
all: ingest forecast causal benchmark ingest-ss forecast-ss causal-ss flex flex-run fl-demo fl-train
