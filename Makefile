.PHONY: setup ingest forecast causal benchmark ingest-ss forecast-ss causal-ss flex flex-run fl-demo fl-train test all
setup:       ; pip install -e '.[notebooks,fl]'
ingest:      ; python -m forecaus_grid_odeon.cli ingest
forecast:    ; python -m forecaus_grid_odeon.cli forecast
causal:      ; python -m forecaus_grid_odeon.cli causal
benchmark:   ; python -m forecaus_grid_odeon.cli benchmark
ingest-ss:   ; python -m forecaus_grid_odeon.cli ingest-ss
forecast-ss: ; python -m forecaus_grid_odeon.cli forecast-ss
causal-ss:   ; python -m forecaus_grid_odeon.cli causal-ss
flex:        ; python -m forecaus_grid_odeon.cli flex
flex-run:    ; python -m forecaus_grid_odeon.cli flex-run
fl-demo:     ; python -m forecaus_grid_odeon.cli fl-demo
fl-train:    ; python -m forecaus_grid_odeon.cli fl-train
test:        ; pytest -q
all: ingest forecast causal benchmark ingest-ss forecast-ss causal-ss flex flex-run fl-demo fl-train
