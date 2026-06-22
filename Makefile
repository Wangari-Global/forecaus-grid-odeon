.PHONY: setup ingest forecast causal benchmark ingest-ss flex fl-demo test all
setup:     ; pip install -e '.[notebooks,fl]'
ingest:    ; python -m forecaus_grid_odeon.cli ingest
forecast:  ; python -m forecaus_grid_odeon.cli forecast
causal:    ; python -m forecaus_grid_odeon.cli causal
benchmark: ; python -m forecaus_grid_odeon.cli benchmark
ingest-ss: ; python -m forecaus_grid_odeon.cli ingest-ss
flex:      ; python -m forecaus_grid_odeon.cli flex
fl-demo:   ; python -m forecaus_grid_odeon.cli fl-demo
test:      ; pytest -q
all: ingest forecast causal benchmark flex fl-demo
