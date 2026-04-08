uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
mkdir -p output
mkdir -p output/saved_configs
mkdir -p output/eval_results