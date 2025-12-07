# Run Spider Evaluation
python scripts/spider/evaluation.py \
  --gold database/gold_195_formatted.sql \
  --pred output/metric_fewshot_llamaindex_deepseek-r1_1.5b_20251207_201406.txt \
  --db data \
  --table scripts/spider/evaluation_examples/examples/tables.json \
  --etype all