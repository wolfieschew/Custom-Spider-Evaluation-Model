# Run Spider Evaluation
python scripts/spider/evaluation.py \
  --gold database/gold_195_formatted.sql \
  --pred output/prediksi_fewshot_smollm2_1.7b_20251207_175347.txt \
  --db data \
  --table scripts/spider/evaluation_examples/examples/tables.json \
  --etype all